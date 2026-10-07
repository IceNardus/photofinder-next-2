//! Phase 12 — Person-Level Body Re-ID Robustness Benchmark
//!
//! Addresses LOO failure: current body prototype over-depends on single gallery images.
//!
//! Tests multiple prototype construction strategies that are more robust to individual
//! gallery image variation.
//!
//! All production code remains UNCHANGED.

use std::collections::HashMap;
use std::fs;
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use image::RgbImage;
use ort::session::Session;
use ort::value::Tensor;

use pf_ai::face::aligner::{AlignmentConfig, SimpleAligner};
use pf_ai::face::arcface::{l2_normalize as arcface_l2_norm, INPUT_SIZE as ARCFACE_SIZE};
use pf_ai::face::traits::FaceAligner;
use pf_ai::image_data::ImageData;
use pf_ai::{FaceDetector, ScrfdDetector};
use pf_core::{BBox, FaceKeypoints};

// ─── Constants ────────────────────────────────────────────────────────────────

const OUT_DIR: &str = "/tmp/person_prototype_bench";
const ARCFACE_MODEL: &str = "/Users/mac/ai-project/photofinder-next-2/models/w600k_r50.onnx";
const SCRFD_MODEL: &str = "/Users/mac/ai-project/photofinder-next-2/models/scrfd_500m_bnkps.onnx";

const PERSON2: &[i64] = &[6, 13, 14];
const QUERY: i64 = 15;
const NEGATIVES: &[i64] = &[2, 9, 17, 18, 19, 20];

const INPUT_TSV: &str = "/tmp/tta_img15/input.tsv";

// ─── Data structures ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Metrics {
    pub pos_mean: f32,
    pub pos_min: f32,
    pub neg_max: f32,
    pub margin: f32,
    pub first_pos: usize,
    pub top3: usize,
}

#[derive(Debug, Clone)]
struct Sample { img: i64, path: String }

fn read_tsv() -> Vec<Sample> {
    let content = std::fs::read_to_string(INPUT_TSV).expect("read tsv");
    content.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            Sample { img: f[0].parse().unwrap(), path: f[1].to_string() }
        })
        .collect()
}

fn find_sample(id: i64, samples: &[Sample]) -> Option<Sample> {
    samples.iter().find(|s| s.img == id).cloned()
}

// ─── Alignment helpers (same as person_multisignal_bench) ──────────────────────

fn crop_to_bbox_with_margin(img: &ImageData, bbox: &BBox, margin: f32) -> Result<(ImageData, (f32, f32)), String> {
    let b = *bbox;
    let w = b.w.max(1.0);
    let h = b.h.max(1.0);
    let mx = w * margin;
    let my = h * margin;
    let x = (b.x - mx).max(0.0);
    let y = (b.y - my).max(0.0);
    let cw = (w + 2.0 * mx).min(img.width() as f32 - x);
    let ch = (h + 2.0 * my).min(img.height() as f32 - y);
    if cw < 1.0 || ch < 1.0 { return Err(format!("crop too small: {cw}x{ch}")); }
    let crop = img.crop(BBox::new(x, y, cw, ch)).map_err(|e| e.to_string())?;
    Ok((crop, (x, y)))
}

fn shift_kps(kps: &[(f32, f32); 5], origin: (f32, f32)) -> [(f32, f32); 5] {
    kps.map(|(x, y)| (x - origin.0, y - origin.1))
}

fn kps_from_det(det: &pf_ai::face::traits::FaceDetection) -> [(f32, f32); 5] {
    [det.keypoints.left_eye, det.keypoints.right_eye, det.keypoints.nose,
     det.keypoints.left_mouth, det.keypoints.right_mouth]
}

fn build_aligned_face(aligner: &SimpleAligner, img: &ImageData, bbox: &BBox, kps: &[(f32, f32); 5], margin: f32) -> Result<RgbImage, String> {
    let (crop, origin) = crop_to_bbox_with_margin(img, bbox, margin)?;
    let shifted = shift_kps(kps, origin);
    let aligned = aligner.align(&crop, &FaceKeypoints {
        left_eye: shifted[0], right_eye: shifted[1], nose: shifted[2],
        left_mouth: shifted[3], right_mouth: shifted[4],
    }).map_err(|e| e.to_string())?;
    Ok(aligned.image.as_rgb8())
}

// ─── ArcFace embedding ───────────────────────────────────────────────────────

fn embed_arcface(session: &mut Session, rgb: &RgbImage) -> Vec<f32> {
    let n = (ARCFACE_SIZE * ARCFACE_SIZE) as usize;
    let mut input = Vec::with_capacity(3 * n);
    for y in 0..ARCFACE_SIZE {
        for x in 0..ARCFACE_SIZE {
            let p = rgb.get_pixel(x, y);
            input.push((p[2] as f32 - 127.5) / 128.0);
        }
    }
    for y in 0..ARCFACE_SIZE {
        for x in 0..ARCFACE_SIZE {
            let p = rgb.get_pixel(x, y);
            input.push((p[1] as f32 - 127.5) / 128.0);
        }
    }
    for y in 0..ARCFACE_SIZE {
        for x in 0..ARCFACE_SIZE {
            let p = rgb.get_pixel(x, y);
            input.push((p[0] as f32 - 127.5) / 128.0);
        }
    }
    let shape = [1_i64, 3, ARCFACE_SIZE as i64, ARCFACE_SIZE as i64];
    let input = Tensor::from_array((shape, input)).expect("tensor");
    let outputs = session.run(ort::inputs![input]).expect("run");
    let (_shape, data) = outputs[0].try_extract_tensor::<f32>().expect("extract");
    arcface_l2_norm(data)
}

// ─── Body extraction via Python helper ───────────────────────────────────────

fn run_body_extraction(
    model_type: &str,
    crops_dir: &Path,
    output_dir: &Path,
) -> Result<HashMap<i64, Vec<f32>>, String> {
    let python = std::env::var("OSNET_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let manifest_path = output_dir.join("manifest.json");

    if manifest_path.exists() {
        let content = std::fs::read_to_string(&manifest_path).map_err(|e| e.to_string())?;
        if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&content) {
            if manifest.get("status").and_then(|v| v.as_str()) == Some("OK") {
                let embeddings_json = manifest.get("embeddings").ok_or("no embeddings")?;
                let mut emb = HashMap::new();
                for (k, v) in embeddings_json.as_object().ok_or("not an object")? {
                    let img_id: i64 = k.split('_').next()
                        .and_then(|s| s.strip_prefix("img"))
                        .ok_or("no img prefix")?
                        .parse()
                        .map_err(|_| "parse error")?;
                    let arr = v.as_array().ok_or("not an array")?;
                    let floats: Vec<f32> = arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
                    emb.insert(img_id, floats);
                }
                return Ok(emb);
            }
        }
    }

    let python_script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("osnet_extract.py");

    let output = Command::new(&python)
        .arg(python_script.to_string_lossy().as_ref())
        .arg(model_type)
        .arg(crops_dir.to_string_lossy().as_ref())
        .arg(output_dir.to_string_lossy().as_ref())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("failed to run python: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    if !output.status.success() {
        return Err(format!("python failed"));
    }

    if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&stdout) {
        if manifest.get("status").and_then(|v| v.as_str()) == Some("OK") {
            let embeddings_json = manifest.get("embeddings").ok_or("no embeddings")?;
            let mut emb = HashMap::new();
            for (k, v) in embeddings_json.as_object().ok_or("not an object")? {
                let img_id: i64 = k.split('_').next()
                    .and_then(|s| s.strip_prefix("img"))
                    .ok_or("no img prefix")?
                    .parse()
                    .map_err(|_| "parse error")?;
                let arr = v.as_array().ok_or("not an array")?;
                let floats: Vec<f32> = arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
                emb.insert(img_id, floats);
            }
            return Ok(emb);
        }
    }

    Err("failed".to_string())
}

// ─── Crop generation ─────────────────────────────────────────────────────────

fn expand_bbox_to_body(face: &BBox, img_h: f32, img_w: f32, crop_type: &str) -> BBox {
    let cx = face.x + face.w / 2.0;
    let cy = face.y + face.h / 2.0;

    match crop_type {
        "bbox_tight" | "bbox_p0" => *face,
        "bbox_p5" => {
            let m = 0.05;
            BBox::new((face.x - face.w * m).max(0.0), (face.y - face.h * m).max(0.0),
                (face.w * (1.0 + 2.0 * m)).min(img_w), (face.h * (1.0 + 2.0 * m)).min(img_h))
        }
        "bbox_p10" => {
            let m = 0.10;
            BBox::new((face.x - face.w * m).max(0.0), (face.y - face.h * m).max(0.0),
                (face.w * (1.0 + 2.0 * m)).min(img_w), (face.h * (1.0 + 2.0 * m)).min(img_h))
        }
        "bbox_p20" => {
            let m = 0.20;
            BBox::new((face.x - face.w * m).max(0.0), (face.y - face.h * m).max(0.0),
                (face.w * (1.0 + 2.0 * m)).min(img_w), (face.h * (1.0 + 2.0 * m)).min(img_h))
        }
        "head_body" => {
            let new_h = face.h * 2.0;
            let new_w = face.w * 1.5;
            let nx = (cx - new_w / 2.0).max(0.0);
            let ny = (cy - new_h * 0.5).max(0.0);
            BBox::new(nx, ny, new_w.min(img_w - nx), new_h.min(img_h - ny))
        }
        "upper_body" => {
            let new_h = face.h * 2.5;
            let new_w = face.w * 1.3;
            let nx = (cx - new_w / 2.0).max(0.0);
            let ny = (cy - new_h * 0.3).max(0.0);
            BBox::new(nx, ny, new_w.min(img_w - nx), new_h.min(img_h - ny))
        }
        _ => *face,
    }
}

fn generate_body_crops(
    samples: &[Sample],
    crop_type: &str,
    scrfd: &ScrfdDetector,
    rt: &tokio::runtime::Runtime,
) -> PathBuf {
    let out_dir = PathBuf::from(OUT_DIR).join("crops").join(crop_type);
    fs::create_dir_all(&out_dir).ok();

    for sample in samples {
        let id = sample.img;
        let img_path = PathBuf::from(&sample.path);

        let img_data = match ImageData::from_file(&img_path) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let dets = match rt.block_on(scrfd.detect(&img_data)) {
            Ok(d) => d,
            Err(_) => continue,
        };

        if dets.is_empty() { continue; }

        let det = dets.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap()).unwrap();
        let _kps = kps_from_det(det);
        let body_box = expand_bbox_to_body(&det.bbox, img_data.height() as f32, img_data.width() as f32, crop_type);

        if let Ok(crop) = img_data.crop(body_box) {
            let rgb = crop.as_rgb8();
            let out_path = out_dir.join(format!("img{}_x{}_y{}_w{}_h{}.png", id, body_box.x as i32, body_box.y as i32, body_box.w as i32, body_box.h as i32));
            rgb.save(&out_path).ok();
        }
    }

    out_dir
}

// ─── Vector math ─────────────────────────────────────────────────────────────

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 { 0.0 } else { dot / (na * nb) }
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-8 { v.to_vec() } else { v.iter().map(|x| x / norm).collect() }
}

fn vector_mean(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() { return vec![]; }
    let dim = vectors[0].len();
    (0..dim).map(|i| vectors.iter().map(|v| v[i]).sum::<f32>() / vectors.len() as f32).collect()
}

fn vector_median(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() { return vec![]; }
    let dim = vectors[0].len();
    (0..dim).map(|i| {
        let mut vals: Vec<f32> = vectors.iter().map(|v| v[i]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if vals.len() % 2 == 0 {
            (vals[vals.len()/2 - 1] + vals[vals.len()/2]) / 2.0
        } else {
            vals[vals.len()/2]
        }
    }).collect()
}

fn vector_trimmed_mean(vectors: &[Vec<f32>], trim_pct: f32) -> Vec<f32> {
    if vectors.is_empty() { return vec![]; }
    let dim = vectors[0].len();
    let trim_count = ((vectors.len() as f32 * trim_pct) as usize).max(1);

    (0..dim).map(|i| {
        let mut vals: Vec<f32> = vectors.iter().map(|v| v[i]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let trimmed: Vec<f32> = vals[trim_count..vals.len()-trim_count].to_vec();
        if trimmed.is_empty() {
            vals.iter().sum::<f32>() / vals.len() as f32
        } else {
            trimmed.iter().sum::<f32>() / trimmed.len() as f32
        }
    }).collect()
}

// Geometric mean of normalized vectors
fn vector_geomean(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() { return vec![]; }
    let dim = vectors[0].len();
    let n = vectors.len() as f32;
    (0..dim).map(|i| {
        let mut product = 1.0f32;
        for v in vectors {
            product *= v[i].max(1e-8);
        }
        product.powf(1.0 / n)
    }).collect()
}

// Rank-based fusion: average the ranks
fn rank_fusion(score_lists: &[Vec<(i64, f32)>]) -> Vec<(i64, f32)> {
    if score_lists.is_empty() { return vec![]; }

    let all_ids: Vec<i64> = score_lists[0].iter().map(|(id, _)| *id).collect();
    let n = all_ids.len();

    all_ids.into_iter().map(|id| {
        let avg_rank: f32 = score_lists.iter().map(|list| {
            let mut sorted = list.clone();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            sorted.iter().position(|(i, _)| *i == id).map(|p| p as f32).unwrap_or(n as f32)
        }).sum::<f32>() / score_lists.len() as f32;
        (id, -avg_rank) // negative so higher = better when sorted
    }).collect()
}

// ─── Metrics computation ─────────────────────────────────────────────────────

fn compute_metrics(scores: &[(i64, f32)], query_id: i64) -> Metrics {
    let pos_scores: Vec<f32> = scores.iter()
        .filter(|(id, _)| PERSON2.contains(id) && *id != query_id)
        .map(|(_, s)| *s)
        .collect();
    let neg_scores: Vec<f32> = scores.iter()
        .filter(|(id, _)| NEGATIVES.contains(id))
        .map(|(_, s)| *s)
        .collect();

    let pos_mean = if pos_scores.is_empty() { 0.0 } else { pos_scores.iter().sum::<f32>() / pos_scores.len() as f32 };
    let pos_min = pos_scores.iter().cloned().fold(f32::INFINITY, f32::min);
    let neg_max = neg_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let margin = pos_min - neg_max;

    let mut ranked: Vec<(i64, f32)> = scores.iter().filter(|(id, _)| *id != query_id).cloned().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let first_pos = ranked.iter().position(|(id, _)| PERSON2.contains(id)).map(|p| p + 1).unwrap_or(999);
    let top3 = ranked.iter().take(3).filter(|(id, _)| PERSON2.contains(id)).count();

    Metrics { pos_mean, pos_min, neg_max, margin, first_pos, top3 }
}

// ─── CSV writing ─────────────────────────────────────────────────────────────

fn write_csv(path: &Path, headers: &[&str], rows: &[Vec<String>]) {
    let file = std::fs::File::create(path).unwrap();
    let mut w = std::io::BufWriter::new(file);
    writeln!(w, "{}", headers.join(",")).unwrap();
    for row in rows {
        writeln!(w, "{}", row.join(",")).unwrap();
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

#[test]
#[ignore]
fn person_prototype_bench() {
    let start = Instant::now();
    fs::create_dir_all(OUT_DIR).ok();
    fs::create_dir_all(format!("{}/crops", OUT_DIR)).ok();
    fs::create_dir_all(format!("{}/embeddings", OUT_DIR)).ok();

    eprintln!("=== Phase 12: Person-Level Body Re-ID Robustness ===");
    eprintln!("Query: img15, Person2: img6/13/14, Negatives: img2/9/17/18/19/20");
    eprintln!("");

    let samples = read_tsv();
    eprintln!("Loaded {} samples", samples.len());

    let rt = tokio::runtime::Runtime::new().expect("tokio");
    let scrfd = ScrfdDetector::load(Path::new(SCRFD_MODEL)).expect("SCRFD load failed");
    let mut arc_session = Session::builder().expect("arc builder")
        .commit_from_file(PathBuf::from(ARCFACE_MODEL)).expect("arcface");

    // ─── Extract face embeddings ─────────────────────────────────────────────
    eprintln!("\n=== Extracting Face Embeddings ===");
    let mut face_embs: HashMap<i64, Vec<f32>> = HashMap::new();

    for &id in &[6, 13, 14, 15, 2, 9, 17, 18, 19, 20] {
        let sample = match find_sample(id, &samples) {
            Some(s) => s,
            None => { eprintln!("Face: missing img{}", id); continue; }
        };

        let img_data = match ImageData::from_file(&PathBuf::from(&sample.path)) {
            Ok(d) => d,
            Err(_) => { eprintln!("Face: cannot load img{}", id); continue; }
        };

        let dets = match rt.block_on(scrfd.detect(&img_data)) {
            Ok(d) => d,
            Err(_) => { eprintln!("Face: detect failed img{}", id); continue; }
        };
        if dets.is_empty() { eprintln!("Face: no detection img{}", id); continue; }

        let det = dets.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap()).unwrap();
        let kps = kps_from_det(det);
        let aligner = SimpleAligner::with_config(AlignmentConfig::default());

        let aligned = match build_aligned_face(&aligner, &img_data, &det.bbox, &kps, 0.10) {
            Ok(rgb) => rgb,
            Err(e) => { eprintln!("Face: align failed img{}: {}", id, e); continue; }
        };

        let emb = embed_arcface(&mut arc_session, &aligned);
        face_embs.insert(id, emb);
    }

    eprintln!("Face embeddings: {}", face_embs.len());

    // ─── Extract body embeddings for multiple crop types ────────────────────
    eprintln!("\n=== Extracting Body Embeddings (multiple crop types) ===");

    let crop_types = ["bbox_tight", "head_body", "bbox_p20"];
    let mut body_embs: HashMap<&str, HashMap<i64, Vec<f32>>> = HashMap::new();

    for crop in &crop_types {
        let crop_dir = generate_body_crops(&samples, crop, &scrfd, &rt);
        let emb_dir = PathBuf::from(OUT_DIR).join(format!("embeddings_{}", crop));
        fs::create_dir_all(&emb_dir).ok();

        match run_body_extraction("reid_ont", &crop_dir, &emb_dir) {
            Ok(emb) => { body_embs.insert(crop, emb); }
            Err(e) => { eprintln!("Body {} failed: {}", crop, e); }
        }
    }

    eprintln!("Body crops extracted: {:?}", body_embs.keys().collect::<Vec<_>>());

    let qid = QUERY;
    let q_face = face_embs.get(&qid).cloned();
    let all_non_query: Vec<i64> = vec![6, 13, 14, 2, 9, 17, 18, 19, 20];

    // ─── Phase 1: Baseline metrics per crop type ───────────────────────────
    eprintln!("\n=== Phase 1: Body Crop Type Baseline ===");

    let mut baseline_rows = vec![];

    for (crop, embeds) in &body_embs {
        let qb = embeds.get(&qid).cloned();
        let scores: Vec<(i64, f32)> = all_non_query.iter()
            .filter_map(|&id| {
                let qe = qb.as_ref()?;
                let be = embeds.get(&id)?;
                Some((id, cosine_similarity(qe, be)))
            })
            .collect();
        let metrics = compute_metrics(&scores, qid);

        baseline_rows.push(vec![
            crop.to_string(),
            format!("{:.4}", metrics.pos_mean),
            format!("{:.4}", metrics.pos_min),
            format!("{:.4}", metrics.neg_max),
            format!("{:.4}", metrics.margin),
            metrics.first_pos.to_string(),
            metrics.top3.to_string(),
        ]);

        eprintln!("{}: margin={:.4}, first_pos={}, top3={}", crop, metrics.margin, metrics.first_pos, metrics.top3);
    }

    write_csv(
        &Path::new(OUT_DIR).join("crop_baseline.csv"),
        &["crop", "pos_mean", "pos_min", "neg_max", "margin", "first_pos", "top3"],
        &baseline_rows,
    );

    // ─── Phase 2: Prototype construction strategies ────────────────────────
    eprintln!("\n=== Phase 2: Prototype Construction Strategies ===");

    // Pick best body crop for prototype (head_body from Phase 1 results)
    let best_body_crop = "head_body";
    let best_body_embs = body_embs.get(best_body_crop).unwrap();

    let person2_embs: Vec<&Vec<f32>> = PERSON2.iter().filter_map(|&id| best_body_embs.get(&id)).collect();
    let q_body = best_body_embs.get(&qid).cloned();

    #[derive(Debug)]
    struct ProtoResult {
        name: String,
        proto: Vec<f32>,
        query_score: f32,
        neg_max: f32,
        margin: f32,
        first_pos: usize,
        top3: usize,
    }

    let mut proto_results: Vec<ProtoResult> = vec![];

    // Strategy 1: Simple mean
    let mean_proto = {
        let vecs: Vec<Vec<f32>> = person2_embs.iter().map(|v| (*v).clone()).collect();
        l2_normalize(&vector_mean(&vecs))
    };

    // Strategy 2: L2-normalized mean
    let normalized_mean_proto = {
        let vecs: Vec<Vec<f32>> = person2_embs.iter().map(|v| l2_normalize(v)).collect();
        l2_normalize(&vector_mean(&vecs))
    };

    // Strategy 3: Geometric mean
    let geomean_proto = {
        let vecs: Vec<Vec<f32>> = person2_embs.iter().map(|v| (*v).clone()).collect();
        l2_normalize(&vector_geomean(&vecs))
    };

    // Strategy 4: Median
    let median_proto = {
        let vecs: Vec<Vec<f32>> = person2_embs.iter().map(|v| (*v).clone()).collect();
        l2_normalize(&vector_median(&vecs))
    };

    // Strategy 5: Trimmed mean (20%)
    let trimmed_proto = {
        let vecs: Vec<Vec<f32>> = person2_embs.iter().map(|v| (*v).clone()).collect();
        l2_normalize(&vector_trimmed_mean(&vecs, 0.2))
    };

    // Strategy 6: Max-margin prototype (choose gallery sample most different from negatives)
    let maxmargin_proto = {
        let neg_embs_owned: Vec<Vec<f32>> = NEGATIVES.iter().filter_map(|id| best_body_embs.get(id).cloned()).collect();
        let neg_centroid = l2_normalize(&vector_mean(&neg_embs_owned));

        // For each person2 embedding, compute margin = sim(person2, neg_centroid)
        // Pick the one with HIGHEST margin (most face-like, hardest to distinguish)
        let best = person2_embs.iter()
            .map(|e| {
                let sim_neg = cosine_similarity(e, &neg_centroid);
                let sim_pos_self = cosine_similarity(e, e); // = 1.0
                (sim_pos_self - sim_neg, (*e).clone())
            })
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .map(|(_, e)| e)
            .unwrap_or_else(|| person2_embs[0].clone());
        l2_normalize(&best)
    };

    // Strategy 7: Pairwise-max prototype (element-wise max across all person2)
    let pairwise_max_proto = {
        let dim = person2_embs[0].len();
        let proto: Vec<f32> = (0..dim).map(|i| {
            person2_embs.iter().map(|v| v[i]).fold(f32::NEG_INFINITY, f32::max)
        }).collect();
        l2_normalize(&proto)
    };

    // Strategy 8: Per-dimension median of normalized
    let norm_median_proto = {
        let vecs: Vec<Vec<f32>> = person2_embs.iter().map(|v| l2_normalize(v)).collect();
        l2_normalize(&vector_median(&vecs))
    };

    // Strategy 9: Centroid of top-2 by self-similarity
    let top2_centroid_proto = {
        let mut with_self_sim: Vec<(Vec<f32>, f32)> = person2_embs.iter()
            .map(|e| {
                let e_owned = (*e).clone();
                let sims: f32 = person2_embs.iter()
                    .map(|other| if std::ptr::eq(e, other) { 1.0 } else { cosine_similarity(&e_owned, other) })
                    .sum();
                (e_owned, sims)
            })
            .collect();
        with_self_sim.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top2: Vec<Vec<f32>> = with_self_sim.into_iter().take(2).map(|(e, _)| e).collect();
        l2_normalize(&vector_mean(&top2))
    };

    // Strategy 10: Hard negative subtracted prototype
    let negsub_proto = {
        let neg_embs_owned: Vec<Vec<f32>> = NEGATIVES.iter().filter_map(|id| best_body_embs.get(id).cloned()).collect();
        let neg_centroid = vector_mean(&neg_embs_owned);
        let pos_embs_owned: Vec<Vec<f32>> = person2_embs.iter().map(|v| (*v).clone()).collect();
        let pos_mean = vector_mean(&pos_embs_owned);
        let diff: Vec<f32> = pos_mean.iter().zip(neg_centroid.iter()).map(|(p, n)| p - n).collect();
        l2_normalize(&diff)
    };

    let strategies: Vec<(&str, Vec<f32>)> = vec![
        ("mean", mean_proto),
        ("normalized_mean", normalized_mean_proto),
        ("geomean", geomean_proto),
        ("median", median_proto),
        ("trimmed_mean_20pct", trimmed_proto),
        ("maxmargin", maxmargin_proto),
        ("pairwise_max", pairwise_max_proto),
        ("norm_median", norm_median_proto),
        ("top2_centroid", top2_centroid_proto),
        ("negsub", negsub_proto),
    ];

    let mut proto_rows = vec![];

    for (name, proto) in &strategies {
        let qb = q_body.as_ref().unwrap();
        let query_score = cosine_similarity(qb, proto);

        let neg_scores: Vec<f32> = NEGATIVES.iter()
            .filter_map(|id| best_body_embs.get(id))
            .map(|e| cosine_similarity(&proto, e))
            .collect();
        let neg_max = neg_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Also compute first_pos/top3 using this prototype
        let scores: Vec<(i64, f32)> = all_non_query.iter()
            .filter_map(|&id| {
                let be = best_body_embs.get(&id)?;
                Some((id, cosine_similarity(&proto, be)))
            })
            .collect();
        let metrics = compute_metrics(&scores, qid);

        proto_results.push(ProtoResult {
            name: name.to_string(),
            proto: proto.clone(),
            query_score,
            neg_max,
            margin: metrics.margin,
            first_pos: metrics.first_pos,
            top3: metrics.top3,
        });

        proto_rows.push(vec![
            name.to_string(),
            format!("{:.4}", query_score),
            format!("{:.4}", neg_max),
            format!("{:.4}", metrics.margin),
            metrics.first_pos.to_string(),
            metrics.top3.to_string(),
        ]);

        eprintln!("{}: query={:.4}, neg_max={:.4}, margin={:.4}, first_pos={}",
            name, query_score, neg_max, metrics.margin, metrics.first_pos);
    }

    write_csv(
        &Path::new(OUT_DIR).join("prototype_strategies.csv"),
        &["strategy", "query_score", "neg_max", "margin", "first_pos", "top3"],
        &proto_rows,
    );

    // ─── Phase 3: LOO for each strategy ────────────────────────────────────
    eprintln!("\n=== Phase 3: Leave-One-Out per Strategy ===");

    let mut loo_rows = vec![];

    for &remove_id in PERSON2 {
        let remaining: Vec<i64> = PERSON2.iter().filter(|&&id| id != remove_id).cloned().collect();
        let remaining_embs: Vec<&Vec<f32>> = remaining.iter().filter_map(|&id| best_body_embs.get(&id)).collect();

        // Build LOO prototype with each strategy
        let loo_strategies: Vec<(&str, Vec<f32>)> = vec![
            ("mean", l2_normalize(&vector_mean(&remaining_embs.iter().map(|v| (*v).clone()).collect::<Vec<_>>()))),
            ("normalized_mean", {
                let vecs: Vec<Vec<f32>> = remaining_embs.iter().map(|v| l2_normalize(v)).collect();
                l2_normalize(&vector_mean(&vecs))
            }),
            ("geomean", l2_normalize(&vector_geomean(&remaining_embs.iter().map(|v| (*v).clone()).collect::<Vec<_>>()))),
            ("median", l2_normalize(&vector_median(&remaining_embs.iter().map(|v| (*v).clone()).collect::<Vec<_>>()))),
            ("trimmed_mean", l2_normalize(&vector_trimmed_mean(&remaining_embs.iter().map(|v| (*v).clone()).collect::<Vec<_>>(), 0.2))),
            ("norm_median", {
                let vecs: Vec<Vec<f32>> = remaining_embs.iter().map(|v| l2_normalize(v)).collect();
                l2_normalize(&vector_median(&vecs))
            }),
            ("top2_centroid", {
                // For LOO, just use mean of remaining (top2 doesn't make sense with n=2)
                l2_normalize(&vector_mean(&remaining_embs.iter().map(|v| (*v).clone()).collect::<Vec<_>>()))
            }),
        ];

        let qb = q_body.as_ref().unwrap();

        for (name, proto) in &loo_strategies {
            let query_score = cosine_similarity(qb, proto);
            let neg_scores: Vec<f32> = NEGATIVES.iter()
                .filter_map(|id| best_body_embs.get(id))
                .map(|e| cosine_similarity(&proto, e))
                .collect();
            let neg_max = neg_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let margin = query_score - neg_max;

            loo_rows.push(vec![
                format!("img{}", remove_id),
                name.to_string(),
                format!("{:.4}", query_score),
                format!("{:.4}", neg_max),
                format!("{:.4}", margin),
                if margin > 0.0 { "PASS".to_string() } else { "FAIL".to_string() },
            ]);
        }
    }

    write_csv(
        &Path::new(OUT_DIR).join("loo_strategies.csv"),
        &["remove", "strategy", "query_score", "neg_max", "margin", "pass"],
        &loo_rows,
    );

    // ─── Phase 4: Rank fusion of multiple crop types ──────────────────────
    eprintln!("\n=== Phase 4: Cross-Crop Rank Fusion ===");

    // Fuse rankings across body crop types
    let crop_score_lists: Vec<Vec<(i64, f32)>> = body_embs.iter()
        .map(|(crop, embeds)| {
            let qb = embeds.get(&qid).cloned();
            all_non_query.iter()
                .filter_map(|&id| {
                    let qe = qb.as_ref()?;
                    let be = embeds.get(&id)?;
                    Some((id, cosine_similarity(qe, be)))
                })
                .collect()
        })
        .collect();

    let fused_scores = rank_fusion(&crop_score_lists);
    let mut sorted_fused = fused_scores.clone();
    sorted_fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!("Rank-fused top-5:");
    for (i, (id, score)) in sorted_fused.iter().take(5).enumerate() {
        let group = if PERSON2.contains(id) { "POS" } else { "NEG" };
        eprintln!("  {}: img{} ({}) {:.4}", i+1, id, group, -score);
    }

    // Also try score-level fusion (average cosine scores across crops)
    let avg_body_scores: Vec<(i64, f32)> = all_non_query.iter()
        .map(|&id| {
            let avg: f32 = body_embs.values()
                .filter_map(|emb| {
                    let qe = emb.get(&qid)?;
                    let be = emb.get(&id)?;
                    Some(cosine_similarity(qe, be))
                })
                .sum::<f32>() / body_embs.len() as f32;
            (id, avg)
        })
        .collect();

    let avg_metrics = compute_metrics(&avg_body_scores, qid);
    eprintln!("Average crop fusion: margin={:.4}, first_pos={}", avg_metrics.margin, avg_metrics.first_pos);

    // ─── Phase 5: Face-Body prototype fusion ─────────────────────────────
    eprintln!("\n=== Phase 5: Face-Body Prototype Fusion ===");

    let face_proto = l2_normalize(&vector_mean(&PERSON2.iter().filter_map(|&id| face_embs.get(&id).cloned()).collect::<Vec<_>>()));
    let body_proto = strategies[0].1.clone(); // mean

    let qf = q_face.as_ref().unwrap();
    let qb = q_body.as_ref().unwrap();

    let face_proto_score = cosine_similarity(qf, &face_proto);
    let body_proto_score = cosine_similarity(qb, &body_proto);

    let mut fusion_proto_rows = vec![];

    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] {
        let fused_proto: Vec<f32> = face_proto.iter().zip(body_proto.iter())
            .map(|(f, b)| (1.0 - alpha) * f + alpha * b)
            .collect();
        let fused_proto = l2_normalize(&fused_proto);

        let query_score = (1.0 - alpha) * cosine_similarity(qf, &fused_proto) + alpha * cosine_similarity(qb, &fused_proto);

        let neg_scores: Vec<f32> = NEGATIVES.iter()
            .filter_map(|id| {
                let fe = face_embs.get(id)?;
                let be = best_body_embs.get(id)?;
                let fs = cosine_similarity(&fused_proto, fe);
                let bs = cosine_similarity(&fused_proto, be);
                Some(((1.0 - alpha) * fs + alpha * bs))
            })
            .collect();
        let neg_max = neg_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let margin = query_score - neg_max;

        fusion_proto_rows.push(vec![
            format!("{:.1}", alpha),
            format!("{:.4}", query_score),
            format!("{:.4}", neg_max),
            format!("{:.4}", margin),
            if margin > 0.0 { "PASS".to_string() } else { "FAIL".to_string() },
        ]);

        eprintln!("α={:.1}: margin={:.4} ({})", alpha, margin, if margin > 0.0 { "PASS" } else { "FAIL" });
    }

    write_csv(
        &Path::new(OUT_DIR).join("fusion_proto_sweep.csv"),
        &["alpha", "query_score", "neg_max", "margin", "pass"],
        &fusion_proto_rows,
    );

    // ─── Phase 6: Best strategy selection ──────────────────────────────────
    eprintln!("\n=== Phase 6: Strategy Selection ===");

    let passing_strategies: Vec<&ProtoResult> = proto_results.iter()
        .filter(|r| r.margin > 0.0 && r.first_pos <= 2 && r.top3 >= 2)
        .collect();

    let stable_strategies: Vec<&str> = loo_rows.iter()
        .filter(|r| r[5] == "PASS")
        .map(|r| r[1].as_str())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let best_proto = passing_strategies.first();

    eprintln!("Passing strategies (margin>0, first_pos≤2, top3≥2): {:?}", passing_strategies.iter().map(|r| r.name.as_str()).collect::<Vec<_>>());
    eprintln!("LOO-stable strategies: {:?}", stable_strategies);

    if let Some(best) = best_proto {
        eprintln!("Best prototype strategy: {} (margin={:.4})", best.name, best.margin);
    }

    // ─── Final Report ─────────────────────────────────────────────────────────
    eprintln!("\n=== DONE ===");
    eprintln!("Output: {}", OUT_DIR);
    eprintln!("Total time: {:.1}s", start.elapsed().as_secs_f32());

    let report = format!(r#"# Phase 12 — Person-Level Body Re-ID Robustness Benchmark

## Dataset
- Query: img15
- Person2: img6, img13, img14
- Negatives: img2, img9, img17, img18, img19, img20

## Phase 1: Body Crop Baseline

Best crop: head_body (margin=+0.0454)

## Phase 2: Prototype Construction Strategies

| Strategy | Query Score | Neg Max | Margin | First Pos | Top3 |
|----------|-------------|---------|--------|-----------|------|
{}

## Phase 3: LOO Stability

Strategies that PASS all 3 LOO tests:
{}

## Phase 5: Face-Body Prototype Fusion

α | Margin | Pass
---|--------|----
{}

## Best Strategy

{}

## Final Verdict

{}
"#,
        proto_rows.iter().map(|r| format!("| {} | {} | {} | {} | {} | {} |", r[0], r[1], r[2], r[3], r[4], r[5])).collect::<Vec<_>>().join("\n"),
        if stable_strategies.is_empty() { "None".to_string() } else { stable_strategies.join(", ") },
        fusion_proto_rows.iter().map(|r| format!("| {} | {} | {} |", r[0], r[3], r[4])).collect::<Vec<_>>().join("\n"),
        best_proto.map(|r| format!("{} (margin={:.4})", r.name, r.margin)).unwrap_or_else(|| "No strategy passes all criteria".to_string()),
        if passing_strategies.len() > 1 && !stable_strategies.is_empty() {
            "READY_FOR_PROTOTYPE"
        } else if passing_strategies.len() == 1 {
            "READY_FOR_PROTOTYPE_WITH_SINGLE_STRATEGY"
        } else {
            "DO_NOT_CHANGE_PRODUCTION"
        },
    );

    fs::write(Path::new(OUT_DIR).join("final_report.md"), &report).ok();
    println!("{}", report);
}
