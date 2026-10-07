//! Multi-Signal Person Identity Benchmark v2
//!
//! Validates ArcFace Face + YouTu Re-ID Body fusion for img15 hard case.
//! All production code remains UNCHANGED.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write as IoWrite};
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

const OUT_DIR: &str = "/tmp/person_multisignal";
const IMG_DIR: &str = "/tmp/person_reid_osnet_bench/images";

const ARCFACE_MODEL: &str = "/Users/mac/ai-project/photofinder-next-2/models/w600k_r50.onnx";
const SCRFD_MODEL: &str = "/Users/mac/ai-project/photofinder-next-2/models/scrfd_500m_bnkps.onnx";

const PERSON2: &[i64] = &[6, 13, 14];
const QUERY: i64 = 15;
const NEGATIVES: &[i64] = &[2, 9, 17, 18, 19, 20];

fn all_ids() -> Vec<i64> {
    let mut ids = vec![];
    ids.extend_from_slice(PERSON2);
    ids.push(QUERY);
    ids.extend_from_slice(NEGATIVES);
    ids
}

fn group_of(id: i64) -> &'static str {
    if PERSON2.contains(&id) { "POS" }
    else if id == QUERY { "QRY" }
    else { "NEG" }
}

fn is_positive(id: i64) -> bool { PERSON2.contains(&id) }
fn is_negative(id: i64) -> bool { NEGATIVES.contains(&id) }
fn is_query(id: i64) -> bool { id == QUERY }

// ─── Data structures ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Metrics {
    pub pos_mean: f32,
    pub pos_min: f32,
    pub pos_max: f32,
    pub neg_max: f32,
    pub neg_mean: f32,
    pub margin: f32,
    pub first_pos: usize,
    pub top3: usize,
    pub top5: usize,
}

// ─── Image loading ───────────────────────────────────────────────────────────

const INPUT_TSV: &str = "/tmp/tta_img15/input.tsv";

#[derive(Debug)]
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

fn find_image(id: i64, samples: &[Sample]) -> Option<PathBuf> {
    samples.iter().find(|s| s.img == id).map(|s| PathBuf::from(&s.path))
}

fn ensure_images() -> PathBuf {
    PathBuf::from("/tmp/tta_img15")
}

// ─── Alignment helpers ───────────────────────────────────────────────────────

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

    // Check cache
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
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("python failed: {}", stderr));
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
        if manifest.get("status").and_then(|v| v.as_str()) == Some("MODEL_NOT_AVAILABLE") {
            return Err("MODEL_NOT_AVAILABLE".to_string());
        }
    }

    Err(format!("failed to parse manifest: {}", stdout))
}

// ─── Crop generation ─────────────────────────────────────────────────────────

fn expand_bbox_to_body(face: &BBox, img_h: f32, img_w: f32, crop_type: &str) -> BBox {
    let cx = face.x + face.w / 2.0;
    let cy = face.y + face.h / 2.0;

    match crop_type {
        "bbox_tight" | "bbox_p0" => *face,
        "bbox_p5" => {
            let m = 0.05;
            BBox::new(
                (face.x - face.w * m).max(0.0),
                (face.y - face.h * m).max(0.0),
                (face.w * (1.0 + 2.0 * m)).min(img_w),
                (face.h * (1.0 + 2.0 * m)).min(img_h),
            )
        }
        "bbox_p10" => {
            let m = 0.10;
            BBox::new(
                (face.x - face.w * m).max(0.0),
                (face.y - face.h * m).max(0.0),
                (face.w * (1.0 + 2.0 * m)).min(img_w),
                (face.h * (1.0 + 2.0 * m)).min(img_h),
            )
        }
        "bbox_p15" => {
            let m = 0.15;
            BBox::new(
                (face.x - face.w * m).max(0.0),
                (face.y - face.h * m).max(0.0),
                (face.w * (1.0 + 2.0 * m)).min(img_w),
                (face.h * (1.0 + 2.0 * m)).min(img_h),
            )
        }
        "bbox_p20" => {
            let m = 0.20;
            BBox::new(
                (face.x - face.w * m).max(0.0),
                (face.y - face.h * m).max(0.0),
                (face.w * (1.0 + 2.0 * m)).min(img_w),
                (face.h * (1.0 + 2.0 * m)).min(img_h),
            )
        }
        "bbox_p25" => {
            let m = 0.25;
            BBox::new(
                (face.x - face.w * m).max(0.0),
                (face.y - face.h * m).max(0.0),
                (face.w * (1.0 + 2.0 * m)).min(img_w),
                (face.h * (1.0 + 2.0 * m)).min(img_h),
            )
        }
        "bbox_p30" => {
            let m = 0.30;
            BBox::new(
                (face.x - face.w * m).max(0.0),
                (face.y - face.h * m).max(0.0),
                (face.w * (1.0 + 2.0 * m)).min(img_w),
                (face.h * (1.0 + 2.0 * m)).min(img_h),
            )
        }
        "bbox_p40" => {
            let m = 0.40;
            BBox::new(
                (face.x - face.w * m).max(0.0),
                (face.y - face.h * m).max(0.0),
                (face.w * (1.0 + 2.0 * m)).min(img_w),
                (face.h * (1.0 + 2.0 * m)).min(img_h),
            )
        }
        "upper_body" => {
            let new_h = face.h * 2.5;
            let new_w = face.w * 1.3;
            let nx = (cx - new_w / 2.0).max(0.0);
            let ny = (cy - new_h * 0.3).max(0.0);
            BBox::new(nx, ny, new_w.min(img_w - nx), new_h.min(img_h - ny))
        }
        "full_body" => {
            let new_h = face.h * 4.0;
            let new_w = face.w * 1.2;
            let nx = (cx - new_w / 2.0).max(0.0);
            let ny = (cy - new_h * 0.15).max(0.0);
            BBox::new(nx, ny, new_w.min(img_w - nx), new_h.min(img_h - ny))
        }
        "head_body" => {
            let new_h = face.h * 2.0;
            let new_w = face.w * 1.5;
            let nx = (cx - new_w / 2.0).max(0.0);
            let ny = (cy - new_h * 0.5).max(0.0);
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

        // Crop the body region (no alignment needed for Re-ID)
        if let Ok(crop) = img_data.crop(body_box) {
            let rgb = crop.as_rgb8();
            let out_path = out_dir.join(format!("img{}_x{}_y{}_w{}_h{}.png", id, body_box.x as i32, body_box.y as i32, body_box.w as i32, body_box.h as i32));
            rgb.save(&out_path).ok();
        }
    }

    out_dir
}

// ─── Metrics computation ─────────────────────────────────────────────────────

fn compute_metrics(scores: &[(i64, f32)], query_id: i64) -> Metrics {
    let pos_scores: Vec<f32> = scores.iter()
        .filter(|(id, _)| is_positive(*id) && *id != query_id)
        .map(|(_, s)| *s)
        .collect();
    let neg_scores: Vec<f32> = scores.iter()
        .filter(|(id, _)| is_negative(*id))
        .map(|(_, s)| *s)
        .collect();

    let pos_mean = if pos_scores.is_empty() { 0.0 } else { pos_scores.iter().sum::<f32>() / pos_scores.len() as f32 };
    let pos_min = pos_scores.iter().cloned().fold(f32::INFINITY, f32::min);
    let pos_max = pos_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let neg_max = neg_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let neg_mean = if neg_scores.is_empty() { 0.0 } else { neg_scores.iter().sum::<f32>() / neg_scores.len() as f32 };
    let margin = pos_min - neg_max;

    let mut ranked: Vec<(i64, f32)> = scores.iter().filter(|(id, _)| *id != query_id).cloned().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let first_pos = ranked.iter().position(|(id, _)| is_positive(*id)).map(|p| p + 1).unwrap_or(999);
    let top3 = ranked.iter().take(3).filter(|(id, _)| is_positive(*id)).count();
    let top5 = ranked.iter().take(5).filter(|(id, _)| is_positive(*id)).count();

    Metrics { pos_mean, pos_min, pos_max, neg_max, neg_mean, margin, first_pos, top3, top5 }
}

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

// ─── CSV Writers ──────────────────────────────────────────────────────────────

fn write_csv(path: &Path, headers: &[&str], rows: &[Vec<String>]) {
    let file = File::create(path).unwrap();
    let mut w = BufWriter::new(file);
    writeln!(w, "{}", headers.join(",")).unwrap();
    for row in rows {
        writeln!(w, "{}", row.join(",")).unwrap();
    }
}

// ─── Pearson/Spearman ────────────────────────────────────────────────────────

fn pearson(xs: &[(f32, f32)]) -> f32 {
    if xs.len() < 2 { return 0.0; }
    let n = xs.len() as f32;
    let mx: f32 = xs.iter().map(|(x, _)| x).sum::<f32>() / n;
    let my: f32 = xs.iter().map(|(_, y)| y).sum::<f32>() / n;
    let num: f32 = xs.iter().map(|(x, y)| (x - mx) * (y - my)).sum();
    let denx: f32 = xs.iter().map(|(x, _)| (x - mx).powi(2)).sum();
    let deny: f32 = xs.iter().map(|(_, y)| (y - my).powi(2)).sum();
    if denx < 1e-8 || deny < 1e-8 { 0.0 } else { num / (denx * deny).sqrt() }
}

fn spearman(xs: &[(f32, f32)]) -> f32 {
    if xs.len() < 2 { return 0.0; }
    let n = xs.len() as f32;
    let mean_r = (n - 1.0) / 2.0;

    let mut indexed: Vec<(usize, f32, f32)> = xs.iter().enumerate().map(|(i, (x, y))| (i, *x, *y)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let ranks_x: HashMap<usize, f32> = indexed.iter().enumerate().map(|(ri, (i, _, _))| (*i, ri as f32)).collect();

    let mut indexed: Vec<(usize, f32, f32)> = xs.iter().enumerate().map(|(i, (x, y))| (i, *x, *y)).collect();
    indexed.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    let ranks_y: HashMap<usize, f32> = indexed.iter().enumerate().map(|(ri, (i, _, _))| (*i, ri as f32)).collect();

    let num: f32 = xs.iter().enumerate().map(|(i, _)| (ranks_x[&i] - mean_r) * (ranks_y[&i] - mean_r)).sum();
    let denx: f32 = xs.iter().enumerate().map(|(i, _)| (ranks_x[&i] - mean_r).powi(2)).sum();
    let deny: f32 = xs.iter().enumerate().map(|(i, _)| (ranks_y[&i] - mean_r).powi(2)).sum();
    if denx < 1e-8 || deny < 1e-8 { 0.0 } else { num / (denx * deny).sqrt() }
}

// ─── Main ────────────────────────────────────────────────────────────────────

#[test]
#[ignore]
fn person_multisignal_bench() {
    let start = Instant::now();
    fs::create_dir_all(OUT_DIR).ok();
    fs::create_dir_all(format!("{}/crops", OUT_DIR)).ok();
    fs::create_dir_all(format!("{}/embeddings", OUT_DIR)).ok();

    eprintln!("=== Multi-Signal Person Identity Benchmark v2 ===");
    eprintln!("Query: img15, Person2: img6/13/14, Negatives: img2/9/17/18/19/20");
    eprintln!("");

    let samples = read_tsv();
    let ids = all_ids();
    eprintln!("Loaded {} samples from TSV", samples.len());

    // Build runtime and models
    let rt = tokio::runtime::Runtime::new().expect("tokio");
    let scrfd = ScrfdDetector::load(Path::new(SCRFD_MODEL)).expect("SCRFD load failed");
    let aligner = SimpleAligner::with_config(AlignmentConfig::default());
    let mut arc_session = Session::builder().expect("arc builder")
        .commit_from_file(PathBuf::from(ARCFACE_MODEL)).expect("arcface");

    eprintln!("Models loaded.");

    // ─── Phase 1: Face Embeddings ───────────────────────────────────────────
    eprintln!("\n=== Phase 1: Face Embedding Extraction ===");
    let mut face_embs: HashMap<i64, Vec<f32>> = HashMap::new();

    for &id in &ids {
        let img_path = match find_image(id, &samples) {
            Some(p) => p,
            None => { eprintln!("Face: missing img{}", id); continue; }
        };

        let img_data = match ImageData::from_file(&img_path) {
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

        let aligned = match build_aligned_face(&aligner, &img_data, &det.bbox, &kps, 0.10) {
            Ok(rgb) => rgb,
            Err(e) => { eprintln!("Face: align failed img{}: {}", id, e); continue; }
        };

        let emb = embed_arcface(&mut arc_session, &aligned);
        face_embs.insert(id, emb);
    }

    eprintln!("Face embeddings: {}/{}", face_embs.len(), ids.len());

    // ─── Phase 2: Body Embeddings (default crop bbox_p20) ─────────────────
    eprintln!("\n=== Phase 2: Body Embedding Extraction ===");
    let default_crop = "bbox_p20";
    let default_crop_dir = generate_body_crops(&samples, default_crop, &scrfd, &rt);
    let default_emb_dir = PathBuf::from(OUT_DIR).join("embeddings_body_default");
    fs::create_dir_all(&default_emb_dir).ok();

    let body_embs = match run_body_extraction("reid_ont", &default_crop_dir, &default_emb_dir) {
        Ok(emb) => emb,
        Err(e) => {
            eprintln!("Body extraction failed: {}", e);
            HashMap::new()
        }
    };
    eprintln!("Body embeddings: {}/{}", body_embs.len(), ids.len());

    let qid = QUERY;
    let all_non_query: Vec<i64> = ids.iter().filter(|&&id| id != qid).cloned().collect();

    // ─── Phase 3: Baseline Verification ────────────────────────────────────
    eprintln!("\n=== Phase 3: Baseline Verification ===");

    let q_face = face_embs.get(&qid).cloned();
    let q_body = body_embs.get(&qid).cloned();

    let face_scores: Vec<(i64, f32)> = all_non_query.iter()
        .filter_map(|&id| {
            let qe = q_face.as_ref()?;
            let fe = face_embs.get(&id)?;
            Some((id, cosine_similarity(qe, fe)))
        })
        .collect();
    let face_metrics = compute_metrics(&face_scores, qid);

    let body_scores: Vec<(i64, f32)> = all_non_query.iter()
        .filter_map(|&id| {
            let qe = q_body.as_ref()?;
            let be = body_embs.get(&id)?;
            Some((id, cosine_similarity(qe, be)))
        })
        .collect();
    let body_metrics = compute_metrics(&body_scores, qid);

    let fusion_scores_04: Vec<(i64, f32)> = all_non_query.iter()
        .filter_map(|&id| {
            let qf = q_face.as_ref()?;
            let qb = q_body.as_ref()?;
            let ff = face_embs.get(&id)?;
            let bb = body_embs.get(&id)?;
            let fs = cosine_similarity(qf, ff);
            let bs = cosine_similarity(qb, bb);
            Some((id, 0.6 * fs + 0.4 * bs))
        })
        .collect();
    let fusion_metrics_04 = compute_metrics(&fusion_scores_04, qid);

    let fm = face_metrics.margin;
    let bm = body_metrics.margin;
    let fum = fusion_metrics_04.margin;
    eprintln!("Face: margin={}, first_pos={}", fm, face_metrics.first_pos);
    eprintln!("Body: margin={}, first_pos={}", bm, body_metrics.first_pos);
    eprintln!("Fusion α=0.4: margin={}, first_pos={}", fum, fusion_metrics_04.first_pos);

    let face_ok = (face_metrics.margin - (-0.1498)).abs() < 0.01;
    let body_ok = (body_metrics.margin - 0.0010).abs() < 0.02;
    let fusion_ok = (fusion_metrics_04.margin - 0.0292).abs() < 0.02;

    if !face_ok || !body_ok || !fusion_ok {
        eprintln!("\n⚠️  WARNING: Results differ from expected!");
        eprintln!("Face expected -0.1498, got {:.4}", face_metrics.margin);
        eprintln!("Body expected +0.0010, got {:.4}", body_metrics.margin);
        eprintln!("Fusion expected +0.0292, got {:.4}", fusion_metrics_04.margin);
    } else {
        eprintln!("\n✓ Baseline verification PASSED");
    }

    // ─── Phase 4: Body Crop Margin Sweep ───────────────────────────────────
    eprintln!("\n=== Phase 4: Body Crop Margin Sweep ===");
    let margin_crops = ["bbox_tight", "bbox_p0", "bbox_p5", "bbox_p10", "bbox_p15", "bbox_p20", "bbox_p25", "bbox_p30", "bbox_p40"];
    let mut crop_sweep_rows = vec![];

    for crop in &margin_crops {
        let crop_dir = generate_body_crops(&samples, crop, &scrfd, &rt);
        let emb_dir = PathBuf::from(OUT_DIR).join(format!("embeddings_{}", crop));
        fs::create_dir_all(&emb_dir).ok();

        let emb = match run_body_extraction("reid_ont", &crop_dir, &emb_dir) {
            Ok(e) => e,
            Err(_) => continue,
        };

        let qb = emb.get(&qid).cloned();
        let scores: Vec<(i64, f32)> = all_non_query.iter()
            .filter_map(|&id| {
                let qe = qb.as_ref()?;
                let be = emb.get(&id)?;
                Some((id, cosine_similarity(qe, be)))
            })
            .collect();
        let metrics = compute_metrics(&scores, qid);

        let pair_to_img6 = scores.iter().find(|(id, _)| *id == 6).map(|(_, s)| *s).unwrap_or(0.0);
        let pair_to_img13 = scores.iter().find(|(id, _)| *id == 13).map(|(_, s)| *s).unwrap_or(0.0);
        let pair_to_img14 = scores.iter().find(|(id, _)| *id == 14).map(|(_, s)| *s).unwrap_or(0.0);

        let mut neg_detail = vec![];
        for nid in NEGATIVES {
            let s = scores.iter().find(|(id, _)| *id == *nid).map(|(_, ss)| *ss).unwrap_or(0.0);
            neg_detail.push(format!("{:.4}", s));
        }

        crop_sweep_rows.push(vec![
            crop.to_string(),
            format!("{:.4}", metrics.pos_mean),
            format!("{:.4}", metrics.pos_min),
            format!("{:.4}", metrics.pos_max),
            format!("{:.4}", metrics.neg_max),
            format!("{:.4}", metrics.neg_mean),
            format!("{:.4}", metrics.margin),
            metrics.first_pos.to_string(),
            metrics.top3.to_string(),
            metrics.top5.to_string(),
            format!("{:.4}", pair_to_img6),
            format!("{:.4}", pair_to_img13),
            format!("{:.4}", pair_to_img14),
            neg_detail.join("|"),
        ]);

        eprintln!("{}: margin={:.4}, first_pos={}, top3={}", crop, metrics.margin, metrics.first_pos, metrics.top3);
    }

    write_csv(
        &Path::new(OUT_DIR).join("body_crop_sweep.csv"),
        &["crop", "pos_mean", "pos_min", "pos_max", "neg_max", "neg_mean", "margin", "first_pos", "top3", "top5", "img6", "img13", "img14", "negatives"],
        &crop_sweep_rows,
    );

    // ─── Phase 5: Body Crop Type ─────────────────────────────────────────────
    eprintln!("\n=== Phase 5: Body Crop Type ===");
    let type_crops = ["bbox_tight", "bbox_p10", "bbox_p20", "bbox_p30", "upper_body", "full_body", "head_body"];
    let mut crop_type_rows = vec![];

    for crop in &type_crops {
        let crop_dir = generate_body_crops(&samples, crop, &scrfd, &rt);
        let emb_dir = PathBuf::from(OUT_DIR).join(format!("embeddings_type_{}", crop));
        fs::create_dir_all(&emb_dir).ok();

        let emb = match run_body_extraction("reid_ont", &crop_dir, &emb_dir) {
            Ok(e) => e,
            Err(_) => continue,
        };

        let qb = emb.get(&qid).cloned();
        let scores: Vec<(i64, f32)> = all_non_query.iter()
            .filter_map(|&id| {
                let qe = qb.as_ref()?;
                let be = emb.get(&id)?;
                Some((id, cosine_similarity(qe, be)))
            })
            .collect();
        let metrics = compute_metrics(&scores, qid);

        crop_type_rows.push(vec![
            crop.to_string(),
            format!("{:.4}", metrics.pos_mean),
            format!("{:.4}", metrics.pos_min),
            format!("{:.4}", metrics.neg_max),
            format!("{:.4}", metrics.margin),
            metrics.first_pos.to_string(),
        ]);
        eprintln!("{}: margin={:.4}, first_pos={}", crop, metrics.margin, metrics.first_pos);
    }

    write_csv(
        &Path::new(OUT_DIR).join("body_crop_type.csv"),
        &["crop_type", "pos_mean", "pos_min", "neg_max", "margin", "first_pos"],
        &crop_type_rows,
    );

    // ─── Phase 6: Fusion Weight Sweep ──────────────────────────────────────
    eprintln!("\n=== Phase 6: Fusion Weight Sweep ===");
    let mut fusion_rows = vec![];
    let mut best_fusion = (0.0f32, Metrics { pos_mean: 0.0, pos_min: 0.0, pos_max: 0.0, neg_max: 0.0, neg_mean: 0.0, margin: f32::NEG_INFINITY, first_pos: 999, top3: 0, top5: 0 });

    for alpha_i in 0..=20 {
        let alpha = alpha_i as f32 / 20.0;
        let scores: Vec<(i64, f32)> = all_non_query.iter()
            .filter_map(|&id| {
                let qf = q_face.as_ref()?;
                let qb = q_body.as_ref()?;
                let ff = face_embs.get(&id)?;
                let bb = body_embs.get(&id)?;
                let fs = cosine_similarity(qf, ff);
                let bs = cosine_similarity(qb, bb);
                Some((id, (1.0 - alpha) * fs + alpha * bs))
            })
            .collect();
        let metrics = compute_metrics(&scores, qid);

        fusion_rows.push(vec![
            format!("{:.2}", alpha),
            format!("{:.4}", metrics.pos_mean),
            format!("{:.4}", metrics.pos_min),
            format!("{:.4}", metrics.pos_max),
            format!("{:.4}", metrics.neg_max),
            format!("{:.4}", metrics.neg_mean),
            format!("{:.4}", metrics.margin),
            metrics.first_pos.to_string(),
            metrics.top3.to_string(),
            metrics.top5.to_string(),
        ]);

        if metrics.margin > best_fusion.1.margin {
            best_fusion = (alpha, Metrics { pos_mean: metrics.pos_mean, pos_min: metrics.pos_min, pos_max: metrics.pos_max, neg_max: metrics.neg_max, neg_mean: metrics.neg_mean, margin: metrics.margin, first_pos: metrics.first_pos, top3: metrics.top3, top5: metrics.top5 });
        }
    }

    write_csv(
        &Path::new(OUT_DIR).join("fusion_sweep.csv"),
        &["alpha", "pos_mean", "pos_min", "pos_max", "neg_max", "neg_mean", "margin", "first_pos", "top3", "top5"],
        &fusion_rows,
    );

    eprintln!("Best fusion: α={:.2}, margin={:.4}, first_pos={}", best_fusion.0, best_fusion.1.margin, best_fusion.1.first_pos);

    // ─── Phase 7: Robustness ─────────────────────────────────────────────────
    eprintln!("\n=== Phase 7: Robustness / Stable Region ===");

    let fusion_metrics_all: Vec<(f32, Metrics)> = (0..=20).map(|alpha_i| {
        let alpha = alpha_i as f32 / 20.0;
        let scores: Vec<(i64, f32)> = all_non_query.iter()
            .filter_map(|&id| {
                let qf = q_face.as_ref()?;
                let qb = q_body.as_ref()?;
                let ff = face_embs.get(&id)?;
                let bb = body_embs.get(&id)?;
                let fs = cosine_similarity(qf, ff);
                let bs = cosine_similarity(qb, bb);
                Some((id, (1.0 - alpha) * fs + alpha * bs))
            })
            .collect();
        (alpha, compute_metrics(&scores, qid))
    }).collect();

    let positive_alphas: Vec<(f32, f32)> = fusion_metrics_all.iter()
        .filter(|(_, m)| m.margin > 0.0)
        .map(|(a, m)| (*a, m.margin))
        .collect();

    let mut robust_regions = vec![];
    let mut i = 0;
    while i < positive_alphas.len() {
        let start_alpha = positive_alphas[i].0;
        let mut j = i;
        while j < positive_alphas.len() && (positive_alphas[j].0 - start_alpha) <= 0.15 {
            j += 1;
        }
        let region_len = j - i;
        if region_len >= 3 {
            robust_regions.push((start_alpha, positive_alphas[j-1].0, region_len));
        }
        i = j;
    }

    let verdict = if best_fusion.1.margin > 0.0 && best_fusion.1.first_pos <= 2 && best_fusion.1.top3 >= 2 {
        if robust_regions.is_empty() { "FACE_BODY_KNIFE_EDGE" } else { "FACE_BODY_ROBUST" }
    } else if best_fusion.1.margin <= 0.0 {
        "FACE_BODY_FAIL"
    } else {
        "FACE_BODY_FIXED"
    };

    eprintln!("Verdict: {}", verdict);
    eprintln!("Best α={:.2}, margin={:.4}, first_pos={}, top3={}", best_fusion.0, best_fusion.1.margin, best_fusion.1.first_pos, best_fusion.1.top3);
    for &(start, end, len) in &robust_regions {
        eprintln!("Robust region: α={:.2}~{:.2} ({} points)", start, end, len);
    }

    // ─── Phase 8: Independence / Correlation ─────────────────────────────────
    eprintln!("\n=== Phase 8: Face-Body Independence ===");

    let face_cos: HashMap<i64, f32> = face_scores.iter().cloned().collect();
    let body_cos: HashMap<i64, f32> = body_scores.iter().cloned().collect();

    let all_pairs: Vec<(f32, f32)> = all_non_query.iter()
        .filter_map(|id| Some((*face_cos.get(id)?, *body_cos.get(id)?)))
        .collect();

    let pos_pairs: Vec<(f32, f32)> = PERSON2.iter()
        .filter(|&&id| id != QUERY)
        .filter_map(|&id| Some((*face_cos.get(&id)?, *body_cos.get(&id)?)))
        .collect();

    let neg_pairs: Vec<(f32, f32)> = NEGATIVES.iter()
        .filter_map(|&id| Some((*face_cos.get(&id)?, *body_cos.get(&id)?)))
        .collect();

    let pearson_all = pearson(&all_pairs);
    let spearman_all = spearman(&all_pairs);
    let pearson_pos = pearson(&pos_pairs);
    let spearman_pos = spearman(&pos_pairs);
    let pearson_neg = pearson(&neg_pairs);
    let spearman_neg = spearman(&neg_pairs);

    write_csv(
        &Path::new(OUT_DIR).join("correlation.csv"),
        &["subset", "n", "pearson", "spearman"],
        &vec![
            vec!["all".to_string(), all_pairs.len().to_string(), format!("{:.4}", pearson_all), format!("{:.4}", spearman_all)],
            vec!["positive".to_string(), pos_pairs.len().to_string(), format!("{:.4}", pearson_pos), format!("{:.4}", spearman_pos)],
            vec!["negative".to_string(), neg_pairs.len().to_string(), format!("{:.4}", pearson_neg), format!("{:.4}", spearman_neg)],
        ],
    );

    eprintln!("Pearson all={:.4}, pos={:.4}, neg={:.4}", pearson_all, pearson_pos, pearson_neg);
    eprintln!("Spearman all={:.4}, pos={:.4}, neg={:.4}", spearman_all, spearman_pos, spearman_neg);

    // ─── Phase 9: Error Complementarity ─────────────────────────────────────
    eprintln!("\n=== Phase 9: Error Complementarity ===");

    let best_alpha = best_fusion.0;
    let mut face_ranked: Vec<(i64, f32)> = face_scores.clone();
    face_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut body_ranked: Vec<(i64, f32)> = body_scores.clone();
    body_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let fusion_scores_best: Vec<(i64, f32)> = all_non_query.iter()
        .filter_map(|&id| {
            let qf = q_face.as_ref()?;
            let qb = q_body.as_ref()?;
            let ff = face_embs.get(&id)?;
            let bb = body_embs.get(&id)?;
            let fs = cosine_similarity(qf, ff);
            let bs = cosine_similarity(qb, bb);
            Some((id, (1.0 - best_alpha) * fs + best_alpha * bs))
        })
        .collect();
    let mut fusion_ranked: Vec<(i64, f32)> = fusion_scores_best.clone();
    fusion_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let face_top1 = face_ranked.first().map(|(id, _)| *id).unwrap_or(0);
    let body_top1 = body_ranked.first().map(|(id, _)| *id).unwrap_or(0);
    let fusion_top1 = fusion_ranked.first().map(|(id, _)| *id).unwrap_or(0);

    let face_correct = is_positive(face_top1);
    let body_correct = is_positive(body_top1);
    let fusion_correct = is_positive(fusion_top1);

    eprintln!("Face top-1: img{} ({})", face_top1, if face_correct { "CORRECT" } else { "WRONG" });
    eprintln!("Body top-1: img{} ({})", body_top1, if body_correct { "CORRECT" } else { "WRONG" });
    eprintln!("Fusion top-1: img{} ({})", fusion_top1, if fusion_correct { "CORRECT" } else { "WRONG" });
    eprintln!("face_wrong_body_correct: {}", !face_correct && body_correct);
    eprintln!("face_correct_body_wrong: {}", face_correct && !body_correct);
    eprintln!("both_wrong: {}", !face_correct && !body_correct);

    let mut ranking_rows = vec![];
    for rank in 0..all_non_query.len().min(10) {
        let (fid, fs) = face_ranked.get(rank).copied().unwrap_or((0, 0.0));
        let (bid, bs) = body_ranked.get(rank).copied().unwrap_or((0, 0.0));
        let (fuid, fus) = fusion_ranked.get(rank).copied().unwrap_or((0, 0.0));
        ranking_rows.push(vec![
            (rank + 1).to_string(),
            format!("img{}", fid), group_of(fid).to_string(), format!("{:.4}", fs),
            format!("img{}", bid), group_of(bid).to_string(), format!("{:.4}", bs),
            format!("img{}", fuid), group_of(fuid).to_string(), format!("{:.4}", fus),
        ]);
    }

    write_csv(
        &Path::new(OUT_DIR).join("ranking.csv"),
        &["rank", "face_img", "face_group", "face_score", "body_img", "body_group", "body_score", "fusion_img", "fusion_group", "fusion_score"],
        &ranking_rows,
    );

    // ─── Phase 10: Prototype Scoring ─────────────────────────────────────────
    eprintln!("\n=== Phase 10: Person Prototype ===");

    // Face prototype
    let face_pos_embs: Vec<&Vec<f32>> = PERSON2.iter().filter_map(|&id| face_embs.get(&id)).collect();
    let face_proto: Vec<f32> = if !face_pos_embs.is_empty() {
        let mean: Vec<f32> = (0..face_pos_embs[0].len())
            .map(|i| face_pos_embs.iter().map(|e| e[i]).sum::<f32>() / face_pos_embs.len() as f32)
            .collect();
        l2_normalize(&mean)
    } else { vec![] };

    let body_pos_embs: Vec<&Vec<f32>> = PERSON2.iter().filter_map(|&id| body_embs.get(&id)).collect();
    let body_proto: Vec<f32> = if !body_pos_embs.is_empty() {
        let mean: Vec<f32> = (0..body_pos_embs[0].len())
            .map(|i| body_pos_embs.iter().map(|e| e[i]).sum::<f32>() / body_pos_embs.len() as f32)
            .collect();
        l2_normalize(&mean)
    } else { vec![] };

    let face_proto_score = q_face.as_ref().and_then(|qf| {
        if face_proto.is_empty() { None } else { Some(cosine_similarity(qf, &face_proto)) }
    }).unwrap_or(0.0);

    let body_proto_score = q_body.as_ref().and_then(|qb| {
        if body_proto.is_empty() { None } else { Some(cosine_similarity(qb, &body_proto)) }
    }).unwrap_or(0.0);

    let neg_face_max = all_non_query.iter()
        .filter(|id| is_negative(**id))
        .filter_map(|id| face_embs.get(id))
        .map(|e| cosine_similarity(&face_proto, e))
        .fold(f32::NEG_INFINITY, f32::max);

    let neg_body_max = all_non_query.iter()
        .filter(|id| is_negative(**id))
        .filter_map(|id| body_embs.get(id))
        .map(|e| cosine_similarity(&body_proto, e))
        .fold(f32::NEG_INFINITY, f32::max);

    write_csv(
        &Path::new(OUT_DIR).join("prototype.csv"),
        &["method", "score", "neg_max", "margin"],
        &vec![
            vec!["face_proto".to_string(), format!("{:.4}", face_proto_score), format!("{:.4}", neg_face_max), format!("{:.4}", face_proto_score - neg_face_max)],
            vec!["body_proto".to_string(), format!("{:.4}", body_proto_score), format!("{:.4}", neg_body_max), format!("{:.4}", body_proto_score - neg_body_max)],
            vec!["fusion_proto".to_string(), format!("{:.4}", 0.6 * face_proto_score + 0.4 * body_proto_score), "N/A".to_string(), "N/A".to_string()],
        ],
    );

    eprintln!("Face proto: score={:.4}, neg_max={:.4}, margin={:.4}", face_proto_score, neg_face_max, face_proto_score - neg_face_max);
    eprintln!("Body proto: score={:.4}, neg_max={:.4}, margin={:.4}", body_proto_score, neg_body_max, body_proto_score - neg_body_max);

    // ─── Phase 11: Leave-One-Out ─────────────────────────────────────────────
    eprintln!("\n=== Phase 11: Leave-One-Out ===");
    let mut loo_rows = vec![];

    for &remove_id in PERSON2 {
        let remaining: Vec<i64> = PERSON2.iter().filter(|&&id| id != remove_id).cloned().collect();

        // Face LOO prototype
        let face_loo_embs: Vec<&Vec<f32>> = remaining.iter().filter_map(|&id| face_embs.get(&id)).collect();
        let face_loo_proto: Vec<f32> = if !face_loo_embs.is_empty() {
            let mean: Vec<f32> = (0..face_loo_embs[0].len())
                .map(|i| face_loo_embs.iter().map(|e| e[i]).sum::<f32>() / face_loo_embs.len() as f32)
                .collect();
            l2_normalize(&mean)
        } else { vec![] };

        let face_loo_score = q_face.as_ref().and_then(|qf| {
            if face_loo_proto.is_empty() { None } else { Some(cosine_similarity(qf, &face_loo_proto)) }
        }).unwrap_or(0.0);
        let face_loo_neg_max = all_non_query.iter()
            .filter(|id| is_negative(**id))
            .filter_map(|id| face_embs.get(id))
            .map(|e| cosine_similarity(&face_loo_proto, e))
            .fold(f32::NEG_INFINITY, f32::max);

        // Body LOO prototype
        let body_loo_embs: Vec<&Vec<f32>> = remaining.iter().filter_map(|&id| body_embs.get(&id)).collect();
        let body_loo_proto: Vec<f32> = if !body_loo_embs.is_empty() {
            let mean: Vec<f32> = (0..body_loo_embs[0].len())
                .map(|i| body_loo_embs.iter().map(|e| e[i]).sum::<f32>() / body_loo_embs.len() as f32)
                .collect();
            l2_normalize(&mean)
        } else { vec![] };

        let body_loo_score = q_body.as_ref().and_then(|qb| {
            if body_loo_proto.is_empty() { None } else { Some(cosine_similarity(qb, &body_loo_proto)) }
        }).unwrap_or(0.0);
        let body_loo_neg_max = all_non_query.iter()
            .filter(|id| is_negative(**id))
            .filter_map(|id| body_embs.get(id))
            .map(|e| cosine_similarity(&body_loo_proto, e))
            .fold(f32::NEG_INFINITY, f32::max);

        let fusion_loo_score = 0.6 * face_loo_score + 0.4 * body_loo_score;
        let fusion_loo_neg_max = 0.6 * face_loo_neg_max + 0.4 * body_loo_neg_max;

        loo_rows.push(vec![
            format!("img{}", remove_id),
            format!("{:.4}", face_loo_score), format!("{:.4}", face_loo_neg_max), format!("{:.4}", face_loo_score - face_loo_neg_max),
            format!("{:.4}", body_loo_score), format!("{:.4}", body_loo_neg_max), format!("{:.4}", body_loo_score - body_loo_neg_max),
            format!("{:.4}", fusion_loo_score), format!("{:.4}", fusion_loo_neg_max), format!("{:.4}", fusion_loo_score - fusion_loo_neg_max),
        ]);

        eprintln!("LOO img{}: face_margin={:.4}, body_margin={:.4}, fusion_margin={:.4}",
            remove_id, face_loo_score - face_loo_neg_max, body_loo_score - body_loo_neg_max, fusion_loo_score - fusion_loo_neg_max);
    }

    write_csv(
        &Path::new(OUT_DIR).join("loo.csv"),
        &["remove", "face_score", "face_neg_max", "face_margin", "body_score", "body_neg_max", "body_margin", "fusion_score", "fusion_neg_max", "fusion_margin"],
        &loo_rows,
    );

    // ─── Phase 12: Dynamic Fusion ───────────────────────────────────────────
    eprintln!("\n=== Phase 12: Dynamic Fusion Strategies ===");

    let strat_a_scores: Vec<(i64, f32)> = all_non_query.iter()
        .filter_map(|&id| {
            let qf = q_face.as_ref()?;
            let qb = q_body.as_ref()?;
            let ff = face_embs.get(&id)?;
            let bb = body_embs.get(&id)?;
            let fs = cosine_similarity(qf, ff);
            let bs = cosine_similarity(qb, bb);
            Some((id, 0.6 * fs + 0.4 * bs))
        })
        .collect();
    let strat_a_metrics = compute_metrics(&strat_a_scores, qid);

    let strat_b_scores: Vec<(i64, f32)> = all_non_query.iter()
        .filter_map(|&id| {
            let qf = q_face.as_ref()?;
            let qb = q_body.as_ref()?;
            let ff = face_embs.get(&id)?;
            let bb = body_embs.get(&id)?;
            let fs = cosine_similarity(qf, ff);
            let bs = cosine_similarity(qb, bb);
            let (fw, bw) = if fs >= 0.80 { (0.8, 0.2) } else if fs >= 0.65 { (0.6, 0.4) } else { (0.4, 0.6) };
            Some((id, fw * fs + bw * bs))
        })
        .collect();
    let strat_b_metrics = compute_metrics(&strat_b_scores, qid);

    write_csv(
        &Path::new(OUT_DIR).join("dynamic_fusion.csv"),
        &["strategy", "weights", "margin", "first_pos", "top3"],
        &vec![
            vec!["A_fixed".to_string(), "0.60|0.40".to_string(), format!("{:.4}", strat_a_metrics.margin), strat_a_metrics.first_pos.to_string(), strat_a_metrics.top3.to_string()],
            vec!["B_adaptive".to_string(), "0.80/0.60/0.40".to_string(), format!("{:.4}", strat_b_metrics.margin), strat_b_metrics.first_pos.to_string(), strat_b_metrics.top3.to_string()],
        ],
    );

    eprintln!("Strategy A (fixed 0.6/0.4): margin={:.4}, first_pos={}", strat_a_metrics.margin, strat_a_metrics.first_pos);
    eprintln!("Strategy B (adaptive): margin={:.4}, first_pos={}", strat_b_metrics.margin, strat_b_metrics.first_pos);

    // ─── Phase 13: Agreement / Conflict ─────────────────────────────────────
    eprintln!("\n=== Phase 13: Agreement / Conflict ===");

    let face_best_person = if is_positive(face_top1) { "PERSON2" } else { "NEGATIVE" };
    let body_best_person = if is_positive(body_top1) { "PERSON2" } else { "NEGATIVE" };
    let fusion_best_person = if is_positive(fusion_top1) { "PERSON2" } else { "NEGATIVE" };
    let agreement = if face_best_person == body_best_person { "AGREEMENT" } else { "CONFLICT" };

    eprintln!("Face→{}, Body→{}, Fusion→{} → {}", face_best_person, body_best_person, fusion_best_person, agreement);

    // ─── Final Report ─────────────────────────────────────────────────────────
    let robust_info_str = if robust_regions.is_empty() {
        "No robust region found".to_string()
    } else {
        robust_regions.iter().map(|(s, e, l)| format!("Robust region: α={:.2}~{:.2} ({} points)", s, e, l)).collect::<Vec<_>>().join("\n")
    };

    let loo_table_str = loo_rows.iter().map(|r| format!("| {} | {} | {} | {} |", r[0], r[3], r[6], r[9])).collect::<Vec<_>>().join("\n");

    let face_m_s = format!("{:.4}", face_metrics.margin);
    let body_m_s = format!("{:.4}", body_metrics.margin);
    let fus_m_s = format!("{:.4}", fusion_metrics_04.margin);
    let ba_s = format!("{:.2}", best_fusion.0);
    let bm_s = format!("{:.4}", best_fusion.1.margin);
    let pa_s = format!("{:.4}", pearson_all);
    let pp_s = format!("{:.4}", pearson_pos);
    let pn_s = format!("{:.4}", pearson_neg);

    let report = format!(r#"# Multi-Signal Person Identity Benchmark v2

## Dataset
- Query: img15
- Person2: img6, img13, img14
- Negatives: img2, img9, img17, img18, img19, img20

## Phase 3: Baseline Verification

| Method | Margin | First Pos | Top3 | Top5 |
|--------|--------|-----------|------|------|
| Face (ArcFace w600k_r50) | {} | {} | {} | {} |
| Body (YouTu Re-ID, bbox+20%) | {} | {} | {} | {} |
| Fusion α=0.4 | {} | {} | {} | {} |

## Phase 6: Best Fusion

Best α = {}, margin = {}, first_pos = {}

## Phase 7: Robustness

Verdict: {}

{}

## Phase 8: Face-Body Independence

Pearson (all): {}
Pearson (positive): {}
Pearson (negative): {}

## Phase 9: Error Complementarity

Face top-1: img{} ({})
Body top-1: img{} ({})
Fusion top-1: img{} ({})
face_wrong_body_correct: {}
face_correct_body_wrong: {}
both_wrong: {}

## Phase 11: LOO

| Removed | Face Margin | Body Margin | Fusion Margin |
|---------|-------------|-------------|---------------|
{}

## Phase 13: Agreement

Face→{}, Body→{}, Fusion→{} → {}

## Final Verdict: {}

## Files

- body_crop_sweep.csv
- body_crop_type.csv
- fusion_sweep.csv
- ranking.csv
- prototype.csv
- loo.csv
- correlation.csv
- dynamic_fusion.csv
"#,
        face_m_s, face_metrics.first_pos, face_metrics.top3, face_metrics.top5,
        body_m_s, body_metrics.first_pos, body_metrics.top3, body_metrics.top5,
        fus_m_s, fusion_metrics_04.first_pos, fusion_metrics_04.top3, fusion_metrics_04.top5,
        ba_s, bm_s, best_fusion.1.first_pos,
        verdict,
        robust_info_str,
        pa_s, pp_s, pn_s,
        face_top1, if face_correct { "CORRECT" } else { "WRONG" },
        body_top1, if body_correct { "CORRECT" } else { "WRONG" },
        fusion_top1, if fusion_correct { "CORRECT" } else { "WRONG" },
        !face_correct && body_correct,
        face_correct && !body_correct,
        !face_correct && !body_correct,
        loo_table_str,
        face_best_person, body_best_person, fusion_best_person, agreement,
        verdict,
    );

    eprintln!("\n=== DONE ===");
    eprintln!("Output: {}", OUT_DIR);
    eprintln!("Total time: {:.1}s", start.elapsed().as_secs_f32());

    fs::write(Path::new(OUT_DIR).join("final_report.md"), &report).ok();
    println!("{}", report);
}
