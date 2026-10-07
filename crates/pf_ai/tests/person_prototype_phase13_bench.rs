//! Phase 13 — Multi-Person Body Prototype Validation
//!
//! Validates YouTu Re-ID + head_body crop + normalized mean prototype approach
//! from Phase 12 on a multi-person LFW dataset.
//!
//! Tests whether the approach generalizes beyond the single hard case (img15)
//! to real Person-level clustering with multiple people.

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

const OUT_DIR: &str = "/tmp/person_prototype_phase13";
const ARCFACE_MODEL: &str = "/Users/mac/ai-project/photofinder-next-2/models/w600k_r50.onnx";
const SCRFD_MODEL: &str = "/Users/mac/ai-project/photofinder-next-2/models/scrfd_500m_bnkps.onnx";
const LFW_ROOT: &str = "/Users/mac/Library/Caches/PhotoFinder/lfw-eval/lfw-deepfunneled/lfw-deepfunneled";

// Three people with 10 images each = 30 images total
// Using first 5 per person for core LOO (gallery=4, query=1)
// Using remaining 5 per person for contamination/prototype expansion tests
const PERSONS: &[(&str, usize)] = &[
    ("George_W_Bush", 10),
    ("Colin_Powell", 10),
    ("Tony_Blair", 10),
];

// ─── Data structures ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Metrics {
    pub pos_mean: f32,
    pub pos_min: f32,
    pub neg_max: f32,
    pub margin: f32,
    pub first_pos: usize,
    pub top3: usize,
    pub top5: usize,
}

#[derive(Debug, Clone)]
struct Sample { person: String, img_idx: usize, path: PathBuf }

fn read_lfw_samples() -> Vec<Sample> {
    let mut samples = vec![];
    for &(person, count) in PERSONS {
        for idx in 0..count {
            let path = PathBuf::from(LFW_ROOT)
                .join(person)
                .join(format!("{}_{:04}.jpg", person, idx + 1));
            if path.exists() {
                samples.push(Sample { person: person.to_string(), img_idx: idx, path });
            }
        }
    }
    samples
}

fn get_person_id(sample: &Sample) -> usize {
    PERSONS.iter().position(|(p, _)| p == &sample.person).unwrap_or(999)
}

fn is_same_person(a: &Sample, b: &Sample) -> bool {
    a.person == b.person
}

// ─── Alignment helpers ─────────────────────────────────────────────────────────

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
) -> Result<HashMap<String, Vec<f32>>, String> {
    let python = std::env::var("OSNET_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let manifest_path = output_dir.join("manifest.json");

    if manifest_path.exists() {
        let content = std::fs::read_to_string(&manifest_path).map_err(|e| e.to_string())?;
        if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&content) {
            if manifest.get("status").and_then(|v| v.as_str()) == Some("OK") {
                let embeddings_json = manifest.get("embeddings").ok_or("no embeddings")?;
                let mut emb = HashMap::new();
                for (k, v) in embeddings_json.as_object().ok_or("not an object")? {
                    let arr = v.as_array().ok_or("not an array")?;
                    let floats: Vec<f32> = arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
                    emb.insert(k.clone(), floats);
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
        return Err(format!("python failed: {}", stdout));
    }

    if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&stdout) {
        if manifest.get("status").and_then(|v| v.as_str()) == Some("OK") {
            let embeddings_json = manifest.get("embeddings").ok_or("no embeddings")?;
            let mut emb = HashMap::new();
            for (k, v) in embeddings_json.as_object().ok_or("not an object")? {
                let arr = v.as_array().ok_or("not an array")?;
                let floats: Vec<f32> = arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
                emb.insert(k.clone(), floats);
            }
            return Ok(emb);
        }
    }

    Err("failed to parse manifest".to_string())
}

// ─── Crop generation ─────────────────────────────────────────────────────────

fn expand_bbox_to_body(face: &BBox, img_h: f32, img_w: f32, crop_type: &str) -> BBox {
    let cx = face.x + face.w / 2.0;
    let cy = face.y + face.h / 2.0;

    match crop_type {
        "bbox_tight" | "bbox_p0" => *face,
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
        let key = format!("{}_{:02}", sample.person, sample.img_idx);
        let img_path = &sample.path;

        let img_data = match ImageData::from_file(img_path) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let dets = match rt.block_on(scrfd.detect(&img_data)) {
            Ok(d) => d,
            Err(_) => continue,
        };

        if dets.is_empty() { continue; }

        let det = dets.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap()).unwrap();
        let body_box = expand_bbox_to_body(&det.bbox, img_data.height() as f32, img_data.width() as f32, crop_type);

        if let Ok(crop) = img_data.crop(body_box) {
            let rgb = crop.as_rgb8();
            let out_path = out_dir.join(format!("{}_x{}_y{}_w{}_h{}.png",
                key, body_box.x as i32, body_box.y as i32, body_box.w as i32, body_box.h as i32));
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

// ─── Metrics computation ─────────────────────────────────────────────────────

fn compute_metrics(scores: &[(String, f32)], query_person: &str, all_samples: &[Sample]) -> Metrics {
    let query_id = all_samples.iter().position(|s| s.person == query_person && s.img_idx == 0).unwrap_or(0);
    let query_sample = &all_samples[query_id];

    let pos_scores: Vec<f32> = scores.iter()
        .filter(|(key, _)| {
            let parts: Vec<&str> = key.split('_').collect();
            let person = parts[0..2].join("_");
            person == query_sample.person && *key != format!("{}_{:02}", query_sample.person, query_sample.img_idx)
        })
        .map(|(_, s)| *s)
        .collect();

    let neg_scores: Vec<f32> = scores.iter()
        .filter(|(key, _)| {
            let parts: Vec<&str> = key.split('_').collect();
            let person = parts[0..2].join("_");
            person != query_sample.person
        })
        .map(|(_, s)| *s)
        .collect();

    let pos_mean = if pos_scores.is_empty() { 0.0 } else { pos_scores.iter().sum::<f32>() / pos_scores.len() as f32 };
    let pos_min = pos_scores.iter().cloned().fold(f32::INFINITY, f32::min);
    let neg_max = neg_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let margin = pos_min - neg_max;

    let mut ranked: Vec<(String, f32)> = scores.iter().cloned().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let first_pos = ranked.iter().position(|(key, _)| {
        let parts: Vec<&str> = key.split('_').collect();
        let person = parts[0..2].join("_");
        person == query_sample.person && *key != format!("{}_{:02}", query_sample.person, query_sample.img_idx)
    }).map(|p| p + 1).unwrap_or(999);

    let top3 = ranked.iter().take(3).filter(|(key, _)| {
        let parts: Vec<&str> = key.split('_').collect();
        let person = parts[0..2].join("_");
        person == query_sample.person
    }).count();

    let top5 = ranked.iter().take(5).filter(|(key, _)| {
        let parts: Vec<&str> = key.split('_').collect();
        let person = parts[0..2].join("_");
        person == query_sample.person
    }).count();

    Metrics { pos_mean, pos_min, neg_max, margin, first_pos, top3, top5 }
}

// Metrics computation using (person_idx, img_idx) keys
fn compute_metrics_idx(scores: &[((usize, usize), f32)], query_person_idx: usize) -> Metrics {
    let pos_scores: Vec<f32> = scores.iter()
        .filter(|((p_idx, _), _)| *p_idx == query_person_idx)
        .map(|(_, s)| *s)
        .collect();

    let neg_scores: Vec<f32> = scores.iter()
        .filter(|((p_idx, _), _)| *p_idx != query_person_idx)
        .map(|(_, s)| *s)
        .collect();

    let pos_mean = if pos_scores.is_empty() { 0.0 } else { pos_scores.iter().sum::<f32>() / pos_scores.len() as f32 };
    let pos_min = pos_scores.iter().cloned().fold(f32::INFINITY, f32::min);
    let neg_max = neg_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let margin = pos_min - neg_max;

    let mut ranked: Vec<((usize, usize), f32)> = scores.to_vec();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let first_pos = ranked.iter().position(|((p_idx, _), _)| *p_idx == query_person_idx).map(|p| p + 1).unwrap_or(999);
    let top3 = ranked.iter().take(3).filter(|((p_idx, _), _)| *p_idx == query_person_idx).count();
    let top5 = ranked.iter().take(5).filter(|((p_idx, _), _)| *p_idx == query_person_idx).count();

    Metrics { pos_mean, pos_min, neg_max, margin, first_pos, top3, top5 }
}

// ─── FAR/FRR/ROC computation ──────────────────────────────────────────────────

fn compute_far_frr(scores: &[(String, f32)], all_samples: &[Sample], threshold: f32) -> (f32, f32) {
    let mut true_positives = 0;
    let mut false_positives = 0;
    let mut true_negatives = 0;
    let mut false_negatives = 0;

    for (i, (key_i, score_i)) in scores.iter().enumerate() {
        for (j, (key_j, score_j)) in scores.iter().enumerate() {
            if i >= j { continue; }

            let parts_i: Vec<&str> = key_i.split('_').collect();
            let parts_j: Vec<&str> = key_j.split('_').collect();
            let person_i = parts_i[0..2].join("_");
            let person_j = parts_j[0..2].join("_");
            let same_person = person_i == person_j;

            let pair_score = (score_i + score_j) / 2.0;
            let predicted_same = pair_score >= threshold;

            if same_person && predicted_same { true_positives += 1; }
            else if same_person && !predicted_same { false_negatives += 1; }
            else if !same_person && predicted_same { false_positives += 1; }
            else { true_negatives += 1; }
        }
    }

    let far = if false_positives + true_negatives > 0 {
        false_positives as f32 / (false_positives + true_negatives) as f32
    } else { 0.0 };

    let frr = if false_negatives + true_positives > 0 {
        false_negatives as f32 / (false_negatives + true_positives) as f32
    } else { 0.0 };

    (far, frr)
}

fn compute_roc_points(scores: &[(String, f32)], all_samples: &[Sample]) -> Vec<(f32, f32, f32)> {
    let mut points = vec![];
    let thresholds: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();

    for &threshold in &thresholds {
        let (far, frr) = compute_far_frr(scores, all_samples, threshold);
        let tar = 1.0 - frr;
        points.push((far, tar, threshold));
    }

    points
}

fn compute_eer(points: &[(f32, f32, f32)]) -> f32 {
    let mut min_diff = f32::INFINITY;
    let mut eer = 0.0;

    for &(far, tar, _) in points {
        let diff = (far - (1.0 - tar)).abs();
        if diff < min_diff {
            min_diff = diff;
            eer = far;
        }
    }

    eer
}

fn compute_auc(points: &[(f32, f32, f32)]) -> f32 {
    let mut auc = 0.0;
    for i in 0..points.len() - 1 {
        let far1 = points[i].0;
        let tar1 = points[i].1;
        let far2 = points[i + 1].0;
        let tar2 = points[i + 1].1;
        auc += (far2 - far1) * (tar1 + tar2) / 2.0;
    }
    auc
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
fn person_prototype_phase13_bench() {
    let start = Instant::now();
    fs::create_dir_all(OUT_DIR).ok();
    fs::create_dir_all(format!("{}/crops", OUT_DIR)).ok();
    fs::create_dir_all(format!("{}/embeddings", OUT_DIR)).ok();

    eprintln!("=== Phase 13: Multi-Person Body Prototype Validation ===");
    eprintln!("Persons: {:?}", PERSONS.iter().map(|(p, c)| format!("{} ({} images)", p, c)).collect::<Vec<_>>());
    eprintln!("");

    let samples = read_lfw_samples();
    eprintln!("Loaded {} samples", samples.len());

    let rt = tokio::runtime::Runtime::new().expect("tokio");
    let scrfd = ScrfdDetector::load(Path::new(SCRFD_MODEL)).expect("SCRFD load failed");
    let mut arc_session = Session::builder().expect("arc builder")
        .commit_from_file(PathBuf::from(ARCFACE_MODEL)).expect("arcface");

    // ─── Extract face embeddings for fusion comparison ───────────────────────
    eprintln!("\n=== Extracting Face Embeddings ===");
    let mut face_embs: HashMap<String, Vec<f32>> = HashMap::new();

    for sample in &samples {
        let key = format!("{}_{:02}", sample.person, sample.img_idx);

        let img_data = match ImageData::from_file(&sample.path) {
            Ok(d) => d,
            Err(_) => { eprintln!("Face: cannot load {}", key); continue; }
        };

        let dets = match rt.block_on(scrfd.detect(&img_data)) {
            Ok(d) => d,
            Err(_) => { eprintln!("Face: detect failed {}", key); continue; }
        };
        if dets.is_empty() { eprintln!("Face: no detection {}", key); continue; }

        let det = dets.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap()).unwrap();
        let kps = kps_from_det(det);
        let aligner = SimpleAligner::with_config(AlignmentConfig::default());

        let aligned = match build_aligned_face(&aligner, &img_data, &det.bbox, &kps, 0.10) {
            Ok(rgb) => rgb,
            Err(e) => { eprintln!("Face: align failed {}: {}", key, e); continue; }
        };

        let emb = embed_arcface(&mut arc_session, &aligned);
        face_embs.insert(key, emb);
    }

    eprintln!("Face embeddings: {}", face_embs.len());

    // ─── Extract body embeddings ─────────────────────────────────────────────
    eprintln!("\n=== Extracting Body Embeddings (head_body crop) ===");

    let crop_type = "head_body";
    let crop_dir = generate_body_crops(&samples, crop_type, &scrfd, &rt);
    let emb_dir = PathBuf::from(OUT_DIR).join(format!("embeddings_{}", crop_type));
    fs::create_dir_all(&emb_dir).ok();

    let body_embs = match run_body_extraction("reid_ont", &crop_dir, &emb_dir) {
        Ok(emb) => emb,
        Err(e) => { eprintln!("Body extraction failed: {}", e); HashMap::new() }
    };

    eprintln!("Body embeddings: {}", body_embs.len());

    // Build mapping from "Person_00" -> actual embedding key (with bbox coords)
    let body_key_map: HashMap<String, String> = body_embs.keys()
        .filter(|k| !k.is_empty())
        .filter_map(|k| {
            // Key format: "Person_Name_00_x..._y..._w..._h..."
            // We want to extract "Person_Name_00" as the short key
            let parts: Vec<&str> = k.split('_').collect();
            if parts.len() >= 3 {
                // Find the index of first part that starts with a number (the img_idx)
                let short_key = if let Some(pos) = parts.iter().position(|p| p.chars().next().unwrap().is_ascii_digit()) {
                    parts[..pos].join("_") + "_" + parts[pos]
                } else {
                    parts[..3].join("_")
                };
                Some((short_key, k.clone()))
            } else {
                None
            }
        })
        .collect();

    eprintln!("Body key mapping: {} entries", body_key_map.len());

    // Build 2D body embedding lookup: (person_idx, img_idx) -> embedding
    // Using nested HashMap: person -> idx -> embedding
    let mut body_emb_2d: HashMap<String, HashMap<usize, Vec<f32>>> = HashMap::new();
    for &(person, count) in PERSONS {
        let mut idx_map = HashMap::new();
        for idx in 0..count {
            let short_key = format!("{}_{:02}", person, idx);
            if let Some(full_key) = body_key_map.get(&short_key) {
                if let Some(emb) = body_embs.get(full_key) {
                    idx_map.insert(idx, emb.clone());
                }
            }
        }
        body_emb_2d.insert(person.to_string(), idx_map);
    }

    // ─── Phase 1: LOO Cross-Validation per Person ────────────────────────────
    eprintln!("\n=== Phase 1: LOO Cross-Validation ===");

    let mut loo_rows = vec![];
    let mut all_loo_metrics = vec![];

    for &(person, count) in PERSONS {
        for query_idx in 0..count {
            // Gallery = all other images of same person
            let gallery_emb_map = body_emb_2d.get(person).unwrap();

            let gallery_embs: Vec<Vec<f32>> = (0..count)
                .filter(|&i| i != query_idx)
                .filter_map(|i| gallery_emb_map.get(&i).cloned())
                .collect();

            if gallery_embs.len() != count - 1 { continue; }

            let proto = l2_normalize(&vector_mean(&gallery_embs));

            // Query
            let query_emb = match gallery_emb_map.get(&query_idx) {
                Some(e) => e,
                None => continue,
            };

            let query_score = cosine_similarity(query_emb, &proto);

            // Get person_idx for this person
            let person_idx = PERSONS.iter().position(|(p, _)| *p == person).unwrap_or(0);

            // All scores against prototype
            let mut scores: Vec<((usize, usize), f32)> = samples.iter()
                .map(|s| {
                    let s_person_idx = PERSONS.iter().position(|(p, _)| p == &s.person).unwrap_or(0);
                    let emb = body_emb_2d.get(&s.person)
                        .and_then(|m| m.get(&s.img_idx))
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; 768]);
                    ((s_person_idx, s.img_idx), cosine_similarity(&emb, &proto))
                })
                .collect();

            let metrics = compute_metrics_idx(&scores, person_idx);

            all_loo_metrics.push(metrics.clone());

            loo_rows.push(vec![
                person.to_string(),
                query_idx.to_string(),
                format!("{:.4}", query_score),
                format!("{:.4}", metrics.pos_min),
                format!("{:.4}", metrics.neg_max),
                format!("{:.4}", metrics.margin),
                metrics.first_pos.to_string(),
                metrics.top3.to_string(),
                metrics.top5.to_string(),
                if metrics.margin > 0.0 { "PASS".to_string() } else { "FAIL".to_string() },
            ]);

            eprintln!("{} img{:02}: query={:.4}, margin={:.4}, first_pos={}, top3={}, top5={}",
                person, query_idx, query_score, metrics.margin, metrics.first_pos, metrics.top3, metrics.top5);
        }
    }

    write_csv(
        &Path::new(OUT_DIR).join("loo_results.csv"),
        &["person", "query_idx", "query_score", "pos_min", "neg_max", "margin", "first_pos", "top3", "top5", "pass"],
        &loo_rows,
    );

    // ─── Phase 2: Aggregate LOO Metrics ───────────────────────────────────────
    eprintln!("\n=== Phase 2: Aggregate LOO Metrics ===");

    let total_queries = all_loo_metrics.len();
    let pass_count = all_loo_metrics.iter().filter(|m| m.margin > 0.0).count();
    let top1_hit = all_loo_metrics.iter().filter(|m| m.first_pos == 1).count();
    let top3_hit = all_loo_metrics.iter().filter(|m| m.top3 >= 1).count();
    let top5_hit = all_loo_metrics.iter().filter(|m| m.top5 >= 1).count();

    let margins: Vec<f32> = all_loo_metrics.iter().map(|m| m.margin).collect();
    let median_margin = {
        let mut sorted = margins.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    };

    let p10_margin = {
        let mut sorted = margins.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[(sorted.len() as f32 * 0.1) as usize].min(0.0_f32)
    };

    eprintln!("Total LOO queries: {}", total_queries);
    eprintln!("PASS rate: {}/{} ({:.1}%)", pass_count, total_queries, 100.0 * pass_count as f32 / total_queries as f32);
    eprintln!("Top-1 accuracy: {}/{} ({:.1}%)", top1_hit, total_queries, 100.0 * top1_hit as f32 / total_queries as f32);
    eprintln!("Top-3 accuracy: {}/{} ({:.1}%)", top3_hit, total_queries, 100.0 * top3_hit as f32 / total_queries as f32);
    eprintln!("Top-5 accuracy: {}/{} ({:.1}%)", top5_hit, total_queries, 100.0 * top5_hit as f32 / total_queries as f32);
    eprintln!("Median margin: {:.4}", median_margin);
    eprintln!("P10 margin: {:.4}", p10_margin);

    // ─── Phase 3: Core Size Experiment (1-5 gallery images) ──────────────────
    eprintln!("\n=== Phase 3: Core Size Experiment ===");

    let mut core_size_rows = vec![];

    for core_size in 1..=5 {
        let mut core_metrics = vec![];

        for &(person, count) in PERSONS {
            for query_idx in 0..count.min(core_size) {
                let person_idx = PERSONS.iter().position(|(p, _)| *p == person).unwrap_or(0);
                let emb_map = body_emb_2d.get(person).unwrap();

                let gallery_embs: Vec<Vec<f32>> = (0..count)
                    .filter(|&i| i != query_idx)
                    .take(core_size)
                    .filter_map(|i| emb_map.get(&i).cloned())
                    .collect();

                if gallery_embs.len() != core_size.min(count - 1) { continue; }

                let proto = l2_normalize(&vector_mean(&gallery_embs));

                let query_emb = match emb_map.get(&query_idx) {
                    Some(e) => e,
                    None => continue,
                };

                let _query_score = cosine_similarity(query_emb, &proto);

                let mut scores: Vec<((usize, usize), f32)> = samples.iter()
                    .map(|s| {
                        let s_idx = PERSONS.iter().position(|(p, _)| p == &s.person).unwrap_or(0);
                        let emb = body_emb_2d.get(&s.person)
                            .and_then(|m| m.get(&s.img_idx))
                            .cloned()
                            .unwrap_or_else(|| vec![0.0; 768]);
                        ((s_idx, s.img_idx), cosine_similarity(&emb, &proto))
                    })
                    .collect();

                let metrics = compute_metrics_idx(&scores, person_idx);
                core_metrics.push(metrics);
            }
        }

        if !core_metrics.is_empty() {
            let avg_margin: f32 = core_metrics.iter().map(|m| m.margin).sum::<f32>() / core_metrics.len() as f32;
            let avg_first_pos: f32 = core_metrics.iter().map(|m| m.first_pos as f32).sum::<f32>() / core_metrics.len() as f32;

            core_size_rows.push(vec![
                core_size.to_string(),
                core_metrics.len().to_string(),
                format!("{:.4}", avg_margin),
                format!("{:.4}", avg_first_pos),
            ]);

            eprintln!("Core size {}: avg_margin={:.4}, avg_first_pos={:.2}", core_size, avg_margin, avg_first_pos);
        }
    }

    write_csv(
        &Path::new(OUT_DIR).join("core_size_experiment.csv"),
        &["core_size", "queries", "avg_margin", "avg_first_pos"],
        &core_size_rows,
    );

    // ─── Phase 4: ROC/FAR/FRR/AUC/EER ───────────────────────────────────────
    eprintln!("\n=== Phase 4: ROC/FAR/FRR/AUC/EER ===");

    // Build all pair scores using body_emb_2d
    let all_embs: Vec<((usize, usize), Vec<f32>)> = samples.iter()
        .filter_map(|s| {
            let person_idx = PERSONS.iter().position(|(p, _)| p == &s.person)?;
            let emb = body_emb_2d.get(&s.person)?.get(&s.img_idx)?.clone();
            Some(((person_idx, s.img_idx), emb))
        })
        .collect();

    // Compute pair scores for same/different person
    let mut same_person_scores = vec![];
    let mut diff_person_scores = vec![];

    for i in 0..all_embs.len() {
        for j in (i + 1)..all_embs.len() {
            let ((p_i, _), emb_i) = &all_embs[i];
            let ((p_j, _), emb_j) = &all_embs[j];

            let pair_sim = cosine_similarity(emb_i, emb_j);

            if p_i == p_j {
                same_person_scores.push(pair_sim);
            } else {
                diff_person_scores.push(pair_sim);
            }
        }
    }

    // Compute ROC points
    let mut roc_rows = vec![];
    for threshold in 0..=100 {
        let t = threshold as f32 / 100.0;

        let tp = same_person_scores.iter().filter(|&&s| s >= t).count() as f32;
        let fn_ = same_person_scores.iter().filter(|&&s| s < t).count() as f32;
        let fp = diff_person_scores.iter().filter(|&&s| s >= t).count() as f32;
        let tn = diff_person_scores.iter().filter(|&&s| s < t).count() as f32;

        let tar = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        let far = if fp + tn > 0.0 { fp / (fp + tn) } else { 0.0 };
        let frr = if tp + fn_ > 0.0 { fn_ / (tp + fn_) } else { 0.0 };

        roc_rows.push(vec![
            format!("{:.2}", t),
            format!("{:.4}", tar),
            format!("{:.4}", far),
            format!("{:.4}", frr),
        ]);
    }

    write_csv(
        &Path::new(OUT_DIR).join("roc_curve.csv"),
        &["threshold", "tar", "far", "frr"],
        &roc_rows,
    );

    // Find EER
    let mut min_diff = f32::MAX;
    let mut eer = 0.0;
    for row in &roc_rows {
        let t = row[0].parse::<f32>().unwrap();
        let far = row[2].parse::<f32>().unwrap();
        let frr = row[3].parse::<f32>().unwrap();
        let diff = (far - frr).abs();
        if diff < min_diff {
            min_diff = diff;
            eer = t;
        }
    }

    // Compute AUC using trapezoidal rule
    let mut auc = 0.0;
    for i in 0..roc_rows.len() - 1 {
        let far1 = roc_rows[i][2].parse::<f32>().unwrap();
        let tar1 = roc_rows[i][1].parse::<f32>().unwrap();
        let far2 = roc_rows[i + 1][2].parse::<f32>().unwrap();
        let tar2 = roc_rows[i + 1][1].parse::<f32>().unwrap();
        auc += (far2 - far1) * (tar1 + tar2) / 2.0;
    }
    auc = 1.0 - auc; // AUC for TPR vs FPR, but we want area under TPR vs FPR curve

    eprintln!("EER (approx): {:.2}", eer);
    eprintln!("AUC: {:.4}", auc);

    // ─── Phase 5: Face vs Body vs Fusion ─────────────────────────────────────
    eprintln!("\n=== Phase 5: Face vs Body vs Fusion ===");

    let mut fusion_rows = vec![];

    // Build per-person prototypes for face and body
    let mut face_protos: HashMap<&str, Vec<f32>> = HashMap::new();
    let mut body_protos_idx: HashMap<usize, Vec<f32>> = HashMap::new();

    for &(person, count) in PERSONS {
        let person_idx = PERSONS.iter().position(|(p, _)| *p == person).unwrap_or(0);
        let keys: Vec<String> = (0..count).map(|i| format!("{}_{:02}", person, i)).collect();

        let face_embs_list: Vec<Vec<f32>> = keys.iter()
            .filter_map(|k| face_embs.get(k).cloned())
            .collect();
        let body_embs_list: Vec<Vec<f32>> = (0..count)
            .filter_map(|i| Some(body_emb_2d.get(person)?.get(&i)?.clone()))
            .collect();

        if !face_embs_list.is_empty() {
            face_protos.insert(person, l2_normalize(&vector_mean(&face_embs_list)));
        }
        if !body_embs_list.is_empty() {
            body_protos_idx.insert(person_idx, l2_normalize(&vector_mean(&body_embs_list)));
        }
    }

    // Test fusion across alpha values
    for alpha in [0.0, 0.25, 0.50, 0.75, 1.0] {
        let mut correct = 0;
        let mut total = 0;
        let mut margins = vec![];

        for &(person, count) in PERSONS {
            let person_idx = PERSONS.iter().position(|(p, _)| *p == person).unwrap_or(0);

            for query_idx in 0..count {
                let query_key = format!("{}_{:02}", person, query_idx);

                let face_q = match face_embs.get(&query_key) {
                    Some(e) => e,
                    None => continue,
                };
                let body_q = match body_emb_2d.get(person).and_then(|m| m.get(&query_idx)) {
                    Some(e) => e,
                    None => continue,
                };

                let face_proto = match face_protos.get(person) {
                    Some(p) => p,
                    None => continue,
                };
                let body_proto = match body_protos_idx.get(&person_idx) {
                    Some(p) => p,
                    None => continue,
                };

                // Compute fused score
                let face_score = cosine_similarity(face_q, face_proto);
                let body_score = cosine_similarity(body_q, body_proto);
                let fused_score = (1.0 - alpha) * face_score + alpha * body_score;

                // Find best match
                let mut best_person_idx = 999;
                let mut best_score = f32::NEG_INFINITY;

                for (idx, &(p, _)) in PERSONS.iter().enumerate() {
                    let fp = match face_protos.get(p) {
                        Some(e) => e,
                        None => continue,
                    };
                    let bp = match body_protos_idx.get(&idx) {
                        Some(e) => e,
                        None => continue,
                    };

                    let fs = cosine_similarity(face_q, fp);
                    let bs = cosine_similarity(body_q, bp);
                    let s = (1.0 - alpha) * fs + alpha * bs;

                    if s > best_score {
                        best_score = s;
                        best_person_idx = idx;
                    }
                }

                if best_person_idx == person_idx {
                    correct += 1;
                }
                total += 1;
                margins.push(fused_score - best_score);
            }
        }

        let accuracy = if total > 0 { 100.0 * correct as f32 / total as f32 } else { 0.0 };

        fusion_rows.push(vec![
            format!("{:.2}", alpha),
            correct.to_string(),
            total.to_string(),
            format!("{:.1}%", accuracy),
        ]);

        eprintln!("α={:.2}: accuracy={:.1}% ({}/{})", alpha, accuracy, correct, total);
    }

    write_csv(
        &Path::new(OUT_DIR).join("fusion_comparison.csv"),
        &["alpha", "correct", "total", "accuracy"],
        &fusion_rows,
    );

    // ─── Phase 6: Prototype Contamination Test ───────────────────────────────
    eprintln!("\n=== Phase 6: Prototype Contamination Test ===");

    let mut contamination_rows = vec![];

    // For each person, add negative samples to their prototype and test degradation
    for &(person, count) in PERSONS {
        let person_idx = PERSONS.iter().position(|(p, _)| *p == person).unwrap_or(0);
        let emb_map = body_emb_2d.get(person).unwrap();

        // Clean prototype
        let clean_embs: Vec<Vec<f32>> = (0..count)
            .filter_map(|i| emb_map.get(&i).cloned())
            .collect();
        let clean_proto = l2_normalize(&vector_mean(&clean_embs));

        // Test with contaminated prototypes (add 1-3 negative samples)
        for n_contaminators in 1..=3 {
            // Pick first N negative samples from other persons
            let other_persons: Vec<&str> = PERSONS.iter()
                .filter(|(p, _)| *p != person)
                .map(|(p, _)| *p)
                .collect();

            let contaminator_embs: Vec<Vec<f32>> = other_persons.iter()
                .take(n_contaminators)
                .filter_map(|p| {
                    let emb_map = body_emb_2d.get(*p)?;
                    Some(emb_map.get(&0)?.clone())
                })
                .collect();

            let mut mixed_embs = clean_embs.clone();
            mixed_embs.extend(contaminator_embs);

            let contaminated_proto = l2_normalize(&vector_mean(&mixed_embs));

            // Test LOO with contaminated prototype
            for query_idx in 0..count.min(3) {
                let query_emb = match emb_map.get(&query_idx) {
                    Some(e) => e,
                    None => continue,
                };

                let clean_score = cosine_similarity(query_emb, &clean_proto);
                let contaminated_score = cosine_similarity(query_emb, &contaminated_proto);
                let degradation = clean_score - contaminated_score;

                contamination_rows.push(vec![
                    person.to_string(),
                    n_contaminators.to_string(),
                    query_idx.to_string(),
                    format!("{:.4}", clean_score),
                    format!("{:.4}", contaminated_score),
                    format!("{:.4}", degradation),
                ]);
            }
        }
    }

    write_csv(
        &Path::new(OUT_DIR).join("contamination_test.csv"),
        &["person", "n_contaminators", "query_idx", "clean_score", "contaminated_score", "degradation"],
        &contamination_rows,
    );

    // ─── Phase 7: Hard Negative Test ─────────────────────────────────────────
    eprintln!("\n=== Phase 7: Hard Negative Test ===");

    // Find most similar negative pairs (hard negatives)
    let mut all_pairs: Vec<((usize, usize), (usize, usize), f32)> = vec![];

    for i in 0..all_embs.len() {
        for j in (i + 1)..all_embs.len() {
            let ((p_i, idx_i), emb_i) = &all_embs[i];
            let ((p_j, idx_j), emb_j) = &all_embs[j];

            if p_i != p_j {
                let sim = cosine_similarity(emb_i, emb_j);
                all_pairs.push(((*p_i, *idx_i), (*p_j, *idx_j), sim));
            }
        }
    }

    all_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    // Build person name lookup
    let person_names: HashMap<usize, &str> = PERSONS.iter().enumerate()
        .map(|(i, (p, _))| (i, *p))
        .collect();

    eprintln!("Top 10 hard negatives (most similar cross-person pairs):");
    for (i, ((p_i, idx_i), (p_j, idx_j), sim)) in all_pairs.iter().take(10).enumerate() {
        let name_i = person_names.get(p_i).unwrap_or(&"Unknown");
        let name_j = person_names.get(p_j).unwrap_or(&"Unknown");
        eprintln!("  {}: {}_{:02} vs {}_{:02} = {:.4}", i + 1, name_i, idx_i, name_j, idx_j, sim);
    }

    let hard_neg_rows: Vec<Vec<String>> = all_pairs.iter().take(20).map(|((p_i, idx_i), (p_j, idx_j), s)| {
        let name_i = person_names.get(p_i).unwrap_or(&"Unknown");
        let name_j = person_names.get(p_j).unwrap_or(&"Unknown");
        vec![format!("{}_{:02}", name_i, idx_i), format!("{}_{:02}", name_j, idx_j), format!("{:.4}", s)]
    }).collect();

    write_csv(
        &Path::new(OUT_DIR).join("hard_negatives.csv"),
        &["img1", "img2", "similarity"],
        &hard_neg_rows,
    );

    // ─── Final Report ─────────────────────────────────────────────────────────
    eprintln!("\n=== DONE ===");
    eprintln!("Output: {}", OUT_DIR);
    eprintln!("Total time: {:.1}s", start.elapsed().as_secs_f32());

    let top1_pct = 100.0 * top1_hit as f32 / total_queries as f32;
    let top3_pct = 100.0 * top3_hit as f32 / total_queries as f32;
    let top5_pct = 100.0 * top5_hit as f32 / total_queries as f32;
    let pass_pct = 100.0 * pass_count as f32 / total_queries as f32;

    // Compute hard negative FAR at threshold 0.5
    let hard_neg_at_05 = all_pairs.iter().filter(|(_, _, s)| *s >= 0.5).count();
    let total_neg_pairs = all_pairs.len();
    let hard_neg_far = 100.0 * hard_neg_at_05 as f32 / total_neg_pairs as f32;

    let verdict = if top1_pct >= 90.0 && hard_neg_far < 5.0 && pass_pct >= 95.0 && median_margin > 0.05 && p10_margin > 0.0 {
        "READY_FOR_PROTOTYPE"
    } else if top1_pct >= 80.0 && median_margin > 0.0 {
        "NEEDS_MORE_DATA"
    } else {
        "BODY_REID_FAIL"
    };

    let report = format!(r#"# Phase 13 — Multi-Person Body Prototype Validation

## Dataset
- Persons: George_W_Bush (10), Colin_Powell (10), Tony_Blair (10)
- Total: 30 images
- Protocol: LOO (gallery=N-1, query=remaining)

## Phase 1: LOO Cross-Validation

| Person | Queries | PASS | Top-1 | Top-3 | Top-5 |
|--------|---------|------|-------|-------|-------|
{}

## Phase 2: Aggregate Metrics

- Total LOO queries: {}
- PASS rate: {}/{} ({:.1}%)
- Top-1 accuracy: {:.1}%
- Top-3 accuracy: {:.1}%
- Top-5 accuracy: {:.1}%
- Median margin: {:.4}
- P10 margin: {:.4}

## Phase 3: Core Size Experiment

| Core Size | Avg Margin | Avg First Pos |
|-----------|------------|---------------|
{}

## Phase 4: ROC/FAR/FRR/AUC/EER

- EER (approx): {:.2}
- AUC: {:.4}

## Phase 5: Face vs Body vs Fusion

| α | Accuracy |
|---|----------|
{}

## Phase 7: Hard Negative Test

Hard negative FAR at τ=0.50: {:.1}% ({}/{} pairs)

## PASS Criteria Check

| Criterion | Threshold | Actual | Status |
|-----------|-----------|--------|--------|
| Top-1 accuracy | ≥90% | {:.1}% | {} |
| Hard-negative FAR | <5% | {:.1}% | {} |
| LOO PASS rate | ≥95% | {:.1}% | {} |
| Median margin | >0.05 | {:.4} | {} |
| P10 margin | >0 | {:.4} | {} |

## Final Verdict

{}
"#,
        loo_rows.iter().map(|r| format!("| {} | {} | {} | {} | {} |", r[0], "1", r[9], r[6], r[7])).collect::<Vec<_>>().join("\n"),
        total_queries,
        pass_count, total_queries, pass_pct,
        top1_pct, top3_pct, top5_pct,
        median_margin,
        p10_margin,
        core_size_rows.iter().map(|r| format!("| {} | {} | {} |", r[0], r[2], r[3])).collect::<Vec<_>>().join("\n"),
        eer,
        auc,
        fusion_rows.iter().map(|r| format!("| {} | {} |", r[0], r[3])).collect::<Vec<_>>().join("\n"),
        hard_neg_far, hard_neg_at_05, total_neg_pairs,
        top1_pct, if top1_pct >= 90.0 { "PASS" } else { "FAIL" },
        hard_neg_far, if hard_neg_far < 5.0 { "PASS" } else { "FAIL" },
        pass_pct, if pass_pct >= 95.0 { "PASS" } else { "FAIL" },
        median_margin, if median_margin > 0.05 { "PASS" } else { "FAIL" },
        p10_margin, if p10_margin > 0.0 { "PASS" } else { "FAIL" },
        verdict,
    );

    fs::write(Path::new(OUT_DIR).join("final_report.md"), &report).ok();
    println!("{}", report);
}
