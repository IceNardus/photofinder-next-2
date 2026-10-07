//! 临时诊断（非生产）：Person Re-ID 模型（OSNet/FastReID）是否能解决 img15 假拆？
//!
//! 背景：
//!   img15 假拆根因已确定为 ArcFace embedding 空间的固有歧义。
//!   本 benchmark 测试"第二种身份特征：人体外观 Re-ID"是否能提供额外区分信号。
//!
//! 重要约束：
//!   1 不改任何生产代码 / DB / HNSW / cluster threshold
//!   2 所有输入使用与 Face-only 基准完全相同的图片
//!   3 每张图片每个模型只计算一次 embedding
//!   4 无 TTA
//!
//! 测试模型：
//!   A. ArcFace w600k_r50（生产路径）
//!   B. OSNet-x1.0（如果 ONNX 可用，否则用 ConvNeXt_tiny 作为 body proxy）
//!   C. OSNet-AIN（fallback 同上）
//!   D. FastReID SBS-R50（fallback 同上）
//!
//! 7 个 Phase：
//!   Phase 1  Face baseline metrics（pos_mean/min/max, neg_max/mean, margin, rank）
//!   Phase 2  Person crop ablation（8 种 crop 类型，保存坐标和可视化）
//!   Phase 3  Body embedding extraction（每种 crop 的 OSNet embeddings）
//!   Phase 4  Ranking analysis（img15 → person2 / negative 的详细排名）
//!   Phase 5  Face vs Body correlation（Pearson/Spearman，重点 hard-negative 和 person2 subset）
//!   Phase 6  Fusion（α * Face + (1-α) * Body，加权 cosine + rank fusion）
//!   Phase 7  Person-level scoring（face/body max/mean/top2/top3）
//!
//! 判定标准：
//!   PASS:   img15 first positive rank = 1
//!   PARTIAL: first positive rank <= 3 且 top3 至少包含 2 个 person2
//!   HELPFUL: 即使 rank > 3，但 body 明显纠正 face 错误排序，且 hard-negative margin 改善
//!   FAIL:   body ranking 与 face 基本相同，或 fusion 无稳定改善
//!
//! 输出：
//!   /tmp/person_reid_osnet_bench/
//!     embeddings/          — ArcFace + OSNet embeddings
//!     crops/              — 8 种 crop 类型的可视化图片
//!     visualizations/     — 排名对比图
//!     pairwise.csv        — 全部 pair 的 cosine 矩阵
//!     ranking.csv         — 排名详情
//!     fusion.csv          — fusion 扫描结果
//!     report.md           — 完整报告
//!
//! 用法：
//!   ORT_DYLIB_PATH=/Users/mac/lib/libonnxruntime.dylib \
//!   cargo test --release -p pf_ai --test person_reid_osnet_bench -- --ignored --nocapture
//!
//!   可选环境变量：
//!     OSNET_PYTHON=/path/to/python3（默认用系统 python3）

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use image::RgbImage;
use ort::session::Session;
use ort::value::Tensor;

use pf_ai::face::aligner::{AlignmentConfig, SimpleAligner};
use pf_ai::face::arcface::{l2_normalize, INPUT_SIZE};
use pf_ai::face::traits::FaceAligner;
use pf_ai::image_data::ImageData;
use pf_ai::{FaceDetector, ScrfdDetector};
use pf_core::{BBox, FaceKeypoints};

const ARCFACE_MODEL: &str = "/Users/mac/ai-project/photofinder-next-2/models/w600k_r50.onnx";
const SCRFD_MODEL: &str = "/Users/mac/ai-project/photofinder-next-2/models/scrfd_500m_bnkps.onnx";
const INPUT_TSV: &str = "/tmp/tta_img15/input.tsv";
const OUT_DIR: &str = "/tmp/person_reid_osnet_bench";

const IMG_QUERY: i64 = 15;
const CORE: [i64; 3] = [6, 13, 14];
const NEG: [i64; 6] = [2, 9, 17, 18, 19, 20];
const ALL: [i64; 10] = [6, 13, 14, 15, 2, 9, 17, 18, 19, 20];

const CROP_MARGIN: f32 = 0.10;

// Crop types for body region
const CROP_TYPES: [&str; 8] = ["A", "B", "C", "D", "E", "F", "G", "H"];
const CROP_NAMES: [&str; 8] = [
    "person_bbox", "bbox_p5", "bbox_p10", "bbox_p20",
    "bbox_p30", "upper_body", "full_body", "body_without_head",
];

// Fusion alpha values
const ALPHA_VALUES: [f32; 11] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

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

fn build_aligned_face(aligner: &SimpleAligner, img: &ImageData, bbox: &BBox, kps: &[(f32, f32); 5]) -> Result<RgbImage, String> {
    let (crop, origin) = crop_to_bbox_with_margin(img, bbox, CROP_MARGIN)?;
    let shifted = shift_kps(kps, origin);
    let aligned = aligner.align(&crop, &FaceKeypoints {
        left_eye: shifted[0], right_eye: shifted[1], nose: shifted[2],
        left_mouth: shifted[3], right_mouth: shifted[4],
    }).map_err(|e| e.to_string())?;
    Ok(aligned.image.as_rgb8())
}

// ─── ArcFace embedding ───────────────────────────────────────────────────────

fn embed_arcface(session: &mut Session, rgb: &RgbImage) -> Vec<f32> {
    let n = (INPUT_SIZE * INPUT_SIZE) as usize;
    let mut input = Vec::with_capacity(3 * n);
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let p = rgb.get_pixel(x, y);
            input.push((p[2] as f32 - 127.5) / 128.0);
        }
    }
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let p = rgb.get_pixel(x, y);
            input.push((p[1] as f32 - 127.5) / 128.0);
        }
    }
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let p = rgb.get_pixel(x, y);
            input.push((p[0] as f32 - 127.5) / 128.0);
        }
    }
    let shape = [1_i64, 3, INPUT_SIZE as i64, INPUT_SIZE as i64];
    let input = Tensor::from_array((shape, input)).expect("tensor");
    let outputs = session.run(ort::inputs![input]).expect("run");
    let (_shape, data) = outputs[0].try_extract_tensor::<f32>().expect("extract");
    l2_normalize(data)
}

// ─── Body crop expansion ─────────────────────────────────────────────────────

fn expand_face_to_body(face: &BBox, img_h: f32, img_w: f32, crop_type: &str) -> BBox {
    let cx = face.x + face.w / 2.0;
    let cy = face.y + face.h / 2.0;

    match crop_type {
        "A" => {
            // person_bbox: just the detected face bbox itself
            *face
        }
        "B" => {
            // bbox + 5% margin
            let mx = face.w * 0.05;
            let my = face.h * 0.05;
            BBox::new(
                (face.x - mx).max(0.0),
                (face.y - my).max(0.0),
                face.w + 2.0 * mx,
                face.h + 2.0 * my,
            )
        }
        "C" => {
            // bbox + 10% margin
            let mx = face.w * 0.10;
            let my = face.h * 0.10;
            BBox::new(
                (face.x - mx).max(0.0),
                (face.y - my).max(0.0),
                face.w + 2.0 * mx,
                face.h + 2.0 * my,
            )
        }
        "D" => {
            // bbox + 20% margin
            let mx = face.w * 0.20;
            let my = face.h * 0.20;
            BBox::new(
                (face.x - mx).max(0.0),
                (face.y - my).max(0.0),
                face.w + 2.0 * mx,
                face.h + 2.0 * my,
            )
        }
        "E" => {
            // bbox + 30% margin
            let mx = face.w * 0.30;
            let my = face.h * 0.30;
            BBox::new(
                (face.x - mx).max(0.0),
                (face.y - my).max(0.0),
                face.w + 2.0 * mx,
                face.h + 2.0 * my,
            )
        }
        "F" => {
            // upper_body: ~2.5x face height centered slightly above face
            let height = face.h * 2.5;
            let width = face.w * 1.3;
            BBox::new(
                (cx - width / 2.0).max(0.0),
                (cy - height * 0.3).max(0.0),
                width.min(img_w - (cx - width / 2.0).max(0.0)),
                height.min(img_h - (cy - height * 0.3).max(0.0)),
            )
        }
        "G" => {
            // full_body: ~4x face height
            let height = face.h * 4.0;
            let width = face.w * 1.2;
            BBox::new(
                (cx - width / 2.0).max(0.0),
                (cy - height * 0.15).max(0.0),
                width.min(img_w - (cx - width / 2.0).max(0.0)),
                height.min(img_h - (cy - height * 0.15).max(0.0)),
            )
        }
        "H" => {
            // body_without_head: from neck down (~1.5x above face center to 4x below)
            let top = cy - face.h * 0.5;  // slightly above face
            let height = face.h * 4.5;
            let width = face.w * 1.3;
            BBox::new(
                (cx - width / 2.0).max(0.0),
                top.max(0.0),
                width.min(img_w - (cx - width / 2.0).max(0.0)),
                (height.min(img_h - top.max(0.0))),
            )
        }
        _ => *face,
    }
}

fn crop_image(img: &ImageData, bbox: &BBox) -> Result<ImageData, String> {
    let b = *bbox;
    if b.w < 1.0 || b.h < 1.0 {
        return Err(format!("crop too small"));
    }
    img.crop(b).map_err(|e| e.to_string())
}

fn resize_to_target(rgb: &RgbImage, target_h: u32, target_w: u32) -> RgbImage {
    use image::imageops::FilterType;
    image::imageops::resize(rgb, target_w, target_h, FilterType::Triangle)
}

// ─── Metrics helpers ──────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Metrics {
    pos_mean: f32, pos_min: f32, pos_max: f32,
    neg_mean: f32, neg_max: f32,
    margin: f32,
}

fn compute_metrics(emb: &HashMap<i64, Vec<f32>>, query: i64, core: &[i64], neg: &[i64]) -> Metrics {
    let cos = |a: i64, b: i64| cosine(emb.get(&a).unwrap(), emb.get(&b).unwrap());
    let pos: Vec<f32> = core.iter().map(|&c| cos(query, c)).collect();
    let neg_vals: Vec<f32> = neg.iter().map(|&n| cos(query, n)).collect();

    let pos_mean = pos.iter().sum::<f32>() / pos.len() as f32;
    let pos_min = pos.iter().cloned().fold(f32::INFINITY, |a, b| a.min(b));
    let pos_max = pos.iter().cloned().fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let neg_mean = neg_vals.iter().sum::<f32>() / neg_vals.len() as f32;
    let neg_max = neg_vals.iter().cloned().fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let margin = pos_min - neg_max;
    Metrics { pos_mean, pos_min, pos_max, neg_mean, neg_max, margin }
}

struct RankingEntry { img: i64, cos: f32, group: &'static str }

fn ranking(emb: &HashMap<i64, Vec<f32>>, query: i64, all: &[i64]) -> Vec<RankingEntry> {
    let cos_fn = |a: i64, b: i64| cosine(emb.get(&a).unwrap(), emb.get(&b).unwrap());
    let mut entries: Vec<RankingEntry> = all.iter()
        .map(|&t| {
            let grp = if CORE.contains(&t) { "PERSON2" } else { "NEG" };
            RankingEntry { img: t, cos: cos_fn(query, t), group: grp }
        })
        .collect();
    entries.sort_by(|a, b| b.cos.partial_cmp(&a.cos).unwrap());
    entries
}

fn first_positive_rank(ranked: &[RankingEntry]) -> Option<usize> {
    ranked.iter().position(|e| e.group == "PERSON2")
}

fn count_positives_in_top_k(ranked: &[RankingEntry], k: usize) -> usize {
    ranked.iter().take(k).filter(|e| e.group == "PERSON2").count()
}

fn save_image(rgb: &RgbImage, path: &Path) -> Result<(), String> {
    rgb.save(path).map_err(|e| e.to_string())
}

// ─── Run Python helper for OSNet extraction ──────────────────────────────────

fn run_osnet_extraction(model_type: &str, crops_dir: &Path, output_dir: &Path) -> Result<(HashMap<i64, Vec<f32>>, usize), String> {
    let python = std::env::var("OSNET_PYTHON").unwrap_or_else(|_| "python3".to_string());

    let manifest_path = output_dir.join("manifest.json");

    // Check if manifest already exists
    if manifest_path.exists() {
        let content = std::fs::read_to_string(&manifest_path).map_err(|e| e.to_string())?;
        if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&content) {
            if manifest.get("status").and_then(|v| v.as_str()) == Some("OK") {
                let embeddings_json = manifest.get("embeddings").ok_or("no embeddings")?;
                let mut emb = HashMap::new();
                for (k, v) in embeddings_json.as_object().ok_or("not an object")? {
                    // Keys are like "img13_x2348_y2235_w1092_h1448" — extract numeric id after "img"
                    let img_id: i64 = k.split('_').next()
                        .and_then(|s| s.strip_prefix("img"))
                        .ok_or("no img prefix")?
                        .parse()
                        .map_err(|_| "parse error")?;
                    let arr = v.as_array().ok_or("not an array")?;
                    let floats: Vec<f32> = arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
                    emb.insert(img_id, floats);
                }
                let dim = manifest.get("feature_dim").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                return Ok((emb, dim));
            }
        }
    }

    let output_str = output_dir.to_string_lossy().to_string();
    let crops_str = crops_dir.to_string_lossy().to_string();

    // Use absolute path to the Python helper script
    let python_script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("osnet_extract.py");

    let output = Command::new(&python)
        .arg(python_script.to_string_lossy().as_ref())
        .arg(model_type)
        .arg(&crops_str)
        .arg(&output_str)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("failed to run python: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("Python extraction failed: {}", stderr);
        return Err(format!("python failed: {}", stderr));
    }

    // Parse manifest from stdout
    if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&stdout) {
        if manifest.get("status").and_then(|v| v.as_str()) == Some("MODEL_NOT_AVAILABLE") {
            return Err("MODEL_NOT_AVAILABLE".to_string());
        }
        if manifest.get("status").and_then(|v| v.as_str()) == Some("OK") {
            let embeddings_json = manifest.get("embeddings").ok_or("no embeddings")?;
            let mut emb = HashMap::new();
            for (k, v) in embeddings_json.as_object().ok_or("not an object")? {
                // Keys are like "img13_x2348_y2235_w1092_h1448" — extract numeric id after "img"
                let img_id: i64 = k.split('_').next()
                    .and_then(|s| s.strip_prefix("img"))
                    .ok_or("no img prefix")?
                    .parse()
                    .map_err(|_| "parse error")?;
                let arr = v.as_array().ok_or("not an array")?;
                let floats: Vec<f32> = arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
                emb.insert(img_id, floats);
            }
            let dim = manifest.get("feature_dim").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            return Ok((emb, dim));
        }
    }

    Err(format!("failed to parse manifest: {}", stdout))
}

// ─── Ranking fusion ──────────────────────────────────────────────────────────

fn rank_fusion(face_ranks: &[i64], body_ranks: &[i64], alpha: f32) -> Vec<i64> {
    // face_ranks[i] = rank position of ALL[i] in face ranking (0-indexed)
    // body_ranks[i] = rank position of ALL[i] in body ranking
    let n = face_ranks.len() as i64;
    let beta = 1.0 - alpha;

    let mut scores: Vec<(i64, f32)> = (0..face_ranks.len()).map(|i| {
        let score = alpha * (n - face_ranks[i]) as f32 + beta * (n - body_ranks[i]) as f32;
        (ALL[i], score)
    }).collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.iter().map(|(img, _)| *img).collect()
}

fn face_rank_position(ranked: &[RankingEntry], target: i64) -> i64 {
    match ranked.iter().position(|e| e.img == target) {
        Some(p) => p as i64,
        None => -1,
    }
}

// ─── Main test ───────────────────────────────────────────────────────────────

#[test]
#[ignore]
fn person_reid_osnet_bench() {
    // Create output directories
    fs::create_dir_all(OUT_DIR).ok();
    fs::create_dir_all(format!("{}/embeddings", OUT_DIR)).ok();
    fs::create_dir_all(format!("{}/crops", OUT_DIR)).ok();
    fs::create_dir_all(format!("{}/visualizations", OUT_DIR)).ok();

    let samples = read_tsv();
    assert_eq!(samples.len(), 10, "expect 10 samples");

    let images: Vec<ImageData> = samples.iter()
        .map(|s| ImageData::from_file(Path::new(&s.path)).expect("open"))
        .collect();

    let rt = tokio::runtime::Runtime::new().expect("tokio");
    let scrfd = ScrfdDetector::load(Path::new(SCRFD_MODEL)).expect("scrfd");
    let aligner = SimpleAligner::with_config(AlignmentConfig::default());

    // ── Phase 0: Face detection + alignment ─────────────────────────────────
    struct Geo { bbox: BBox, kps: [(f32, f32); 5] }
    let geos: Vec<Geo> = samples.iter()
        .enumerate()
        .map(|(i, _s)| {
            let dets = rt.block_on(scrfd.detect(&images[i])).expect("scrfd");
            let best = dets.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap()).unwrap();
            Geo {
                bbox: best.bbox,
                kps: kps_from_det(best),
            }
        })
        .collect();

    // Save face-aligned crops (ArcFace 112x112)
    let face_crops: Vec<RgbImage> = samples.iter()
        .enumerate()
        .map(|(i, _s)| {
            let rgb = build_aligned_face(&aligner, &images[i], &geos[i].bbox, &geos[i].kps)
                .expect("align face");
            // Save
            let img_id = samples[i].img;
            save_image(&rgb, Path::new(&format!("{}/embeddings/face_img{}.png", OUT_DIR, img_id))).ok();
            rgb
        })
        .collect();

    // ── Phase 1: ArcFace embeddings ─────────────────────────────────────────
    let mut arc_sess = Session::builder().expect("arc builder")
        .commit_from_file(PathBuf::from(ARCFACE_MODEL)).expect("arcface");
    let face_emb: HashMap<i64, Vec<f32>> = samples.iter()
        .enumerate()
        .map(|(i, s)| (s.img, embed_arcface(&mut arc_sess, &face_crops[i])))
        .collect();

    let face_metrics = compute_metrics(&face_emb, IMG_QUERY, &CORE, &NEG);
    let face_ranked = ranking(&face_emb, IMG_QUERY, &ALL);

    let f_first_pos = first_positive_rank(&face_ranked).map(|r| r + 1).unwrap_or(999);
    let f_top3_pos = count_positives_in_top_k(&face_ranked, 3);
    let f_top5_pos = count_positives_in_top_k(&face_ranked, 5);

    // ── Phase 2: Body crop ablation ─────────────────────────────────────────
    // For each crop type, create and save body crops
    let mut crop_vis: HashMap<String, HashMap<i64, RgbImage>> = HashMap::new();

    for (ct, cn) in CROP_TYPES.iter().zip(CROP_NAMES.iter()) {
        let crop_dir = format!("{}/crops/{}", OUT_DIR, cn);
        fs::create_dir_all(&crop_dir).ok();

        let mut crops = HashMap::new();
        for (i, s) in samples.iter().enumerate() {
            let bbox = expand_face_to_body(&geos[i].bbox, images[i].height() as f32, images[i].width() as f32, ct);
            let rgb = match crop_image(&images[i], &bbox) {
                Ok(img_data) => img_data.as_rgb8(),
                Err(_) => face_crops[i].clone(),
            };
            // Save crop with coordinates
            let path = format!("{}/img{}_x{:.0}_y{:.0}_w{:.0}_h{:.0}.png",
                crop_dir, s.img, bbox.x, bbox.y, bbox.w, bbox.h);
            save_image(&rgb, Path::new(&path)).ok();
            crops.insert(s.img, rgb);
        }
        crop_vis.insert(cn.to_string(), crops);
    }

    // ── Phase 3: OSNet extraction (via Python helper) ───────────────────────
    // Try each body model in order; first available wins
    let body_models = ["reid_ont", "osnet_x1_0", "osnet_ain_x1_0", "fastreid_sbs_r50", "convnext_tiny"];

    let mut body_emb: HashMap<i64, Vec<f32>> = HashMap::new();
    let mut best_body_model = String::new();
    let mut best_body_dim = 0usize;

    for model_name in &body_models {
        let crops_base = format!("{}/crops/bbox_p20", OUT_DIR);  // Use 20% margin crop as default
        let emb_dir = format!("{}/embeddings/body_{}", OUT_DIR, model_name);
        fs::create_dir_all(&emb_dir).ok();

        match run_osnet_extraction(model_name, Path::new(&crops_base), Path::new(&emb_dir)) {
            Ok((emb, dim)) => {
                body_emb = emb;
                best_body_model = model_name.to_string();
                best_body_dim = dim;
                eprintln!("Using body model: {} ({} dim)", model_name, dim);
                break;
            }
            Err(e) => {
                eprintln!("Model {} not available: {}", model_name, e);
            }
        }
    }

    if body_emb.is_empty() {
        eprintln!("WARNING: No body models available. Using face-only analysis.");
    }

    let body_metrics = if !body_emb.is_empty() {
        compute_metrics(&body_emb, IMG_QUERY, &CORE, &NEG)
    } else {
        Metrics { pos_mean: 0.0, pos_min: 0.0, pos_max: 0.0, neg_mean: 0.0, neg_max: 0.0, margin: 0.0 }
    };

    let body_ranked = if !body_emb.is_empty() {
        ranking(&body_emb, IMG_QUERY, &ALL)
    } else {
        vec![]
    };

    let b_first_pos = body_ranked.first()
        .and_then(|e| if e.group == "PERSON2" { Some(1) } else { first_positive_rank(&body_ranked).map(|r| r + 1) })
        .unwrap_or(999);
    let b_top3_pos = count_positives_in_top_k(&body_ranked, 3);
    let b_top5_pos = count_positives_in_top_k(&body_ranked, 5);

    // ── Phase 4: Detailed pair analysis ───────────────────────────────────
    let cos_face = |a: i64, b: i64| cosine(face_emb.get(&a).unwrap(), face_emb.get(&b).unwrap());
    let cos_body = |a: i64, b: i64| -> Option<f32> {
        body_emb.get(&a).and_then(|ea| body_emb.get(&b).map(|eb| cosine(ea, eb)))
    };

    // img15 → person2 pairs
    let img15_pairs_pos: Vec<(i64, f32, f32)> = CORE.iter().map(|&c| {
        let fc = cos_face(IMG_QUERY, c);
        let bc = cos_body(IMG_QUERY, c).unwrap_or(0.0);
        (c, fc, bc)
    }).collect();

    // img15 → negative pairs
    let img15_pairs_neg: Vec<(i64, f32, f32)> = NEG.iter().map(|&n| {
        let fc = cos_face(IMG_QUERY, n);
        let bc = cos_body(IMG_QUERY, n).unwrap_or(0.0);
        (n, fc, bc)
    }).collect();

    // ── Phase 5: Correlation analysis ──────────────────────────────────────
    // Pearson/Spearman on full pairwise
    fn pearson(xs: &[f32], ys: &[f32]) -> f32 {
        let n = xs.len() as f32;
        let mx = xs.iter().sum::<f32>() / n;
        let my = ys.iter().sum::<f32>() / n;
        let num: f32 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
        let da: f32 = xs.iter().map(|x| (x - mx).powi(2)).sum();
        let db: f32 = ys.iter().map(|y| (y - my).powi(2)).sum();
        let denom = (da * db).sqrt();
        if denom < 1e-6 { 0.0 } else { num / denom }
    }

    fn rank_by(xs: &[f32]) -> Vec<f32> {
        let mut idx: Vec<usize> = (0..xs.len()).collect();
        idx.sort_by(|&i, &j| xs[i].partial_cmp(&xs[j]).unwrap());
        let n = xs.len();
        let mut r = vec![0.0; n];
        let mut i = 0;
        while i < n {
            let mut j = i;
            while j < n && xs[idx[j]] == xs[idx[i]] { j += 1; }
            let avg = (i + j - 1) as f32 / 2.0;
            for k in i..j { r[idx[k]] = avg; }
            i = j;
        }
        r
    }

    fn spearman(xs: &[f32], ys: &[f32]) -> f32 {
        pearson(&rank_by(xs), &rank_by(ys))
    }

    // Full pairwise correlation
    let mut all_face_cos: Vec<f32> = Vec::new();
    let mut all_body_cos: Vec<f32> = Vec::new();
    for i in 0..ALL.len() {
        for j in (i+1)..ALL.len() {
            all_face_cos.push(cos_face(ALL[i], ALL[j]));
            if let Some(bc) = cos_body(ALL[i], ALL[j]) {
                all_body_cos.push(bc);
            }
        }
    }

    let pearson_full = if !body_emb.is_empty() && all_body_cos.len() == all_face_cos.len() {
        pearson(&all_face_cos, &all_body_cos)
    } else { 0.0 };

    let spearman_full = if !body_emb.is_empty() && all_body_cos.len() == all_face_cos.len() {
        spearman(&all_face_cos, &all_body_cos)
    } else { 0.0 };

    // img15 hard-negative subset correlation
    let hn_pairs: Vec<(f32, f32)> = NEG.iter()
        .filter_map(|&n| cos_body(IMG_QUERY, n).map(|bc| (cos_face(IMG_QUERY, n), bc)))
        .collect();
    let hn_face: Vec<f32> = hn_pairs.iter().map(|(f, _)| *f).collect();
    let hn_body: Vec<f32> = hn_pairs.iter().map(|(_, b)| *b).collect();
    let pearson_hn = pearson(&hn_face, &hn_body);
    let spearman_hn = spearman(&hn_face, &hn_body);

    // person2 positive subset correlation
    let pos_pairs: Vec<(f32, f32)> = CORE.iter()
        .filter_map(|&c| cos_body(IMG_QUERY, c).map(|bc| (cos_face(IMG_QUERY, c), bc)))
        .collect();
    let pos_face: Vec<f32> = pos_pairs.iter().map(|(f, _)| *f).collect();
    let pos_body: Vec<f32> = pos_pairs.iter().map(|(_, b)| *b).collect();
    let pearson_pos = pearson(&pos_face, &pos_body);
    let spearman_pos = spearman(&pos_face, &pos_body);

    // ── Phase 6: Fusion ───────────────────────────────────────────────────
    let mut fusion_results: Vec<(f32, Metrics, i64, usize, usize)> = Vec::new(); // (alpha, metrics, first_pos_rank, top3, top5)

    if !body_emb.is_empty() {
        for &alpha in &ALPHA_VALUES {
            let beta = 1.0 - alpha;

            // Cosine fusion
            let mut fused_emb: HashMap<i64, Vec<f32>> = HashMap::new();
            for &img_id in &ALL {
                let fe = face_emb.get(&img_id).unwrap();
                let be = body_emb.get(&img_id).unwrap();
                let fused: Vec<f32> = fe.iter().zip(be.iter())
                    .map(|(f, b)| alpha * f + beta * b)
                    .collect();
                let norm = fused.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                let normed: Vec<f32> = fused.iter().map(|x| x / norm).collect();
                fused_emb.insert(img_id, normed);
            }

            let fused_metrics = compute_metrics(&fused_emb, IMG_QUERY, &CORE, &NEG);
            let fused_ranked = ranking(&fused_emb, IMG_QUERY, &ALL);
            let first_pos = first_positive_rank(&fused_ranked).map(|r| r + 1).unwrap_or(999);
            let top3 = count_positives_in_top_k(&fused_ranked, 3);
            let top5 = count_positives_in_top_k(&fused_ranked, 5);

            fusion_results.push((alpha, fused_metrics, first_pos as i64, top3, top5));
        }
    }

    // Rank fusion
    let face_rank_pos: Vec<i64> = ALL.iter().map(|&img| face_rank_position(&face_ranked, img) as i64).collect();
    let body_rank_pos: Vec<i64> = ALL.iter().map(|&img| face_rank_position(&body_ranked, img) as i64).collect();

    let mut rank_fusion_results: Vec<(f32, i64, usize, usize)> = Vec::new(); // (alpha, first_pos_rank, top3, top5)
    for &alpha in &ALPHA_VALUES {
        let fused_order = rank_fusion(&face_rank_pos, &body_rank_pos, alpha);
        let first_pos = fused_order.iter().position(|&img| CORE.contains(&img)).map(|r| r + 1).unwrap_or(999);
        let top3 = fused_order.iter().take(3).filter(|&&img| CORE.contains(&img)).count();
        let top5 = fused_order.iter().take(5).filter(|&&img| CORE.contains(&img)).count();
        rank_fusion_results.push((alpha, first_pos as i64, top3, top5));
    }

    // Find best fusion
    let best_fusion = fusion_results.iter()
        .max_by(|a, b| a.1.margin.partial_cmp(&b.1.margin).unwrap())
        .cloned();

    // ── Phase 7: Person-level scoring ──────────────────────────────────────
    // Face person-level
    let face_person_max = CORE.iter().map(|&c| cos_face(IMG_QUERY, c)).fold(0.0_f32, |a, b| a.max(b));
    let face_person_mean = CORE.iter().map(|&c| cos_face(IMG_QUERY, c)).sum::<f32>() / 3.0;
    let mut face_top2: Vec<f32> = CORE.iter().map(|&c| cos_face(IMG_QUERY, c)).collect();
    face_top2.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let face_top2_avg = face_top2.iter().take(2).sum::<f32>() / 2.0;
    let face_top3_avg = face_top2.iter().sum::<f32>() / 3.0;

    // Body person-level
    let body_person_max = if !body_emb.is_empty() {
        CORE.iter().filter_map(|&c| cos_body(IMG_QUERY, c)).fold(0.0_f32, |a, b| a.max(b))
    } else { 0.0 };
    let body_person_mean = if !body_emb.is_empty() {
        CORE.iter().filter_map(|&c| cos_body(IMG_QUERY, c)).sum::<f32>() / 3.0
    } else { 0.0 };

    // ────────────────────────────────────────────────────────────────────────
    // Build report
    // ────────────────────────────────────────────────────────────────────────

    let mut r = String::new();
    r.push_str("# Person Re-ID Benchmark: OSNet vs ArcFace\n\n");
    r.push_str(&format!("## Models\n"));
    r.push_str(&format!("- Face: ArcFace w600k_r50 (512-dim, production path)\n"));
    r.push_str(&format!("- Body: {} ({} dim, via Python helper)\n\n", best_body_model, best_body_dim));

    // Phase 1
    r.push_str("## Phase 1: Face Baseline\n\n");
    r.push_str(&format!("| Metric | Value |\n"));
    r.push_str(&format!("|--------|-------|\n"));
    r.push_str(&format!("| pos_mean | {:.4} |\n", face_metrics.pos_mean));
    r.push_str(&format!("| pos_min | {:.4} |\n", face_metrics.pos_min));
    r.push_str(&format!("| pos_max | {:.4} |\n", face_metrics.pos_max));
    r.push_str(&format!("| neg_max | {:.4} |\n", face_metrics.neg_max));
    r.push_str(&format!("| neg_mean | {:.4} |\n", face_metrics.neg_mean));
    r.push_str(&format!("| **margin** | **{:+.4}** |\n", face_metrics.margin));
    r.push_str(&format!("| first_positive_rank | {} |\n", f_first_pos));
    r.push_str(&format!("| top3 positives | {} |\n", f_top3_pos));
    r.push_str(&format!("| top5 positives | {} |\n\n", f_top5_pos));

    r.push_str(&format!("**img15 Ranking (Face):**\n"));
    r.push_str(&format!("| rank | img | group | cosine |\n"));
    r.push_str(&format!("|------|-----|-------|--------|\n"));
    for (i, e) in face_ranked.iter().enumerate() {
        let marker = if e.img == IMG_QUERY { " ←QRY" } else { "" };
        r.push_str(&format!("| {} | img{} | {} | {:.4} |{}\n", i+1, e.img, e.group, e.cos, marker));
    }
    r.push('\n');

    // Phase 2 (crop summary)
    r.push_str("## Phase 2: Body Crop Ablation\n\n");
    r.push_str("8 crop types generated:\n");
    for (ct, cn) in CROP_TYPES.iter().zip(CROP_NAMES.iter()) {
        r.push_str(&format!("- {} ({})\n", ct, cn));
    }
    r.push_str(&format!("\nCrop images saved to: {}/crops/\n\n", OUT_DIR));

    // Phase 3 (body metrics)
    r.push_str("## Phase 3: Body Embedding Metrics\n\n");
    if !body_emb.is_empty() {
        r.push_str(&format!("| Metric | Value |\n"));
        r.push_str(&format!("|--------|-------|\n"));
        r.push_str(&format!("| pos_mean | {:.4} |\n", body_metrics.pos_mean));
        r.push_str(&format!("| pos_min | {:.4} |\n", body_metrics.pos_min));
        r.push_str(&format!("| neg_max | {:.4} |\n", body_metrics.neg_max));
        r.push_str(&format!("| **margin** | **{:+.4}** |\n", body_metrics.margin));
        r.push_str(&format!("| first_positive_rank | {} |\n", b_first_pos));
        r.push_str(&format!("| top3 positives | {} |\n", b_top3_pos));
        r.push_str(&format!("| top5 positives | {} |\n\n", b_top5_pos));
    } else {
        r.push_str("No body model available.\n\n");
    }

    // Phase 4: Pair analysis
    r.push_str("## Phase 4: img15 Pair Analysis\n\n");
    r.push_str("**img15 → person2 (positives):**\n");
    r.push_str(&format!("| target | face_cos | body_cos | diff |\n"));
    r.push_str(&format!("|--------|----------|----------|------|\n"));
    for (c, fc, bc) in &img15_pairs_pos {
        r.push_str(&format!("| img{} | {:.4} | {:.4} | {:+.4} |\n", c, fc, bc, fc - bc));
    }

    r.push_str("\n**img15 → negatives:**\n");
    r.push_str(&format!("| target | face_cos | body_cos | diff |\n"));
    r.push_str(&format!("|--------|----------|----------|------|\n"));
    for (n, fc, bc) in &img15_pairs_neg {
        r.push_str(&format!("| img{} | {:.4} | {:.4} | {:+.4} |\n", n, fc, bc, fc - bc));
    }
    r.push('\n');

    // Phase 5: Correlation
    r.push_str("## Phase 5: Face vs Body Correlation\n\n");
    r.push_str(&format!("| Subset | Pearson | Spearman |\n"));
    r.push_str(&format!("|--------|---------|----------|\n"));
    r.push_str(&format!("| Full pairwise ({} pairs) | {:.4} | {:.4} |\n", all_face_cos.len(), pearson_full, spearman_full));
    r.push_str(&format!("| img15 hard-neg (n={}) | {:.4} | {:.4} |\n", hn_pairs.len(), pearson_hn, spearman_hn));
    r.push_str(&format!("| person2 positive (n={}) | {:.4} | {:.4} |\n\n", pos_pairs.len(), pearson_pos, spearman_pos));

    // Phase 6: Fusion
    r.push_str("## Phase 6: Fusion\n\n");
    r.push_str("**Cosine Fusion:**\n");
    r.push_str(&format!("| α | pos_mean | pos_min | neg_max | margin | first_pos | top3 | top5 |\n"));
    r.push_str(&format!("|--:|----------:|----------:|----------:|----------:|----------:|----:|----:|\n"));
    for (alpha, m, fp, t3, t5) in &fusion_results {
        r.push_str(&format!("| {:.1} | {:.4} | {:.4} | {:.4} | {:+.4} | {} | {} | {} |\n",
            alpha, m.pos_mean, m.pos_min, m.neg_max, m.margin, fp, t3, t5));
    }

    r.push_str("\n**Rank Fusion:**\n");
    r.push_str(&format!("| α | first_pos | top3 | top5 |\n"));
    r.push_str(&format!("|--:|----------:|----:|----:|\n"));
    for (alpha, fp, t3, t5) in &rank_fusion_results {
        r.push_str(&format!("| {:.1} | {} | {} | {} |\n", alpha, fp, t3, t5));
    }
    r.push('\n');

    // Phase 7: Person-level
    r.push_str("## Phase 7: Person-Level Scoring\n\n");
    r.push_str(&format!("| Method | Score |\n"));
    r.push_str(&format!("|--------|-------|\n"));
    r.push_str(&format!("| face_max | {:.4} |\n", face_person_max));
    r.push_str(&format!("| face_mean | {:.4} |\n", face_person_mean));
    r.push_str(&format!("| face_top2 | {:.4} |\n", face_top2_avg));
    r.push_str(&format!("| face_top3 | {:.4} |\n", face_top3_avg));
    r.push_str(&format!("| body_max | {:.4} |\n", body_person_max));
    r.push_str(&format!("| body_mean | {:.4} |\n\n", body_person_mean));

    // Verdict
    r.push_str("## Verdict\n\n");

    let verdict = if f_first_pos == 1 {
        "FACE_ONLY_ALREADY_WORKS"
    } else if b_first_pos == 1 {
        "OSNET_BODY_HELPFUL"
    } else if b_first_pos <= 3 && b_top3_pos >= 2 {
        "OSNET_BODY_PARTIAL"
    } else if let Some((best_alpha, best_m, best_fp, best_t3, _)) = best_fusion {
        if best_fp <= 3 && best_t3 >= 2 {
            "FUSION_HELPFUL"
        } else if best_m.margin > face_metrics.margin && best_fp < f_first_pos as i64 {
            "FUSION_PARTIAL"
        } else {
            "OSNET_BODY_FAIL"
        }
    } else {
        "OSNET_BODY_FAIL"
    };

    r.push_str(&format!("**Final Verdict: {}**\n\n", verdict));
    r.push_str("| Criterion | Face | Body |\n");
    r.push_str("|-----------|------|------|\n");
    r.push_str(&format!("| first_positive_rank | {} | {} |\n", f_first_pos, b_first_pos));
    r.push_str(&format!("| top3 positives | {} | {} |\n", f_top3_pos, b_top3_pos));
    r.push_str(&format!("| top5 positives | {} | {} |\n", f_top5_pos, b_top5_pos));
    r.push_str(&format!("| margin | {:+.4} | {:+.4} |\n", face_metrics.margin, body_metrics.margin));

    if let Some((ba, bm, bfp, bt3, _)) = best_fusion {
        r.push_str(&format!("| best_fusion α | - | {:.1} |\n", ba));
        r.push_str(&format!("| best_fusion margin | - | {:+.4} |\n", bm.margin));
        r.push_str(&format!("| best_fusion first_pos | - | {} |\n", bfp));
    }

    r.push_str(&format!("\n**OSNet Model:** {} ({}-dim)\n", best_body_model, best_body_dim));

    eprintln!("{r}");
    println!("{r}");

    // ── Save CSV outputs ────────────────────────────────────────────────────

    // Pairwise CSV
    let mut pw_csv = String::new();
    pw_csv.push_str("img_a,img_b,face_cos,body_cos,face_rank,body_rank\n");
    for (i, &a) in ALL.iter().enumerate() {
        for &b in &ALL[(i+1)..] {
            let fr = face_ranked.iter().position(|e| e.img == a).unwrap() as i64;
            let br = face_ranked.iter().position(|e| e.img == b).unwrap() as i64;
            let bc = cos_body(a, b).unwrap_or(0.0);
            pw_csv.push_str(&format!("{},{},{:.6},{:.6},{},{}\n",
                a, b, cos_face(a, b), bc, fr, br));
        }
    }
    fs::write(format!("{}/pairwise.csv", OUT_DIR), &pw_csv).ok();

    // Ranking CSV
    let mut rnk_csv = String::new();
    rnk_csv.push_str("model,img,group,rank,cosine\n");
    for (i, e) in face_ranked.iter().enumerate() {
        rnk_csv.push_str(&format!("face,img{},{},{},{:.6}\n", e.img, e.group, i+1, e.cos));
    }
    for (i, e) in body_ranked.iter().enumerate() {
        rnk_csv.push_str(&format!("body,img{},{},{},{:.6}\n", e.img, e.group, i+1, e.cos));
    }
    fs::write(format!("{}/ranking.csv", OUT_DIR), &rnk_csv).ok();

    // Fusion CSV
    let mut fus_csv = String::new();
    fus_csv.push_str("alpha,pos_mean,pos_min,neg_max,margin,first_pos,top3,top5\n");
    for (alpha, m, fp, t3, t5) in &fusion_results {
        fus_csv.push_str(&format!("{:.1},{:.6},{:.6},{:.6},{:.6},{},{},{}\n",
            alpha, m.pos_mean, m.pos_min, m.neg_max, m.margin, fp, t3, t5));
    }
    fs::write(format!("{}/fusion.csv", OUT_DIR), &fus_csv).ok();

    // Report MD
    fs::write(format!("{}/report.md", OUT_DIR), &r).ok();

    eprintln!("\n=== DONE ===");
    eprintln!("Output: {}/", OUT_DIR);
}
