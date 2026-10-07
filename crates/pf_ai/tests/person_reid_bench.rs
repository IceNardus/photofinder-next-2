//! 临时诊断（非生产）：Body Appearance (MobileNetV4/CLIP) 是否能解决 img15 假拆？
//!
//! 重要约束：
//!   1 不改生产代码 / DB / HNSW / cluster threshold
//!   2 所有输入使用与 Face-only 基准完全相同的图片和检测框
//!   3 每张图片每个模型只计算一次 embedding
//!   4 无 TTA
//!
//! 背景：
//!   img15 假拆根因已确定为 ArcFace embedding 空间的固有歧义。
//!   本 benchmark 测试"第二种身份特征：人体外观"是否能提供额外区分信号。
//!
//! 5 个 Phase：
//!   Phase 1  ArcFace Face-only 基线（same metrics as dual_identity_bench）
//!   Phase 2  Body Appearance cosine matrix（MobileNetV4 / CLIP features）
//!   Phase 3  Body-only ranking for img15
//!   Phase 4  Face + Body 融合（加权 cosine）
//!   Phase 5  Verdict（BODY_REID_SUCCESS / PARTIAL / FAIL）
//!
//! Body Crop 类型（8种）：
//!   A. face_tight   - face bbox + 10% margin
//!   B. face_loose   - face bbox + 30% margin
//!   C. upper_body   - face bbox 展开到上半身（约 2.5x face height）
//!   D. full_upper   - face bbox 展开到完整上半身（约 3x face height）
//!   E. loose        - face bbox 展开到宽松区域（约 3x h, 1.5x w）
//!   F. square       - 以 face center 为中心，最大边长为边长的正方形
//!   G. face_center  - face bbox 不扩展，只 center-crop 到 112x112
//!   H. full_body    - face bbox 展开到全身（约 4x face height）
//!
//! 模型：
//!   - Face: w600k_r50（生产路径，同 dual_identity_bench）
//!   - Body: mobileclip_s2.onnx（CLIP image encoder, 512-dim, via ort）
//!
//! 用法：
//!   ORT_DYLIB_PATH=/Users/mac/lib/libonnxruntime.dylib \
//!   cargo test --release -p pf_ai --test person_reid_bench -- --ignored --nocapture
//!
//! 只读。不改生产代码 / DB / HNSW / threshold。

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

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
const MOBILECLIP_MODEL: &str = "/Users/mac/ai-project/photofinder-next-2/models/mobileclip_s2.onnx";
const INPUT_TSV: &str = "/tmp/tta_img15/input.tsv";
const OUT_DIR: &str = "/tmp/person_reid_bench";
const OUT_CSV: &str = "/tmp/person_reid_bench/face_body_metrics.csv";
const OUT_MD: &str = "/tmp/person_reid_bench/face_body_metrics.md";

const IMG_QUERY: i64 = 15;
const CORE: [i64; 3] = [6, 13, 14];
const NEG: [i64; 6] = [2, 9, 17, 18, 19, 20];
const ALL: [i64; 10] = [6, 13, 14, 15, 2, 9, 17, 18, 19, 20];
const CROP_MARGIN: f32 = 0.10;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

// ─── Alignment helpers (same as dual_identity_bench) ─────────────────────────

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

/// MobileNetV4 / CLIP body embedding: 512-dim, BGR -> RGB with ImageNet norm
fn embed_mobileclip(session: &mut Session, rgb: &RgbImage) -> Vec<f32> {
    let h = rgb.height();
    let w = rgb.width();
    let n = (h as usize) * (w as usize);

    // MobileClip-S2 input: BGR -> RGB, scale to [0, 1], then ImageNet normalize
    // MobileClip uses mean=[0.48145466, 0.45782759, 0.40821073] and
    // std=[0.26862954, 0.26130259, 0.27577711]
    let mean = [0.48145466_f32, 0.45782759, 0.40821073];
    let std = [0.26862954, 0.26130259, 0.27577711];

    let mut input = Vec::with_capacity(3 * n);
    for y in 0..h {
        for x in 0..w {
            let p = rgb.get_pixel(x, y);
            // RGB order
            input.push((p[0] as f32 / 255.0 - mean[0]) / std[0]);
        }
    }
    for y in 0..h {
        for x in 0..w {
            let p = rgb.get_pixel(x, y);
            input.push((p[1] as f32 / 255.0 - mean[1]) / std[1]);
        }
    }
    for y in 0..h {
        for x in 0..w {
            let p = rgb.get_pixel(x, y);
            input.push((p[2] as f32 / 255.0 - mean[2]) / std[2]);
        }
    }

    let shape = [1_i64, 3, h as i64, w as i64];
    let input = Tensor::from_array((shape, input)).expect("tensor mobileclip");
    let outputs = session.run(ort::inputs![input]).expect("run mobileclip");
    let (_shape, data) = outputs[0].try_extract_tensor::<f32>().expect("extract mobileclip");
    l2_normalize(data)
}

// ─── Body Crop Types ──────────────────────────────────────────────────────────

fn expand_face_to_body(face: &BBox, img_h: f32, img_w: f32, crop_type: &str) -> BBox {
    let cx = face.x + face.w / 2.0;
    let cy = face.y + face.h / 2.0;
    let margin = 0.10_f32;

    match crop_type {
        "A" => {
            // face_tight: face + 10% margin
            let mx = face.w * margin;
            let my = face.h * margin;
            BBox::new(
                (face.x - mx).max(0.0),
                (face.y - my).max(0.0),
                face.w + 2.0 * mx,
                face.h + 2.0 * my,
            )
        }
        "B" => {
            // face_loose: face + 30% margin
            let mx = face.w * 0.30;
            let my = face.h * 0.30;
            BBox::new(
                (face.x - mx).max(0.0),
                (face.y - my).max(0.0),
                face.w + 2.0 * mx,
                face.h + 2.0 * my,
            )
        }
        "C" => {
            // upper_body: 2.5x face height, 1.3x face width, centered at face
            let height = face.h * 2.5;
            let width = face.w * 1.3;
            let x = (cx - width / 2.0).max(0.0);
            let y = (cy - height * 0.3).max(0.0); // slightly above face center
            BBox::new(x, y, width.min(img_w - x), height.min(img_h - y))
        }
        "D" => {
            // full_upper: 3x face height, 1.2x face width
            let height = face.h * 3.0;
            let width = face.w * 1.2;
            let x = (cx - width / 2.0).max(0.0);
            let y = (cy - height * 0.25).max(0.0);
            BBox::new(x, y, width.min(img_w - x), height.min(img_h - y))
        }
        "E" => {
            // loose: 3x h, 1.5x w
            let height = face.h * 3.0;
            let width = face.w * 1.5;
            let x = (cx - width / 2.0).max(0.0);
            let y = (cy - height * 0.2).max(0.0);
            BBox::new(x, y, width.min(img_w - x), height.min(img_h - y))
        }
        "F" => {
            // square: face center, max边长, 正方形
            let size = face.w.max(face.h) * 1.2;
            let x = (cx - size / 2.0).max(0.0);
            let y = (cy - size / 2.0).max(0.0);
            BBox::new(x, y, size.min(img_w - x), size.min(img_h - y))
        }
        "G" => {
            // face_center: face region, no expand, center-crop to 1:1
            let size = face.w.max(face.h);
            let x = (cx - size / 2.0).max(0.0);
            let y = (face.y).max(0.0);
            BBox::new(x, y, size.min(img_w - x), size.min(img_h - y))
        }
        "H" => {
            // full_body: 4x face height, 1.2x face width
            let height = face.h * 4.0;
            let width = face.w * 1.2;
            let x = (cx - width / 2.0).max(0.0);
            let y = (cy - height * 0.15).max(0.0);
            BBox::new(x, y, width.min(img_w - x), height.min(img_h - y))
        }
        _ => *face,
    }
}

fn crop_image(img: &ImageData, bbox: &BBox) -> Result<ImageData, String> {
    let b = *bbox;
    if b.w < 1.0 || b.h < 1.0 {
        return Err(format!("crop too small: {:?}", b));
    }
    img.crop(b).map_err(|e| e.to_string())
}

fn resize_to_target(rgb: RgbImage, target_h: u32, target_w: u32) -> RgbImage {
    use image::imageops::FilterType;
    image::imageops::resize(&rgb, target_w, target_h, FilterType::Triangle)
}

// ─── Metrics helpers ──────────────────────────────────────────────────────────

fn compute_metrics(emb: &HashMap<i64, Vec<f32>>, query: i64, core: &[i64], neg: &[i64]) -> (f32, f32, f32, f32, f32) {
    let cos = |a: i64, b: i64| cosine(emb.get(&a).unwrap(), emb.get(&b).unwrap());

    let pos: Vec<f32> = core.iter().map(|&c| cos(query, c)).collect();
    let neg_vals: Vec<f32> = neg.iter().map(|&n| cos(query, n)).collect();

    let pos_mean = pos.iter().sum::<f32>() / pos.len() as f32;
    let pos_min = pos.iter().cloned().fold(f32::INFINITY, |a, b| a.min(b));
    let neg_max = neg_vals.iter().cloned().fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let neg_mean = neg_vals.iter().sum::<f32>() / neg_vals.len() as f32;
    let margin = pos_min - neg_max;
    (pos_mean, pos_min, neg_max, neg_mean, margin)
}

fn ranking(emb: &HashMap<i64, Vec<f32>>, query: i64, all: &[i64]) -> Vec<(i64, f32)> {
    let cos = |a: i64, b: i64| cosine(emb.get(&a).unwrap(), emb.get(&b).unwrap());
    let mut ranked: Vec<(i64, f32)> = all.iter().map(|&t| (t, cos(query, t))).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ranked
}

// ─── Main test ────────────────────────────────────────────────────────────────

#[test]
#[ignore]
fn person_reid_bench() {
    fs::create_dir_all(OUT_DIR).expect("mkdir");
    let samples = read_tsv();
    let images: Vec<ImageData> = samples.iter()
        .map(|s| ImageData::from_file(std::path::Path::new(&s.path)).expect("open"))
        .collect();

    let rt = tokio::runtime::Runtime::new().expect("tokio");
    let scrfd = ScrfdDetector::load(std::path::Path::new(SCRFD_MODEL)).expect("load scrfd");
    let aligner = SimpleAligner::with_config(AlignmentConfig::default());

    // ── Phase 1: Face detections + ArcFace embeddings (production path) ──────
    // Store bbox + keypoints for alignment
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

    // Align face crops (112x112) using production alignment
    let face_crops: Vec<RgbImage> = samples.iter()
        .enumerate()
        .map(|(i, _s)| {
            build_aligned_face(&aligner, &images[i], &geos[i].bbox, &geos[i].kps)
                .expect("align face")
        })
        .collect();

    // ArcFace embeddings
    let mut arc_sess = Session::builder().expect("arc builder")
        .commit_from_file(PathBuf::from(ARCFACE_MODEL)).expect("arcface");
    let face_emb: HashMap<i64, Vec<f32>> = samples.iter()
        .enumerate()
        .map(|(i, s)| (s.img, embed_arcface(&mut arc_sess, &face_crops[i])))
        .collect();

    // MobileClip session
    let mut clip_sess = Session::builder().expect("clip builder")
        .commit_from_file(PathBuf::from(MOBILECLIP_MODEL)).expect("mobileclip");

    let cos_face = |a: i64, b: i64| cosine(face_emb.get(&a).unwrap(), face_emb.get(&b).unwrap());

    // ── Phase 2-8: Body crops with MobileClip ────────────────────────────────
    let crop_types = ["A", "B", "C", "D", "E", "F", "G", "H"];
    let crop_names = [
        "face_tight", "face_loose", "upper_body", "full_upper",
        "loose", "square", "face_center", "full_body",
    ];

    let mut r = String::new();
    r.push_str("\n============================================================\n");
    r.push_str("PERSON Re-ID BENCHMARK: Body Appearance vs Face-only\n");
    r.push_str("============================================================\n\n");

    // Phase 1: Face-only baseline
    let (f_pos_mean, f_pos_min, f_neg_max, f_neg_mean, f_margin) =
        compute_metrics(&face_emb, IMG_QUERY, &CORE, &NEG);
    r.push_str(&format!("Phase 1: Face-only Baseline (ArcFace w600k_r50)\n"));
    r.push_str(&format!("  pos_mean={:.4}  pos_min={:.4}  neg_max={:.4}  neg_mean={:.4}  margin={:+.4}\n\n",
        f_pos_mean, f_pos_min, f_neg_max, f_neg_mean, f_margin));

    // Body appearance embeddings per crop type
    let mut best_body_crop = String::new();
    let mut best_body_margin = f32::NEG_INFINITY;
    let mut best_body_emb = HashMap::new();
    let mut all_body_metrics: Vec<(String, String, f32, f32, f32, f32, f32)> = Vec::new();

    for (ct, cn) in crop_types.iter().zip(crop_names.iter()) {
        // Create body crops from face bbox expansion
        let body_crops: Vec<RgbImage> = samples.iter()
            .enumerate()
            .map(|(i, _s)| {
                let bbox = expand_face_to_body(&geos[i].bbox, images[i].height() as f32, images[i].width() as f32, ct);
                // Try body crop, fall back to face crop on any failure
                let rgb = match crop_image(&images[i], &bbox) {
                    Ok(img_data) => img_data.as_rgb8(),
                    Err(_) => {
                        // Fallback: use the already-aligned face crop
                        face_crops[i].clone()
                    }
                };
                // Resize to 224x224 for MobileClip
                resize_to_target(rgb, 224, 224)
            })
            .collect();

        // MobileClip embeddings
        let body_emb: HashMap<i64, Vec<f32>> = samples.iter()
            .enumerate()
            .map(|(i, s)| (s.img, embed_mobileclip(&mut clip_sess, &body_crops[i])))
            .collect();

        let (b_pos_mean, b_pos_min, b_neg_max, b_neg_mean, b_margin) =
            compute_metrics(&body_emb, IMG_QUERY, &CORE, &NEG);

        all_body_metrics.push((
            ct.to_string(),
            cn.to_string(),
            b_pos_mean, b_pos_min, b_neg_max, b_neg_mean, b_margin,
        ));

        if b_margin > best_body_margin {
            best_body_margin = b_margin;
            best_body_crop = cn.to_string();
            best_body_emb = body_emb;
        }
    }

    // Print all body crop results
    r.push_str("Phase 2: Body Appearance Metrics by Crop Type (MobileClip 512-dim)\n");
    r.push_str(&format!("{:<12} {:>12} {:>12} {:>12} {:>12} {:>12}\n",
        "crop", "pos_mean", "pos_min", "neg_max", "neg_mean", "margin"));
    for (ct, cn, pos_mean, pos_min, neg_max, neg_mean, margin) in &all_body_metrics {
        r.push_str(&format!("{:<12} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>+12.4}\n",
            format!("{}({})", ct, cn), pos_mean, pos_min, neg_max, neg_mean, margin));
    }
    r.push('\n');

    // Phase 3: img15 ranking for best body crop
    r.push_str(&format!("Phase 3: img15 Ranking (Best Body Crop: {})\n", best_body_crop));
    let best_ranked = ranking(&best_body_emb, IMG_QUERY, &ALL);
    r.push_str(&format!("{:<6} {:<12} {:>10}\n", "rank", "img", "group"));
    for (i, (img, cos)) in best_ranked.iter().enumerate() {
        let grp = if CORE.contains(img) { "PERSON2" } else { "NEG" };
        r.push_str(&format!("{:>4}  {:<12} {:<10} {:>8.4}\n", i + 1, format!("img{}", img), grp, cos));
    }
    r.push('\n');

    // Phase 4: Face + Body fusion
    r.push_str("Phase 4: Face + Body Fusion (weighted cosine)\n");
    let fusion_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    r.push_str(&format!("{:<8} {:>12} {:>12} {:>12} {:>12}\n",
        "w_face", "pos_mean", "pos_min", "neg_max", "margin"));

    let mut best_fw = 0.0_f32;
    let mut best_fusion_margin = f32::NEG_INFINITY;
    let mut best_fusion_metrics = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);

    for &fw in &fusion_weights {
        let bw = 1.0 - fw;
        // Compute fused embedding
        let mut fused_emb: HashMap<i64, Vec<f32>> = HashMap::new();
        for &img_id in &ALL {
            let fe = face_emb.get(&img_id).unwrap();
            let be = best_body_emb.get(&img_id).unwrap();
            let fused: Vec<f32> = fe.iter().zip(be.iter())
                .map(|(f, b)| fw * f + bw * b)
                .collect();
            // Renormalize
            let norm = fused.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            let normed: Vec<f32> = fused.iter().map(|x| x / norm).collect();
            fused_emb.insert(img_id, normed);
        }

        let (pos_mean, pos_min, neg_max, neg_mean, margin) =
            compute_metrics(&fused_emb, IMG_QUERY, &CORE, &NEG);

        if margin > best_fusion_margin {
            best_fusion_margin = margin;
            best_fw = fw;
            best_fusion_metrics = (pos_mean, pos_min, neg_max, neg_mean, margin);
        }

        r.push_str(&format!("{:>8.1} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>+12.4}\n",
            fw, pos_mean, pos_min, neg_max, neg_mean, margin));
    }
    r.push('\n');

    // Phase 5: Verdict
    r.push_str("Phase 5: Verdict\n");
    let face_only_pass = f_margin > 0.0 && best_ranked[0].0 == IMG_QUERY;
    let body_only_best = best_body_margin > 0.0;
    let fusion_pass = best_fusion_margin > 0.0;

    let img15_in_positives_body = best_body_emb.get(&IMG_QUERY)
        .map(|qe| {
            let cos_to_core: Vec<f32> = CORE.iter()
                .map(|&c| cosine(qe, best_body_emb.get(&c).unwrap()))
                .collect();
            let cos_to_neg: Vec<f32> = NEG.iter()
                .map(|&n| cosine(qe, best_body_emb.get(&n).unwrap()))
                .collect();
            let pos_min = cos_to_core.iter().cloned().fold(f32::INFINITY, |a, b| a.min(b));
            let neg_max = cos_to_neg.iter().cloned().fold(f32::NEG_INFINITY, |a, b| a.max(b));
            pos_min > neg_max
        }).unwrap_or(false);

    r.push_str(&format!("  Face-only (ArcFace):  margin={:+.4}  img15_rank1={}  PASS={}\n",
        f_margin, best_ranked[0].0 == IMG_QUERY, face_only_pass));
    r.push_str(&format!("  Body-only (MobileClip, {}):  margin={:+.4}  best_crop={}  PASS={}\n",
        best_body_crop, best_body_margin, best_body_crop, body_only_best));
    r.push_str(&format!("  Face+Body fusion (w_face={:.1}):  margin={:+.4}  PASS={}\n",
        best_fw, best_fusion_margin, fusion_pass));
    r.push_str(&format!("  img15 in positive region (body):  {}\n\n", img15_in_positives_body));

    // Determine overall verdict
    let verdict = if face_only_pass {
        "BODY_REID_NOT_NEEDED (face-only already works)".to_string()
    } else if fusion_pass && best_fw < 0.9 {
        "BODY_REID_SUCCESS (fusion resolves img15)".to_string()
    } else if body_only_best || img15_in_positives_body {
        "BODY_REID_PARTIAL (body helps but insufficient alone)".to_string()
    } else {
        "BODY_REID_FAIL (body appearance provides no additional signal)".to_string()
    };

    r.push_str(&format!("VERDICT: {}\n", verdict));

    // img15 body ranking details
    r.push_str(&format!("\nimg15 Body Ranking (crop={}):\n", best_body_crop));
    for (i, (img, cos)) in best_ranked.iter().enumerate() {
        let grp = if CORE.contains(img) { "PERSON2" } else { "NEG" };
        let marker = if *img == IMG_QUERY { " ← QUERY" } else { "" };
        r.push_str(&format!("  {:>2}. img{:<4} {:<10} {:>8.4}{}\n",
            i + 1, img, grp, cos, marker));
    }

    // Face ranking for comparison
    let face_ranked = ranking(&face_emb, IMG_QUERY, &ALL);
    r.push_str(&format!("\nimg15 Face Ranking:\n"));
    for (i, (img, cos)) in face_ranked.iter().enumerate() {
        let grp = if CORE.contains(img) { "PERSON2" } else { "NEG" };
        let marker = if *img == IMG_QUERY { " ← QUERY" } else { "" };
        r.push_str(&format!("  {:>2}. img{:<4} {:<10} {:>8.4}{}\n",
            i + 1, img, grp, cos, marker));
    }

    r.push_str("\n=== DONE ===\n");
    eprintln!("{r}");

    // CSV output
    let mut csv = String::new();
    csv.push_str("phase,metric,value\n");
    csv.push_str(&format_args!("face_baseline,pos_mean,{:.4}\n", f_pos_mean).to_string());
    csv.push_str(&format_args!("face_baseline,pos_min,{:.4}\n", f_pos_min).to_string());
    csv.push_str(&format_args!("face_baseline,neg_max,{:.4}\n", f_neg_max).to_string());
    csv.push_str(&format_args!("face_baseline,neg_mean,{:.4}\n", f_neg_mean).to_string());
    csv.push_str(&format_args!("face_baseline,margin,{:.4}\n", f_margin).to_string());
    csv.push_str(&format_args!("body_best,crop,{}\n", best_body_crop).to_string());
    csv.push_str(&format_args!("body_best,margin,{:.4}\n", best_body_margin).to_string());
    csv.push_str(&format_args!("fusion,best_w_face,{:.1}\n", best_fw).to_string());
    csv.push_str(&format_args!("fusion,margin,{:.4}\n", best_fusion_margin).to_string());
    csv.push_str(&format_args!("verdict,{},0\n", verdict).to_string());

    fs::write(OUT_CSV, &csv).expect("write csv");
    fs::write(OUT_MD, &r).expect("write md");
}
