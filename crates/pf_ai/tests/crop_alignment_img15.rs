//! 临时诊断（非生产）：img15 Crop / Landmark / Alignment 根因定位。
//!
//! 目标：定位 img15 的 embedding 漂移究竟来自 SCRFD bbox / crop / landmark
//! 坐标变换 / alignment，还是 ArcFace 本身。
//!
//! 12 个 Phase：
//!   Phase 1  生产基准：Path A（生产复刻）→ cos vs DB embedding 必须 ≥ 0.999，否则停
//!   Phase 2  四张 jaqor 完整几何信息（image size / bbox / ratio / center / landmarks）
//!   Phase 3  crop landmark 坐标变换验证：reconstructed == original，误差 < 0.01px
//!   Phase 4  三条 Alignment Path：A 生产 crop / B 绕过 crop / C 严格 crop
//!   Phase 5  A↔B / A↔C / B↔C 单图 embedding 一致性
//!   Phase 6  三个 Path 的 jaqor pair matrix
//!   Phase 7  加入负例 {2,9,20,17,18,19}，pos_min/pos_mean/neg_max/margin
//!   Phase 8  Crop Margin Sweep 0–30%（观察是否存在稳定趋势）
//!   Phase 9  保存 112×112 对齐结果 + 诊断图（含 canonical/placed landmarks）
//!   Phase 10 Alignment Error（placed vs canonical，绝对 + 归一化 / 眼距）
//!   Phase 11 img15 人工 canonical landmark（绕过 SCRFD 5 点）→ 是否回到 jaqor 空间
//!   Phase 12 最终判定（YES/NO 清单 + A–F）
//!
//! 输入：/tmp/tta_img15/input.tsv（img<TAB>path<TAB>bx by bw bh<TAB>le re nose lm rm<TAB>512 db embedding）
//!   bbox / 5 点来自 SCRFD-at-ingest（即 DB 实际使用的值，lfw_path_verify 已证与
//!   实时 SCRFD detect 逐字节一致，关键点误差 0.0px）。
//! 输出：/tmp/img15_diag/*.png
//!
//! 用法：
//!   ORT_DYLIB_PATH=/Users/mac/lib/libonnxruntime.dylib \
//!   cargo test --release -p pf_ai --test crop_alignment_img15 -- --ignored --nocapture
//!
//! 只读库 / 模型推理。不改生产代码 / 不改 DB / 不重聚类。

use std::fs;
use std::path::{Path, PathBuf};

use image::{Rgb, RgbImage};
use ort::session::Session;
use ort::value::Tensor;

use pf_ai::face::aligner::{AlignmentConfig, SimpleAligner, REF_LANDMARKS_RAW};
use pf_ai::face::arcface::{l2_normalize, EMBEDDING_DIM, INPUT_SIZE};
use pf_ai::face::pose::estimate_yaw_pitch_roll;
use pf_ai::face::traits::FaceAligner;
use pf_ai::image_data::ImageData;
use pf_ai::quality::{assess, blur_score_from_aligned, combined_quality};
use pf_ai::{FaceDetector, QualityFilter, ScrfdDetector};
use pf_core::{BBox, FaceKeypoints};

const ARCFACE_MODEL: &str = "/Users/mac/ai-project/photofinder-next-2/models/w600k_r50.onnx";
const SCRFD_MODEL: &str = "/Users/mac/ai-project/photofinder-next-2/models/scrfd_500m_bnkps.onnx";
const INPUT_TSV: &str = "/tmp/tta_img15/input.tsv";
const OUT_DIR: &str = "/tmp/img15_diag";

const SAME: [i64; 4] = [6, 13, 14, 15];
const NEG: [i64; 6] = [2, 9, 20, 17, 18, 19];

fn is_same(a: i64, b: i64) -> bool {
    SAME.contains(&a) && SAME.contains(&b)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

// ---- 数据 ----

struct Sample {
    img: i64,
    path: String,
    bbox: BBox,
    kps: [(f32, f32); 5], // le re nose lm rm（原图坐标）
    db_emb: Vec<f32>,
}

fn read_tsv() -> Vec<Sample> {
    let content = std::fs::read_to_string(INPUT_TSV).expect("read tsv");
    let mut out = Vec::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        let tof = |i: usize| f[i].parse::<f32>().unwrap();
        let img: i64 = f[0].parse().unwrap();
        let path = f[1].to_string();
        let bbox = BBox::new(tof(2), tof(3), tof(4), tof(5));
        let mut kps = [(0.0f32, 0.0f32); 5];
        for k in 0..5 {
            kps[k] = (tof(6 + 2 * k), tof(7 + 2 * k));
        }
        let db_emb: Vec<f32> = f[16..16 + EMBEDDING_DIM]
            .iter()
            .map(|s| s.parse::<f32>().unwrap())
            .collect();
        out.push(Sample {
            img,
            path,
            bbox,
            kps,
            db_emb,
        });
    }
    out
}

// ---- ArcFace 1-crop 推理（复刻 arcface.rs build_input + run_once）----

fn embed_one(session: &mut Session, rgb: &RgbImage) -> Vec<f32> {
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
    let outputs = session.run(ort::inputs![input]).expect("arcface run");
    let (_shape, data) = outputs[0].try_extract_tensor::<f32>().expect("extract");
    l2_normalize(data)
}

// ---- alignment 数学（复刻 aligner.rs）----

fn ref_landmarks(cfg: &AlignmentConfig) -> [(f32, f32); 5] {
    let mut out = REF_LANDMARKS_RAW;
    for p in &mut out {
        p.0 *= cfg.align_scale;
        p.1 = p.1 * cfg.align_scale + cfg.align_y_offset;
    }
    out
}

fn apply_transform(t: &[f32; 6], pts: &[(f32, f32); 5]) -> [(f32, f32); 5] {
    let mut out = [(0.0f32, 0.0f32); 5];
    for i in 0..5 {
        out[i] = (
            t[0] * pts[i].0 + t[1] * pts[i].1 + t[2],
            t[3] * pts[i].0 + t[4] * pts[i].1 + t[5],
        );
    }
    out
}

/// 输入 kps（与传给 aligner 的图同一坐标系）在 112×112 输出中的实际落点。
fn placed_landmarks(input_kps: &[(f32, f32); 5], cfg: &AlignmentConfig) -> [(f32, f32); 5] {
    let dst = ref_landmarks(cfg);
    let t = SimpleAligner::compute_similarity_transform(input_kps, &dst);
    apply_transform(&t, input_kps)
}

fn kps_from_array(a: &[(f32, f32); 5]) -> FaceKeypoints {
    FaceKeypoints {
        left_eye: a[0],
        right_eye: a[1],
        nose: a[2],
        left_mouth: a[3],
        right_mouth: a[4],
    }
}

// ---- crop（生产私有 crop_to_bbox_with_margin 的复刻，margin 参数化）----

fn crop_to_bbox_with_margin(
    img: &ImageData,
    bbox: &BBox,
    margin: f32,
) -> Result<(ImageData, (f32, f32)), String> {
    let b = *bbox;
    let w = b.w.max(1.0);
    let h = b.h.max(1.0);
    let mx = w * margin;
    let my = h * margin;
    let x = (b.x - mx).max(0.0);
    let y = (b.y - my).max(0.0);
    let cw = (w + 2.0 * mx).min(img.width() as f32 - x);
    let ch = (h + 2.0 * my).min(img.height() as f32 - y);
    if cw < 1.0 || ch < 1.0 {
        return Err(format!("crop too small: {cw}x{ch}"));
    }
    let crop = img
        .crop(BBox::new(x, y, cw, ch))
        .map_err(|e| e.to_string())?;
    Ok((crop, (x, y)))
}

fn shift_kps(kps: &[(f32, f32); 5], origin: (f32, f32)) -> [(f32, f32); 5] {
    kps.map(|(x, y)| (x - origin.0, y - origin.1))
}

/// Path A：生产路径复刻 = crop(margin) + kp 平移 + SimpleAligner + (外部 embed)。
fn path_a(
    aligner: &SimpleAligner,
    cfg: &AlignmentConfig,
    img: &ImageData,
    bbox: &BBox,
    kps: &[(f32, f32); 5],
    margin: f32,
) -> Result<(RgbImage, [(f32, f32); 5]), String> {
    let (crop, origin) = crop_to_bbox_with_margin(img, bbox, margin)?;
    let shifted = shift_kps(kps, origin);
    let aligned = aligner.align(&crop, &kps_from_array(&shifted)).map_err(|e| e.to_string())?;
    let placed = placed_landmarks(&shifted, cfg);
    Ok((aligned.image.as_rgb8(), placed))
}

/// Path B：绕过 crop = 原图 + 原始 kps 直接 SimpleAligner。
fn path_b(
    aligner: &SimpleAligner,
    cfg: &AlignmentConfig,
    img: &ImageData,
    kps: &[(f32, f32); 5],
) -> Result<(RgbImage, [(f32, f32); 5]), String> {
    let aligned = aligner.align(img, &kps_from_array(kps)).map_err(|e| e.to_string())?;
    let placed = placed_landmarks(kps, cfg);
    Ok((aligned.image.as_rgb8(), placed))
}

/// Path C：crop + 独立严格坐标变换（内嵌 reconstruct 校验，误差必须 < 0.01px）。
fn path_c(
    aligner: &SimpleAligner,
    cfg: &AlignmentConfig,
    img: &ImageData,
    bbox: &BBox,
    kps: &[(f32, f32); 5],
    label: i64,
) -> Result<(RgbImage, [(f32, f32); 5]), String> {
    let (crop, origin) = crop_to_bbox_with_margin(img, bbox, 0.10)?;
    let mut shifted = [(0.0f32, 0.0f32); 5];
    for k in 0..5 {
        shifted[k] = (kps[k].0 - origin.0, kps[k].1 - origin.1);
        let rx = origin.0 + shifted[k].0;
        let ry = origin.1 + shifted[k].1;
        let err = ((rx - kps[k].0).powi(2) + (ry - kps[k].1).powi(2)).sqrt();
        assert!(err < 0.01, "img{label} kp{k} reconstruct err {err:.4} >= 0.01px");
    }
    let aligned = aligner.align(&crop, &kps_from_array(&shifted)).map_err(|e| e.to_string())?;
    let placed = placed_landmarks(&shifted, cfg);
    Ok((aligned.image.as_rgb8(), placed))
}

/// 实时 SCRFD 路径（复刻 lfw_path_verify::extract_best，embedding 用 embed_one）：
/// 用于 Phase 1 二分——同一张图，存储 bbox/kps 路径 vs 实时 detect 路径。
fn live_extract_best(
    rt: &tokio::runtime::Runtime,
    scrfd: &ScrfdDetector,
    aligner: &SimpleAligner,
    qf: &QualityFilter,
    session: &mut Session,
    image: &ImageData,
) -> Option<(Vec<f32>, BBox, [(f32, f32); 5])> {
    let dets = rt.block_on(scrfd.detect(image)).ok()?;
    let mut best: Option<(Vec<f32>, BBox, [(f32, f32); 5], f32)> = None;
    for det in dets {
        let ypr = estimate_yaw_pitch_roll(&det.keypoints);
        let (ci, origin) = match crop_to_bbox_with_margin(image, &det.bbox, 0.10) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let shifted = [
            (det.keypoints.left_eye.0 - origin.0, det.keypoints.left_eye.1 - origin.1),
            (det.keypoints.right_eye.0 - origin.0, det.keypoints.right_eye.1 - origin.1),
            (det.keypoints.nose.0 - origin.0, det.keypoints.nose.1 - origin.1),
            (det.keypoints.left_mouth.0 - origin.0, det.keypoints.left_mouth.1 - origin.1),
            (det.keypoints.right_mouth.0 - origin.0, det.keypoints.right_mouth.1 - origin.1),
        ];
        let aligned = match aligner.align(&ci, &kps_from_array(&shifted)) {
            Ok(a) => a,
            Err(_) => continue,
        };
        let gray = aligned.image.to_luma8();
        let blur = blur_score_from_aligned(&gray);
        let q = assess(&det, blur, ypr);
        if !q.passes() {
            continue;
        }
        if det.score < qf.min_detector_score {
            continue;
        }
        let quality = combined_quality(det.score, q.face_area_score, q.blur_score, q.pose_score);
        let emb = embed_one(session, &aligned.image.as_rgb8());
        let kps = [
            det.keypoints.left_eye,
            det.keypoints.right_eye,
            det.keypoints.nose,
            det.keypoints.left_mouth,
            det.keypoints.right_mouth,
        ];
        if best.as_ref().map(|b| quality > b.3).unwrap_or(true) {
            best = Some((emb, det.bbox, kps, quality));
        }
    }
    best.map(|(e, b, k, _)| (e, b, k))
}

// ---- 绘图（诊断图，手写像素，无新依赖）----

fn in_bounds(img: &RgbImage, x: i32, y: i32) -> bool {
    x >= 0 && y >= 0 && x < img.width() as i32 && y < img.height() as i32
}

fn draw_dot(img: &mut RgbImage, cx: f32, cy: f32, r: i32, color: [u8; 3]) {
    let cxi = cx.round() as i32;
    let cyi = cy.round() as i32;
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                let x = cxi + dx;
                let y = cyi + dy;
                if in_bounds(img, x, y) {
                    img.put_pixel(x as u32, y as u32, Rgb(color));
                }
            }
        }
    }
}

fn draw_rect(img: &mut RgbImage, b: &BBox, color: [u8; 3], t: i32) {
    let x0 = b.x as i32;
    let y0 = b.y as i32;
    let x1 = (b.x + b.w) as i32;
    let y1 = (b.y + b.h) as i32;
    for th in 0..t {
        for x in (x0 - th)..=(x1 + th) {
            if in_bounds(img, x, y0 - th) {
                img.put_pixel(x as u32, (y0 - th) as u32, Rgb(color));
            }
            if in_bounds(img, x, y1 + th) {
                img.put_pixel(x as u32, (y1 + th) as u32, Rgb(color));
            }
        }
        for y in (y0 - th)..=(y1 + th) {
            if in_bounds(img, x0 - th, y) {
                img.put_pixel((x0 - th) as u32, y as u32, Rgb(color));
            }
            if in_bounds(img, x1 + th, y) {
                img.put_pixel((x1 + th) as u32, y as u32, Rgb(color));
            }
        }
    }
}

fn draw_kps(img: &mut RgbImage, kps: &[(f32, f32); 5], color: [u8; 3], r: i32) {
    for k in kps {
        draw_dot(img, k.0, k.1, r, color);
    }
}

fn composite(panels: &[&RgbImage], cols: usize, cell: u32) -> RgbImage {
    let gap = 4u32;
    let rows = (panels.len() + cols - 1) / cols;
    let w = cols as u32 * cell + (cols as u32 + 1) * gap;
    let h = rows as u32 * cell + (rows as u32 + 1) * gap;
    let mut out = RgbImage::from_pixel(w, h, Rgb([32, 32, 32]));
    for (n, p) in panels.iter().enumerate() {
        let thumb = image::imageops::resize(*p, cell, cell, image::imageops::FilterType::Triangle);
        let x = gap + (n % cols) as u32 * (cell + gap);
        let y = gap + (n / cols) as u32 * (cell + gap);
        image::imageops::replace(&mut out, &thumb, x as i64, y as i64);
    }
    out
}

// ---- 指标 ----

fn full_metrics(scores: &[(i64, i64, f32)]) -> (f32, f32, f32, f32) {
    let mut same: Vec<f32> = Vec::new();
    let mut neg: Vec<f32> = Vec::new();
    for &(a, b, s) in scores {
        if is_same(a, b) {
            same.push(s);
        } else {
            neg.push(s);
        }
    }
    let same_min = same.iter().copied().fold(f32::INFINITY, f32::min);
    let same_mean = same.iter().sum::<f32>() / same.len() as f32;
    let hard_neg_max = neg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (same_min, same_mean, hard_neg_max, same_min - hard_neg_max)
}

fn img15_metrics(scores: &[(i64, i64, f32)]) -> (f32, f32, f32, f32) {
    let mut pos: Vec<f32> = Vec::new();
    let mut neg: Vec<f32> = Vec::new();
    for &(a, b, s) in scores {
        if a == 15 || b == 15 {
            if is_same(a, b) {
                pos.push(s);
            } else {
                neg.push(s);
            }
        }
    }
    let pos_min = pos.iter().copied().fold(f32::INFINITY, f32::min);
    let pos_mean = pos.iter().sum::<f32>() / pos.len() as f32;
    let neg_max = neg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (pos_min, pos_mean, neg_max, pos_min - neg_max)
}

fn print_matrix(label: &str, imgs: &[i64], embs: &[(i64, Vec<f32>)]) {
    let get = |x: i64| embs.iter().find(|(i, _)| *i == x).unwrap().1.clone();
    let hdr: String = imgs.iter().map(|i| format!("img{i:>3}")).collect::<Vec<_>>().join(" ");
    eprintln!("\n[{label}]");
    eprintln!("        {hdr}");
    for a in imgs {
        let row: Vec<String> = imgs
            .iter()
            .map(|b| {
                if a == b {
                    "1.000".to_string()
                } else {
                    format!("{:.3}", cosine(&get(*a), &get(*b)))
                }
            })
            .collect();
        eprintln!("img{a:>2}  {}", row.join(" "));
    }
}

#[test]
#[ignore]
fn crop_alignment_img15() {
    let samples = read_tsv();
    assert_eq!(samples.len(), 10, "expect 10 samples");
    let cfg = AlignmentConfig::default();
    let aligner = SimpleAligner::with_config(cfg.clone());
    let ref_pts = ref_landmarks(&cfg);
    let ref_eye = ref_pts[1].0 - ref_pts[0].0; // 25.0

    let mut session = Session::builder()
        .expect("builder")
        .commit_from_file(PathBuf::from(ARCFACE_MODEL))
        .expect("load arcface");

    let images: Vec<ImageData> = samples
        .iter()
        .map(|s| ImageData::from_file(Path::new(&s.path)).expect("open"))
        .collect();
    let idx = |img: i64| samples.iter().position(|s| s.img == img).expect("idx");

    // ============ Phase 1: 生产基准 (实时 SCRFD 全精度 kps) + TSV-kps 截断对照 ============
    // 生产 DB 存全精度 f32 kps；实时 detect 逐字节复现（lfw_path_verify 已证）。TSV 导出时
    // 把 kps 截断到 ~0.001px（bbox 保留全精度），对 img15 这类 ill-conditioned landmark 会
    // 放大到 embedding 位移 ~0.008 cos。故生产基准必须用实时全精度 kps，TSV-kps 仅作对照。
    eprintln!("=== Phase 1: 生产路径复刻 (实时 detect 全精度 kps → crop10 → align → embed) vs DB embedding ===");
    let rt = tokio::runtime::Runtime::new().expect("tokio");
    let scrfd = ScrfdDetector::load(Path::new(SCRFD_MODEL)).expect("scrfd");
    let qf = QualityFilter::from_config(0.5, 20, 0.5, 90.0);
    let mut p1_min = f32::MAX;
    for s in &samples {
        let i = idx(s.img);
        // (a) 生产基准：实时 detect（全精度 f32）→ crop10 → align → embed
        let live = live_extract_best(&rt, &scrfd, &aligner, &qf, &mut session, &images[i]);
        let live = match live {
            Some(v) => v,
            None => {
                eprintln!("  img{:<3} live extract: None（无 face 通过质量过滤）", s.img);
                continue;
            }
        };
        let (emb_prod, lb, lk) = live;
        let c_prod = cosine(&emb_prod, &s.db_emb);
        p1_min = p1_min.min(c_prod);
        // (b) 对照：TSV 存储 kps（导出截断 ~0.001px）→ 同一路径
        let (rgb_stored, _) = path_a(&aligner, &cfg, &images[i], &s.bbox, &s.kps, 0.10).expect("path A");
        let emb_stored = embed_one(&mut session, &rgb_stored);
        let c_stored = cosine(&emb_stored, &s.db_emb);
        let c_sl = cosine(&emb_stored, &emb_prod);
        let kp_err = (0..5).map(|k| dist(lk[k], s.kps[k])).fold(0.0f32, f32::max);
        let bd = (lb.x - s.bbox.x)
            .abs()
            .max((lb.y - s.bbox.y).abs())
            .max((lb.w - s.bbox.w).abs())
            .max((lb.h - s.bbox.h).abs());
        eprintln!(
            "  img{:<3} cos(prod,DB)={:.6} cos(TSVkps,DB)={:.6} cos(TSVkps,prod)={:.6} | TSV_kp_err={:.4}px | bbox_delta={:.4}",
            s.img, c_prod, c_stored, c_sl, kp_err, bd
        );
    }
    eprintln!("  >> min cos(prod, DB) = {p1_min:.6}  (要求 >= 0.999)");
    if p1_min < 0.999 {
        eprintln!("  !! Phase 1 未达标：先定位生产路径 vs DB 差异来源。");
    }

    // ---- img15 debug: 存储路径 vs 实时路径 对齐图逐像素对比 ----
    {
        let s = &samples[idx(15)];
        let (rgb_a, _pa) = path_a(&aligner, &cfg, &images[idx(15)], &s.bbox, &s.kps, 0.10).expect("A");
        let dets = rt.block_on(scrfd.detect(&images[idx(15)])).expect("detect");
        eprintln!("  [debug] img15 检测数 = {}", dets.len());
        for (di, d) in dets.iter().enumerate() {
            eprintln!("  [debug]   det{di}: score={:.4} bbox=({:.6},{:.6},{:.6},{:.6}) kps_le=({:.6},{:.6})", d.score, d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h, d.keypoints.left_eye.0, d.keypoints.left_eye.1);
        }
        let best = dets.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap()).expect("best");
        let (ci, origin) = crop_to_bbox_with_margin(&images[idx(15)], &best.bbox, 0.10).expect("crop");
        let (crop_a, origin_a) = crop_to_bbox_with_margin(&images[idx(15)], &s.bbox, 0.10).expect("cropA");
        eprintln!("  [debug] crop[TSVkps]: origin=({:.6},{:.6}) dims={}x{}  crop[live]: origin=({:.6},{:.6}) dims={}x{}",
            origin_a.0, origin_a.1, crop_a.width(), crop_a.height(),
            origin.0, origin.1, ci.width(), ci.height());
        let shifted = [
            (best.keypoints.left_eye.0 - origin.0, best.keypoints.left_eye.1 - origin.1),
            (best.keypoints.right_eye.0 - origin.0, best.keypoints.right_eye.1 - origin.1),
            (best.keypoints.nose.0 - origin.0, best.keypoints.nose.1 - origin.1),
            (best.keypoints.left_mouth.0 - origin.0, best.keypoints.left_mouth.1 - origin.1),
            (best.keypoints.right_mouth.0 - origin.0, best.keypoints.right_mouth.1 - origin.1),
        ];
        let aligned_l = aligner.align(&ci, &kps_from_array(&shifted)).expect("align L");
        let rgb_l = aligned_l.image.as_rgb8();
        eprintln!("  [debug] stored bbox=({:.6},{:.6},{:.6},{:.6}) live bbox=({:.6},{:.6},{:.6},{:.6})", s.bbox.x, s.bbox.y, s.bbox.w, s.bbox.h, best.bbox.x, best.bbox.y, best.bbox.w, best.bbox.h);
        eprintln!("  [debug] stored kps={:?}", s.kps);
        eprintln!("  [debug] live  kps={:?}", [
            (best.keypoints.left_eye.0, best.keypoints.left_eye.1),
            (best.keypoints.right_eye.0, best.keypoints.right_eye.1),
            (best.keypoints.nose.0, best.keypoints.nose.1),
            (best.keypoints.left_mouth.0, best.keypoints.left_mouth.1),
            (best.keypoints.right_mouth.0, best.keypoints.right_mouth.1),
        ]);
        let mut diff_sum = 0u64;
        let mut n = 0u64;
        let mut max_diff = 0u64;
        let mut n_large = 0u64;
        for y in 0..112 {
            for x in 0..112 {
                let pa = rgb_a.get_pixel(x, y);
                let pb = rgb_l.get_pixel(x, y);
                for c in 0..3 {
                    let d = (pa[c] as i32 - pb[c] as i32).unsigned_abs() as u64;
                    diff_sum += d;
                    max_diff = max_diff.max(d);
                    if d > 20 {
                        n_large += 1;
                    }
                    n += 1;
                }
            }
        }
        eprintln!("  [debug] img15 aligned [TSVkps] vs [live]: mean_abs={:.4} max={} n_diff>20={}  (of {} px*ch)", diff_sum as f32 / n as f32, max_diff, n_large, n);
        let emb_a2 = embed_one(&mut session, &rgb_a);
        let emb_l2 = embed_one(&mut session, &rgb_l);
        eprintln!("  [debug] cos(aligned[TSVkps]_emb, aligned[live]_emb) = {:.6}", cosine(&emb_a2, &emb_l2));
        eprintln!("  [debug] cos(aligned[TSVkps]_emb, db) = {:.6}  cos(aligned[live]_emb, db) = {:.6}", cosine(&emb_a2, &s.db_emb), cosine(&emb_l2, &s.db_emb));
        // 保存两张对齐图
        rgb_a.save(Path::new(OUT_DIR).join("debug_img15_TSVkps.png")).expect("save");
        rgb_l.save(Path::new(OUT_DIR).join("debug_img15_live.png")).expect("save");
    }

    // ============ Phase 2: 几何信息 ============
    eprintln!("\n=== Phase 2: 四张 jaqor 几何信息 ===");
    for &img in &SAME {
        let i = idx(img);
        let s = &samples[i];
        let im = &images[i];
        let (w, h) = (im.width() as f32, im.height() as f32);
        let b = s.bbox;
        let ratio = b.w * b.h / (w * h);
        let cx = b.x + b.w / 2.0;
        let cy = b.y + b.h / 2.0;
        let [le, re, nose, lm, rm] = s.kps;
        let eye_d = dist(le, re);
        let eye_c = ((le.0 + re.0) / 2.0, (le.1 + re.1) / 2.0);
        let nose2eye = dist(nose, eye_c);
        let mouth_c = ((lm.0 + rm.0) / 2.0, (lm.1 + rm.1) / 2.0);
        let mouth2eye = dist(mouth_c, eye_c);
        eprintln!("img{img}: image={w:.0}x{h:.0}");
        eprintln!("  raw bbox x1={:.1} y1={:.1} x2={:.1} y2={:.1}  w={:.1} h={:.1}", b.x, b.y, b.x + b.w, b.y + b.h, b.w, b.h);
        eprintln!("  bbox/img ratio={:.4}  center=({:.1},{:.1})  norm=({:.4},{:.4})", ratio, cx, cy, cx / w, cy / h);
        eprintln!("  le=({:.1},{:.1}) re=({:.1},{:.1}) nose=({:.1},{:.1}) lm=({:.1},{:.1}) rm=({:.1},{:.1})", le.0, le.1, re.0, re.1, nose.0, nose.1, lm.0, lm.1, rm.0, rm.1);
        eprintln!("  eye_dist={:.1}  eye_center=({:.1},{:.1})  nose2eye={:.1}  mouth2eye={:.1}", eye_d, eye_c.0, eye_c.1, nose2eye, mouth2eye);
    }

    // ============ Phase 3: crop landmark 坐标变换验证 ============
    eprintln!("\n=== Phase 3: crop landmark 坐标变换（reconstructed == original, <0.01px）===");
    let mut p3_max = 0.0f32;
    for &img in &SAME {
        let i = idx(img);
        let s = &samples[i];
        let (_, origin) = crop_to_bbox_with_margin(&images[i], &s.bbox, 0.10).expect("crop");
        let mut m = 0.0f32;
        for (ox, oy) in s.kps.iter() {
            let (sx, sy) = (ox - origin.0, oy - origin.1);
            let (rx, ry) = (origin.0 + sx, origin.1 + sy);
            let e = ((rx - ox).powi(2) + (ry - oy).powi(2)).sqrt();
            m = m.max(e);
        }
        p3_max = p3_max.max(m);
        eprintln!("  img{img}: crop_origin=({:.2},{:.2})  max_reconstruct_err={:.6}px", origin.0, origin.1, m);
    }
    let crop_correct = p3_max < 0.01;
    eprintln!("  >> crop 坐标变换正确: {}", if crop_correct { "YES" } else { "NO" });

    // ============ Phase 4-7: 三条 Path 的 embedding ============
    let mut embs_a: Vec<(i64, Vec<f32>)> = Vec::new();
    let mut embs_b: Vec<(i64, Vec<f32>)> = Vec::new();
    let mut embs_c: Vec<(i64, Vec<f32>)> = Vec::new();
    for s in &samples {
        let i = idx(s.img);
        let (rgb_a, _) = path_a(&aligner, &cfg, &images[i], &s.bbox, &s.kps, 0.10).expect("A");
        let (rgb_b, _) = path_b(&aligner, &cfg, &images[i], &s.kps).expect("B");
        let (rgb_c, _) = path_c(&aligner, &cfg, &images[i], &s.bbox, &s.kps, s.img).expect("C");
        embs_a.push((s.img, embed_one(&mut session, &rgb_a)));
        embs_b.push((s.img, embed_one(&mut session, &rgb_b)));
        embs_c.push((s.img, embed_one(&mut session, &rgb_c)));
    }
    let get = |embs: &[(i64, Vec<f32>)], x: i64| embs.iter().find(|(i, _)| *i == x).unwrap().1.clone();

    eprintln!("\n=== Phase 5: A↔B / A↔C / B↔C 单图一致性 ===");
    eprintln!("Image   A↔B     A↔C     B↔C");
    for s in &samples {
        let a = get(&embs_a, s.img);
        let b = get(&embs_b, s.img);
        let c = get(&embs_c, s.img);
        eprintln!("img{:<3} {:.4}  {:.4}  {:.4}", s.img, cosine(&a, &b), cosine(&a, &c), cosine(&b, &c));
    }

    eprintln!("\n=== Phase 6: jaqor pair matrix (三 Path) ===");
    print_matrix("Path A (生产 crop)", &SAME, &embs_a);
    print_matrix("Path B (无 crop)", &SAME, &embs_b);
    print_matrix("Path C (严格 crop)", &SAME, &embs_c);

    eprintln!("\n=== Phase 7: 负例指标 (SAME={:?} NEG={:?}) ===", SAME, NEG);
    let mk_scores = |embs: &[(i64, Vec<f32>)]| -> Vec<(i64, i64, f32)> {
        let mut sc = Vec::new();
        for i in 0..embs.len() {
            for j in (i + 1)..embs.len() {
                sc.push((embs[i].0, embs[j].0, cosine(&embs[i].1, &embs[j].1)));
            }
        }
        sc
    };
    eprintln!("{:<22} {:>8} {:>8} {:>12} {:>11} {:>8} {:>8} {:>10} {:>12}", "Path", "SameMin", "SameMean", "HardNegMax", "Separation", "15PosMin", "15PosMean", "15NegMax", "15Margin");
    for (name, embs) in [("A: 生产 crop", &embs_a), ("B: 无 crop", &embs_b), ("C: 严格 crop", &embs_c)] {
        let scores = mk_scores(embs);
        let (sm, sme, nm, sep) = full_metrics(&scores);
        let (pmin, pmean, negmax, mar) = img15_metrics(&scores);
        eprintln!("{name:<22} {:>8.4} {:>8.4} {:>12.4} {:>11.4} {:>8.4} {:>8.4} {:>10.4} {:>12.4}", sm, sme, nm, sep, pmin, pmean, negmax, mar);
    }

    // ============ Phase 8: Crop Margin Sweep ============
    eprintln!("\n=== Phase 8: Crop Margin Sweep (仅观察稳定趋势, 非选参) ===");
    eprintln!("{:<6} {:>8} {:>8} {:>8} {:>7} {:>7} {:>8} {:>8}", "Margin", "15↔6", "15↔13", "15↔14", "PosMin", "NegMax", "Margin", "SameMin");
    for margin in [0.0f32, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30] {
        let mut embs: Vec<(i64, Vec<f32>)> = Vec::new();
        for s in &samples {
            let i = idx(s.img);
            let (rgb, _) = path_a(&aligner, &cfg, &images[i], &s.bbox, &s.kps, margin).expect("A sweep");
            embs.push((s.img, embed_one(&mut session, &rgb)));
        }
        let e15 = get(&embs, 15);
        let s6 = cosine(&e15, &get(&embs, 6));
        let s13 = cosine(&e15, &get(&embs, 13));
        let s14 = cosine(&e15, &get(&embs, 14));
        let pos_min = s6.min(s13).min(s14);
        let neg_max = NEG.iter().map(|&n| cosine(&e15, &get(&embs, n))).fold(f32::NEG_INFINITY, f32::max);
        let mut same_all: Vec<f32> = Vec::new();
        for a in 0..embs.len() {
            for b in (a + 1)..embs.len() {
                if is_same(embs[a].0, embs[b].0) {
                    same_all.push(cosine(&embs[a].1, &embs[b].1));
                }
            }
        }
        let same_min = same_all.iter().copied().fold(f32::INFINITY, f32::min);
        let pct = (margin * 100.0) as i32;
        eprintln!("{pct:>3}%    {:>8.3} {:>8.3} {:>8.3} {:>7.3} {:>7.3} {:>8.3} {:>8.3}", s6, s13, s14, pos_min, neg_max, pos_min - neg_max, same_min);
    }

    // ============ Phase 9: 保存对齐图 + 诊断图 ============
    fs::create_dir_all(OUT_DIR).expect("mkdir");
    eprintln!("\n=== Phase 9: 保存对齐图到 {OUT_DIR} ===");
    for &img in &SAME {
        let i = idx(img);
        let s = &samples[i];
        let im = &images[i];

        // 原图 + bbox + SCRFD landmarks
        let mut orig = im.as_rgb8();
        draw_rect(&mut orig, &s.bbox, [0, 255, 0], 3);
        draw_kps(&mut orig, &s.kps, [255, 0, 0], 6);
        orig.save(Path::new(OUT_DIR).join(format!("img{img}_orig.png"))).expect("save orig");

        // crop + shifted landmarks
        let (crop, origin) = crop_to_bbox_with_margin(im, &s.bbox, 0.10).expect("crop");
        let shifted = shift_kps(&s.kps, origin);
        let mut crp = crop.as_rgb8();
        draw_kps(&mut crp, &shifted, [255, 0, 0], 6);
        crp.save(Path::new(OUT_DIR).join(format!("img{img}_crop.png"))).expect("save crop");

        // Path A / B aligned + canonical(green) vs placed(red)
        let (rgb_a, placed_a) = path_a(&aligner, &cfg, im, &s.bbox, &s.kps, 0.10).expect("A");
        let (rgb_b, placed_b) = path_b(&aligner, &cfg, im, &s.kps).expect("B");
        for (name, rgb, placed) in [("A", &rgb_a, &placed_a), ("B", &rgb_b, &placed_b)] {
            let mut a = (*rgb).clone();
            draw_kps(&mut a, placed, [255, 0, 0], 2); // 实际落点（红）
            draw_kps(&mut a, &ref_pts, [0, 255, 0], 4); // canonical（绿，外圈）
            a.save(Path::new(OUT_DIR).join(format!("img{img}_{name}_aligned.png"))).expect("save aligned");
        }

        // 2×2 诊断图: [orig, crop / aligned A, aligned B]
        let diag = composite(&[&orig, &crp, &rgb_a, &rgb_b], 2, 280);
        diag.save(Path::new(OUT_DIR).join(format!("img{img}_diag.png"))).expect("save diag");
        eprintln!("  saved img{img}: orig / crop / A_aligned / B_aligned / diag");
    }

    // ============ Phase 10: Alignment Error ============
    eprintln!("\n=== Phase 10: Alignment Error (placed vs canonical, 归一化=误差/ref_eye_dist={ref_eye:.1}px) ===");
    eprintln!("Image  Path  AlignError(px)  Normalized");
    let mut align_errs: Vec<(i64, &str, f32, f32)> = Vec::new();
    for &img in &SAME {
        let i = idx(img);
        let s = &samples[i];
        for (name, placed) in [
            ("A", placed_landmarks(&shift_kps(&s.kps, crop_to_bbox_with_margin(&images[i], &s.bbox, 0.10).expect("crop").1), &cfg)),
            ("B", placed_landmarks(&s.kps, &cfg)),
        ] {
            let err = (0..5).map(|k| dist(placed[k], ref_pts[k])).sum::<f32>() / 5.0;
            let norm = err / ref_eye;
            align_errs.push((img, name, err, norm));
            eprintln!("img{img}  {name}   {err:.4}          {norm:.4}");
        }
    }
    let err15_a = align_errs.iter().find(|(i, p, _, _)| *i == 15 && *p == "A").unwrap().2;
    let med_other = {
        let mut v: Vec<f32> = align_errs
            .iter()
            .filter(|(i, p, _, _)| *i != 15 && *p == "A")
            .map(|(_, _, e, _)| *e)
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let landmark_abnormal = err15_a > 2.0 * med_other.max(1e-6);

    // ============ Phase 11: img15 人工 canonical landmark ============
    eprintln!("\n=== Phase 11: img15 人工 canonical landmark (绕过 SCRFD 5 点) ===");
    let s15 = &samples[idx(15)];
    let im15 = &images[idx(15)];
    let (crop15, _origin) = crop_to_bbox_with_margin(im15, &s15.bbox, 0.10).expect("crop15");
    let (cw, ch) = (crop15.width() as f32, crop15.height() as f32);

    // (a) plain resize: crop → 112×112（无任何 landmark 仿射）。
    //     注意不能用 ImageData::resize（DynamicImage::resize 保纵横比，img15 会得 94×112）；
    //     用 imageops::resize 精确拉伸到 112×112。
    let resized = image::imageops::resize(
        &crop15.as_rgb8(),
        INPUT_SIZE,
        INPUT_SIZE,
        image::imageops::FilterType::Lanczos3,
    );
    let emb_resize = embed_one(&mut session, &resized);

    // (b) canonical landmark: 把 canonical 参考点映射进 crop，居中放置且全部落在 crop 内，
    //     强制 img15 对齐到 canonical 几何（绕过 SCRFD 5 点）。
    //     不能直接从原点按 (cw/112, ch/112) 缩放——img15 的 mouth 会超出 crop 下界。
    let (rx_min, ry_min) = ref_pts.iter().fold((f32::MAX, f32::MAX), |(mx, my), p| (mx.min(p.0), my.min(p.1)));
    let (rx_max, ry_max) = ref_pts.iter().fold((f32::MIN, f32::MIN), |(mx, my), p| (mx.max(p.0), my.max(p.1)));
    let (rw, rh) = (rx_max - rx_min, ry_max - ry_min);
    let scale_c = (cw / rw).min(ch / rh) * 0.8; // 保纵横比，留 20% 边
    let (ccx, ccy) = (cw / 2.0, ch / 2.0);
    let (crx, cry) = ((rx_min + rx_max) / 2.0, (ry_min + ry_max) / 2.0);
    let canon_crop: [(f32, f32); 5] = ref_pts.map(|(x, y)| (ccx + (x - crx) * scale_c, ccy + (y - cry) * scale_c));
    let aligned_c = aligner
        .align(&crop15, &kps_from_array(&canon_crop))
        .expect("canon align");
    let emb_canon = embed_one(&mut session, &aligned_c.image.as_rgb8());

    let a15 = get(&embs_a, 15);
    eprintln!("img15 生产(Path A) 参照值:");
    for &t in &[6, 13, 14] {
        let ea = get(&embs_a, t);
        eprintln!("  img15[A] ↔ img{t}: {:.4}", cosine(&a15, &ea));
    }
    eprintln!("manual 对齐后:");
    for &t in &[6, 13, 14] {
        let ea = get(&embs_a, t);
        eprintln!("  img15[resize] ↔ img{t}: {:.4}", cosine(&emb_resize, &ea));
        eprintln!("  img15[canon ] ↔ img{t}: {:.4}", cosine(&emb_canon, &ea));
    }
    eprintln!("manual 对齐后 vs 负例:");
    for &n in &NEG {
        let en = get(&embs_a, n);
        eprintln!("  img15[resize] ↔ img{n}: {:.4}", cosine(&emb_resize, &en));
        eprintln!("  img15[canon ] ↔ img{n}: {:.4}", cosine(&emb_canon, &en));
    }
    let base_13 = cosine(&a15, &get(&embs_a, 13));
    let manual_13 = cosine(&emb_resize, &get(&embs_a, 13)).max(cosine(&emb_canon, &get(&embs_a, 13)));
    let manual_improves = manual_13 - base_13 >= 0.10;
    let manual_crosses_075 = manual_13 >= 0.75;

    // ============ Phase 12: 判定 ============
    eprintln!("\n=== Phase 12: 最终判定 ===");
    let b_15_jaqor = get(&embs_b, 15);
    let a_15_jaqor = get(&embs_a, 15);
    let no_crop_improve = {
        let pa: Vec<f32> = SAME.iter().filter(|&&x| x != 15).map(|&x| cosine(&a_15_jaqor, &get(&embs_a, x))).collect();
        let pb: Vec<f32> = SAME.iter().filter(|&&x| x != 15).map(|&x| cosine(&b_15_jaqor, &get(&embs_b, x))).collect();
        let ma = pa.iter().sum::<f32>() / pa.len() as f32;
        let mb = pb.iter().sum::<f32>() / pb.len() as f32;
        eprintln!("  img15↔jaqor: A mean={ma:.4}  B(无crop) mean={mb:.4}  Δ={:+.4}", mb - ma);
        mb - ma >= 0.10
    };
    eprintln!("  img15 归一化 alignment error: {:.4} (other jaqor median {:.4}, ratio {:.2}x)", err15_a / ref_eye, med_other / ref_eye, err15_a / med_other.max(1e-6));

    eprintln!("\n========================================");
    eprintln!("IMG15 CROP / ALIGNMENT DIAGNOSIS");
    eprintln!("========================================");
    eprintln!("1. Crop 坐标变换是否正确: {}", if crop_correct { "YES" } else { "NO" });
    eprintln!("2. SCRFD landmarks 是否异常: {}", if landmark_abnormal { "YES" } else { "NO" });
    eprintln!("3. crop 是否造成明显 alignment distortion: (见 Phase 5 B↔C 与 Phase 6/7 差异)");
    eprintln!("4. 不 crop 是否明显改善 img15: {}", if no_crop_improve { "YES" } else { "NO" });
    eprintln!("5. crop margin 是否存在稳定最佳区间: (见 Phase 8 表格, 看单调趋势)");
    eprintln!("6. manual landmark 是否改善: {} (Δ={:+.4}, cross0.75={})", if manual_improves { "YES" } else { "NO" }, manual_13 - base_13, manual_crosses_075);
    eprintln!("----------------------------------------");
    eprintln!("关键数据: 生产 img15↔img13 = {base_13:.4}; manual(最优)↔img13 = {manual_13:.4}");

    // ============ Phase 13: img15 输入变体矩阵 (A-H) — 是否存在真分离 ============
    // 目标：逐一换输入生成方式，计算 img15 → 9 个样本 cosine 与
    //   PosMin / PosMean / NegMax / Margin = PosMin - NegMax。
    // 关键判据：能否 positive↑ / negative↓（真分离），而非所有 cosine 一起下降。
    // 若所有预处理都无法 positive > negative → 问题进入 ArcFace embedding 层。
    eprintln!("\n=== Phase 13: img15 输入变体矩阵 (A-H) ===");

    // 全精度 bbox/kps：实时 detect（最高分 det，= DB 存储值），避免 TSV 截断
    struct Geo {
        bbox: BBox,
        kps: [(f32, f32); 5],
    }
    let mut geos: Vec<Geo> = Vec::new();
    for s in &samples {
        let i = idx(s.img);
        let dets = rt.block_on(scrfd.detect(&images[i])).expect("detect");
        let best = dets
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .expect("best det");
        geos.push(Geo {
            bbox: best.bbox,
            kps: [
                best.keypoints.left_eye,
                best.keypoints.right_eye,
                best.keypoints.nose,
                best.keypoints.left_mouth,
                best.keypoints.right_mouth,
            ],
        });
    }

    enum Input {
        AProd,   // 生产：crop10 + SimpleAligner
        BRaw0,   // 原始 bbox crop(margin0) + align
        C20,     // bbox + 20% margin + align
        D30,     // bbox + 30% margin + align
        ECanon,  // 只按 landmarks canonical align（全图，无 crop）
        FResize, // 原始人脸区域(margin0) → 112×112，无 align（lanczos）
        GMirror, // 生产对齐图 左右镜像（全部样本统一镜像）
        H(image::imageops::FilterType), // 原始人脸区域(margin0) → resize 插值扫描
    }
    fn variant_image(
        img: &ImageData,
        geo: &Geo,
        aligner: &SimpleAligner,
        cfg: &AlignmentConfig,
        v: &Input,
    ) -> Result<RgbImage, String> {
        match v {
            Input::AProd => path_a(aligner, cfg, img, &geo.bbox, &geo.kps, 0.10).map(|(rgb, _)| rgb),
            Input::BRaw0 => path_a(aligner, cfg, img, &geo.bbox, &geo.kps, 0.00).map(|(rgb, _)| rgb),
            Input::C20 => path_a(aligner, cfg, img, &geo.bbox, &geo.kps, 0.20).map(|(rgb, _)| rgb),
            Input::D30 => path_a(aligner, cfg, img, &geo.bbox, &geo.kps, 0.30).map(|(rgb, _)| rgb),
            Input::ECanon => {
                let a = aligner.align(img, &kps_from_array(&geo.kps)).map_err(|e| e.to_string())?;
                Ok(a.image.as_rgb8())
            }
            Input::FResize => {
                let (crop, _) = crop_to_bbox_with_margin(img, &geo.bbox, 0.00)?;
                Ok(image::imageops::resize(
                    &crop.as_rgb8(),
                    INPUT_SIZE,
                    INPUT_SIZE,
                    image::imageops::FilterType::Lanczos3,
                ))
            }
            Input::H(f) => {
                let (crop, _) = crop_to_bbox_with_margin(img, &geo.bbox, 0.00)?;
                Ok(image::imageops::resize(&crop.as_rgb8(), INPUT_SIZE, INPUT_SIZE, *f))
            }
            Input::GMirror => {
                let (rgb, _) = path_a(aligner, cfg, img, &geo.bbox, &geo.kps, 0.10)?;
                let mut out = RgbImage::new(INPUT_SIZE, INPUT_SIZE);
                for y in 0..INPUT_SIZE {
                    for x in 0..INPUT_SIZE {
                        out.put_pixel(x, y, *rgb.get_pixel(INPUT_SIZE - 1 - x, y));
                    }
                }
                Ok(out)
            }
        }
    }

    let variants: Vec<(&str, Input)> = vec![
        ("A 生产crop10+align", Input::AProd),
        ("B crop0+align", Input::BRaw0),
        ("C crop20+align", Input::C20),
        ("D crop30+align", Input::D30),
        ("E 只landmark全图align", Input::ECanon),
        ("F 原始人脸resize", Input::FResize),
        ("G 镜像(A)", Input::GMirror),
        ("H nearest", Input::H(image::imageops::FilterType::Nearest)),
        ("H triangle(bilinear)", Input::H(image::imageops::FilterType::Triangle)),
        ("H catmullrom(bicubic)", Input::H(image::imageops::FilterType::CatmullRom)),
        ("H lanczos", Input::H(image::imageops::FilterType::Lanczos3)),
    ];

    // 参考（生产 A）的 margin，用于比较方向
    let mut ref_margin = 0.0f32;
    eprintln!("  img15→:  {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "jaq6", "jaq13", "jaq14", "neg2", "neg9", "neg20", "neg17", "neg18", "neg19");
    for (name, v) in &variants {
        let mut embs: Vec<(i64, Vec<f32>)> = Vec::new();
        let mut failed = false;
        for gi in 0..samples.len() {
            match variant_image(&images[gi], &geos[gi], &aligner, &cfg, v) {
                Ok(rgb) => embs.push((samples[gi].img, embed_one(&mut session, &rgb))),
                Err(e) => {
                    eprintln!("  [err] {name} img{}: {e}", samples[gi].img);
                    failed = true;
                }
            }
        }
        if failed {
            continue;
        }
        let gete = |x: i64| embs.iter().find(|(i, _)| *i == x).unwrap().1.clone();
        let e15 = gete(15);
        let mut row = String::new();
        let mut pos = Vec::new();
        for &t in &[6, 13, 14] {
            let c = cosine(&e15, &gete(t));
            pos.push(c);
            row.push_str(&format!("{:>8.4}", c));
        }
        let mut neg = Vec::new();
        for &n in &NEG {
            let c = cosine(&e15, &gete(n));
            neg.push(c);
            row.push_str(&format!("{:>8.4}", c));
        }
        let pos_min = pos.iter().cloned().fold(f32::INFINITY, f32::min);
        let pos_mean = pos.iter().sum::<f32>() / pos.len() as f32;
        let neg_max = neg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let margin = pos_min - neg_max;
        if matches!(*v, Input::AProd) {
            ref_margin = margin;
        }
        let delta = margin - ref_margin;
        let sep = if margin > 0.0 {
            "正分离"
        } else if delta >= 0.05 {
            "↑"
        } else if delta <= -0.05 {
            "↓"
        } else {
            "~"
        };
        eprintln!("  {name:<22}{row}  | PosMin={:.4} PosMean={:.4} NegMax={:.4} Margin={:+.4} ({sep})", pos_min, pos_mean, neg_max, margin);
    }
    eprintln!("  >> 若所有变体 Margin<=0 且相对生产无稳定 ↑，则确认问题已进入 ArcFace embedding 层。");

    eprintln!("\n=== DONE ===");
}
