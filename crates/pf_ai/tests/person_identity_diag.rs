//! 临时诊断（非生产）：Person-level identity diagnosis for img15（Phase 14–18）。
//!
//! 前置结论：A/B 换模型（glintr100）不解决 img15（Margin -0.1498 → -0.1892，更差），
//! 故本阶段从【人脸/身份质量】与【person prototype 聚合】角度检查，回答三个问题：
//!   Q1. img15 是不是 pose/quality 问题？
//!   Q2. person2 多 prototype 是否能正确吸收 img15？
//!   Q3. 是否应把"单张 face → person"决策改成"face → person prototypes"？
//!
//! Phase 14 — Face Quality CSV（quality/det/yaw/pitch/roll/face_w/h/eye_dist/blur/align_err）
//! Phase 15 — Aligned Face Diagnostic PNG（原图/bbox/5pts/crop10/aligned/mask）+ 逐项检查
//! Phase 16 — Leave-One-Out Prototype Test（P6/P13/P14 单 prototype）
//! Phase 17 — Person Prototype Aggregation（max/mean/top2/centroid/q-centroid/q-top2）
//! Phase 18 — Prototype Contamination Test（按 quality 加入，跟踪 centroid 漂移）
//!
//! 用法：
//!   ORT_DYLIB_PATH=/Users/mac/lib/libonnxruntime.dylib \
//!   cargo test --release -p pf_ai --test person_identity_diag -- --ignored --nocapture
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
use pf_ai::quality::{assess, blur_score_from_aligned, QualityBreakdown};
use pf_ai::{FaceDetector, ScrfdDetector};
use pf_core::{BBox, FaceKeypoints};

const ARCFACE_MODEL: &str = "/Users/mac/ai-project/photofinder-next-2/models/w600k_r50.onnx";
const SCRFD_MODEL: &str = "/Users/mac/ai-project/photofinder-next-2/models/scrfd_500m_bnkps.onnx";
const INPUT_TSV: &str = "/tmp/tta_img15/input.tsv";
const OUT_DIR: &str = "/tmp/person_diag";

const PERSON2: [i64; 4] = [6, 13, 14, 15];
const NEG: [i64; 6] = [2, 9, 20, 17, 18, 19];
const FOCUS: [i64; 4] = [6, 13, 14, 15];

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
}

fn read_tsv() -> Vec<Sample> {
    let content = std::fs::read_to_string(INPUT_TSV).expect("read tsv");
    let mut out = Vec::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        out.push(Sample {
            img: f[0].parse().unwrap(),
            path: f[1].to_string(),
        });
    }
    out
}

// ---- 推理 / 对齐（生产路径复刻）----

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

/// 用指定 aligner 走生产对齐路径：crop(margin=0.10) + kp 平移 + align → 112×112 RGB。
fn path_a(
    aligner: &SimpleAligner,
    img: &ImageData,
    bbox: &BBox,
    kps: &[(f32, f32); 5],
) -> Result<RgbImage, String> {
    let (crop, origin) = crop_to_bbox_with_margin(img, bbox, 0.10)?;
    let shifted = shift_kps(kps, origin);
    let aligned = aligner.align(&crop, &kps_from_array(&shifted)).map_err(|e| e.to_string())?;
    Ok(aligned.image.as_rgb8())
}

// ---- 椭圆 mask 几何（复刻 aligner::apply_ellipse_mask 常量）----
const MASK_CX: f32 = 56.0;
const MASK_CY: f32 = 53.0;
const MASK_RX: f32 = INPUT_SIZE as f32 * 0.40; // 44.8
const MASK_RY: f32 = INPUT_SIZE as f32 * 0.48; // 53.76

fn in_mask(x: f32, y: f32) -> bool {
    let dx = (x - MASK_CX) / MASK_RX;
    let dy = (y - MASK_CY) / MASK_RY;
    dx * dx + dy * dy <= 1.0
}

// ---- 单样本信息 ----

struct FaceInfo {
    img: i64,
    bbox: BBox,
    kps: [(f32, f32); 5],
    ypr: Option<(f32, f32, f32)>,
    q: QualityBreakdown,
    aligned_prod: RgbImage,
    aligned_raw: RgbImage,
    crop: RgbImage,
    emb: Vec<f32>,
    align_err: f32,
    align_err_norm: f32,
    placed: [(f32, f32); 5],
    face_area_ratio: f32,
}

/// 椭球内非黑像素比例（= 有效保留区域）。
fn mask_effective(aligned: &RgbImage) -> f32 {
    let mut n = 0u32;
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let p = aligned.get_pixel(x, y);
            if p[0] > 8 || p[1] > 8 || p[2] > 8 {
                n += 1;
            }
        }
    }
    n as f32 / (INPUT_SIZE * INPUT_SIZE) as f32
}

/// 亮度统计（用于 illumination 检查）。入参为原始像素图（未 histeq）。
fn luma_stats(rgb: &RgbImage) -> (f32, f32, f32, f32) {
    let (w, h) = (rgb.width(), rgb.height());
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut left = 0.0f64;
    let mut right = 0.0f64;
    let mut nl = 0.0f64;
    let mut nr = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            let p = rgb.get_pixel(x, y);
            let lum = (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) as f64;
            sum += lum;
            sum_sq += lum * lum;
            if x < w / 2 {
                left += lum;
                nl += 1.0;
            } else {
                right += lum;
                nr += 1.0;
            }
        }
    }
    let n = (w * h) as f64;
    let mean = sum / n;
    let var = (sum_sq / n) - mean * mean;
    (
        mean as f32,
        var.sqrt() as f32,
        (left / nl) as f32,
        (right / nr) as f32,
    )
}

/// 构建所有样本的 FaceInfo（实时 SCRFD 全精度 bbox/kps）。
fn build_infos(
    rt: &tokio::runtime::Runtime,
    scrfd: &ScrfdDetector,
    aligner_prod: &SimpleAligner,
    aligner_raw: &SimpleAligner,
    cfg: &AlignmentConfig,
    session: &mut Session,
    samples: &[Sample],
    images: &[ImageData],
) -> Vec<FaceInfo> {
    let mut infos = Vec::new();
    for s in samples {
        let i = samples.iter().position(|x| x.img == s.img).unwrap();
        let im = &images[i];
        let dets = rt.block_on(scrfd.detect(im)).expect("detect");
        let best = dets
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .expect("best det");
        let bbox = best.bbox;
        let kps: [(f32, f32); 5] = [
            best.keypoints.left_eye,
            best.keypoints.right_eye,
            best.keypoints.nose,
            best.keypoints.left_mouth,
            best.keypoints.right_mouth,
        ];
        let ypr = estimate_yaw_pitch_roll(&best.keypoints);
        let aligned_prod = path_a(aligner_prod, im, &bbox, &kps).expect("path_a prod");
        let aligned_raw = path_a(aligner_raw, im, &bbox, &kps).expect("path_a raw");
        let (crop, _) = crop_to_bbox_with_margin(im, &bbox, 0.10).expect("crop");
        let crop_rgb = crop.as_rgb8();
        let gray = image::DynamicImage::from(aligned_prod.clone()).to_luma8();
        let blur = blur_score_from_aligned(&gray);
        let q = assess(best, blur, ypr);
        let emb = embed_one(session, &aligned_prod);
        let shifted = shift_kps(&kps, crop_to_bbox_with_margin(im, &bbox, 0.10).expect("crop").1);
        let placed = placed_landmarks(&shifted, cfg);
        let ref_pts = ref_landmarks(cfg);
        let err = (0..5).map(|k| dist(placed[k], ref_pts[k])).sum::<f32>() / 5.0;
        let ref_eye = ref_pts[1].0 - ref_pts[0].0; // 25.0
        let face_area_ratio = bbox.w * bbox.h / (im.width() as f32 * im.height() as f32);
        infos.push(FaceInfo {
            img: s.img,
            bbox,
            kps,
            ypr,
            q,
            aligned_prod,
            aligned_raw,
            crop: crop_rgb,
            emb,
            align_err: err,
            align_err_norm: err / ref_eye,
            placed,
            face_area_ratio,
        });
    }
    infos
}

// ---- 绘图 ----

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

/// 绘制 mask 椭圆边界（在给定图上）。
fn draw_mask_ellipse(img: &mut RgbImage, color: [u8; 3], step: u32) {
    for y in 0..=INPUT_SIZE {
        let yf = y as f32;
        for x in 0..=INPUT_SIZE {
            let xf = x as f32;
            let d = ((xf - MASK_CX) / MASK_RX).powi(2) + ((yf - MASK_CY) / MASK_RY).powi(2);
            if (d - 1.0).abs() < 0.06 {
                img.put_pixel(x % INPUT_SIZE, y % INPUT_SIZE, Rgb(color));
            }
        }
    }
    let _ = step;
}

#[test]
#[ignore]
fn person_identity_diag() {
    fs::create_dir_all(OUT_DIR).expect("mkdir");
    let samples = read_tsv();
    assert_eq!(samples.len(), 10, "expect 10 samples");
    let cfg = AlignmentConfig::default();
    let aligner_prod = SimpleAligner::with_config(cfg.clone());
    let aligner_raw = SimpleAligner::with_config(AlignmentConfig {
        use_ellipse_mask: false,
        use_histogram_eq: false,
        ..cfg.clone()
    });
    let images: Vec<ImageData> = samples
        .iter()
        .map(|s| ImageData::from_file(Path::new(&s.path)).expect("open"))
        .collect();

    let rt = tokio::runtime::Runtime::new().expect("tokio");
    let scrfd = ScrfdDetector::load(Path::new(SCRFD_MODEL)).expect("scrfd");
    let mut session = Session::builder()
        .expect("builder")
        .commit_from_file(PathBuf::from(ARCFACE_MODEL))
        .expect("load arcface");

    let infos = build_infos(
        &rt, &scrfd, &aligner_prod, &aligner_raw, &cfg, &mut session, &samples, &images,
    );
    let get = |x: i64| infos.iter().find(|f| f.img == x).expect("info");

    // ============ Phase 14: Face Quality CSV ============
    eprintln!("\n=== Phase 14: Face Quality / Identity Quality ===");
    let mut csv = String::new();
    csv.push_str("img,quality,det_score,yaw,pitch,roll,face_width,face_height,eye_distance,blur_score,alignment_error,alignment_error_norm,face_area_score,pose_score,face_area_ratio\n");
    for f in &infos {
        let (yaw, pitch, roll) = f.ypr.unwrap_or((0.0, 0.0, 0.0));
        csv.push_str(&format!(
            "{},{:.4},{:.4},{:.2},{:.2},{:.2},{:.1},{:.1},{:.1},{:.4},{:.4},{:.4},{:.4},{:.4},{:.6}\n",
            f.img, f.q.quality, f.q.detector_score, yaw, pitch, roll,
            f.bbox.w, f.bbox.h, f.q.eye_distance, f.q.blur_score,
            f.align_err, f.align_err_norm, f.q.face_area_score, f.q.pose_score, f.face_area_ratio,
        ));
    }
    fs::write(Path::new(OUT_DIR).join("phase14_quality.csv"), &csv).expect("write csv");

    eprintln!("{:<4} {:>7} {:>7} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8} {:>7} {:>8} {:>7}", "img", "quality", "det", "yaw", "pitch", "roll", "f_w", "f_h", "eyeD", "blur", "alErr", "alErrN");
    for f in &infos {
        let (yaw, pitch, roll) = f.ypr.unwrap_or((0.0, 0.0, 0.0));
        eprintln!(
            "{:<4} {:>7.3} {:>7.3} {:>6.1} {:>6.1} {:>6.1} {:>8.0} {:>8.0} {:>8.1} {:>7.3} {:>8.2} {:>7.3}",
            f.img, f.q.quality, f.q.detector_score, yaw, pitch, roll,
            f.bbox.w, f.bbox.h, f.q.eye_distance, f.q.blur_score, f.align_err, f.align_err_norm,
        );
    }
    // 重点对比 img15 vs img6/13/14
    eprintln!("\n  重点对比（img15 vs 同身份 img6/13/14）:");
    eprintln!("  {:<5} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}", "img", "quality", "det", "yaw", "roll", "blur", "alErrN");
    for &t in &FOCUS {
        let f = get(t);
        let (yaw, _, roll) = f.ypr.unwrap_or((0.0, 0.0, 0.0));
        eprintln!(
            "  {:<5} {:>8.3} {:>8.3} {:>8.1} {:>8.1} {:>8.3} {:>8.3}",
            format!("img{t}"), f.q.quality, f.q.detector_score, yaw, roll, f.q.blur_score, f.align_err_norm,
        );
    }

    // ============ Phase 15: Aligned Face Diagnostic ============
    eprintln!("\n=== Phase 15: Aligned Face Diagnostic ===");
    for f in &infos {
        let img = f.img;
        let im = &images[samples.iter().position(|s| s.img == img).unwrap()];
        // 原图 + bbox + 5 点
        let mut orig = im.as_rgb8();
        draw_rect(&mut orig, &f.bbox, [0, 255, 0], 3);
        draw_kps(&mut orig, &f.kps, [255, 0, 0], 6);
        // crop + shifted kps
        let (crop, origin) = crop_to_bbox_with_margin(im, &f.bbox, 0.10).expect("crop");
        let shifted = shift_kps(&f.kps, origin);
        let mut crp = crop.as_rgb8();
        draw_kps(&mut crp, &shifted, [255, 0, 0], 6);
        // aligned_prod + canonical(green)/placed(red)
        let ref_pts = ref_landmarks(&cfg);
        let mut apr = f.aligned_prod.clone();
        draw_kps(&mut apr, &f.placed, [255, 0, 0], 2);
        draw_kps(&mut apr, &ref_pts, [0, 255, 0], 4);
        // aligned_raw
        let mut raw = f.aligned_raw.clone();
        draw_kps(&mut raw, &f.placed, [255, 0, 0], 2);
        draw_kps(&mut raw, &ref_pts, [0, 255, 0], 4);
        // mask 边界 overlay on raw
        let mut maskv = f.aligned_raw.clone();
        draw_mask_ellipse(&mut maskv, [255, 0, 255], 0);
        // 保存
        orig.save(Path::new(OUT_DIR).join(format!("img{img}_orig.png"))).expect("save");
        crp.save(Path::new(OUT_DIR).join(format!("img{img}_crop.png"))).expect("save");
        apr.save(Path::new(OUT_DIR).join(format!("img{img}_aligned_prod.png"))).expect("save");
        raw.save(Path::new(OUT_DIR).join(format!("img{img}_aligned_raw.png"))).expect("save");
        maskv.save(Path::new(OUT_DIR).join(format!("img{img}_mask.png"))).expect("save");
        let diag = composite(&[&orig, &crp, &apr, &raw, &maskv], 3, 300);
        diag.save(Path::new(OUT_DIR).join(format!("img{img}_diag.png"))).expect("save");

        // ---- 逐项检查（文本）----
        let (yaw, pitch, roll) = f.ypr.unwrap_or((0.0, 0.0, 0.0));
        // 1. 脸部占比
        let face_ratio = f.face_area_ratio;
        let eye_f = f.q.eye_distance;
        let eye_frac = eye_f / (f.bbox.w.max(f.bbox.h));
        // 2. 左右脸可见度（yaw 符号 + placed nose 相对中心偏移 + 左右有效像素）
        let nose_x = f.placed[2].0;
        let eye_cx = (f.placed[0].0 + f.placed[1].0) / 2.0;
        let nose_off = nose_x - eye_cx; // + = nose 偏右（可能左脸更可见）
        let mut left_ok = 0u32;
        let mut right_ok = 0u32;
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let p = f.aligned_prod.get_pixel(x, y);
                let vis = p[0] > 8 || p[1] > 8 || p[2] > 8;
                if x < INPUT_SIZE / 2 {
                    left_ok += vis as u32;
                } else {
                    right_ok += vis as u32;
                }
            }
        }
        // 3. landmark 几何（placed 坐标）
        let (ple, pre) = (f.placed[0], f.placed[1]);
        let placed_eye = dist(ple, pre);
        // 4. mask 有效区域
        let eff = mask_effective(&f.aligned_prod);
        // 6/7. 遮挡与光照（在 raw crop 上测，未 histeq）
        let (lum_mean, lum_std, lum_left, lum_right) = luma_stats(&f.crop);
        let illum_asym = (lum_left - lum_right).abs() / (lum_left + lum_right).max(1e-6);
        // 5 个 placed landmark 是否落在 mask 椭圆内
        let in_mask_n = f.placed.iter().filter(|p| in_mask(p.0, p.1)).count();
        eprintln!(
            "img{img}: face_ratio={:.5} eye_frac_in_bbox={:.3} yaw={yaw:.1} pitch={pitch:.1} roll={roll:.1} | nose_off={:+.1}px Lvis/Rvis={left_ok}/{right_ok} | placed_eye={placed_eye:.1}px mask_eff={eff:.3} | lum mean={lum_mean:.0} std={lum_std:.0} L={lum_left:.0} R={lum_right:.0} asym={illum_asym:.3} | placed_in_mask={in_mask_n}/5",
            face_ratio, eye_frac, nose_off,
        );
    }

    // ============ Phase 16: Leave-One-Out Prototype Test ============
    eprintln!("\n=== Phase 16: Leave-One-Out Prototype Test ===");
    eprintln!("  person2 = {{6,13,14}}；P6=emb6, P13=emb13, P14=emb14");
    eprintln!("{:<16} {:>8} {:>8} {:>8}", "query → proto", "P6", "P13", "P14");
    for q in [15, 6, 13, 14] {
        let eq = &get(q).emb;
        let row: Vec<String> = [6, 13, 14]
            .iter()
            .map(|&p| format!("{:>8.4}", cosine(eq, &get(p).emb)))
            .collect();
        eprintln!("{:<16} {}", format!("img{q} →"), row.join(" "));
    }
    eprintln!("  img15 最近单 prototype: P{}={:.4}（vs asian-man NegMax={:.4}）",
        [6, 13, 14].iter().max_by(|&&a, &&b| cosine(&get(15).emb, &get(a).emb).partial_cmp(&cosine(&get(15).emb, &get(b).emb)).unwrap()).unwrap(),
        [6, 13, 14].iter().map(|&p| cosine(&get(15).emb, &get(p).emb)).fold(f32::NEG_INFINITY, f32::max),
        NEG.iter().map(|&n| cosine(&get(15).emb, &get(n).emb)).fold(f32::NEG_INFINITY, f32::max),
    );

    // ============ Phase 17: Person Prototype Aggregation ============
    eprintln!("\n=== Phase 17: Person Prototype Aggregation ===");
    // 正例: person2 {6,13,14} 的 prototype 聚合分数（query=img15）
    // 负例: 6 个独立硬负例（无聚合）。NegMax 跨所有方法相同。
    let neg_max = NEG
        .iter()
        .map(|&n| cosine(&get(15).emb, &get(n).emb))
        .fold(f32::NEG_INFINITY, f32::max);
    let protos: Vec<(Vec<f32>, f32)> = [6, 13, 14]
        .iter()
        .map(|&p| (get(p).emb.clone(), get(p).q.quality))
        .collect();
    let c6 = cosine(&get(15).emb, &get(6).emb);
    let c13 = cosine(&get(15).emb, &get(13).emb);
    let c14 = cosine(&get(15).emb, &get(14).emb);
    let pos_min = c6.min(c13).min(c14);
    let pos_mean = (c6 + c13 + c14) / 3.0;
    eprintln!("  img15→person2 单 prototype: P6={c6:.4} P13={c13:.4} P14={c14:.4} | PosMin={pos_min:.4} PosMean={pos_mean:.4} NegMax={neg_max:.4} Margin_singleMin={:+.4}",
        pos_min - neg_max);

    let agg = |method: &str| -> f32 {
        let mut cos_all: Vec<f32> = protos.iter().map(|(e, _)| cosine(&get(15).emb, e)).collect();
        match method {
            "A_max" => cos_all.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            "B_mean" => cos_all.iter().sum::<f32>() / cos_all.len() as f32,
            "C_top2mean" => {
                cos_all.sort_by(|a, b| b.partial_cmp(a).unwrap());
                (cos_all[0] + cos_all[1]) / 2.0
            }
            "D_centroid" => {
                let mut c = vec![0.0f32; EMBEDDING_DIM];
                for (e, _) in &protos {
                    for (i, v) in e.iter().enumerate() {
                        c[i] += v;
                    }
                }
                let c = l2_normalize(&c);
                cosine(&get(15).emb, &c)
            }
            "E_qcentroid" => {
                let mut c = vec![0.0f32; EMBEDDING_DIM];
                let mut wsum = 0.0f32;
                for (e, q) in &protos {
                    wsum += q;
                    for (i, v) in e.iter().enumerate() {
                        c[i] += v * q;
                    }
                }
                for v in &mut c {
                    *v /= wsum.max(1e-6);
                }
                let c = l2_normalize(&c);
                cosine(&get(15).emb, &c)
            }
            "F_qtop2" => {
                // 按 quality 取 top-2 prototypes，做 weighted mean
                let mut idx: Vec<usize> = (0..protos.len()).collect();
                idx.sort_by(|&a, &b| protos[b].1.partial_cmp(&protos[a].1).unwrap());
                let mut c = vec![0.0f32; EMBEDDING_DIM];
                let mut wsum = 0.0f32;
                for &k in idx.iter().take(2) {
                    let (e, q) = &protos[k];
                    wsum += q;
                    for (i, v) in e.iter().enumerate() {
                        c[i] += v * q;
                    }
                }
                for v in &mut c {
                    *v /= wsum.max(1e-6);
                }
                let c = l2_normalize(&c);
                cosine(&get(15).emb, &c)
            }
            _ => unreachable!(),
        }
    };
    eprintln!("{:<14} {:>12} {:>12} {:>12}", "Method", "person_score", "Margin_person", "Δvs singleMin");
    let base_margin = pos_min - neg_max;
    for m in ["A_max", "B_mean", "C_top2mean", "D_centroid", "E_qcentroid", "F_qtop2"] {
        let s = agg(m);
        let mar = s - neg_max;
        eprintln!("{:<14} {:>12.4} {:>+12.4} {:>+12.4}", m, s, mar, mar - base_margin);
    }

    // ============ Phase 18: Prototype Contamination Test ============
    eprintln!("\n=== Phase 18: Prototype Contamination Test ===");
    // 按 quality DESC 排列 person2 {6,13,14,15}
    let mut order: Vec<i64> = PERSON2.to_vec();
    order.sort_by(|&a, &b| get(b).q.quality.partial_cmp(&get(a).q.quality).unwrap());
    eprintln!("  person2 按 quality DESC 加入顺序: {}", order.iter().map(|x| format!("img{x}(q={:.3})", get(*x).q.quality)).collect::<Vec<_>>().join(" → "));

    let centroid_of = |imgs: &[i64]| -> Vec<f32> {
        let mut c = vec![0.0f32; EMBEDDING_DIM];
        for &x in imgs {
            for (i, v) in get(x).emb.iter().enumerate() {
                c[i] += v;
            }
        }
        l2_normalize(&c)
    };

    // 逐步记录
    let mut acc: Vec<i64> = Vec::new();
    for &x in &order {
        acc.push(x);
        let c = centroid_of(&acc);
        eprintln!("  加入 img{x}: person2 = {{{}}}  centroid={:?}",
            acc.iter().map(|v| format!("img{v}")).collect::<Vec<_>>().join(","),
            acc.iter().map(|v| format!("{}", v)).collect::<Vec<_>>().join("+"),
        );
        // 相对加入前 centroid 的漂移
        if acc.len() > 1 {
            let prev = centroid_of(&acc[..acc.len() - 1]);
            eprintln!("    cos(centroid_new, centroid_prev) = {:.6}", cosine(&c, &prev));
        }
    }

    // 关键：img15 加入前（{6,13,14}）→ 加入后（{6,13,14,15}）的漂移
    let before = [6, 13, 14];
    let after = [6, 13, 14, 15];
    let cb = centroid_of(&before);
    let ca = centroid_of(&after);
    let drift = cosine(&cb, &ca);
    let drift_dist = (1.0 - drift).sqrt(); // chordal-ish proxy
    eprintln!("\n  img15 加入前 person2 = {{6,13,14}}");
    eprintln!("  img15 加入后 person2 = {{6,13,14,15}}");
    eprintln!("  cos(centroid_before, centroid_after) = {drift:.6}  (1-cos={:.6})", 1.0 - drift);
    // 是否向 asian-man 漂移？
    let toward2_before = cosine(&cb, &get(2).emb);
    let toward2_after = cosine(&ca, &get(2).emb);
    let toward9_before = cosine(&cb, &get(9).emb);
    let toward9_after = cosine(&ca, &get(9).emb);
    eprintln!("  person2 centroid ↔ asian-man img2: before={toward2_before:.4} after={toward2_after:.4} Δ={:+.4}", toward2_after - toward2_before);
    eprintln!("  person2 centroid ↔ asian-man img9: before={toward9_before:.4} after={toward9_after:.4} Δ={:+.4}", toward9_after - toward9_before);
    // 加入 img15 后 person2 centroid 仍能代表 img6/13/14 吗？
    eprintln!("  加入后 centroid ↔ img6={:.4} img13={:.4} img14={:.4}（加入前 ↔ 各自={:.4}/{:.4}/{:.4}）",
        cosine(&ca, &get(6).emb), cosine(&ca, &get(13).emb), cosine(&ca, &get(14).emb),
        cosine(&cb, &get(6).emb), cosine(&cb, &get(13).emb), cosine(&cb, &get(14).emb),
    );
    let _ = drift_dist;

    eprintln!("\n=== DONE === 输出目录 {OUT_DIR}（phase14_quality.csv + img*_*.png）");
}
