//! 临时诊断（非生产）：SCRFD 500M vs 10G 人脸检测对比。
//!
//! 目标图片：img6/13/14/15 (person2) + img2/9/17/18/19/20 (negative)
//! 每张图分别用 500M 和 10G 检测：
//!   bbox_diff     : 两 bbox 中心距 + IoU
//!   landmark_diff : 5 个关键点欧氏距离
//!   face_quality  : 两模型 det_score 对比
//!   embedding     : w600k_r50（生产路径），分别用两套 bbox/kps 对齐后的 cosine matrix
//!   metrics       : pos_mean / pos_min / neg_max / margin / rank
//!
//! 用法：
//!   ORT_DYLIB_PATH=/Users/mac/lib/libonnxruntime.dylib \
//!   cargo test --release -p pf_ai --test scrfd_detector_compare -- --ignored --nocapture

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use image::RgbImage;
use ort::session::Session;
use ort::value::Tensor;

use pf_ai::face::aligner::{AlignmentConfig, SimpleAligner};
use pf_ai::face::arcface::{l2_normalize, INPUT_SIZE};
use pf_ai::face::traits::FaceAligner;
use pf_ai::face::FaceDetection;
use pf_ai::image_data::ImageData;
use pf_ai::{FaceDetector, ScrfdDetector};
use pf_core::{BBox, FaceKeypoints};

const SCRFD_500M: &str = "/Users/mac/ai-project/photofinder-next-2/models/scrfd_500m_bnkps.onnx";
const SCRFD_10G: &str = "/Users/mac/ai-project/photofinder-next-2/models/scrfd_10g_bnkps.onnx";
const ARCFACE_MODEL: &str = "/Users/mac/ai-project/photofinder-next-2/models/w600k_r50.onnx";
const INPUT_TSV: &str = "/tmp/tta_img15/input.tsv";
const OUT_DIR: &str = "/tmp/scrfd_compare";
const CROP_MARGIN: f32 = 0.10;

const CORE: [i64; 3] = [6, 13, 14];
const IMG_QUERY: i64 = 15;
const NEG: [i64; 6] = [2, 9, 17, 18, 19, 20];
const ALL: [i64; 10] = [6, 13, 14, 15, 2, 9, 17, 18, 19, 20];

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
    let outputs = session.run(ort::inputs![input]).expect("run");
    let (_shape, data) = outputs[0].try_extract_tensor::<f32>().expect("extract");
    l2_normalize(data)
}

fn kps_from_det(det: &FaceDetection) -> [(f32, f32); 5] {
    let k = &det.keypoints;
    [k.left_eye, k.right_eye, k.nose, k.left_mouth, k.right_mouth]
}

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
    if cw < 1.0 || ch < 1.0 { return Err(format!("crop too small")); }
    let crop = img.crop(BBox::new(x, y, cw, ch)).map_err(|e| e.to_string())?;
    Ok((crop, (x, y)))
}

fn shift_kps(kps: &[(f32, f32); 5], origin: (f32, f32)) -> [(f32, f32); 5] {
    kps.map(|(x, y)| (x - origin.0, y - origin.1))
}

fn build_production(aligner: &SimpleAligner, img: &ImageData, bbox: &BBox, kps: &[(f32, f32); 5]) -> Result<RgbImage, String> {
    let (crop, origin) = crop_to_bbox_with_margin(img, bbox, CROP_MARGIN)?;
    let shifted = shift_kps(kps, origin);
    let aligned = aligner.align(&crop, &FaceKeypoints {
        left_eye: shifted[0], right_eye: shifted[1], nose: shifted[2],
        left_mouth: shifted[3], right_mouth: shifted[4],
    }).map_err(|e| e.to_string())?;
    Ok(aligned.image.as_rgb8())
}

fn bbox_center(b: &BBox) -> (f32, f32) { (b.x + b.w / 2.0, b.y + b.h / 2.0) }

fn bbox_iou(a: &BBox, b: &BBox) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.w).min(b.x + b.w);
    let y2 = (a.y + a.h).min(b.y + b.h);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = a.w * a.h;
    let area_b = b.w * b.h;
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

fn kps_dist_mean(a: &[(f32, f32); 5], b: &[(f32, f32); 5]) -> (f32, [f32; 5]) {
    let mut sum = 0.0f32;
    let mut per = [0.0f32; 5];
    for i in 0..5 {
        let dx = a[i].0 - b[i].0;
        let dy = a[i].1 - b[i].1;
        per[i] = (dx * dx + dy * dy).sqrt();
        sum += per[i];
    }
    (sum / 5.0, per)
}

#[test]
#[ignore]
fn scrfd_detector_compare() {
    fs::create_dir_all(OUT_DIR).expect("mkdir");
    let samples = read_tsv();
    let images: Vec<ImageData> = samples
        .iter()
        .map(|s| ImageData::from_file(std::path::Path::new(&s.path)).expect("open"))
        .collect();

    let rt = tokio::runtime::Runtime::new().expect("tokio");
    let scrfd_500m = ScrfdDetector::load(std::path::Path::new(SCRFD_500M)).expect("load 500m");
    let scrfd_10g = ScrfdDetector::load(std::path::Path::new(SCRFD_10G)).expect("load 10g");
    let aligner = SimpleAligner::with_config(AlignmentConfig::default());

    struct Det {
        bbox: BBox,
        kps: [(f32, f32); 5],
        score: f32,
    }

    let dets_500m: Vec<Det> = samples.iter()
        .enumerate()
        .map(|(i, _s)| {
            let results = rt.block_on(scrfd_500m.detect(&images[i])).expect("500m");
            let best = results.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap()).unwrap();
            Det { bbox: best.bbox, kps: kps_from_det(best), score: best.score }
        })
        .collect();

    let dets_10g: Vec<Det> = samples.iter()
        .enumerate()
        .map(|(i, _s)| {
            let results = rt.block_on(scrfd_10g.detect(&images[i])).expect("10g");
            let best = results.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap()).unwrap();
            Det { bbox: best.bbox, kps: kps_from_det(best), score: best.score }
        })
        .collect();

    // 两套对齐图
    let aligned_500m: Vec<RgbImage> = samples.iter()
        .enumerate()
        .map(|(i, _s)| build_production(&aligner, &images[i], &dets_500m[i].bbox, &dets_500m[i].kps).expect("align 500m"))
        .collect();

    let aligned_10g: Vec<RgbImage> = samples.iter()
        .enumerate()
        .map(|(i, _s)| build_production(&aligner, &images[i], &dets_10g[i].bbox, &dets_10g[i].kps).expect("align 10g"))
        .collect();

    // ArcFace embeddings
    let mut arc1 = Session::builder().expect("builder")
        .commit_from_file(PathBuf::from(ARCFACE_MODEL)).expect("arc1");
    let emb_500m: HashMap<i64, Vec<f32>> = samples.iter()
        .enumerate()
        .map(|(i, s)| (s.img, embed_one(&mut arc1, &aligned_500m[i])))
        .collect();

    let mut arc2 = Session::builder().expect("builder")
        .commit_from_file(PathBuf::from(ARCFACE_MODEL)).expect("arc2");
    let emb_10g: HashMap<i64, Vec<f32>> = samples.iter()
        .enumerate()
        .map(|(i, s)| (s.img, embed_one(&mut arc2, &aligned_10g[i])))
        .collect();

    let cos_500m = |a: i64, b: i64| cosine(emb_500m.get(&a).unwrap(), emb_500m.get(&b).unwrap());
    let cos_10g  = |a: i64, b: i64| cosine(emb_10g.get(&a).unwrap(),  emb_10g.get(&b).unwrap());

    let mut r = String::new();

    r.push_str("\n==================================================\n");
    r.push_str("SCRFD 500M vs 10G COMPARISON\n");
    r.push_str("==================================================\n\n");

    // BBox comparison
    r.push_str("1. BBox Comparison\n");
    r.push_str(&format!("{:<6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}\n",
        "img", "500m_cx", "10g_cx", "delta_cx", "delta_cy", "dist_px", "IoU"));
    for (i, s) in samples.iter().enumerate() {
        let (c5x, c5y) = bbox_center(&dets_500m[i].bbox);
        let (c10x, c10y) = bbox_center(&dets_10g[i].bbox);
        let dcx = (c5x - c10x).abs();
        let dcy = (c5y - c10y).abs();
        let dist = (dcx * dcx + dcy * dcy).sqrt();
        let iou = bbox_iou(&dets_500m[i].bbox, &dets_10g[i].bbox);
        r.push_str(&format!("{:<6} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.4}\n",
            format!("img{}", s.img), c5x, c10x, dcx, dcy, dist, iou));
    }
    r.push('\n');

    // Landmark comparison
    r.push_str("2. Landmark Comparison (mean of 5 kps Euclidean distance)\n");
    r.push_str(&format!("{:<6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}\n",
        "img", "mean_px", "LE", "RE", "Nose", "LM", "RM", "max"));
    for (i, s) in samples.iter().enumerate() {
        let (mean, per) = kps_dist_mean(&dets_500m[i].kps, &dets_10g[i].kps);
        let mx = per.iter().cloned().fold(f32::NEG_INFINITY, |a, b| a.max(b));
        r.push_str(&format!("{:<6} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4}\n",
            format!("img{}", s.img), mean, per[0], per[1], per[2], per[3], per[4], mx));
    }
    r.push('\n');

    // Face quality
    r.push_str("3. Face Quality (detector score)\n");
    r.push_str(&format!("{:<6} {:>12} {:>12} {:>12} {:>10}\n",
        "img", "500M_score", "10G_score", "delta", "winner"));
    for (i, s) in samples.iter().enumerate() {
        let d = dets_500m[i].score - dets_10g[i].score;
        r.push_str(&format!("{:<6} {:>12.4} {:>12.4} {:>+12.4} {:>10}\n",
            format!("img{}", s.img), dets_500m[i].score, dets_10g[i].score, d,
            if d > 0.0 { "500M" } else { "10G" }));
    }
    r.push('\n');

    // Cosine matrix
    r.push_str("4. Embedding Cosine Matrix\n");
    r.push_str(&format!("{:<8}", ""));
    for &img in &ALL { r.push_str(&format!("{:>8}", format!("img{}", img))); }
    r.push_str("  (model)\n");

    for lbl in &["500M", "10G"] {
        r.push_str(&format!("{:<8}", format!("({})", lbl)));
        for &row in &ALL {
            let v = if *lbl == "500M" { cos_500m(IMG_QUERY, row) } else { cos_10g(IMG_QUERY, row) };
            r.push_str(&format!("{:>8.4}", if row == IMG_QUERY { 1.0 } else { v }));
        }
        r.push('\n');
    }
    r.push('\n');

    // Metrics
    r.push_str("5. Metrics Summary\n");
    r.push_str(&format!("{:>8} {:>12} {:>12} {:>12} {:>12} {:>10}\n",
        "model", "pos_mean", "pos_min", "neg_max", "neg_mean", "margin"));
    for lbl in &["500M", "10G"] {
        let pos: Vec<f32> = CORE.iter().map(|&c| if *lbl == "500M" { cos_500m(IMG_QUERY, c) } else { cos_10g(IMG_QUERY, c) }).collect();
        let neg: Vec<f32> = NEG.iter().map(|&n| if *lbl == "500M" { cos_500m(IMG_QUERY, n) } else { cos_10g(IMG_QUERY, n) }).collect();
        let pos_mean = pos.iter().sum::<f32>() / 3.0;
        let pos_min = pos.iter().cloned().fold(f32::INFINITY, |a, b| a.min(b));
        let neg_max = neg.iter().cloned().fold(f32::NEG_INFINITY, |a, b| a.max(b));
        let neg_mean = neg.iter().sum::<f32>() / 6.0;
        let margin = pos_min - neg_max;
        r.push_str(&format!("{:>8} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>+10.4}\n",
            lbl, pos_mean, pos_min, neg_max, neg_mean, margin));
    }
    r.push('\n');

    // Ranking
    r.push_str("6. img15 Ranking\n");
    for lbl in &["500M", "10G"] {
        let mut ranked: Vec<(i64, f32)> = ALL.iter()
            .map(|&t| (t, if *lbl == "500M" { cos_500m(IMG_QUERY, t) } else { cos_10g(IMG_QUERY, t) }))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        r.push_str(&format!("--- {} ---\n", lbl));
        r.push_str(&format!("{:>4} {:<8} {:<10} {:>8}\n", "rank", "img", "group", "cosine"));
        for (i, (img, s)) in ranked.iter().enumerate() {
            let grp = if CORE.contains(img) { "PERSON2" } else { "NEGATIVE" };
            r.push_str(&format!("{:>4} {:<8} {:<10} {:>8.4}\n", i + 1, format!("img{}", img), grp, s));
        }
        r.push('\n');
    }

    r.push_str("=== DONE ===\n");
    eprintln!("{r}");

    // CSV
    let mut csv = String::new();
    csv.push_str("img,500m_cx,10g_cx,dcx,500m_cy,10g_cy,dcy,center_dist,iou,kps_mean,LE,RE,Nose,LM,RM,max_kps,500m_score,10g_score,score_delta\n");
    for (i, s) in samples.iter().enumerate() {
        let (c5x, c5y) = bbox_center(&dets_500m[i].bbox);
        let (c10x, c10y) = bbox_center(&dets_10g[i].bbox);
        let dcx = c5x - c10x;
        let dcy = c5y - c10y;
        let cdist = (dcx * dcx + dcy * dcy).sqrt();
        let iou = bbox_iou(&dets_500m[i].bbox, &dets_10g[i].bbox);
        let (kps_mean, kps_per) = kps_dist_mean(&dets_500m[i].kps, &dets_10g[i].kps);
        let mx = kps_per.iter().cloned().fold(f32::NEG_INFINITY, |a, b| a.max(b));
        let score_500m = dets_500m[i].score;
        let score_10g = dets_10g[i].score;
        let score_delta = score_500m - score_10g;
        csv.push_str(&format_args!(
            "{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
            s.img, c5x, c10x, dcx, c5y, c10y, dcy, cdist, iou, kps_mean,
            kps_per[0], kps_per[1], kps_per[2], kps_per[3], kps_per[4], mx,
            score_500m, score_10g, score_delta
        ).to_string());
    }
    fs::write(std::path::Path::new(OUT_DIR).join("scrfd_compare.csv"), &csv).expect("write csv");
    fs::write(std::path::Path::new(OUT_DIR).join("scrfd_compare.txt"), &r).expect("write txt");
}
