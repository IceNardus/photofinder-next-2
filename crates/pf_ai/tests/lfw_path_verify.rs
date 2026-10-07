//! 临时诊断（非生产）：验证 lfw_eval 的预处理路径是否忠实于生产。
//!
//! Part A：用 lfw_eval 的提取路径（SCRFD 实时 detect → crop 10% → SimpleAligner →
//!         ArcFace 1-crop）跑 stock 的 10 张已知样本，对比 DB 里存的 embedding。
//!         若 cos≈0.99（同 tta_img15 的 stored-bbox 路径），则 lfw_eval 路径无 bug，
//!         证明 LFW 相似度分布是真实数据集属性而非 harness 缺陷。
//! Part B：同一路径跑 LFW 子集（前 N 个 ≥8 张的人 × 8 张），输出
//!         同人 vs 跨人 cosine 分布 + max-F1，复核原 LFW 校准（same-mean≈0.34）。
//!
//! 输入（stock 样本）：/tmp/tta_img15/input.tsv
//! LFW 数据：默认 ~/Library/Caches/PhotoFinder/lfw-eval/lfw-deepfunneled/lfw-deepfunneled
//!
//! 用法：
//!   ORT_DYLIB_PATH=/Users/mac/lib/libonnxruntime.dylib \
//!   cargo test --release -p pf_ai --test lfw_path_verify -- --ignored --nocapture
//!
//! 不改生产代码 / 不改 DB / 不重聚类。

use std::path::{Path, PathBuf};

use pf_ai::face::pose::estimate_yaw_pitch_roll;
use pf_ai::quality::{assess, blur_score_from_aligned, combined_quality};
use pf_ai::{
    ArcFaceEmbedder, FaceAligner, FaceDetection, FaceDetector, FaceEmbedder, ImageData,
    QualityFilter, ScrfdDetector, SimpleAligner,
};
use pf_core::{BBox, FaceKeypoints};

const SCRFD: &str = "/Users/mac/ai-project/photofinder-next-2/models/scrfd_500m_bnkps.onnx";
const ARCFACE: &str = "/Users/mac/ai-project/photofinder-next-2/models/w600k_r50.onnx";
const INPUT_TSV: &str = "/tmp/tta_img15/input.tsv";
const LFW_ROOT: &str =
    "/Users/mac/Library/Caches/PhotoFinder/lfw-eval/lfw-deepfunneled/lfw-deepfunneled";

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn l2(v: &[f32]) -> Vec<f32> {
    let n = dot(v, v).sqrt().max(1e-12);
    v.iter().map(|x| x / n).collect()
}

/// 复刻 lfw_eval `crop_to_bbox_with_margin`（= 生产私有函数拷贝）。
fn crop_to_bbox_with_margin(image: &ImageData, det: &FaceDetection) -> Result<(ImageData, FaceKeypoints), ()> {
    let b = det.bbox;
    let w = b.w.max(1.0);
    let h = b.h.max(1.0);
    let mx = w * 0.1;
    let my = h * 0.1;
    let x = (b.x - mx).max(0.0);
    let y = (b.y - my).max(0.0);
    let cw = (w + 2.0 * mx).min(image.width() as f32 - x);
    let ch = (h + 2.0 * my).min(image.height() as f32 - y);
    if cw < 1.0 || ch < 1.0 {
        return Err(());
    }
    let crop_bbox = BBox::new(x, y, cw, ch);
    let cropped = image.crop(crop_bbox).map_err(|_| ())?;
    let shifted = FaceKeypoints {
        left_eye: (det.keypoints.left_eye.0 - x, det.keypoints.left_eye.1 - y),
        right_eye: (det.keypoints.right_eye.0 - x, det.keypoints.right_eye.1 - y),
        nose: (det.keypoints.nose.0 - x, det.keypoints.nose.1 - y),
        left_mouth: (det.keypoints.left_mouth.0 - x, det.keypoints.left_mouth.1 - y),
        right_mouth: (det.keypoints.right_mouth.0 - x, det.keypoints.right_mouth.1 - y),
    };
    Ok((cropped, shifted))
}

/// lfw_eval 的 extract_best_face（1-crop）：SCRFD 实时 detect → crop → align → embed。
/// 返回 (embedding, detection bbox/keypoints) 以便对比存储值。
fn extract_best(
    rt: &tokio::runtime::Runtime,
    scrfd: &ScrfdDetector,
    aligner: &SimpleAligner,
    e1: &ArcFaceEmbedder,
    qf: &QualityFilter,
    image: &ImageData,
) -> Option<(Vec<f32>, BBox, [(f32, f32); 5])> {
    let dets = rt.block_on(scrfd.detect(image)).ok()?;
    let mut best: Option<(Vec<f32>, BBox, [(f32, f32); 5], f32)> = None;
    for det in dets {
        let ypr = estimate_yaw_pitch_roll(&det.keypoints);
        let (ci, sk) = match crop_to_bbox_with_margin(image, &det) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let aligned = match aligner.align(&ci, &sk) {
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
        let emb = rt.block_on(e1.embed(&aligned)).ok()?.values;
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

struct StockSample {
    img: i64,
    path: String,
    bbox: BBox,
    kps: [(f32, f32); 5],
    db_emb: Vec<f32>,
}

fn read_tsv() -> Vec<StockSample> {
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
        let db_emb: Vec<f32> = f[16..16 + 512].iter().map(|s| s.parse::<f32>().unwrap()).collect();
        out.push(StockSample { img, path, bbox, kps, db_emb });
    }
    out
}

fn cosine_dist(embs: &[Vec<f32>], ids: &[&str], label: &str) {
    let n = embs.len();
    let mut same: Vec<f32> = Vec::new();
    let mut diff: Vec<f32> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let s = dot(&embs[i], &embs[j]);
            if ids[i] == ids[j] {
                same.push(s);
            } else {
                diff.push(s);
            }
        }
    }
    same.sort_by(|a, b| a.partial_cmp(b).unwrap());
    diff.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |v: &Vec<f32>, q: f32| v[(q * (v.len().max(2) - 1) as f32) as usize];
    let mean = |v: &Vec<f32>| v.iter().sum::<f32>() / v.len() as f32;
    eprintln!(
        "  {label}: same(n={}) mean={:.4} p50={:.4} p90={:.4} | diff(n={}) mean={:.4} p50={:.4} p90={:.4}",
        same.len(), mean(&same), p(&same, 0.5), p(&same, 0.9),
        diff.len(), mean(&diff), p(&diff, 0.5), p(&diff, 0.9)
    );
}

fn max_f1(embs: &[Vec<f32>], ids: &[&str], label: &str) {
    let n = embs.len();
    let mut pairs: Vec<(f32, bool)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((dot(&embs[i], &embs[j]), ids[i] == ids[j]));
        }
    }
    let mut best = (0.0f32, 0.0f32);
    for t100 in (30..=90).step_by(1) {
        let tau = t100 as f32 / 100.0;
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fnc = 0usize;
        for &(s, same) in &pairs {
            if same {
                if s >= tau {
                    tp += 1;
                } else {
                    fnc += 1;
                }
            } else if s >= tau {
                fp += 1;
            }
        }
        let prec = if tp + fp == 0 { 0.0 } else { tp as f32 / (tp + fp) as f32 };
        let rec = if tp + fnc == 0 { 0.0 } else { tp as f32 / (tp + fnc) as f32 };
        let f1 = if prec + rec == 0.0 { 0.0 } else { 2.0 * prec * rec / (prec + rec) };
        if f1 > best.1 {
            best = (tau, f1);
        }
    }
    eprintln!("  {label}: max-F1 = {:.4} @tau = {:.2}", best.1, best.0);
}

#[test]
#[ignore]
fn lfw_path_verify() {
    let rt = tokio::runtime::Runtime::new().expect("tokio");
    let scrfd = ScrfdDetector::load(Path::new(SCRFD)).expect("scrfd");
    let aligner = SimpleAligner::new();
    let e1 = ArcFaceEmbedder::load(Path::new(ARCFACE)).expect("arcface"); // 默认 1-crop
    let qf = QualityFilter::from_config(0.5, 20, 0.5, 90.0);

    // ============ Part A: stock 交叉验证 ============
    eprintln!("=== Part A: lfw_eval 路径跑 stock 10 张已知样本 ===");
    let samples = read_tsv();
    let mut a_embs: Vec<Vec<f32>> = Vec::new();
    let mut a_ids: Vec<&str> = Vec::new();
    let mut sum_cos = 0.0f32;
    let mut min_cos = f32::MAX;
    for s in &samples {
        let img = ImageData::from_file(Path::new(&s.path)).expect("open");
        let (emb, det_bbox, det_kps) = match extract_best(&rt, &scrfd, &aligner, &e1, &qf, &img) {
            Some(v) => v,
            None => {
                eprintln!("  img{}: lfw_eval 路径无 face 通过过滤!", s.img);
                continue;
            }
        };
        let c = dot(&emb, &s.db_emb);
        sum_cos += c;
        min_cos = min_cos.min(c);
        // 检测 bbox/kps vs 存储值
        let db = &s.bbox;
        let kp_err = (0..5)
            .map(|k| {
                let dx = det_kps[k].0 - s.kps[k].0;
                let dy = det_kps[k].1 - s.kps[k].1;
                (dx * dx + dy * dy).sqrt()
            })
            .fold(0.0f32, f32::max);
        eprintln!(
            "  img{}: cos(实时路径,DB)={:.4} | 检测bbox=({:.0},{:.0},{:.0},{:.0}) vs 存储=({:.0},{:.0},{:.0},{:.0}) | max_kp_err={:.1}px",
            s.img, c, det_bbox.x, det_bbox.y, det_bbox.w, det_bbox.h,
            db.x, db.y, db.w, db.h, kp_err
        );
        a_embs.push(l2(&emb));
        a_ids.push(if s.img == 6 || s.img == 13 || s.img == 14 || s.img == 15 { "jaqor" } else { "other" });
    }
    let n = a_embs.len();
    eprintln!(
        "  >> 实时路径 vs DB embedding: cos mean={:.4} min={:.4} (tta_img15 stored 路径为 0.99+)",
        sum_cos / n as f32, min_cos
    );
    eprintln!("  -- 实时路径 1-crop 矩阵（验证 jaqor 0.85+ 是否复现）:");
    for (i, s) in samples.iter().enumerate() {
        if i >= a_embs.len() {
            break;
        }
        let row: Vec<String> = samples
            .iter()
            .enumerate()
            .map(|(j, t)| {
                if i == j {
                    "1.000".to_string()
                } else {
                    format!("{:.3}", dot(&a_embs[i], &a_embs[j]))
                }
            })
            .collect();
        eprintln!("  img{:>2} {}", s.img, row.join(" "));
    }

    // ============ Part B: LFW 子集分布 ============
    eprintln!("\n=== Part B: LFW 子集（同一路径）===");
    let mut people: Vec<(String, Vec<PathBuf>)> = Vec::new();
    for entry in std::fs::read_dir(LFW_ROOT).expect("lfw root") {
        let entry = entry.expect("entry");
        if !entry.path().is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let mut jpgs: Vec<PathBuf> = std::fs::read_dir(entry.path())
            .expect("dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|e| e == "jpg").unwrap_or(false))
            .collect();
        jpgs.sort();
        if jpgs.len() >= 8 {
            jpgs.truncate(8);
            people.push((name, jpgs));
        }
        if people.len() >= 6 {
            break;
        }
    }
    let mut lfw_embs: Vec<Vec<f32>> = Vec::new();
    let mut lfw_ids: Vec<&str> = Vec::new();
    let mut proc = 0usize;
    let mut ok = 0usize;
    for (name, paths) in &people {
        for p in paths {
            proc += 1;
            let bytes = std::fs::read(p).expect("read");
            let img = ImageData::from_bytes(&bytes).expect("decode");
            if let Some((emb, _, _)) = extract_best(&rt, &scrfd, &aligner, &e1, &qf, &img) {
                ok += 1;
                lfw_embs.push(l2(&emb));
                lfw_ids.push(name);
            }
        }
    }
    eprintln!("  LFW 子集: {} people, {} images processed, {} faces extracted (rate {:.3})", people.len(), proc, ok, ok as f32 / proc as f32);
    if ok < 30 {
        eprintln!("  !! 提取率过低，检查 SCRFD 是否适合 250×250 图");
    }
    cosine_dist(&lfw_embs, &lfw_ids, "LFW same/diff");
    max_f1(&lfw_embs, &lfw_ids, "LFW max-F1");

    eprintln!("\n=== DONE ===");
}
