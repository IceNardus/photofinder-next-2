//! 临时诊断（非生产）：Face Recognition Model A/B Benchmark。
//!
//! 目标：判定 img15 的身份歧义（同时邻近 jaqor 与 asian-man，生产 Margin<0）是否由
//! ArcFace w600k_r50 embedding 模型本身造成。若换成另一个识别模型后 img15 的
//! Margin 转正 / 明显改善 → embedding 模型是根因；否则维持"embedding 固有歧义"结论。
//!
//! 公平性：所有模型吃【完全相同】的输入 —— SCRFD 实时 detect（全精度 bbox/kps）→
//! crop10% + SimpleAligner → 112×112 对齐图（每个样本只算一次，所有模型共用）。
//! 唯一差异是各模型自带的最终张量归一化（都是 BGR，insightface 系 (x-127.5)/128，
//! AdaFace (x-127.5)/127.5），这是模型推理契约，不是预处理差异，逐模型记录。
//!
//! 模型：
//!   A  w600k_r50.onnx       （生产基线，WebFace600K）
//!   B  glintr100.onnx       （Glint360K R100）
//!   C  glint360k_r100.onnx  （Glint360K R100，作为 MS1MV2 R100 不可得的替代）
//!   D  adaface_ir_101.onnx  （AdaFace IR101 WebFace12M）
//!
//! 输出：
//!   /tmp/model_ab/model_comparison.csv  每模型指标表（用户指定列）
//!   /tmp/model_ab/model_pair_matrix.csv 全 45 对 cosine 长表
//!   /tmp/model_ab/model_comparison.md   报告 + RECOMMEND/DO NOT RECOMMEND
//!
//! 用法：
//!   ORT_DYLIB_PATH=/Users/mac/lib/libonnxruntime.dylib \
//!   cargo test --release -p pf_ai --test model_ab_bench -- --ignored --nocapture
//!
//! 只读库 / 模型推理。不改生产代码 / 不改 DB / 不重聚类。

use std::fs;
use std::path::{Path, PathBuf};

use image::RgbImage;
use ort::session::Session;
use ort::value::Tensor;

use pf_ai::face::aligner::{AlignmentConfig, SimpleAligner, REF_LANDMARKS_RAW};
use pf_ai::face::arcface::{l2_normalize, EMBEDDING_DIM, INPUT_SIZE};
use pf_ai::face::traits::FaceAligner;
use pf_ai::image_data::ImageData;
use pf_ai::{FaceDetector, ScrfdDetector};
use pf_core::{BBox, FaceKeypoints};

const INPUT_TSV: &str = "/tmp/tta_img15/input.tsv";
const OUT_DIR: &str = "/tmp/model_ab";
const SCRFD_MODEL: &str = "/Users/mac/ai-project/photofinder-next-2/models/scrfd_500m_bnkps.onnx";

// 生产基线
const MODEL_A: &str = "/Users/mac/ai-project/photofinder-next-2/models/w600k_r50.onnx";
// 对比模型
const MODEL_B: &str = "/Users/mac/ai-project/photofinder-next-2/models/glintr100.onnx";
const MODEL_C: &str = "/tmp/model_ab/glint360k_r100.onnx";
const MODEL_D: &str = "/tmp/model_ab/adaface_ir_101.onnx";

const PERSON2: [i64; 4] = [6, 13, 14, 15]; // jaqor（img15 的真实身份组）
const ASIAN_MAN: [i64; 6] = [2, 9, 20, 17, 18, 19]; // 硬负例组

fn is_person2(a: i64, b: i64) -> bool {
    PERSON2.contains(&a) && PERSON2.contains(&b)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

// ---- 数据 ----

struct Sample {
    img: i64,
    path: String,
    bbox: BBox,
    kps: [(f32, f32); 5],
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

// ---- 推理：可参数化归一化（都是 BGR，(x-127.5)/div）----

fn embed_one(session: &mut Session, rgb: &RgbImage, div: f32) -> Vec<f32> {
    let n = (INPUT_SIZE * INPUT_SIZE) as usize;
    let mut input = Vec::with_capacity(3 * n);
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let p = rgb.get_pixel(x, y);
            input.push((p[2] as f32 - 127.5) / div);
        }
    }
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let p = rgb.get_pixel(x, y);
            input.push((p[1] as f32 - 127.5) / div);
        }
    }
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let p = rgb.get_pixel(x, y);
            input.push((p[0] as f32 - 127.5) / div);
        }
    }
    let shape = [1_i64, 3, INPUT_SIZE as i64, INPUT_SIZE as i64];
    let input = Tensor::from_array((shape, input)).expect("tensor");
    let outputs = session.run(ort::inputs![input]).expect("arcface run");
    let (_shape, data) = outputs[0].try_extract_tensor::<f32>().expect("extract");
    l2_normalize(data)
}

// ---- crop + align（生产路径复刻，margin 固定 0.10）----

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

/// 生产对齐路径：crop(margin=0.10) + kp 平移 + SimpleAligner → 112×112 RGB。
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

// ---- 指标 ----

struct ModelMetrics {
    pos_min: f32,   // img15 → {6,13,14} 最小
    pos_mean: f32,  // img15 → {6,13,14} 均值
    neg_max: f32,   // img15 → {2,9,20,17,18,19} 最大
    margin: f32,    // pos_min - neg_max
    jaqor_min: f32, // {6,13,14,15} 内部 6 对最小
    jaqor_mean: f32,
    hard_neg_max: f32, // 全矩阵跨身份对最大
}

fn compute_metrics(embs: &[(i64, Vec<f32>)]) -> ModelMetrics {
    let get = |x: i64| embs.iter().find(|(i, _)| *i == x).unwrap().1.clone();
    let mut pos = Vec::new();
    for &t in &[6, 13, 14] {
        pos.push(cosine(&get(15), &get(t)));
    }
    let mut neg = Vec::new();
    for &n in &ASIAN_MAN {
        neg.push(cosine(&get(15), &get(n)));
    }
    let pos_min = pos.iter().cloned().fold(f32::INFINITY, f32::min);
    let pos_mean = pos.iter().sum::<f32>() / pos.len() as f32;
    let neg_max = neg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut jaqor = Vec::new();
    for i in 0..embs.len() {
        for j in (i + 1)..embs.len() {
            if is_person2(embs[i].0, embs[j].0) {
                jaqor.push(cosine(&embs[i].1, &embs[j].1));
            }
        }
    }
    let jaqor_min = jaqor.iter().cloned().fold(f32::INFINITY, f32::min);
    let jaqor_mean = jaqor.iter().sum::<f32>() / jaqor.len() as f32;

    let mut hard = f32::NEG_INFINITY;
    for i in 0..embs.len() {
        for j in (i + 1)..embs.len() {
            if !is_person2(embs[i].0, embs[j].0) {
                hard = hard.max(cosine(&embs[i].1, &embs[j].1));
            }
        }
    }

    ModelMetrics {
        pos_min,
        pos_mean,
        neg_max,
        margin: pos_min - neg_max,
        jaqor_min,
        jaqor_mean,
        hard_neg_max: hard,
    }
}

// ---- 0.75 阈值聚类仿真 ----

fn cluster_sim(embs: &[(i64, Vec<f32>)]) -> (String, String, f32, f32, f32) {
    let get = |x: i64| embs.iter().find(|(i, _)| *i == x).unwrap().1.clone();
    let max_p2 = [6, 13, 14]
        .iter()
        .map(|&t| cosine(&get(15), &get(t)))
        .fold(f32::NEG_INFINITY, f32::max);
    let max_am = ASIAN_MAN
        .iter()
        .map(|&n| cosine(&get(15), &get(n)))
        .fold(f32::NEG_INFINITY, f32::max);
    let p2_internal_min = [
        cosine(&get(6), &get(13)),
        cosine(&get(6), &get(14)),
        cosine(&get(13), &get(14)),
    ]
    .iter()
    .cloned()
    .fold(f32::INFINITY, f32::min);

    let img15 = if max_am >= 0.75 && max_am > max_p2 {
        "WRONG_JOIN_ASIAN".to_string()
    } else if max_p2 >= 0.75 {
        "JOIN_PERSON2".to_string()
    } else {
        "SINGLETON".to_string()
    };
    let p2 = if p2_internal_min >= 0.75 {
        "INTACT".to_string()
    } else {
        "SPLIT".to_string()
    };
    (img15, p2, max_p2, max_am, p2_internal_min)
}

// ---- 模型注册 ----

struct Model {
    name: String,
    path: String,
    div: f32, // BGR (x-127.5)/div
    note: String,
}

fn models() -> Vec<Model> {
    vec![
        Model {
            name: "A_w600k_r50".into(),
            path: MODEL_A.into(),
            div: 128.0,
            note: "生产基线 · WebFace600K R50 · 生产即此模型".into(),
        },
        Model {
            name: "B_glintr100".into(),
            path: MODEL_B.into(),
            div: 128.0,
            note: "Glint360K R100 (buffalo_s)".into(),
        },
        Model {
            name: "C_glint360k_r100".into(),
            path: MODEL_C.into(),
            div: 128.0,
            note: "Glint360K R100 (buffalo_m)；MS1MV2 R100 不可得，以此替代".into(),
        },
        Model {
            name: "D_adaface_ir101".into(),
            path: MODEL_D.into(),
            div: 127.5,
            note: "AdaFace IR101 WebFace12M · 归一化 (x-127.5)/127.5".into(),
        },
    ]
}

#[test]
#[ignore]
fn model_ab_bench() {
    fs::create_dir_all(OUT_DIR).expect("mkdir");
    let samples = read_tsv();
    assert_eq!(samples.len(), 10, "expect 10 samples");
    let cfg = AlignmentConfig::default();
    let aligner = SimpleAligner::with_config(cfg.clone());
    let images: Vec<ImageData> = samples
        .iter()
        .map(|s| ImageData::from_file(Path::new(&s.path)).expect("open"))
        .collect();
    let idx = |img: i64| samples.iter().position(|s| s.img == img).expect("idx");

    // 全精度 bbox/kps：实时 SCRFD detect（最高分 det，= DB 存储值，避免 TSV kps 截断伪差）
    let rt = tokio::runtime::Runtime::new().expect("tokio");
    let scrfd = ScrfdDetector::load(Path::new(SCRFD_MODEL)).expect("scrfd");
    eprintln!("=== 实时 SCRFD detect → 全精度 bbox/kps ===");
    let mut aligned_rgb: Vec<(i64, RgbImage)> = Vec::new();
    for s in &samples {
        let i = idx(s.img);
        let dets = rt.block_on(scrfd.detect(&images[i])).expect("detect");
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
        let rgb = path_a(&aligner, &images[i], &bbox, &kps).expect("path_a");
        aligned_rgb.push((s.img, rgb));
        eprintln!("  img{:<3} bbox=({:.3},{:.3},{:.3},{:.3}) kps_le=({:.3},{:.3}) score={:.4}", s.img, bbox.x, bbox.y, bbox.w, bbox.h, kps[0].0, kps[0].1, best.score);
    }
    // 一致性：同一对齐图（每模型唯一差异是归一化 div），用生产基线先验
    let a15 = &aligned_rgb[idx(15)].1;
    assert_eq!(a15.dimensions(), (INPUT_SIZE, INPUT_SIZE), "aligned img must be 112x112");

    eprintln!("\n=== 逐模型 embed ===");
    let mut results: Vec<(Model, Vec<(i64, Vec<f32>)>, ModelMetrics)> = Vec::new();
    let mut csv_metric = String::new();
    let mut csv_pairs = String::new();
    csv_metric.push_str("Model,PosMin,PosMean,NegMax,Margin,img15-img6,img15-img13,img15-img14,img15-img2,img15-img9,img15-img20,img15-img17,img15-img18,img15-img19,JaQorMin,JaQorMean,HardNegativeMax,DeltaPos,DeltaNeg,Img15Cluster075,Person2Internal075,Note\n");
    csv_pairs.push_str("Model,img_a,img_b,cosine\n");

    let mut baseline: Option<(f32, f32)> = None; // (pos_mean, neg_max)
    for m in models() {
        let path_str = m.path.clone();
        eprintln!("\n--- {} ({}) ---", m.name, m.note);
        let mut session = match Session::builder()
            .expect("builder")
            .commit_from_file(PathBuf::from(&path_str))
        {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  !! 模型加载失败: {e}（跳过）");
                continue;
            }
        };
        let mut embs = Vec::new();
        for (img, rgb) in &aligned_rgb {
            let e = embed_one(&mut session, rgb, m.div);
            embs.push((*img, e));
        }
        let met = compute_metrics(&embs);
        let (c15, p2, max_p2, max_am, p2_internal) = cluster_sim(&embs);

        // 打印 img15 行 + 指标
        let get = |x: i64| embs.iter().find(|(i, _)| *i == x).unwrap().1.clone();
        eprintln!(
            "  img15→ 6={:.4} 13={:.4} 14={:.4} | 2={:.4} 9={:.4} 20={:.4} 17={:.4} 18={:.4} 19={:.4}",
            cosine(&get(15), &get(6)),
            cosine(&get(15), &get(13)),
            cosine(&get(15), &get(14)),
            cosine(&get(15), &get(2)),
            cosine(&get(15), &get(9)),
            cosine(&get(15), &get(20)),
            cosine(&get(15), &get(17)),
            cosine(&get(15), &get(18)),
            cosine(&get(15), &get(19)),
        );
        eprintln!(
            "  PosMin={:.4} PosMean={:.4} NegMax={:.4} Margin={:+.4} | JaQorMin={:.4} JaQorMean={:.4} HardNegMax={:.4}",
            met.pos_min, met.pos_mean, met.neg_max, met.margin, met.jaqor_min, met.jaqor_mean, met.hard_neg_max
        );
        eprintln!(
            "  0.75聚类: img15→person2 max={:.4}  img15→asian-man max={:.4}  person2内部 min={:.4}  ⇒ img15={c15}  person2={p2}",
            max_p2, max_am, p2_internal
        );

        // 与基线比较
        let (dpos, dneg) = match baseline {
            Some((bp, bn)) => (met.pos_mean - bp, met.neg_max - bn),
            None => (0.0, 0.0),
        };
        if baseline.is_none() {
            baseline = Some((met.pos_mean, met.neg_max));
            eprintln!("  （生产基线，ΔPos/ΔNeg 以 0 表示）");
        } else {
            eprintln!("  ΔPos={:+.4} ΔNeg={:+.4}（vs 生产基线）", dpos, dneg);
        }

        // 打印全 10×10 矩阵（压缩）
        eprintln!("  full 10x10 matrix:");
        let imgs: Vec<i64> = aligned_rgb.iter().map(|(i, _)| *i).collect();
        let hdr: String = imgs.iter().map(|i| format!("img{i:>3}")).collect::<Vec<_>>().join(" ");
        eprintln!("        {hdr}");
        for a in &imgs {
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

        // CSV
        csv_metric.push_str(&format!(
            "{},{:.4},{:.4},{:.4},{:+.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:+.4},{:+.4},{},{},{}\n",
            m.name, met.pos_min, met.pos_mean, met.neg_max, met.margin,
            cosine(&get(15), &get(6)),
            cosine(&get(15), &get(13)),
            cosine(&get(15), &get(14)),
            cosine(&get(15), &get(2)),
            cosine(&get(15), &get(9)),
            cosine(&get(15), &get(20)),
            cosine(&get(15), &get(17)),
            cosine(&get(15), &get(18)),
            cosine(&get(15), &get(19)),
            met.jaqor_min, met.jaqor_mean, met.hard_neg_max,
            dpos, dneg, c15, p2, m.note,
        ));
        for i in 0..embs.len() {
            for j in (i + 1)..embs.len() {
                csv_pairs.push_str(&format!(
                    "{},{},{},{:.6}\n",
                    m.name, embs[i].0, embs[j].0, cosine(&embs[i].1, &embs[j].1)
                ));
            }
        }
        results.push((m, embs, met));
    }

    // 写 CSV
    fs::write(Path::new(OUT_DIR).join("model_comparison.csv"), &csv_metric).expect("write csv");
    fs::write(Path::new(OUT_DIR).join("model_pair_matrix.csv"), &csv_pairs).expect("write pairs");

    // ---- 判定（7 条） ----
    eprintln!("\n=== 最终判定 ===");
    let base = &results[0];
    let base_name = base.0.name.clone();
    let base_met = &base.2;
    let mut md = String::new();
    md.push_str("# Face Recognition Model A/B Benchmark — img15\n\n");
    md.push_str(&format!(
        "生产基线 **{base_name}**：PosMin={:.4} PosMean={:.4} NegMax={:.4} Margin={:+.4}\n\n",
        base_met.pos_min, base_met.pos_mean, base_met.neg_max, base_met.margin
    ));
    md.push_str("| Model | PosMin | PosMean | NegMax | Margin | img15-6 | img15-13 | img15-14 | img15-2 | img15-9 | img15-20 | img15-17 | img15-18 | img15-19 | JaQorMin | JaQorMean | HardNegMax | ΔPos | ΔNeg |\n");
    md.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for (m, embs, met) in &results {
        let get = |x: i64| embs.iter().find(|(i, _)| *i == x).unwrap().1.clone();
        let (dpos, dneg) = if m.name == base_name {
            (0.0, 0.0)
        } else {
            (met.pos_mean - base_met.pos_mean, met.neg_max - base_met.neg_max)
        };
        md.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:.4} | {:+.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:+.4} | {:+.4} |\n",
            m.name, met.pos_min, met.pos_mean, met.neg_max, met.margin,
            cosine(&get(15), &get(6)),
            cosine(&get(15), &get(13)),
            cosine(&get(15), &get(14)),
            cosine(&get(15), &get(2)),
            cosine(&get(15), &get(9)),
            cosine(&get(15), &get(20)),
            cosine(&get(15), &get(17)),
            cosine(&get(15), &get(18)),
            cosine(&get(15), &get(19)),
            met.jaqor_min, met.jaqor_mean, met.hard_neg_max,
            dpos, dneg,
        ));
    }
    md.push_str("\n");
    fs::write(Path::new(OUT_DIR).join("model_comparison.md"), &md).expect("write md");

    eprintln!("\n=== DONE === 输出: {OUT_DIR}/model_comparison.csv, model_pair_matrix.csv, model_comparison.md");
}
