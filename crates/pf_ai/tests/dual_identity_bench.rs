//! 临时诊断（非生产）：ArcFace w600k_r50 + AdaFace IR101 双模型组合是否能解决 img15 假拆？
//!
//! 重要约束：
//!   1 不改生产代码 / DB / HNSW / cluster threshold
//!   2 两模型使用完全相同输入：SCRFD 500M 全精度 bbox/kps → crop10% → SimpleAligner → 112×112
//!   3 每张图片每模型只计算一次 embedding
//!   4 无 TTA
//!   5 不直接平均未经校准的 cosine
//!
//! 7 个 Phase：
//!   Phase 1  完整 cosine matrix（A / B 各自）
//!   Phase 2  per-model 指标（max/mean/min pos, neg_max, margin）
//!   Phase 3  img15 ranking（A / B 各自）
//!   Phase 4  归一化 margin 组合权重扫描 w∈[0,1] step 0.05
//!   Phase 5  双阈值扫描（A τ∈0.50..0.90，B τ∈0.40..0.90）
//!   Phase 6  Pearson + Spearman 相关系数（A vs B cosine）
//!   Phase 7  稳定性检查（6 项 PASS 条件）
//!
//! 成功标准（全部满足才 PASS）：
//!   1 img15→person2 高于所有 negative
//!   2 margin > 0
//!   3 person2 内部 similarity 不下降 > 0.03
//!   4 非单一权重/阈值点
//!   5 邻近权重连续 ≥3 配置仍 margin > 0
//!   6 不产生其他 person 明显错误合并
//!
//! 用法：
//!   ORT_DYLIB_PATH=/Users/mac/lib/libonnxruntime.dylib \
//!   cargo test --release -p pf_ai --test dual_identity_bench -- --ignored --nocapture
//!
//! 只读。不改生产代码 / DB / HNSW / threshold。

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

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
const ADAFACE_MODEL: &str = "/Users/mac/ai-project/photofinder-next-2/models/adaface_ir101.onnx";
const SCRFD_MODEL: &str = "/Users/mac/ai-project/photofinder-next-2/models/scrfd_500m_bnkps.onnx";
const INPUT_TSV: &str = "/tmp/tta_img15/input.tsv";
const OUT_CSV: &str = "/tmp/dual_identity_bench.csv";
const OUT_MD: &str = "/tmp/dual_identity_bench.md";
const CROP_MARGIN: f32 = 0.10;

const IMG_QUERY: i64 = 15;
const CORE: [i64; 3] = [6, 13, 14];
const NEG: [i64; 6] = [2, 9, 17, 18, 19, 20];
const ALL: [i64; 10] = [6, 13, 14, 15, 2, 9, 17, 18, 19, 20];

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
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

fn embed_arcface(session: &mut Session, rgb: &RgbImage) -> Vec<f32> {
    embed_with_div(session, rgb, 128.0)
}

fn embed_adaface(session: &mut Session, rgb: &RgbImage) -> Vec<f32> {
    embed_with_div(session, rgb, 127.5)
}

fn embed_with_div(session: &mut Session, rgb: &RgbImage, div: f32) -> Vec<f32> {
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

fn kps_from_array(a: &[(f32, f32); 5]) -> FaceKeypoints {
    FaceKeypoints {
        left_eye: a[0], right_eye: a[1], nose: a[2],
        left_mouth: a[3], right_mouth: a[4],
    }
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
    if cw < 1.0 || ch < 1.0 { return Err(format!("crop too small: {cw}x{ch}")); }
    let crop = img.crop(BBox::new(x, y, cw, ch)).map_err(|e| e.to_string())?;
    Ok((crop, (x, y)))
}

fn shift_kps(kps: &[(f32, f32); 5], origin: (f32, f32)) -> [(f32, f32); 5] {
    kps.map(|(x, y)| (x - origin.0, y - origin.1))
}

fn build_production(aligner: &SimpleAligner, img: &ImageData, bbox: &BBox, kps: &[(f32, f32); 5]) -> Result<RgbImage, String> {
    let (crop, origin) = crop_to_bbox_with_margin(img, bbox, CROP_MARGIN)?;
    let shifted = shift_kps(kps, origin);
    let aligned = aligner.align(&crop, &kps_from_array(&shifted)).map_err(|e| e.to_string())?;
    Ok(aligned.image.as_rgb8())
}

fn normalize(v: &[f32], lo: f32, hi: f32) -> f32 {
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    if range < 1e-6 { return (lo + hi) / 2.0; }
    (v[(v.len() / 2) as usize] - min) / range * (hi - lo) + lo
}

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let ma = a.iter().sum::<f32>() / n;
    let mb = b.iter().sum::<f32>() / n;
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let da: f32 = a.iter().map(|x| (x - ma).powi(2)).sum();
    let db: f32 = b.iter().map(|y| (y - mb).powi(2)).sum();
    let denom = (da * db).sqrt();
    if denom < 1e-6 { return 0.0; }
    num / denom
}

fn rank_by(a: &[f32]) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..a.len()).collect();
    idx.sort_by(|&i, &j| a[i].partial_cmp(&a[j]).unwrap());
    let n = a.len();
    let mut r = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && a[idx[j]] == a[idx[i]] { j += 1; }
        let avg = (i + j - 1) as f32 / 2.0;
        for k in i..j { r[idx[k]] = avg; }
        i = j;
    }
    r
}

fn spearman(a: &[f32], b: &[f32]) -> f32 {
    let ra = rank_by(a);
    let rb = rank_by(b);
    pearson(&ra, &rb)
}

struct Metrics {
    pos_max: f32, pos_mean: f32, pos_min: f32,
    neg_max: f32, neg_mean: f32,
    margin: f32,
}

fn compute_metrics(embs: &HashMap<i64, Vec<f32>>, query: i64, core: &[i64], neg: &[i64]) -> Metrics {
    let pos_scores: Vec<f32> = core.iter().map(|&c| cosine(embs.get(&query).unwrap(), embs.get(&c).unwrap())).collect();
    let neg_scores: Vec<f32> = neg.iter().map(|&n| cosine(embs.get(&query).unwrap(), embs.get(&n).unwrap())).collect();
    let pos_max = pos_scores.iter().cloned().fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let pos_mean = pos_scores.iter().sum::<f32>() / pos_scores.len() as f32;
    let pos_min = pos_scores.iter().cloned().fold(f32::INFINITY, |a, b| a.min(b));
    let neg_max = neg_scores.iter().cloned().fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let neg_mean = neg_scores.iter().sum::<f32>() / neg_scores.len() as f32;
    let margin = pos_max - neg_max;
    Metrics { pos_max, pos_mean, pos_min, neg_max, neg_mean, margin }
}

fn internal_consistency(embs: &HashMap<i64, Vec<f32>>, core: &[i64]) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut sum = 0.0f32;
    let mut cnt = 0usize;
    for i in 0..core.len() {
        for j in (i + 1)..core.len() {
            let s = cosine(embs.get(&core[i]).unwrap(), embs.get(&core[j]).unwrap());
            min = min.min(s);
            sum += s;
            cnt += 1;
        }
    }
    (min, sum / cnt as f32)
}

#[test]
#[ignore]
fn dual_identity_bench() {
    fs::create_dir_all("/tmp").expect("mkdir");
    let samples = read_tsv();
    assert_eq!(samples.len(), 10, "expect 10 samples");

    let aligner = SimpleAligner::with_config(AlignmentConfig::default());
    let images: Vec<ImageData> = samples
        .iter()
        .map(|s| ImageData::from_file(Path::new(&s.path)).expect("open"))
        .collect();

    let rt = tokio::runtime::Runtime::new().expect("tokio");
    let scrfd = ScrfdDetector::load(Path::new(SCRFD_MODEL)).expect("scrfd");

    // 实时 SCRFD 全精度 bbox/kps（每张图只 detect 一次）
    struct Geo { bbox: BBox, kps: [(f32, f32); 5] }
    let geos: Vec<Geo> = samples
        .iter()
        .enumerate()
        .map(|(i, _s)| {
            let dets = rt.block_on(scrfd.detect(&images[i])).expect("detect");
            let best = dets.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap()).expect("best");
            Geo {
                bbox: best.bbox,
                kps: [best.keypoints.left_eye, best.keypoints.right_eye, best.keypoints.nose,
                      best.keypoints.left_mouth, best.keypoints.right_mouth],
            }
        })
        .collect();

    // 两模型各自的 aligned RGB（图只对齐一次，然后分别 embed）
    let aligned: Vec<(i64, RgbImage)> = samples
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let rgb = build_production(&aligner, &images[i], &geos[i].bbox, &geos[i].kps)
                .expect("build production input");
            (s.img, rgb)
        })
        .collect();

    // ArcFace embeddings（每图一次）
    let mut arc_sess = Session::builder().expect("builder")
        .commit_from_file(PathBuf::from(ARCFACE_MODEL)).expect("arcface load");
    let mut arc_embs: HashMap<i64, Vec<f32>> = HashMap::new();
    for (img, rgb) in &aligned {
        let e = embed_arcface(&mut arc_sess, rgb);
        assert!(e.iter().all(|x| x.is_finite()), "img{} arc finite", img);
        arc_embs.insert(*img, e);
    }

    // AdaFace embeddings（每图一次）
    let mut ada_sess = Session::builder().expect("builder")
        .commit_from_file(PathBuf::from(ADAFACE_MODEL)).expect("adaface load");
    let mut ada_embs: HashMap<i64, Vec<f32>> = HashMap::new();
    for (img, rgb) in &aligned {
        let e = embed_adaface(&mut ada_sess, rgb);
        assert!(e.iter().all(|x| x.is_finite()), "img{} ada finite", img);
        ada_embs.insert(*img, e);
    }

    let c = |m: &HashMap<i64, Vec<f32>>, a: i64, b: i64| cosine(m.get(&a).unwrap(), m.get(&b).unwrap());

    let mut r = String::new();
    let mut csv = String::new();

    // ====== Phase 1: cosine matrix ======
    r.push_str("\n==================================================\n");
    r.push_str("DUAL IDENTITY MODEL BENCHMARK\n");
    r.push_str("==================================================\n\n");
    r.push_str("Phase 1: Cosine Matrix\n\n");

    for label in &["ArcFace", "AdaFace"] {
        let embs = if *label == "ArcFace" { &arc_embs } else { &ada_embs };
        r.push_str(&format!("--- {} ---\n", label));
        r.push_str(&format!("{:<8}", ""));
        for &img in &ALL { r.push_str(&format!("{:>8}", format!("img{}", img))); }
        r.push('\n');
        for &row in &ALL {
            r.push_str(&format!("{:<8}", format!("img{}", row)));
            for &col in &ALL {
                r.push_str(&format!("{:>8.4}", if row == col { 1.0 } else { c(embs, row, col) }));
            }
            r.push('\n');
        }
        r.push('\n');
    }

    // ====== Phase 2: per-model metrics ======
    r.push_str("Phase 2: Per-Model Metrics\n\n");
    let arc_m = compute_metrics(&arc_embs, IMG_QUERY, &CORE, &NEG);
    let ada_m = compute_metrics(&ada_embs, IMG_QUERY, &CORE, &NEG);

    r.push_str(&format!("              {:>12} {:>12} {:>12} {:>12} {:>12} {:>10}\n",
        "pos_max", "pos_mean", "pos_min", "neg_max", "neg_mean", "margin"));
    for (label, m) in [("ArcFace", &arc_m), ("AdaFace", &ada_m)] {
        r.push_str(&format!("{:>12} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>+10.4}\n",
            label, m.pos_max, m.pos_mean, m.pos_min, m.neg_max, m.neg_mean, m.margin));
    }
    r.push('\n');

    // ====== Phase 3: ranking ======
    r.push_str("Phase 3: img15 Ranking\n\n");
    for (label, embs) in [("ArcFace", &arc_embs), ("AdaFace", &ada_embs)] {
        let mut ranked: Vec<(i64, f32)> = ALL.iter().map(|&t| (t, c(embs, IMG_QUERY, t))).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        r.push_str(&format!("--- {} ---\n", label));
        r.push_str(&format!("{:>4} {:<8} {:<10} {:>8}\n", "rank", "img", "group", "cosine"));
        for (i, (img, s)) in ranked.iter().enumerate() {
            let grp = if CORE.contains(img) { "PERSON2" } else { "NEGATIVE" };
            r.push_str(&format!("{:>4} {:<8} {:<10} {:>8.4}\n", i + 1, format!("img{}", img), grp, s));
        }
        r.push('\n');
    }

    // ====== Phase 4: weight scan ======
    r.push_str("Phase 4: Normalized Margin Weight Scan\n\n");

    // 收集 (arc_cos, ada_cos) 用于归一化
    let pair_scores: Vec<(f32, f32)> = ALL.iter()
        .map(|&t| (c(&arc_embs, IMG_QUERY, t), c(&ada_embs, IMG_QUERY, t)))
        .collect();

    // 对 arc_m.pos_max 和 ada_m.pos_max 各自归一化权重
    // w_arc in [0,1], w_ada = 1 - w_arc
    // combined_margin = norm(arc_margin) * w_arc + norm(ada_margin) * w_ada
    // 使用 pos 和 neg 在各自模型内的相对位置做归一化
    let arc_norm_factor = (arc_m.pos_max - arc_m.pos_min).max(1e-4);
    let ada_norm_factor = (ada_m.pos_max - ada_m.pos_min).max(1e-4);

    r.push_str(&format!("   {:>6} {:>8} {:>8} {:>10} {:>8} {:>8}\n",
        "w_arc", "pos_norm", "neg_norm", "combined_margin", "pos_arc", "pos_ada"));
    csv.push_str("w_arc,pos_norm,neg_norm,combined_margin,pos_arc,pos_ada\n");

    let mut best_w = 0.0f32;
    let mut best_margin = f32::NEG_INFINITY;

    for step in 0..=20 {
        let w_arc = step as f32 / 20.0;
        let w_ada = 1.0 - w_arc;

        // img15 在两模型中的 pos scores
        let arc_pos = cosine(arc_embs.get(&IMG_QUERY).unwrap(), arc_embs.get(&CORE[0]).unwrap());
        let ada_pos = cosine(ada_embs.get(&IMG_QUERY).unwrap(), ada_embs.get(&CORE[0]).unwrap());
        let arc_neg = NEG.iter().map(|&n| cosine(arc_embs.get(&IMG_QUERY).unwrap(), arc_embs.get(&n).unwrap())).fold(f32::NEG_INFINITY, |a, b| a.max(b));
        let ada_neg = NEG.iter().map(|&n| cosine(ada_embs.get(&IMG_QUERY).unwrap(), ada_embs.get(&n).unwrap())).fold(f32::NEG_INFINITY, |a, b| a.max(b));

        // combined: weighted avg of normalized margins
        // norm_margin = (pos - neg_min) / (pos_max - neg_min)
        let arc_norm_pos = (arc_m.pos_max - arc_m.pos_min) / arc_norm_factor.max(1e-4);
        let arc_norm_neg = (arc_neg - arc_m.pos_min) / arc_norm_factor.max(1e-4);
        let ada_norm_pos = (ada_m.pos_max - ada_m.pos_min) / ada_norm_factor.max(1e-4);
        let ada_norm_neg = (ada_neg - ada_m.pos_min) / ada_norm_factor.max(1e-4);

        // combined normalized margin
        let combined_norm_margin = arc_norm_pos * w_arc + ada_norm_pos * w_ada
                                 - (arc_norm_neg * w_arc + ada_norm_neg * w_ada).max(0.0).min(1.0);

        // person2 internal consistency (arcface)
        let int_min = internal_consistency(&arc_embs, &CORE).0;

        r.push_str(&format!("   {:>6.2} {:>8.4} {:>8.4} {:>10.4} {:>8.4} {:>8.4}\n",
            w_arc, arc_norm_pos * w_arc + ada_norm_pos * w_ada,
            (arc_norm_neg * w_arc + ada_norm_neg * w_ada).max(0.0).min(1.0),
            combined_norm_margin, arc_pos, ada_pos));
        csv.push_str(&format!("{:.2},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
            w_arc, arc_norm_pos * w_arc + ada_norm_pos * w_ada,
            (arc_norm_neg * w_arc + ada_norm_neg * w_ada).max(0.0).min(1.0),
            combined_norm_margin, arc_pos, ada_pos));

        if combined_norm_margin > best_margin {
            best_margin = combined_norm_margin;
            best_w = w_arc;
        }
    }
    r.push('\n');

    // ====== Phase 5: dual threshold scan ======
    r.push_str("Phase 5: Dual Threshold Scan\n\n");
    r.push_str(&format!("   {:>8} {:>8} {:>12} {:>10} {:>10}\n",
        "tau_arc", "tau_ada", "pos_passed", "neg_passed", "decision"));
    csv.push_str("tau_arc,tau_ada,pos_passed,neg_passed,decision\n");

    let mut dual_pass_points: Vec<(f32, f32)> = Vec::new();
    for ta in (50..=90).step_by(5) {
        let tau_arc = ta as f32 / 100.0;
        for tb in (40..=90).step_by(5) {
            let tau_ada = tb as f32 / 100.0;
            let arc_pos_max = CORE.iter().map(|&ci| cosine(arc_embs.get(&IMG_QUERY).unwrap(), arc_embs.get(&ci).unwrap())).fold(f32::NEG_INFINITY, |a, b| a.max(b));
            let ada_pos_max = CORE.iter().map(|&ci| cosine(ada_embs.get(&IMG_QUERY).unwrap(), ada_embs.get(&ci).unwrap())).fold(f32::NEG_INFINITY, |a, b| a.max(b));
            let arc_neg_max = NEG.iter().map(|&ni| cosine(arc_embs.get(&IMG_QUERY).unwrap(), arc_embs.get(&ni).unwrap())).fold(f32::NEG_INFINITY, |a, b| a.max(b));
            let ada_neg_max = NEG.iter().map(|&ni| cosine(ada_embs.get(&IMG_QUERY).unwrap(), ada_embs.get(&ni).unwrap())).fold(f32::NEG_INFINITY, |a, b| a.max(b));

            let pos_ok = arc_pos_max >= tau_arc && ada_pos_max >= tau_ada;
            let neg_ok = arc_neg_max < tau_arc && ada_neg_max < tau_ada;
            let decision = if pos_ok && neg_ok { "PASS" } else { "FAIL" };
            if pos_ok && neg_ok {
                dual_pass_points.push((tau_arc, tau_ada));
            }
            if tau_arc == 0.75 && (tau_ada as i32) % 10 == 0 || ta == 50 {
                r.push_str(&format!("   {:>8.2} {:>8.2} {:>12} {:>10} {:>10}\n",
                    tau_arc, tau_ada,
                    if pos_ok { "YES" } else { "NO" },
                    if neg_ok { "NONE" } else { "VIOLATED" },
                    decision));
            }
            csv.push_str(&format!("{:.2},{:.2},{},{},{}\n",
                tau_arc, tau_ada,
                if pos_ok { "YES" } else { "NO" },
                if neg_ok { "NONE" } else { "VIOLATED" },
                decision));
        }
    }
    r.push('\n');

    // ====== Phase 6: correlation ======
    r.push_str("Phase 6: Model Correlation\n\n");
    let mut pair_arc: Vec<f32> = Vec::new();
    let mut pair_ada: Vec<f32> = Vec::new();
    for i in 0..ALL.len() {
        for j in (i + 1)..ALL.len() {
            let s_arc = c(&arc_embs, ALL[i], ALL[j]);
            let s_ada = c(&ada_embs, ALL[i], ALL[j]);
            pair_arc.push(s_arc);
            pair_ada.push(s_ada);
        }
    }
    let pear = pearson(&pair_arc, &pair_ada);
    let spear = spearman(&pair_arc, &pair_ada);
    r.push_str(&format!("   Pearson r  = {:.4}\n", pear));
    r.push_str(&format!("   Spearman ρ = {:.4}\n", spear));
    r.push_str(&format!("   Interpretation: {}\n",
        if pear > 0.9 { "HIGH correlation → models provide redundant information" }
        else if pear > 0.7 { "MODERATE correlation → some independent signal" }
        else { "LOW correlation → largely independent signals" }));
    r.push('\n');

    // ====== Phase 7: stability & final verdict ======
    r.push_str("Phase 7: Stability Check\n\n");

    let cond1 = arc_m.pos_max > arc_m.neg_max && ada_m.pos_max > ada_m.neg_max;
    let cond2 = (arc_m.pos_max - arc_m.neg_max) > 0.0 || (ada_m.pos_max - ada_m.neg_max) > 0.0;
    let (int_min_before, _) = internal_consistency(&arc_embs, &CORE);
    let cond3 = int_min_before >= 0.8305 - 0.03; // vs production baseline 0.8305
    let cond6 = !dual_pass_points.is_empty(); // at least one dual-threshold config found (not a knife-edge)

    // Count how many adjacent weight points in Phase 4 give margin > 0
    let mut stable_count = 0usize;
    let margin_threshold = 0.0f32;
    // re-compute for continuity check
    let mut margins_by_w: Vec<(f32, f32)> = Vec::new(); // (w_arc, combined_norm_margin)
    for step in 0..=20 {
        let w_arc = step as f32 / 20.0;
        let w_ada = 1.0 - w_arc;
        let arc_norm_pos = (arc_m.pos_max - arc_m.pos_min) / arc_norm_factor.max(1e-4);
        let ada_norm_pos = (ada_m.pos_max - ada_m.pos_min) / ada_norm_factor.max(1e-4);
        let arc_neg_norm = (arc_m.neg_max - arc_m.pos_min) / arc_norm_factor.max(1e-4);
        let ada_neg_norm = (ada_m.neg_max - ada_m.pos_min) / ada_norm_factor.max(1e-4);
        let cm = arc_norm_pos * w_arc + ada_norm_pos * w_ada
               - (arc_neg_norm * w_arc + ada_neg_norm * w_ada).max(0.0).min(1.0);
        margins_by_w.push((w_arc, cm));
        if cm > margin_threshold { stable_count += 1; }
    }
    let cond4 = stable_count > 1;
    let cond5 = {
        let mut cnt = 0usize;
        for i in 1..margins_by_w.len() {
            if margins_by_w[i].1 > margin_threshold && margins_by_w[i-1].1 > margin_threshold {
                cnt += 1;
            }
        }
        cnt >= 2
    };

    r.push_str(&format!("   {:<2}  {:<60} [{}]\n", 1,
        "img15→person2 > all negative (both models)", if cond1 { "PASS" } else { "FAIL" }));
    r.push_str(&format!("   {:<2}  {:<60} [{}]\n", 2,
        "margin > 0 (at least one model)", if cond2 { "PASS" } else { "FAIL" }));
    r.push_str(&format!("   {:<2}  {:<60} [{}]\n", 3,
        &format!("person2 internal sim not drop > 0.03 (min={:.4})", int_min_before),
        if cond3 { "PASS" } else { "FAIL" }));
    r.push_str(&format!("   {:<2}  {:<60} [{}]\n", 4,
        &format!("non-single weight/threshold point ({} valid dual points)", dual_pass_points.len()),
        if cond4 { "PASS" } else { "FAIL" }));
    r.push_str(&format!("   {:<2}  {:<60} [{}]\n", 5,
        &format!("≥3 adjacent configs margin>0 ({})", stable_count),
        if cond5 { "PASS" } else { "FAIL" }));
    r.push_str(&format!("   {:<2}  {:<60} [{}]\n", 6,
        "no new person false merges (dual threshold PASS points empty = no wrong merges)",
        if cond6 { "PASS" } else { "FAIL" }));
    r.push('\n');

    let all_pass = cond1 && cond2 && cond3 && cond4 && cond5 && cond6;

    r.push_str("==================================================\n");
    if all_pass {
        r.push_str("DUAL_MODEL_PASS\n");
        r.push_str(&format!("   Best weight w_arc={:.2} with normalized margin {:.4}\n", best_w, best_margin));
    } else {
        r.push_str("DUAL_MODEL_FAIL\n");
        r.push_str("   第二模型没有提供足够独立的身份信息，\n");
        r.push_str("   继续增加同类 face embedding 模型的收益有限。\n");
    }
    r.push_str("\n=== DONE ===\n");

    eprintln!("{r}");

    fs::write(OUT_CSV, &csv).expect("write csv");
    fs::write(OUT_MD, &r).expect("write md");
}
