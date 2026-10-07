//! 临时诊断（非生产）：img15 是否存在 person-level evidence 归属 person2？
//!
//! 全部走生产 pipeline：SCRFD live detect → crop10% → SimpleAligner(1.25, -10, mask, histeq)
//! → w600k_r50 → L2 normalize。全部 10 张重新 embedding（禁止 DB embedding / TTA），每图只算一次。
//!
//! 9 部分：
//!   1 pairwise matrix     img15 → 全部 9 张
//!   2 positive core       P={6,13,14}: max / top2_mean / top3_mean / median / mean
//!   3 prototype           normalize(mean(6,13,14)) → cos(img15, ·)
//!   4 medoid              argmax_i mean(sim(i,j))（core 内）→ img15 → medoid
//!   5 leave-one-out       prototype 缺 6/13/14 各一次 → cos(img15, ·)
//!   6 pos-vs-neg          6 种 scoring × (positive, negative_max, margin)
//!   7 ranking             9 张按 cosine 降序，标 PERSON2 / NEGATIVE
//!   8 cluster consistency core{6,13,14} 加 img15 前后: centroid/medoid/intra/drift/sim-to-neg
//!   9 core expansion      Core1{6} → Core2{6,13} → Core3{6,13,14}，candidate img15
//!
//! 成功标准（不是 img15>0.75）：存在 person-level scoring 满足
//!   1 img15→person2 高于所有 negative  2 加 img13/14 evidence 不下降
//!   3 不污染 person2 core  4 不导致 6/13/14 被重新错误聚类
//!   5 margin 在不同 core 子集下保持正值  6 非单样本 knife-edge
//! 若所有 person-level 方法均无法形成稳定正 margin → 明确报告需引入第二种身份特征。
//!
//! 用法：
//!   ORT_DYLIB_PATH=/Users/mac/lib/libonnxruntime.dylib \
//!   cargo test --release -p pf_ai --test img15_person_evidence -- --ignored --nocapture
//!
//! 只读库 / 模型推理。不改生产代码 / 不改 DB / 不重聚类。

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use image::RgbImage;
use ort::session::Session;
use ort::value::Tensor;

use pf_ai::face::aligner::{AlignmentConfig, SimpleAligner};
use pf_ai::face::arcface::{l2_normalize, EMBEDDING_DIM, INPUT_SIZE};
use pf_ai::face::traits::FaceAligner;
use pf_ai::image_data::ImageData;
use pf_ai::{FaceDetector, ScrfdDetector};
use pf_core::{BBox, FaceKeypoints};

const ARCFACE_MODEL: &str = "/Users/mac/ai-project/photofinder-next-2/models/w600k_r50.onnx";
const SCRFD_MODEL: &str = "/Users/mac/ai-project/photofinder-next-2/models/scrfd_500m_bnkps.onnx";
const INPUT_TSV: &str = "/tmp/tta_img15/input.tsv";
const OUT_DIR: &str = "/tmp/img15_person_evidence";
const CROP_MARGIN: f32 = 0.10;

const CORE: [i64; 3] = [6, 13, 14]; // person2 正核心
const CANDIDATE: i64 = 15; // 待判定样本
const NEG: [i64; 6] = [2, 9, 17, 18, 19, 20]; // 6 个不同 negative 人

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

// ---- 推理 / 对齐 ----

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

/// 生产输入：crop10% → SimpleAligner（含椭圆 mask + histeq）→ 112×112。
fn build_production(
    aligner: &SimpleAligner,
    img: &ImageData,
    bbox: &BBox,
    kps: &[(f32, f32); 5],
) -> Result<RgbImage, String> {
    let (crop, origin) = crop_to_bbox_with_margin(img, bbox, CROP_MARGIN)?;
    let shifted = shift_kps(kps, origin);
    let aligned = aligner.align(&crop, &kps_from_array(&shifted)).map_err(|e| e.to_string())?;
    Ok(aligned.image.as_rgb8())
}

// ---- person-level 计算 ----

fn cosine_map(m: &HashMap<i64, Vec<f32>>, a: i64, b: i64) -> f32 {
    cosine(m.get(&a).unwrap(), m.get(&b).unwrap())
}

/// 归一化均值原型。
fn prototype(m: &HashMap<i64, Vec<f32>>, core: &[i64]) -> Vec<f32> {
    let d = m.get(&core[0]).unwrap().len();
    let mut acc = vec![0f32; d];
    for &i in core {
        let v = m.get(&i).unwrap();
        for (k, x) in v.iter().enumerate() {
            acc[k] += x;
        }
    }
    let n = core.len() as f32;
    for v in &mut acc {
        *v /= n;
    }
    l2_normalize(&acc)
}

/// medoid：core 内 mean(sim(i,j)) 最大（排除自身）。
fn medoid(m: &HashMap<i64, Vec<f32>>, core: &[i64]) -> (i64, f32) {
    let mut best = core[0];
    let mut best_s = f32::NEG_INFINITY;
    for &i in core {
        let mut acc = 0.0f32;
        let mut cnt = 0usize;
        for &j in core {
            if i == j {
                continue;
            }
            acc += cosine_map(m, i, j);
            cnt += 1;
        }
        let mean = acc / cnt as f32;
        if mean > best_s {
            best_s = mean;
            best = i;
        }
    }
    (best, best_s)
}

/// 候选（embedding 或 sample id）对 negatives 的最大 cosine + 对应样本。
fn neg_max_from_emb(m: &HashMap<i64, Vec<f32>>, cand: &[f32], neg: &[i64]) -> (f32, i64) {
    let mut mx = f32::NEG_INFINITY;
    let mut arg = neg[0];
    for &n in neg {
        let c = cosine(cand, m.get(&n).unwrap());
        if c > mx {
            mx = c;
            arg = n;
        }
    }
    (mx, arg)
}

fn neg_max_from_id(m: &HashMap<i64, Vec<f32>>, cand: i64, neg: &[i64]) -> (f32, i64) {
    neg_max_from_emb(m, m.get(&cand).unwrap(), neg)
}

fn stats3(v: &[f32]) -> (f32, f32, f32, f32, f32) {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let max = *s.last().unwrap();
    let top2 = (s[2] + s[1]) / 2.0;
    let top3 = s.iter().sum::<f32>() / 3.0;
    let median = s[1];
    let mean = top3;
    (max, top2, top3, median, mean)
}

#[test]
#[ignore]
fn img15_person_evidence() {
    fs::create_dir_all(OUT_DIR).expect("mkdir");
    let samples = read_tsv();
    assert_eq!(samples.len(), 10, "expect 10 samples");
    let aligner = SimpleAligner::with_config(AlignmentConfig::default());
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

    // 实时全精度 bbox/5pts（= DB 存储值）
    struct Geo {
        bbox: BBox,
        kps: [(f32, f32); 5],
    }
    let mut geos = Vec::new();
    for i in 0..samples.len() {
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

    // 每张图只 embedding 一次（生产路径）
    let mut embs: HashMap<i64, Vec<f32>> = HashMap::new();
    for (si, s) in samples.iter().enumerate() {
        let rgb = build_production(&aligner, &images[si], &geos[si].bbox, &geos[si].kps)
            .expect("build production input");
        let e = embed_one(&mut session, &rgb);
        assert!(e.iter().all(|x| x.is_finite()), "img{} emb finite", s.img);
        embs.insert(s.img, e);
    }
    assert_eq!(embs.len(), 10, "10 unique embeddings");
    assert_eq!(embs.get(&CANDIDATE).unwrap().len(), EMBEDDING_DIM);

    let c = |a: i64, b: i64| cosine_map(&embs, a, b);
    let mut report = String::new();
    let mut csv = String::new();

    // ============ 1. pairwise matrix ============
    report.push_str("\n==================================================\n");
    report.push_str("IMG15 PERSON-LEVEL EVIDENCE\n");
    report.push_str("==================================================\n\n");
    report.push_str("1. Pairwise matrix (img15 对全部 9 张, 其余为参考)\n");
    let cols = [6, 13, 14, 2, 9, 17, 18, 19, 20];
    report.push_str(format!("{:<8}", "").as_str());
    for &col in &cols {
        report.push_str(format!("{:<8}", format!("img{}", col)).as_str());
    }
    report.push('\n');
    for &row in &[CANDIDATE, 6, 13, 14, 2, 9, 17, 18, 19, 20] {
        report.push_str(format!("{:<8}", format!("img{}", row)).as_str());
        for &col in &cols {
            report.push_str(format!("{:<8.4}", if row == col { 1.0 } else { c(row, col) }).as_str());
        }
        report.push('\n');
    }
    report.push('\n');

    // ============ 2. positive core ============
    let core_sims: Vec<f32> = CORE.iter().map(|&t| c(CANDIDATE, t)).collect();
    let (max_s, top2_mean, top3_mean, median_s, mean_s) = stats3(&core_sims);
    report.push_str("2. Positive core  P = {img6, img13, img14}   img15 →\n");
    report.push_str(format!("   img6={:.4}  img13={:.4}  img14={:.4}\n", core_sims[0], core_sims[1], core_sims[2]).as_str());
    report.push_str(format!("   max={:.4}  top2_mean={:.4}  top3_mean={:.4}  median={:.4}  mean={:.4}\n",
        max_s, top2_mean, top3_mean, median_s, mean_s).as_str());
    report.push('\n');

    // ============ 3. prototype ============
    let proto = prototype(&embs, &CORE);
    let proto_cos = cosine(&embs[&CANDIDATE], &proto);
    report.push_str(format!("3. Prototype  prototype_mean = normalize(mean(img6,img13,img14))\n   cos(img15, prototype) = {:.4}\n", proto_cos).as_str());
    report.push('\n');

    // ============ 4. medoid ============
    let (med, med_sim) = medoid(&embs, &CORE);
    report.push_str(format!("4. Medoid   argmax_i mean(sim(i,j)) in core → img{} (内部 mean {:.4})\n   img15 → medoid = {:.4}\n",
        med, med_sim, c(CANDIDATE, med)).as_str());
    report.push('\n');

    // ============ 5. leave-one-out prototype ============
    report.push_str("5. Leave-one-out prototype   img15 →\n");
    let mut loo = Vec::new();
    for &ex in &CORE {
        let rest: Vec<i64> = CORE.iter().copied().filter(|&x| x != ex).collect();
        let p = prototype(&embs, &rest);
        let s = cosine(&embs[&CANDIDATE], &p);
        loo.push((ex, s));
        report.push_str(format!("   prototype_without_img{} ({}) → {:.4}\n",
            ex, rest.iter().map(|x| format!("img{}", x)).collect::<Vec<_>>().join(","), s).as_str());
    }
    report.push('\n');

    // ============ 6. positive-vs-negative evidence ============
    let (neg_max, neg_arg) = neg_max_from_id(&embs, CANDIDATE, &NEG);
    report.push_str(format!("6. Positive-vs-negative evidence   (negative_max = img15→img{} = {:.4})\n", neg_arg, neg_max).as_str());
    report.push_str(format!("   {:<12} {:>10} {:>10} {:>9}\n", "method", "positive", "negative", "margin").as_str());
    let mut methods: Vec<(&str, f32)> = Vec::new();
    methods.push(("max", max_s));
    methods.push(("top2_mean", top2_mean));
    methods.push(("top3_mean", top3_mean));
    methods.push(("median", median_s));
    methods.push(("prototype", proto_cos));
    methods.push(("medoid", c(CANDIDATE, med)));
    for (name, pos) in &methods {
        let margin = pos - neg_max;
        report.push_str(format!("   {:<12} {:>10.4} {:>10.4} {:>+9.4}\n", name, pos, neg_max, margin).as_str());
        csv.push_str(&format!("method,{},{:.4},{:.4},{:+.4}\n", name, pos, neg_max, margin));
    }
    report.push('\n');

    // ============ 7. ranking ============
    report.push_str("7. Ranking  img15 对全部测试图片按 cosine 降序\n");
    let mut ranked: Vec<(i64, f32)> = cols
        .iter()
        .map(|&t| (t, c(CANDIDATE, t)))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    report.push_str(format!("   {:<4} {:<8} {:<10} {:>8}\n", "rank", "img", "group", "cosine").as_str());
    for (i, (img, s)) in ranked.iter().enumerate() {
        let group = if CORE.contains(img) { "PERSON2" } else { "NEGATIVE" };
        let mark = if CORE.contains(img) && *s >= 0.75 { " ✓(≥0.75)" } else { "" };
        report.push_str(format!("   {:<4} {:<8} {:<10} {:>8.4}{}\n", i + 1, format!("img{}", img), group, s, mark).as_str());
    }
    report.push('\n');

    // ============ 8. cluster consistency ============
    report.push_str("8. Cluster consistency   core{img6,img13,img14} 加 img15 前后\n");
    let ctr_before = prototype(&embs, &CORE);
    let (med_before, _) = medoid(&embs, &CORE);
    let intra_pairs = [(6, 13), (6, 14), (13, 14)];
    let intra_before: Vec<f32> = intra_pairs.iter().map(|&(a, b)| c(a, b)).collect();
    let intra_min_before = intra_before.iter().cloned().fold(f32::INFINITY, f32::min);
    let intra_mean_before = intra_before.iter().sum::<f32>() / intra_before.len() as f32;
    let (nb_sim, nb_arg) = neg_max_from_emb(&embs, &ctr_before, &NEG);
    report.push_str(format!("   [前] centroid 对 negative max = {:.4} (img{})\n", nb_sim, nb_arg).as_str());
    report.push_str(format!("       medoid=img{}   intra min={:.4} mean={:.4}\n", med_before, intra_min_before, intra_mean_before).as_str());

    let core_plus: Vec<i64> = vec![6, 13, 14, 15];
    let ctr_after = prototype(&embs, &core_plus);
    let (med_after, _) = medoid(&embs, &core_plus);
    let intra_after_pairs = [
        (6, 13), (6, 14), (6, 15), (13, 14), (13, 15), (14, 15),
    ];
    let intra_after: Vec<f32> = intra_after_pairs.iter().map(|&(a, b)| c(a, b)).collect();
    let intra_min_after = intra_after.iter().cloned().fold(f32::INFINITY, f32::min);
    let intra_mean_after = intra_after.iter().sum::<f32>() / intra_after.len() as f32;
    let drift = cosine(&ctr_before, &ctr_after);
    let (na_sim, na_arg) = neg_max_from_emb(&embs, &ctr_after, &NEG);
    report.push_str(format!("   [后] centroid 对 negative max = {:.4} (img{})\n", na_sim, na_arg).as_str());
    report.push_str(format!("       medoid=img{}   intra min={:.4} mean={:.4}\n", med_after, intra_min_after, intra_mean_after).as_str());
    report.push_str(format!("       centroid drift  cos(before,after) = {:.4}\n", drift).as_str());
    report.push_str(format!("       Δ negative max = {:+.4}\n", na_sim - nb_sim).as_str());
    report.push_str("   核心成员自身: cos(成员, centroid_after) vs 成员对 negative max:\n");
    let mut core_member_safe = true;
    for &x in &CORE {
        let s2c = cosine(&embs[&x], &ctr_after);
        let (s2n, narg) = neg_max_from_id(&embs, x, &NEG);
        let safe = s2c > s2n;
        core_member_safe &= safe;
        report.push_str(format!("       img{}: cos→centroid_after={:.4}  neg_max={:.4} (img{})  [{}]\n",
            x, s2c, s2n, narg, if safe { "SAFE" } else { "AT_RISK" }).as_str());
    }
    report.push('\n');

    // ============ 9. positive core expansion ============
    report.push_str("9. Positive Core Expansion   candidate img15\n");
    report.push_str(format!("   {:<18} {:>10} {:>10} {:>10} {:>9}\n", "core", "positive", "neg_max", "neg_mean", "margin").as_str());
    let mut expansion: Vec<(i64, f32)> = Vec::new();
    for k in 1..=3 {
        let core: Vec<i64> = CORE[..k].to_vec();
        let p = prototype(&embs, &core);
        let pos = cosine(&embs[&CANDIDATE], &p);
        let (nmax, narg) = neg_max_from_emb(&embs, &embs[&CANDIDATE], &NEG);
        let nmean = NEG.iter().map(|&n| cosine(&embs[&CANDIDATE], &embs[&n])).sum::<f32>() / NEG.len() as f32;
        let margin = pos - nmax;
        expansion.push((k as i64, pos));
        let core_lbl = core.iter().map(|x| format!("img{}", x)).collect::<Vec<_>>().join(",");
        report.push_str(format!("   {:<18} {:>10.4} {:>10.4} {:>10.4} {:>+9.4}  (neg_max=img{})\n",
            format!("Core{}={{}}", core_lbl), pos, nmax, nmean, margin, narg).as_str());
        csv.push_str(&format!("expansion,Core{},{:.4},{:.4},{:.4},{:+.4}\n", k, pos, nmax, nmean, margin));
    }
    report.push('\n');

    // ============ 成功标准 ============
    let neg_of = |pos: f32| pos - neg_max;
    let cond1 = methods.iter().any(|(_, pos)| neg_of(*pos) > 0.0);
    let p1 = expansion[0].1;
    let p2 = expansion[1].1;
    let p3 = expansion[2].1;
    let cond2 = p2 >= p1 - 1e-4 && p3 >= p2 - 1e-4;
    let cond3 = drift >= 0.97 && (na_sim - nb_sim) <= 0.01;
    let cond4 = core_member_safe;
    let cond5 = expansion.iter().all(|(_, pos)| neg_of(*pos) > 0.0);
    let loo_margins: Vec<f32> = loo.iter().map(|(_, s)| neg_of(*s)).collect();
    let cond6 = loo_margins.iter().all(|m| *m > 0.0);

    report.push_str("==================================================\n");
    report.push_str("SUCCESS CRITERIA (person-level, 非 img15>0.75)\n");
    report.push_str("==================================================\n");
    report.push_str(format!("  1. 存在 person scoring 使 img15→person2 高于所有 negative  [{}]  (6 方法中正 margin {} 个)\n",
        if cond1 { "PASS" } else { "FAIL" }, methods.iter().filter(|(_, p)| neg_of(*p) > 0.0).count()).as_str());
    report.push_str(format!("  2. 加 img13/14 evidence 后 positive 不下降  [{}]  (Core1 {:.4} → Core2 {:.4} → Core3 {:.4})\n",
        if cond2 { "PASS" } else { "FAIL" }, p1, p2, p3).as_str());
    report.push_str(format!("  3. 不污染 person2 core  [{}]  (centroid drift cos={:.4}, Δneg_max={:+.4})\n",
        if cond3 { "PASS" } else { "FAIL" }, drift, na_sim - nb_sim).as_str());
    report.push_str(format!("  4. 不导致 6/13/14 重新错误聚类  [{}]\n", if cond4 { "PASS" } else { "FAIL" }).as_str());
    report.push_str(format!("  5. margin 跨 core 子集保持正值  [{}]  (Core1/2/3 margins: {:+.4} / {:+.4} / {:+.4})\n",
        if cond5 { "PASS" } else { "FAIL" }, neg_of(p1), neg_of(p2), neg_of(p3)).as_str());
    report.push_str(format!("  6. 非单样本 knife-edge (leave-one-out margins)  [{}]  ({:+.4} / {:+.4} / {:+.4})\n",
        if cond6 { "PASS" } else { "FAIL" }, loo_margins[0], loo_margins[1], loo_margins[2]).as_str());
    report.push('\n');

    let all_pass = cond1 && cond2 && cond3 && cond4 && cond5 && cond6;
    report.push_str("VERDICT:\n");
    if all_pass {
        let best = methods.iter().max_by(|a, b| neg_of(a.1).partial_cmp(&neg_of(b.1)).unwrap()).unwrap();
        report.push_str(format!("  ✓ 存在稳定 person-level evidence。最佳方法: {} (margin {:+.4})。\n", best.0, neg_of(best.1)).as_str());
    } else {
        report.push_str("  ✗ Face embedding 本身无法提供足够证据，\n");
        report.push_str("    下一阶段必须引入第二种身份特征。\n");
    }
    report.push_str("\n=== DONE === 输出目录 ");
    report.push_str(OUT_DIR);
    report.push('\n');

    eprintln!("{report}");

    fs::write(Path::new(OUT_DIR).join("person_evidence_report.txt"), &report).expect("write report");
    fs::write(Path::new(OUT_DIR).join("person_evidence.csv"), &csv).expect("write csv");
}
