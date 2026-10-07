//! 临时诊断（非生产）：img15 轻量 TTA 实测。
//!
//! 离线复刻生产管线（crop 10% margin → SimpleAligner → ArcFace 1-crop），
//! 对 10 张已确认身份的图生成 4 个视图（orig / flip / rot+5 / rot-5）embedding，
//! 比较 1-crop / +flip / +rot / 4-view mean / weighted / pairwise-max，
//! 输出全矩阵 + img15 排名 + 同人/跨人分离指标。
//!
//! 输入：/tmp/tta_img15/input.tsv（由 dump_tta_input.py 从 live DB 导出）：
//!   img<TAB>path<TAB>bx<TAB>by<TAB>bw<TAB>bh<TAB>le_x<le_y re_x re_y n_x n_y lm_x lm_y rm_x rm_y<TAB>512 db embedding
//!
//! 用法：
//!   ORT_DYLIB_PATH=/Users/mac/lib/libonnxruntime.dylib \
//!   cargo test --release -p pf_ai --test tta_img15 -- --ignored --nocapture
//!
//! 读库 + 模型推理，不改生产代码 / 不改 DB / 不重聚类。

use std::path::PathBuf;

use image::RgbImage;
use ort::session::Session;
use ort::value::Tensor;

use pf_ai::face::aligner::SimpleAligner;
use pf_ai::face::arcface::{l2_normalize, EMBEDDING_DIM, INPUT_SIZE};
use pf_ai::face::traits::FaceAligner;
use pf_ai::image_data::ImageData;
use pf_core::BBox;

const ARCFACE_MODEL: &str =
    "/Users/mac/ai-project/photofinder-next-2/models/w600k_r50.onnx";
const INPUT_TSV: &str = "/tmp/tta_img15/input.tsv";

/// 同人（jaqor）集合：img6/13/14/15。
const SAME: [i64; 4] = [6, 13, 14, 15];

fn is_same(a: i64, b: i64) -> bool {
    SAME.contains(&a) && SAME.contains(&b)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

fn l2(v: &[f32]) -> Vec<f32> {
    l2_normalize(v)
}

fn norm_add(parts: &[Vec<f32>]) -> Vec<f32> {
    let mut out = vec![0.0f32; EMBEDDING_DIM];
    for p in parts {
        for (o, x) in out.iter_mut().zip(p.iter()) {
            *o += x;
        }
    }
    l2(&out)
}

fn weighted(orig: &[f32], flip: &[f32], rotp: &[f32], rotm: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; EMBEDDING_DIM];
    for i in 0..EMBEDDING_DIM {
        out[i] = 0.50 * orig[i] + 0.20 * flip[i] + 0.15 * rotp[i] + 0.15 * rotm[i];
    }
    l2(&out)
}

// ---- ArcFace 单视图推理（复刻 arcface.rs 的 build_input + session.run）----

fn flip_horizontal(face: &RgbImage) -> RgbImage {
    let mut flipped = RgbImage::new(INPUT_SIZE, INPUT_SIZE);
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let p = face.get_pixel(INPUT_SIZE - 1 - x, y);
            flipped.put_pixel(x, y, *p);
        }
    }
    flipped
}

fn rotate_nearest(face: &RgbImage, angle_rad: f32) -> RgbImage {
    let w = face.width() as i32;
    let h = face.height() as i32;
    let cx = w as f32 * 0.5;
    let cy = h as f32 * 0.5;
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();
    let black = image::Rgb([0_u8, 0, 0]);
    let mut out = RgbImage::new(face.width(), face.height());
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let sx = (dx * cos_a + dy * sin_a + cx).round() as i32;
            let sy = (-dx * sin_a + dy * cos_a + cy).round() as i32;
            let p = if sx >= 0 && sx < w && sy >= 0 && sy < h {
                *face.get_pixel(sx as u32, sy as u32)
            } else {
                black
            };
            out.put_pixel(x as u32, y as u32, p);
        }
    }
    out
}

fn build_input(face: &RgbImage, flipped: bool) -> Vec<f32> {
    let n = (INPUT_SIZE * INPUT_SIZE) as usize;
    let mut input = Vec::with_capacity(3 * n);
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let p = if flipped {
                face.get_pixel(INPUT_SIZE - 1 - x, y)
            } else {
                face.get_pixel(x, y)
            };
            input.push((p[2] as f32 - 127.5) / 128.0);
        }
    }
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let p = if flipped {
                face.get_pixel(INPUT_SIZE - 1 - x, y)
            } else {
                face.get_pixel(x, y)
            };
            input.push((p[1] as f32 - 127.5) / 128.0);
        }
    }
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let p = if flipped {
                face.get_pixel(INPUT_SIZE - 1 - x, y)
            } else {
                face.get_pixel(x, y)
            };
            input.push((p[0] as f32 - 127.5) / 128.0);
        }
    }
    input
}

fn embed_one(session: &mut Session, face: &RgbImage, flipped: bool) -> Vec<f32> {
    let input_data = build_input(face, flipped);
    let shape = [1_i64, 3, INPUT_SIZE as i64, INPUT_SIZE as i64];
    let input = Tensor::from_array((shape, input_data)).expect("tensor");
    let outputs = session
        .run(ort::inputs![input])
        .expect("arcface run");
    let (_shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .expect("extract");
    l2_normalize(data)
}

// ---- 数据 ----

struct Sample {
    img: i64,
    path: String,
    bbox: BBox,
    kps: [(f32, f32); 5], // le re nose lm rm
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

/// 复刻 traits.rs crop_to_bbox_with_margin：10% margin crop + kp 平移。
fn crop_and_align(
    aligner: &SimpleAligner,
    s: &Sample,
) -> Result<RgbImage, String> {
    let img = ImageData::from_file(std::path::Path::new(&s.path)).map_err(|e| e.to_string())?;
    let w = img.width() as f32;
    let h = img.height() as f32;
    let bw = s.bbox.w.max(1.0);
    let bh = s.bbox.h.max(1.0);
    let mx = bw * 0.1;
    let my = bh * 0.1;
    let x = (s.bbox.x - mx).max(0.0);
    let y = (s.bbox.y - my).max(0.0);
    let cw = (bw + 2.0 * mx).min(w - x);
    let ch = (bh + 2.0 * my).min(h - y);
    if cw < 1.0 || ch < 1.0 {
        return Err(format!("crop too small: {cw}x{ch}"));
    }
    let crop = img
        .crop(BBox::new(x, y, cw, ch))
        .map_err(|e| e.to_string())?;
    let kps = pf_core::FaceKeypoints {
        left_eye: (s.kps[0].0 - x, s.kps[0].1 - y),
        right_eye: (s.kps[1].0 - x, s.kps[1].1 - y),
        nose: (s.kps[2].0 - x, s.kps[2].1 - y),
        left_mouth: (s.kps[3].0 - x, s.kps[3].1 - y),
        right_mouth: (s.kps[4].0 - x, s.kps[4].1 - y),
    };
    let aligned = aligner.align(&crop, &kps).map_err(|e| e.to_string())?;
    Ok(aligned.image.as_rgb8())
}

// ---- 指标 ----

struct Metrics {
    same_min: f32,
    same_mean: f32,
    hard_neg_max: f32,
    separation: f32,      // same_min - hard_neg_max
    img15_margin: f32,    // pos_min(15) - neg_max(15)
}

fn compute_metrics(scores: &[(i64, i64, f32)]) -> Metrics {
    let mut same_vals: Vec<f32> = Vec::new();
    let mut neg_vals: Vec<f32> = Vec::new();
    let mut img15_pos: Vec<f32> = Vec::new();
    let mut img15_neg: Vec<f32> = Vec::new();
    for &(a, b, s) in scores {
        if is_same(a, b) {
            same_vals.push(s);
            if a == 15 || b == 15 {
                img15_pos.push(s);
            }
        } else {
            neg_vals.push(s);
            if a == 15 || b == 15 {
                img15_neg.push(s);
            }
        }
    }
    let same_min = same_vals.iter().copied().fold(f32::INFINITY, f32::min);
    let same_mean = same_vals.iter().sum::<f32>() / same_vals.len() as f32;
    let hard_neg_max = neg_vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let img15_margin = img15_pos.iter().copied().fold(f32::INFINITY, f32::min)
        - img15_neg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    Metrics {
        same_min,
        same_mean,
        hard_neg_max,
        separation: same_min - hard_neg_max,
        img15_margin,
    }
}

fn accuracy_at(scores: &[(i64, i64, f32)], tau: f32) -> (usize, usize, usize, usize, f32, f32, f32) {
    let mut tp = 0usize;
    let mut tn = 0usize;
    let mut fp = 0usize;
    let mut fn_ = 0usize;
    for &(a, b, s) in scores {
        let pred_same = s >= tau;
        let true_same = is_same(a, b);
        match (true_same, pred_same) {
            (true, true) => tp += 1,
            (false, false) => tn += 1,
            (false, true) => fp += 1,
            (true, false) => fn_ += 1,
        }
    }
    let prec = tp as f32 / (tp + fp) as f32;
    let rec = tp as f32 / (tp + fn_) as f32;
    let f1 = if prec + rec > 0.0 {
        2.0 * prec * rec / (prec + rec)
    } else {
        0.0
    };
    (tp, tn, fp, fn_, prec, rec, f1)
}

#[test]
#[ignore]
fn tta_img15_experiment() {
    let samples = read_tsv();
    eprintln!("loaded {} samples: {:?}", samples.len(), samples.iter().map(|s| s.img).collect::<Vec<_>>());
    assert_eq!(samples.len(), 10, "expect 10 samples");

    let mut session = Session::builder()
        .expect("builder")
        .commit_from_file(PathBuf::from(ARCFACE_MODEL))
        .expect("load arcface");
    let aligner = SimpleAligner::new();
    let rot = std::f32::consts::PI * 5.0 / 180.0;

    // per-image: 4 view embeddings
    let mut views: Vec<(i64, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new(); // img, orig, flip, rotp, rotm
    let mut max_db_cos = 0.0f32;
    let mut min_db_cos = f32::MAX;
    for s in &samples {
        let rgb = crop_and_align(&aligner, s).unwrap_or_else(|e| panic!("img{}: {e}", s.img));
        let orig = embed_one(&mut session, &rgb, false);
        let flip = embed_one(&mut session, &flip_horizontal(&rgb), false);
        let rotp = embed_one(&mut session, &rotate_nearest(&rgb, rot), false);
        let rotm = embed_one(&mut session, &rotate_nearest(&rgb, -rot), false);
        let db_cos = cosine(&orig, &s.db_emb);
        max_db_cos = max_db_cos.max(db_cos);
        min_db_cos = min_db_cos.min(db_cos);
        views.push((s.img, orig, flip, rotp, rotm));
    }
    eprintln!("sanity orig-vs-DB: cos range [{:.4}, {:.4}]", min_db_cos, max_db_cos);
    assert!(min_db_cos > 0.99, "offline repro does not match DB (min={min_db_cos})");

    // modes: name -> per-image single embedding
    let modes: Vec<(&str, Vec<Vec<f32>>)> = vec![
        ("1-crop", views.iter().map(|v| v.1.clone()).collect()),
        ("+Flip", views.iter().map(|v| norm_add(&[v.1.clone(), v.2.clone()])).collect()),
        ("+Rot±5", views.iter().map(|v| norm_add(&[v.1.clone(), v.3.clone(), v.4.clone()])).collect()),
        ("4-view Mean", views.iter().map(|v| norm_add(&[v.1.clone(), v.2.clone(), v.3.clone(), v.4.clone()])).collect()),
        ("Weighted", views.iter().map(|v| weighted(&v.1, &v.2, &v.3, &v.4)).collect()),
    ];

    let imgs: Vec<i64> = samples.iter().map(|s| s.img).collect();
    let label = |a: i64| -> &'static str { if SAME.contains(&a) { "SAME" } else { "diff" } };

    // ============ 1. full pairwise matrix per mode ============
    for (name, embs) in &modes {
        eprintln!("\n=== Matrix ({name}) ===");
        let header: String = imgs.iter().map(|i| format!("img{i:>3}")).collect::<Vec<_>>().join(" ");
        eprintln!("            {header}");
        for (i, a) in imgs.iter().enumerate() {
            let row: Vec<String> = imgs
                .iter()
                .map(|b| format!("{:.3}", if *a == *b { 1.0 } else { cosine(&embs[i], &embs[imgs.iter().position(|x| x == b).unwrap()]) }))
                .collect();
            eprintln!("img{a:>2}  {}", row.join(" "));
        }
    }

    // ============ 2. img15 ranking per mode ============
    for (name, embs) in &modes {
        let i15 = imgs.iter().position(|x| *x == 15).unwrap();
        let mut rank: Vec<(i64, f32)> = imgs
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i15)
            .map(|(j, b)| (*b, cosine(&embs[i15], &embs[j])))
            .collect();
        rank.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("\n=== Query=img15 ({name}) ===");
        for (rank_i, (img, s)) in rank.iter().enumerate() {
            eprintln!("  rank {:<2} img{:<2} ({:<4})  {:.4}", rank_i + 1, img, label(*img), s);
        }
    }

    // ============ 3. metrics per mode ============
    eprintln!("\n=== Core metrics per mode ===");
    eprintln!("{:<14} {:>8} {:>8} {:>14} {:>11} {:>13}", "Mode", "SameMin", "SameMean", "HardNegMax", "Separation", "img15Margin");
    for (name, embs) in &modes {
        let mut scores = Vec::new();
        for i in 0..imgs.len() {
            for j in (i + 1)..imgs.len() {
                scores.push((imgs[i], imgs[j], cosine(&embs[i], &embs[j])));
            }
        }
        let m = compute_metrics(&scores);
        eprintln!("{:<14} {:>8.4} {:>8.4} {:>14.4} {:>11.4} {:>13.4}", name, m.same_min, m.same_mean, m.hard_neg_max, m.separation, m.img15_margin);
    }

    // ============ 4. accuracy sweep per mode ============
    eprintln!("\n=== Pairwise accuracy sweep ===");
    for tau in [0.55_f32, 0.60, 0.65, 0.70, 0.75, 0.80] {
        eprintln!("tau={:.2}:", tau);
        for (name, embs) in &modes {
            let mut scores = Vec::new();
            for i in 0..imgs.len() {
                for j in (i + 1)..imgs.len() {
                    scores.push((imgs[i], imgs[j], cosine(&embs[i], &embs[j])));
                }
            }
            let (tp, tn, fp, fn_, p, r, f1) = accuracy_at(&scores, tau);
            eprintln!("  {:<14} TP={:<2} TN={:<2} FP={:<2} FN={:<2} P={:.3} R={:.3} F1={:.3}",
                name, tp, tn, fp, fn_, p, r, f1);
        }
    }

    // ============ 5. pairwise-max (实验对照) ============
    eprintln!("\n=== Pairwise-max (experimental control, 16 view-combos) ===");
    let mut pm_scores = Vec::new();
    for i in 0..imgs.len() {
        for j in (i + 1)..imgs.len() {
            let va = &views[i];
            let vb = &views[j];
            let mut best = 0.0f32;
            let v_a = [&va.1, &va.2, &va.3, &va.4];
            let v_b = [&vb.1, &vb.2, &vb.3, &vb.4];
            for x in &v_a {
                for y in &v_b {
                    best = best.max(cosine(x, y));
                }
            }
            pm_scores.push((imgs[i], imgs[j], best));
        }
    }
    let pm = compute_metrics(&pm_scores);
    eprintln!("pairwise-max: SameMin={:.4} SameMean={:.4} HardNegMax={:.4} Sep={:.4} img15Margin={:.4}",
        pm.same_min, pm.same_mean, pm.hard_neg_max, pm.separation, pm.img15_margin);

    // ============ 6. img15 per-view table ============
    eprintln!("\n=== Query img15 per-view ===");
    let i15 = views.iter().position(|v| v.0 == 15).unwrap();
    let cols = [6, 13, 14, 2, 9, 20, 17];
    let rows: Vec<(&str, Vec<Vec<f32>>)> = vec![
        ("Original", views.iter().map(|v| v.1.clone()).collect()),
        ("Flip", views.iter().map(|v| v.2.clone()).collect()),
        ("Rot+5", views.iter().map(|v| v.3.clone()).collect()),
        ("Rot-5", views.iter().map(|v| v.4.clone()).collect()),
        ("Mean4", modes[3].1.clone()),
        ("Weighted", modes[4].1.clone()),
    ];
    let col_hdr: String = cols.iter().map(|c| format!("img{c:>3}")).collect::<Vec<_>>().join(" ");
    eprintln!("{:<10} {col_hdr}", "");
    for (rname, embs) in &rows {
        let vals: Vec<String> = cols
            .iter()
            .map(|c| {
                let j = views.iter().position(|v| v.0 == *c).unwrap();
                format!("{:.3}", cosine(&embs[i15], &embs[j]))
            })
            .collect();
        eprintln!("{rname:<10} {}", vals.join(" "));
    }
    // img15 pos vs neg per row
    eprintln!("{}", "-".repeat(60));
    eprintln!("{:<10} {:>8} {:>8} {:>10}", "Row", "PosMin", "PosMean", "NegMax");
    for (rname, embs) in &rows {
        let mut pos: Vec<f32> = Vec::new();
        let mut neg: Vec<f32> = Vec::new();
        for (j, v) in views.iter().enumerate() {
            if j == i15 {
                continue;
            }
            let s = cosine(&embs[i15], &embs[j]);
            if is_same(15, v.0) {
                pos.push(s);
            } else {
                neg.push(s);
            }
        }
        let pmin = pos.iter().copied().fold(f32::INFINITY, f32::min);
        let pmean = pos.iter().sum::<f32>() / pos.len() as f32;
        let nmax = neg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("{rname:<10} {:>8.4} {:>8.4} {:>10.4}", pmin, pmean, nmax);
    }

    eprintln!("\n=== DONE ===");
}
