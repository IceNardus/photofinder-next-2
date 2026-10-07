//! 临时诊断（非生产）：定位 img15 embedding 漂移来源。
//!
//! 对 production 对齐管线逐环节做单因素消融（其余保持生产默认）：
//!   A. align_scale / y_offset（AlignmentConfig）
//!   B. histogram equalization on/off
//!   C. ellipse mask on/off
//!   D. crop margin 大小
//!   E. 输入归一化（(x-offset)/div）
//!   F. 其他（直接全图对齐、histeq+mask 同时关闭）
//!
//! 每配置对 10 张图（jaqor {6,13,14,15} + 负例 {2,9,20,17,18,19}）重新
//! crop→align→ArcFace 1-crop，报告 img15 的 margin（pos_min − neg_max）与
//! img15↔{6,13,14} / img15↔{2,9,20} 相似度，以及相对 DB embedding 的漂移量。
//!
//! 输入：/tmp/tta_img15/input.tsv（同 tta_img15）。
//! 用法：
//!   ORT_DYLIB_PATH=/Users/mac/lib/libonnxruntime.dylib \
//!   cargo test --release -p pf_ai --test alignment_ablation -- --ignored --nocapture
//!
//! 不改生产代码 / 不改 DB / 不重聚类。

use std::path::PathBuf;

use image::RgbImage;
use ort::session::Session;
use ort::value::Tensor;

use pf_ai::face::aligner::{AlignmentConfig, SimpleAligner};
use pf_ai::face::arcface::{l2_normalize, EMBEDDING_DIM, INPUT_SIZE};
use pf_ai::face::traits::FaceAligner;
use pf_ai::image_data::ImageData;
use pf_core::{BBox, FaceKeypoints};

const ARCFACE_MODEL: &str = "/Users/mac/ai-project/photofinder-next-2/models/w600k_r50.onnx";
const INPUT_TSV: &str = "/tmp/tta_img15/input.tsv";

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
        out.push(Sample { img, path, bbox, kps, db_emb });
    }
    out
}

/// 每配置的参数面。
struct PreConfig {
    name: &'static str,
    align_scale: f32,
    y_offset: f32,
    mask: bool,
    histeq: bool,
    crop_margin: f32,
    norm_offset: f32,
    norm_div: f32,
    use_crop: bool, // F: 直接全图对齐
}

const BASE: PreConfig = PreConfig {
    name: "base(生产)",
    align_scale: 1.25,
    y_offset: -10.0,
    mask: true,
    histeq: true,
    crop_margin: 0.10,
    norm_offset: 127.5,
    norm_div: 128.0,
    use_crop: true,
};

fn variants() -> Vec<PreConfig> {
    vec![
        BASE,
        // A. alignment 参数
        PreConfig { name: "A-scale=1.00", align_scale: 1.00, ..BASE },
        PreConfig { name: "A-scale=1.50", align_scale: 1.50, ..BASE },
        PreConfig { name: "A-yoff=0", y_offset: 0.0, ..BASE },
        PreConfig { name: "A-yoff=-20", y_offset: -20.0, ..BASE },
        // B. histogram equalization
        PreConfig { name: "B-histeq=off", histeq: false, ..BASE },
        // C. ellipse mask
        PreConfig { name: "C-mask=off", mask: false, ..BASE },
        // D. crop margin
        PreConfig { name: "D-margin=0.05", crop_margin: 0.05, ..BASE },
        PreConfig { name: "D-margin=0.15", crop_margin: 0.15, ..BASE },
        PreConfig { name: "D-margin=0.20", crop_margin: 0.20, ..BASE },
        PreConfig { name: "D-margin=0.30", crop_margin: 0.30, ..BASE },
        // E. 输入归一化（对照：模型训练即 (x-127.5)/128，变更只会退化）
        PreConfig { name: "E-norm127.5", norm_div: 127.5, ..BASE },
        // F. 其他：直接全图对齐（不等价于生产，仅作对照）
        PreConfig { name: "F-no-crop", use_crop: false, ..BASE },
        // B+C 交互：histeq+mask 同时关
        PreConfig { name: "BC-both-off", mask: false, histeq: false, ..BASE },
    ]
}

fn build_input_scaled(face: &RgbImage, offset: f32, div: f32) -> Vec<f32> {
    let n = (INPUT_SIZE * INPUT_SIZE) as usize;
    let mut input = Vec::with_capacity(3 * n);
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let p = face.get_pixel(x, y);
            input.push((p[2] as f32 - offset) / div);
        }
    }
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let p = face.get_pixel(x, y);
            input.push((p[1] as f32 - offset) / div);
        }
    }
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let p = face.get_pixel(x, y);
            input.push((p[0] as f32 - offset) / div);
        }
    }
    input
}

fn embed_scaled(session: &mut Session, face: &RgbImage, offset: f32, div: f32) -> Vec<f32> {
    let input_data = build_input_scaled(face, offset, div);
    let shape = [1_i64, 3, INPUT_SIZE as i64, INPUT_SIZE as i64];
    let input = Tensor::from_array((shape, input_data)).expect("tensor");
    let outputs = session.run(ort::inputs![input]).expect("arcface run");
    let (_shape, data) = outputs[0].try_extract_tensor::<f32>().expect("extract");
    l2_normalize(data)
}

/// crop(可选 margin) → align。返回 (RgbImage, source_crop_dims)。
fn crop_and_align_cfg(
    cfg: &PreConfig,
    s: &Sample,
) -> Result<(RgbImage, (u32, u32)), String> {
    let img = ImageData::from_file(std::path::Path::new(&s.path)).map_err(|e| e.to_string())?;
    let aligner = SimpleAligner::with_config(AlignmentConfig {
        align_scale: cfg.align_scale,
        align_y_offset: cfg.y_offset,
        use_ellipse_mask: cfg.mask,
        use_histogram_eq: cfg.histeq,
    });
    if !cfg.use_crop {
        let kps = FaceKeypoints {
            left_eye: s.kps[0],
            right_eye: s.kps[1],
            nose: s.kps[2],
            left_mouth: s.kps[3],
            right_mouth: s.kps[4],
        };
        let aligned = aligner.align(&img, &kps).map_err(|e| e.to_string())?;
        return Ok((aligned.image.as_rgb8(), (img.width(), img.height())));
    }
    let w = img.width() as f32;
    let h = img.height() as f32;
    let bw = s.bbox.w.max(1.0);
    let bh = s.bbox.h.max(1.0);
    let mx = bw * cfg.crop_margin;
    let my = bh * cfg.crop_margin;
    let x = (s.bbox.x - mx).max(0.0);
    let y = (s.bbox.y - my).max(0.0);
    let cw = (bw + 2.0 * mx).min(w - x);
    let ch = (bh + 2.0 * my).min(h - y);
    if cw < 1.0 || ch < 1.0 {
        return Err(format!("crop too small: {cw}x{ch}"));
    }
    let crop = img.crop(BBox::new(x, y, cw, ch)).map_err(|e| e.to_string())?;
    let kps = FaceKeypoints {
        left_eye: (s.kps[0].0 - x, s.kps[0].1 - y),
        right_eye: (s.kps[1].0 - x, s.kps[1].1 - y),
        nose: (s.kps[2].0 - x, s.kps[2].1 - y),
        left_mouth: (s.kps[3].0 - x, s.kps[3].1 - y),
        right_mouth: (s.kps[4].0 - x, s.kps[4].1 - y),
    };
    let aligned = aligner.align(&crop, &kps).map_err(|e| e.to_string())?;
    Ok((aligned.image.as_rgb8(), (cw as u32, ch as u32)))
}

#[test]
#[ignore]
fn alignment_ablation() {
    let samples = read_tsv();
    assert_eq!(samples.len(), 10, "expect 10 samples");
    let imgs: Vec<i64> = samples.iter().map(|s| s.img).collect();
    eprintln!("samples: {:?}", imgs);

    let mut session = Session::builder()
        .expect("builder")
        .commit_from_file(PathBuf::from(ARCFACE_MODEL))
        .expect("load arcface");

    // 每配置输出一块
    for cfg in variants() {
        let mut embs: Vec<Vec<f32>> = Vec::new();
        let mut db_cos_sum = 0.0f32;
        let mut db_cos_min = f32::MAX;
        for s in &samples {
            let (rgb, dims) = crop_and_align_cfg(&cfg, s).unwrap_or_else(|e| panic!("{}: {e}", s.img));
            let emb = embed_scaled(&mut session, &rgb, cfg.norm_offset, cfg.norm_div);
            let dbc = cosine(&emb, &s.db_emb);
            db_cos_sum += dbc;
            db_cos_min = db_cos_min.min(dbc);
            embs.push(emb);
            if s.img == 15 {
                eprintln!("[{}] img15 crop={}x{}", cfg.name, dims.0, dims.1);
            }
        }
        let db_cos_mean = db_cos_sum / samples.len() as f32;

        let i15 = imgs.iter().position(|x| *x == 15).unwrap();
        let idx = |i: i64| imgs.iter().position(|x| *x == i).unwrap();

        let p6 = cosine(&embs[i15], &embs[idx(6)]);
        let p13 = cosine(&embs[i15], &embs[idx(13)]);
        let p14 = cosine(&embs[i15], &embs[idx(14)]);
        let n2 = cosine(&embs[i15], &embs[idx(2)]);
        let n9 = cosine(&embs[i15], &embs[idx(9)]);
        let n20 = cosine(&embs[i15], &embs[idx(20)]);
        let j613 = cosine(&embs[idx(6)], &embs[idx(13)]);
        let j614 = cosine(&embs[idx(6)], &embs[idx(14)]);
        let j1314 = cosine(&embs[idx(13)], &embs[idx(14)]);

        let pos_min = p6.min(p13).min(p14);
        let pos_mean = (p6 + p13 + p14) / 3.0;
        let neg_max = n2.max(n9).max(n20);
        let margin = pos_min - neg_max;

        // full-set metrics
        let mut same_min = f32::INFINITY;
        let mut same_sum = 0.0f32;
        let mut neg_max_all = f32::NEG_INFINITY;
        let mut n_same = 0usize;
        for i in 0..imgs.len() {
            for j in (i + 1)..imgs.len() {
                let s = cosine(&embs[i], &embs[j]);
                if is_same(imgs[i], imgs[j]) {
                    same_min = same_min.min(s);
                    same_sum += s;
                    n_same += 1;
                } else {
                    neg_max_all = neg_max_all.max(s);
                }
            }
        }
        let same_mean = same_sum / n_same as f32;

        // img6 在 img15 邻居中的排名
        let mut rank: Vec<(i64, f32)> = imgs
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i15)
            .map(|(j, b)| (*b, cosine(&embs[i15], &embs[j])))
            .collect();
        rank.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let r6 = rank.iter().position(|(img, _)| *img == 6).unwrap() + 1;

        eprintln!(
            "[{}] pos6={:.3} pos13={:.3} pos14={:.3} | neg2={:.3} neg9={:.3} neg20={:.3} | posMin={:.3} posMean={:.3} negMax={:.3} margin={:+.3} | 6-13={:.3} 6-14={:.3} 13-14={:.3} | SameMin={:.3} SameMean={:.3} HardNegMax={:.3} Sep={:+.3} | img6rank#{} | dbCos={:.3}/min{:.3}",
            cfg.name, p6, p13, p14, n2, n9, n20, pos_min, pos_mean, neg_max, margin,
            j613, j614, j1314,
            same_min, same_mean, neg_max_all, same_min - neg_max_all, r6, db_cos_mean, db_cos_min
        );
    }

    eprintln!("\n=== 按 img15 margin 排序（越大越好；base 应为 -0.126）===");
    let mut rows: Vec<(String, f32)> = Vec::new();
    for cfg in variants() {
        let mut embs: Vec<Vec<f32>> = Vec::new();
        for s in &samples {
            let (rgb, _dims) = crop_and_align_cfg(&cfg, s).unwrap();
            embs.push(embed_scaled(&mut session, &rgb, cfg.norm_offset, cfg.norm_div));
        }
        let i15 = imgs.iter().position(|x| *x == 15).unwrap();
        let idx = |i: i64| imgs.iter().position(|x| *x == i).unwrap();
        let pos_min = [6, 13, 14].iter().map(|&i| cosine(&embs[i15], &embs[idx(i)])).fold(f32::INFINITY, f32::min);
        let neg_max = [2, 9, 20].iter().map(|&i| cosine(&embs[i15], &embs[idx(i)])).fold(f32::NEG_INFINITY, f32::max);
        rows.push((cfg.name.to_string(), pos_min - neg_max));
    }
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (name, m) in &rows {
        eprintln!("  {name:<18} margin={m:+.4}");
    }

    eprintln!("\n=== DONE ===");
}
