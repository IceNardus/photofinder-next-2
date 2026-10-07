//! 临时诊断（非生产）：SimpleAligner landmark 几何变换是否是 img15 漂移来源？
//!
//! 5 种输入模式，全部共用【同一 SCRFD 实时全精度 bbox/5pts】+【同一 w600k_r50】+
//! 【统一 112×112】+ 10 张图全部重新 embedding（禁止混用 DB embedding）+ 禁止 TTA：
//!   A  production        ：crop10% + SimpleAligner + histEq + ellipse mask
//!   B  bbox-only-10      ：bbox+10% 直接 resize 112（无 landmark / histEq / mask）
//!   C  bbox-only-20      ：bbox+20% 直接 resize 112（无 landmark / histEq / mask）
//!   D  square-bbox       ：bbox+10% → 正方形 crop + 必要时 padding → resize 112（无其它）
//!   E  square-bbox+histEq：D 完全一致，仅追加 production 直方图均衡（无 mask）
//!
//! 每模式输出：img15→9 样本 cosine；PosMin/PosMean/NegMax/Margin；jaqor 内部 3 对；
//! 0.65/0.70/0.75 三阈值归并判定；bbox/crop 元数据；112 图存 /tmp/img15_bbox_diag/。
//!
//! 全局成功条件（非只看 img15）：
//!   ① img15→jaqor PosMean ↑  ② img15→asian-man NegMax ↓
//!   ③ jaqor 内部 SameMin 不降 >0.03  ④ Margin 稳定 ↑  ⑤ 无 knife-edge
//! 若 B/C/D 任一项满足 → 对 bbox margin 扫 0/5/10/15/20/25/30% 出完整曲线。
//!
//! 用法：
//!   ORT_DYLIB_PATH=/Users/mac/lib/libonnxruntime.dylib \
//!   cargo test --release -p pf_ai --test bbox_vs_landmark -- --ignored --nocapture
//!
//! 只读库 / 模型推理。不改生产代码 / 不改 DB / 不重聚类。

use std::fs;
use std::path::{Path, PathBuf};

use image::{Rgb, RgbImage};
use ort::session::Session;
use ort::value::Tensor;

use pf_ai::face::aligner::{AlignmentConfig, SimpleAligner, REF_LANDMARKS_RAW};
use pf_ai::face::arcface::{l2_normalize, EMBEDDING_DIM, INPUT_SIZE};
use pf_ai::face::traits::FaceAligner;
use pf_ai::image_data::ImageData;
use pf_ai::{FaceDetector, ScrfdDetector};
use pf_core::{BBox, FaceKeypoints};

const ARCFACE_MODEL: &str = "/Users/mac/ai-project/photofinder-next-2/models/w600k_r50.onnx";
const SCRFD_MODEL: &str = "/Users/mac/ai-project/photofinder-next-2/models/scrfd_500m_bnkps.onnx";
const INPUT_TSV: &str = "/tmp/tta_img15/input.tsv";
const OUT_DIR: &str = "/tmp/img15_bbox_diag";

const PERSON2: [i64; 4] = [6, 13, 14, 15];
const NEG: [i64; 6] = [2, 9, 20, 17, 18, 19];

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

fn ref_landmarks(cfg: &AlignmentConfig) -> [(f32, f32); 5] {
    let mut out = REF_LANDMARKS_RAW;
    for p in &mut out {
        p.0 *= cfg.align_scale;
        p.1 = p.1 * cfg.align_scale + cfg.align_y_offset;
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

// ---- 模式元数据 ----

#[derive(Clone, Copy)]
struct ModeMeta {
    bbox: BBox,
    crop_x: f32,
    crop_y: f32,
    crop_w: f32,
    crop_h: f32,
    face_cx: f32,
    face_cy: f32,
    face_w: f32,
    face_h: f32,
    padded: bool,
}

/// production 直方图均衡（复刻 aligner::apply_histogram_equalization，作用于 112 图）。
fn apply_histeq(img: &mut RgbImage) {
    let n = (INPUT_SIZE * INPUT_SIZE) as usize;
    let mut gray = vec![0u32; n];
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let p = img.get_pixel(x, y);
            let lum = (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) as u32;
            gray[(y * INPUT_SIZE + x) as usize] = lum;
        }
    }
    let mut hist = [0u32; 256];
    for &v in &gray {
        hist[v as usize] += 1;
    }
    let mut cdf = [0u32; 256];
    cdf[0] = hist[0];
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + hist[i];
    }
    let mut cdf_min = 0;
    for i in 0..256 {
        if cdf[i] > 0 {
            cdf_min = cdf[i];
            break;
        }
    }
    let total = (INPUT_SIZE * INPUT_SIZE) as u32;
    let denom = total.saturating_sub(cdf_min);
    if denom == 0 {
        return;
    }
    let mut lut = [0u8; 256];
    for i in 0..256 {
        lut[i] = ((cdf[i] - cdf_min) as f32 / denom as f32 * 255.0).round() as u8;
    }
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let p = *img.get_pixel(x, y);
            let lum = 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
            let eq_lum = lut[lum.min(255.0) as usize] as f32;
            if lum > 0.0 {
                let ratio = eq_lum / lum;
                let r = (p[0] as f32 * ratio).clamp(0.0, 255.0) as u8;
                let g = (p[1] as f32 * ratio).clamp(0.0, 255.0) as u8;
                let b = (p[2] as f32 * ratio).clamp(0.0, 255.0) as u8;
                img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    A,
    B,
    C,
    D,
    E,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::A => "A_production",
            Mode::B => "B_bbox10",
            Mode::C => "C_bbox20",
            Mode::D => "D_square",
            Mode::E => "E_square_histeq",
        }
    }
    fn is_landmark(self) -> bool {
        self == Mode::A
    }
}

/// 生成某模式的 112×112 输入 + 元数据。
fn build_input(
    mode: Mode,
    aligner: &SimpleAligner,
    img: &ImageData,
    bbox: &BBox,
    kps: &[(f32, f32); 5],
    margin: f32,
) -> Result<(RgbImage, ModeMeta), String> {
    let meta = ModeMeta {
        bbox: *bbox,
        crop_x: 0.0,
        crop_y: 0.0,
        crop_w: 0.0,
        crop_h: 0.0,
        face_cx: bbox.x + bbox.w / 2.0,
        face_cy: bbox.y + bbox.h / 2.0,
        face_w: bbox.w,
        face_h: bbox.h,
        padded: false,
    };
    let mut m = meta;
    match mode {
        Mode::A => {
            let (crop, origin) = crop_to_bbox_with_margin(img, bbox, margin)?;
            m.crop_x = origin.0;
            m.crop_y = origin.1;
            m.crop_w = crop.width() as f32;
            m.crop_h = crop.height() as f32;
            let shifted = shift_kps(kps, origin);
            let aligned = aligner.align(&crop, &kps_from_array(&shifted)).map_err(|e| e.to_string())?;
            Ok((aligned.image.as_rgb8(), m))
        }
        Mode::B | Mode::C => {
            let (crop, origin) = crop_to_bbox_with_margin(img, bbox, margin)?;
            m.crop_x = origin.0;
            m.crop_y = origin.1;
            m.crop_w = crop.width() as f32;
            m.crop_h = crop.height() as f32;
            let rgb = image::imageops::resize(
                &crop.as_rgb8(),
                INPUT_SIZE,
                INPUT_SIZE,
                image::imageops::FilterType::Lanczos3,
            );
            Ok((rgb, m))
        }
        Mode::D | Mode::E => {
            // bbox 扩展 margin → 正方形 crop（以 bbox 中心为中心）→ 越界 padding 黑色 → resize
            let bw = bbox.w.max(1.0);
            let bh = bbox.h.max(1.0);
            let mx = bw * margin;
            let my = bh * margin;
            let ex0 = bbox.x - mx;
            let ey0 = bbox.y - my;
            let ew = bw + 2.0 * mx;
            let eh = bh + 2.0 * my;
            let s = ew.max(eh);
            let sq0x = (ex0 + ew / 2.0) - s / 2.0;
            let sq0y = (ey0 + eh / 2.0) - s / 2.0;
            let iw = img.width() as f32;
            let ih = img.height() as f32;
            let cl0x = sq0x.max(0.0);
            let cl0y = sq0y.max(0.0);
            let cl1x = (sq0x + s).min(iw);
            let cl1y = (sq0y + s).min(ih);
            m.padded = cl0x > sq0x || cl0y > sq0y || cl1x < sq0x + s || cl1y < sq0y + s;
            m.crop_x = cl0x;
            m.crop_y = cl0y;
            m.crop_w = cl1x - cl0x;
            m.crop_h = cl1y - cl0y;
            let offx = (cl0x - sq0x) as u32;
            let offy = (cl0y - sq0y) as u32;
            let cw = (cl1x - cl0x) as u32;
            let ch = (cl1y - cl0y) as u32;
            let irgb = img.as_rgb8();
            let mut canvas = RgbImage::from_pixel(s as u32, s as u32, Rgb([0, 0, 0]));
            for py in 0..ch {
                for px in 0..cw {
                    let p = *irgb.get_pixel((cl0x + px as f32) as u32, (cl0y + py as f32) as u32);
                    canvas.put_pixel(offx + px, offy + py, p);
                }
            }
            let mut rgb = image::imageops::resize(
                &canvas,
                INPUT_SIZE,
                INPUT_SIZE,
                image::imageops::FilterType::Lanczos3,
            );
            if mode == Mode::E {
                apply_histeq(&mut rgb);
            }
            Ok((rgb, m))
        }
    }
}

#[test]
#[ignore]
fn bbox_vs_landmark() {
    fs::create_dir_all(OUT_DIR).expect("mkdir");
    let samples = read_tsv();
    assert_eq!(samples.len(), 10, "expect 10 samples");
    let cfg = AlignmentConfig::default();
    let aligner = SimpleAligner::with_config(cfg.clone());
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

    // 实时全精度 bbox/kps（= DB 存储值）
    struct Geo {
        bbox: BBox,
        kps: [(f32, f32); 5],
    }
    let mut geos = Vec::new();
    for s in &samples {
        let i = samples.iter().position(|x| x.img == s.img).unwrap();
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

    let modes = [Mode::A, Mode::B, Mode::C, Mode::D, Mode::E];
    let margins: Vec<(Mode, f32)> = vec![
        (Mode::A, 0.10),
        (Mode::B, 0.10),
        (Mode::C, 0.20),
        (Mode::D, 0.10),
        (Mode::E, 0.10),
    ];

    struct Result {
        mode: Mode,
        margin: f32,
        embs: Vec<(i64, Vec<f32>)>,
        metas: Vec<ModeMeta>,
    }
    let mut results = Vec::new();

    for (mi, mode) in modes.iter().enumerate() {
        let margin = margins[mi].1;
        let mut embs = Vec::new();
        let mut metas = Vec::new();
        for (si, s) in samples.iter().enumerate() {
            let (rgb, meta) = build_input(*mode, &aligner, &images[si], &geos[si].bbox, &geos[si].kps, margin)
                .expect("build input");
            let e = embed_one(&mut session, &rgb);
            embs.push((s.img, e));
            metas.push(meta);
            // 保存 112×112
            rgb.save(Path::new(OUT_DIR).join(format!("{}_img{}.png", mode.name(), s.img)))
                .expect("save");
        }
        results.push(Result {
            mode: *mode,
            margin,
            embs,
            metas,
        });
    }

    // ---- 逐模式输出 ----
    let get = |res: &Result, x: i64| res.embs.iter().find(|(i, _)| *i == x).unwrap().1.clone();
    let mut summary = String::new();
    summary.push_str("Mode,Margin,PosMin,PosMean,NegMax,MarginValue,img15-6,img15-13,img15-14,img15-2,img15-9,img15-20,SameMin6-13-14,DeltaPosMean,DeltaNegMax,DeltaMargin,DeltaSameMin,PASS\n");
    let base = &results[0];
    let bget = |x: i64| base.embs.iter().find(|(i, _)| *i == x).unwrap().1.clone();
    let base_pos_mean = {
        let p: Vec<f32> = [6, 13, 14].iter().map(|&t| cosine(&bget(15), &bget(t))).collect();
        p.iter().sum::<f32>() / p.len() as f32
    };
    let base_neg_max = NEG.iter().map(|&n| cosine(&bget(15), &bget(n))).fold(f32::NEG_INFINITY, f32::max);
    let base_same_min = [6, 13, 14]
        .iter()
        .enumerate()
        .flat_map(|(i, &a)| [6, 13, 14][i + 1..].iter().map(move |&b| (a, b)))
        .map(|(a, b)| cosine(&bget(a), &bget(b)))
        .fold(f32::INFINITY, f32::min);
    let base_margin = {
        let p: Vec<f32> = [6, 13, 14].iter().map(|&t| cosine(&bget(15), &bget(t))).collect();
        let pm = p.iter().cloned().fold(f32::INFINITY, f32::min);
        pm - base_neg_max
    };

    for res in &results {
        let e15 = get(res, 15);
        let mut pos = Vec::new();
        let mut neg = Vec::new();
        for &t in &[6, 13, 14] {
            pos.push(cosine(&e15, &get(res, t)));
        }
        for &n in &NEG {
            neg.push(cosine(&e15, &get(res, n)));
        }
        let pos_min = pos.iter().cloned().fold(f32::INFINITY, f32::min);
        let pos_mean = pos.iter().sum::<f32>() / pos.len() as f32;
        let neg_max = neg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let margin = pos_min - neg_max;
        // jaqor 内部
        let same = [cosine(&get(res, 6), &get(res, 13)), cosine(&get(res, 6), &get(res, 14)), cosine(&get(res, 13), &get(res, 14))];
        let same_min = same.iter().cloned().fold(f32::INFINITY, f32::min);

        eprintln!("\n--- {} (margin {:.0}%) ---", res.mode.name(), res.margin * 100.0);
        eprintln!("  img15→ 6={:.4} 13={:.4} 14={:.4} | 2={:.4} 9={:.4} 20={:.4} | 17={:.4} 18={:.4} 19={:.4}",
            pos[0], pos[1], pos[2], neg[0], neg[1], neg[2], neg[3], neg[4], neg[5]);
        eprintln!("  PosMin={:.4} PosMean={:.4} NegMax={:.4} Margin={:+.4}", pos_min, pos_mean, neg_max, margin);
        eprintln!("  jaqor 内部: 6-13={:.4} 6-14={:.4} 13-14={:.4}  SameMin={:.4}", same[0], same[1], same[2], same_min);

        // 元数据（以 img15 为代表，附整组 crop 尺寸范围）
        let m15 = &res.metas[samples.iter().position(|s| s.img == 15).unwrap()];
        eprintln!("  [img15] bbox=({:.1},{:.1},{:.1},{:.1}) center=({:.1},{:.1}) face={:.0}x{:.0} crop=({:.1},{:.1}) {}x{} padded={}",
            m15.bbox.x, m15.bbox.y, m15.bbox.w, m15.bbox.h, m15.face_cx, m15.face_cy, m15.face_w, m15.face_h,
            m15.crop_x, m15.crop_y, m15.crop_w, m15.crop_h, m15.padded);
        // 是否发生 padding（任一图）
        let any_pad = res.metas.iter().any(|m| m.padded);
        if any_pad {
            let pads: Vec<String> = samples
                .iter()
                .zip(res.metas.iter())
                .filter(|(_, m)| m.padded)
                .map(|(s, _)| format!("img{}", s.img))
                .collect();
            eprintln!("  [组] 发生 padding 的图: {}", pads.join(","));
        }

        // 0.65/0.70/0.75 归并判定
        let max_p2 = [6, 13, 14].iter().map(|&t| cosine(&e15, &get(res, t))).fold(f32::NEG_INFINITY, f32::max);
        let max_p1 = NEG.iter().map(|&n| cosine(&e15, &get(res, n))).fold(f32::NEG_INFINITY, f32::max);
        eprintln!("  归并判定  maxP2(img15→person2)={max_p2:.4}  maxP1(img15→asian-man)={max_p1:.4}:");
        for &tau in &[0.65f32, 0.70, 0.75] {
            let verdict = if max_p1 >= tau && max_p2 >= tau {
                "AMBIG(双超)"
            } else if max_p2 >= tau {
                "JOIN_P2"
            } else if max_p1 >= tau {
                "WRONG_JOIN_P1"
            } else {
                "SINGLETON"
            };
            eprintln!("    τ={tau:.2}: {verdict}");
        }

        // 全局成功条件 vs A
        let dpos = pos_mean - base_pos_mean;
        let dneg = neg_max - base_neg_max;
        let dmargin = margin - base_margin;
        let dsame = same_min - base_same_min;
        let pass = dpos > 0.0 && dneg < 0.0 && dmargin > 0.0 && dsame >= -0.03;
        eprintln!("  vs A: ΔPosMean={dpos:+.4} ΔNegMax={dneg:+.4} ΔMargin={dmargin:+.4} ΔSameMin={dsame:+.4}  [{}]",
            if pass { "PASS" } else { "NO" });
        summary.push_str(&format!(
            "{},{:.0}%,{:.4},{:.4},{:.4},{:+.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:+.4},{:+.4},{:+.4},{:+.4},{}\n",
            res.mode.name(), res.margin * 100.0, pos_min, pos_mean, neg_max, margin,
            pos[0], pos[1], pos[2], neg[0], neg[1], neg[2], same_min,
            dpos, dneg, dmargin, dsame, if pass { "PASS" } else { "NO" },
        ));
    }
    fs::write(Path::new(OUT_DIR).join("modes_summary.csv"), &summary).expect("write csv");

    // ---- 触发 margin 扫描：B/C/D 任一满足 PASS ----
    eprintln!("\n=== Margin 扫描触发判定（B/C/D 任一 PASS）===");
    let trig: Vec<Mode> = [Mode::B, Mode::C, Mode::D]
        .into_iter()
        .filter(|&m| {
            results.iter().find(|r| r.mode == m).map(|r| {
                let e15 = get(r, 15);
                let p: Vec<f32> = [6, 13, 14].iter().map(|&t| cosine(&e15, &get(r, t))).collect();
                let pm = p.iter().cloned().fold(f32::INFINITY, f32::min);
                let pmean = p.iter().sum::<f32>() / p.len() as f32;
                let nm = NEG.iter().map(|&n| cosine(&e15, &get(r, n))).fold(f32::NEG_INFINITY, f32::max);
                let sm = [6, 13, 14].iter().enumerate().flat_map(|(i, &a)| [6, 13, 14][i + 1..].iter().map(move |&b| (a, b)))
                    .map(|(a, b)| cosine(&get(r, a), &get(r, b))).fold(f32::INFINITY, f32::min);
                let dmargin = (pm - nm) - base_margin;
                (pmean - base_pos_mean > 0.0) && (nm - base_neg_max < 0.0) && (dmargin > 0.0) && (sm - base_same_min >= -0.03)
            }).unwrap_or(false)
        })
        .collect();
    eprintln!("  触发扫描的模式: {}", if trig.is_empty() { "无（B/C/D 均未满足全局成功条件）".to_string() } else { trig.iter().map(|m| m.name()).collect::<Vec<_>>().join(", ") });

    if !trig.is_empty() {
        let mut curve = String::new();
        curve.push_str("Mode,MarginPct,PosMin,PosMean,NegMax,MarginValue,SameMin,img15-6,img15-13,img15-14\n");
        for &m in &trig {
            eprintln!("\n--- {}({}) bbox margin 扫描 0..30% ---", m.name(), if m.is_landmark() { "landmark 对齐" } else { "无 landmark" });
            eprintln!("{:<6} {:>7} {:>7} {:>7} {:>7} {:>9}", "margin", "PosMin", "PosMean", "NegMax", "Margin", "SameMin");
            for pct in [0.0f32, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30] {
                let mut embs = Vec::new();
                for (si, s) in samples.iter().enumerate() {
                    let (rgb, _) = build_input(m, &aligner, &images[si], &geos[si].bbox, &geos[si].kps, pct).expect("build");
                    embs.push((s.img, embed_one(&mut session, &rgb)));
                }
                let g = |x: i64| embs.iter().find(|(i, _)| *i == x).unwrap().1.clone();
                let e15 = g(15);
                let p: Vec<f32> = [6, 13, 14].iter().map(|&t| cosine(&e15, &g(t))).collect();
                let pm = p.iter().cloned().fold(f32::INFINITY, f32::min);
                let pmean = p.iter().sum::<f32>() / p.len() as f32;
                let nm = NEG.iter().map(|&n| cosine(&e15, &g(n))).fold(f32::NEG_INFINITY, f32::max);
                let sm = [6, 13, 14].iter().enumerate().flat_map(|(i, &a)| [6, 13, 14][i + 1..].iter().map(move |&b| (a, b)))
                    .map(|(a, b)| cosine(&g(a), &g(b))).fold(f32::INFINITY, f32::min);
                eprintln!("{:<6} {:>7.4} {:>7.4} {:>7.4} {:>+7.4} {:>9.4}", format!("{:.0}%", pct * 100.0), pm, pmean, nm, pm - nm, sm);
                curve.push_str(&format!("{},{:.0}%,{:.4},{:.4},{:.4},{:+.4},{:.4},{:.4},{:.4},{:.4}\n",
                    m.name(), pct * 100.0, pm, pmean, nm, pm - nm, sm, p[0], p[1], p[2]));
            }
        }
        fs::write(Path::new(OUT_DIR).join("margin_scan.csv"), &curve).expect("write margin csv");
    }

    eprintln!("\n=== 参考（A 基线）===");
    eprintln!("  A: PosMin={:.4} PosMean={base_pos_mean:.4} NegMax={base_neg_max:.4} Margin={base_margin:+.4} SameMin={base_same_min:.4}",
        {
            let e15 = bget(15);
            let p: Vec<f32> = [6, 13, 14].iter().map(|&t| cosine(&e15, &bget(t))).collect();
            p.iter().cloned().fold(f32::INFINITY, f32::min)
        });
    eprintln!("=== DONE === 输出目录 {OUT_DIR}");
}
