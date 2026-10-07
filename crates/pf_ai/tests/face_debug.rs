//! 临时 debug 测试 — 复现"两张不同图得到相同 ArcFace embedding"的问题。
//!
//! 用法:`cargo test --release -p pf_ai --test face_debug -- --ignored --nocapture`

use std::path::PathBuf;
use std::sync::Arc;

use pf_ai::image_data::ImageData;
use pf_ai::{ArcFaceEmbedder, FaceAligner, FaceDetector, FaceEmbedder, FacePipeline, QualityFilter,
            ScrfdDetector, SimpleAligner};

const FACE_MODEL: &str = "/Users/mac/Library/Application Support/PhotoFinderNext/resources/models/w600k_r50.onnx";
const SCRFD_MODEL: &str = "/Users/mac/Library/Application Support/PhotoFinderNext/resources/models/scrfd_500m_bnkps.onnx";

fn bytes_to_image(bytes: &[u8]) -> ImageData {
    ImageData::from_bytes(bytes).expect("decode")
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// Test 1: 仅 embedder(已知是好的) — sanity check
#[test]
#[ignore]
fn debug_embedder_only() {
    let model = ArcFaceEmbedder::load(&PathBuf::from(FACE_MODEL)).expect("load");
    let bytes_a = std::fs::read("/Users/mac/Downloads/pexels-jaqor-33601831.jpg").unwrap();
    let bytes_b = std::fs::read("/Users/mac/Downloads/pexels-jaqor-33601835.jpg").unwrap();
    let bytes_c = std::fs::read("/Users/mac/Downloads/pexels-jaqor-33601811.jpg").unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mk = |b: &[u8]| pf_ai::AlignedFace {
        image: bytes_to_image(b),
        original_bbox: pf_core::BBox::new(0.0, 0.0, 0.0, 0.0),
    };

    let a = rt.block_on(model.embed(&mk(&bytes_a))).unwrap();
    let b = rt.block_on(model.embed(&mk(&bytes_b))).unwrap();
    let c = rt.block_on(model.embed(&mk(&bytes_c))).unwrap();

    eprintln!("=== Embedder-only on full jaqor images ===");
    eprintln!("cos(a,b) = {:.4}", cosine(&a.values, &b.values));
    eprintln!("cos(a,c) = {:.4}", cosine(&a.values, &c.values));
    eprintln!("cos(b,c) = {:.4}", cosine(&b.values, &c.values));

    assert!(cosine(&a.values, &b.values) < 0.9999);
    assert!(cosine(&a.values, &c.values) < 0.9999);
    assert!(cosine(&b.values, &c.values) < 0.9999);
}

/// Test 2: full pipeline (detect → align → embed) — 看完整链路
#[test]
#[ignore]
fn debug_full_pipeline() {
    let scrfd = ScrfdDetector::load(&PathBuf::from(SCRFD_MODEL)).expect("load scrfd");
    let arcface = ArcFaceEmbedder::load(&PathBuf::from(FACE_MODEL)).expect("load arcface");
    let aligner = Arc::new(SimpleAligner::new());
    let qf = QualityFilter::from_config(0.5, 20, 0.5, 90.0);
    let pipeline = FacePipeline::new(scrfd, aligner, arcface, qf);

    let bytes_a = std::fs::read("/Users/mac/Downloads/pexels-jaqor-33601831.jpg").unwrap();
    let bytes_b = std::fs::read("/Users/mac/Downloads/pexels-jaqor-33601835.jpg").unwrap();
    let bytes_c = std::fs::read("/Users/mac/Downloads/pexels-jaqor-33601811.jpg").unwrap();

    let img_a = bytes_to_image(&bytes_a);
    let img_b = bytes_to_image(&bytes_b);
    let img_c = bytes_to_image(&bytes_c);

    let rt = tokio::runtime::Runtime::new().unwrap();
    let feat_a = rt.block_on(pipeline.process(&img_a)).unwrap();
    let feat_b = rt.block_on(pipeline.process(&img_b)).unwrap();
    let feat_c = rt.block_on(pipeline.process(&img_c)).unwrap();

    eprintln!("=== Full pipeline output ===");
    if let Some(f) = feat_a.first() {
        eprintln!("image A: {} faces, bbox = ({:.0},{:.0},{}x{})",
            feat_a.len(),
            f.detection.bbox.x, f.detection.bbox.y,
            f.detection.bbox.w as i32, f.detection.bbox.h as i32);
    } else {
        eprintln!("image A: 0 faces");
    }
    if let Some(f) = feat_b.first() {
        eprintln!("image B: {} faces, bbox = ({:.0},{:.0},{}x{})",
            feat_b.len(),
            f.detection.bbox.x, f.detection.bbox.y,
            f.detection.bbox.w as i32, f.detection.bbox.h as i32);
    } else {
        eprintln!("image B: 0 faces (filtered out, e.g. tilted)");
    }
    if let Some(f) = feat_c.first() {
        eprintln!("image C: {} faces, bbox = ({:.0},{:.0},{}x{})",
            feat_c.len(),
            f.detection.bbox.x, f.detection.bbox.y,
            f.detection.bbox.w as i32, f.detection.bbox.h as i32);
    } else {
        eprintln!("image C: 0 faces");
    }

    let emb_a = &feat_a[0].embedding.values;
    let emb_b = if feat_b.is_empty() { &vec![][..] } else { &feat_b[0].embedding.values };
    let emb_c = if feat_c.is_empty() { &vec![][..] } else { &feat_c[0].embedding.values };

    eprintln!("cos(A,B) = {:.4}", cosine(emb_a, emb_b));
    eprintln!("cos(A,C) = {:.4}", cosine(emb_a, emb_c));
    eprintln!("cos(B,C) = {:.4}", cosine(emb_b, emb_c));

    eprintln!("keypoint A left_eye: ({:.1},{:.1})", feat_a[0].detection.keypoints.left_eye.0, feat_a[0].detection.keypoints.left_eye.1);
    if let Some(f) = feat_b.first() {
        eprintln!("keypoint B left_eye: ({:.1},{:.1})", f.detection.keypoints.left_eye.0, f.detection.keypoints.left_eye.1);
    }
    if let Some(f) = feat_c.first() {
        eprintln!("keypoint C left_eye: ({:.1},{:.1})", f.detection.keypoints.left_eye.0, f.detection.keypoints.left_eye.1);
    }

    // Bug guard: A/B/C 必须有不同的 embedding（face_id 边界保证）。
    // B/C 可能被 pose filter 拒绝(face=[])——只对非空集合做断言。
    assert!(cosine(emb_a, emb_a) > 0.9999, "self-similarity sanity");
    if !feat_b.is_empty() {
        assert!(cosine(emb_a, emb_b) < 0.9999, "A == B (BUG in detect/align)");
    }
    if !feat_c.is_empty() {
        assert!(cosine(emb_a, emb_c) < 0.9999, "A == C (BUG)");
    }
    if !feat_b.is_empty() && !feat_c.is_empty() {
        assert!(cosine(emb_b, emb_c) < 0.9999, "B == C (BUG)");
    }
}

/// Test 3: 隔离 detector — 看 detector 返回的 bboxes 和 keypoints 是否相同
#[test]
#[ignore]
fn debug_scrfd_only() {
    let scrfd = ScrfdDetector::load(&PathBuf::from(SCRFD_MODEL)).expect("load");
    let bytes_a = std::fs::read("/Users/mac/Downloads/pexels-jaqor-33601831.jpg").unwrap();
    let bytes_b = std::fs::read("/Users/mac/Downloads/pexels-jaqor-33601835.jpg").unwrap();
    let bytes_c = std::fs::read("/Users/mac/Downloads/pexels-jaqor-33601811.jpg").unwrap();
    let img_a = bytes_to_image(&bytes_a);
    let img_b = bytes_to_image(&bytes_b);
    let img_c = bytes_to_image(&bytes_c);

    let rt = tokio::runtime::Runtime::new().unwrap();
    let det_a = rt.block_on(scrfd.detect(&img_a)).unwrap();
    let det_b = rt.block_on(scrfd.detect(&img_b)).unwrap();
    let det_c = rt.block_on(scrfd.detect(&img_c)).unwrap();

    eprintln!("=== SCRFD detector only ===");
    for (name, dets) in [("A", &det_a), ("B", &det_b), ("C", &det_c)] {
        eprintln!("img {}: {} faces", name, dets.len());
        for d in dets {
            eprintln!("  bbox=({:.0},{:.0},{}x{}) score={:.3} kps.l_eye=({:.1},{:.1})",
                d.bbox.x, d.bbox.y, d.bbox.w as i32, d.bbox.h as i32, d.score,
                d.keypoints.left_eye.0, d.keypoints.left_eye.1);
        }
    }

    assert!(det_a.len() >= 1 && det_b.len() >= 1 && det_c.len() >= 1);
    let ba = &det_a[0].bbox;
    let bb = &det_b[0].bbox;
    let bc = &det_c[0].bbox;
    eprintln!("bbox_a: ({:.0},{:.0},{}x{})", ba.x, ba.y, ba.w as i32, ba.h as i32);
    eprintln!("bbox_b: ({:.0},{:.0},{}x{})", bb.x, bb.y, bb.w as i32, bb.h as i32);
    eprintln!("bbox_c: ({:.0},{:.0},{}x{})", bc.x, bc.y, bc.w as i32, bc.h as i32);
    assert_ne!((ba.x as i32, ba.y as i32), (bb.x as i32, bb.y as i32), "bbox positions identical");
}

/// Test 4: aligner 单独跑 — 看不同 keypoints 是否产生不同 aligned face
#[test]
#[ignore]
fn debug_aligner_only() {
    let aligner = SimpleAligner::new();

    let bytes_a = std::fs::read("/Users/mac/Downloads/pexels-jaqor-33601831.jpg").unwrap();
    let bytes_b = std::fs::read("/Users/mac/Downloads/pexels-jaqor-33601835.jpg").unwrap();
    let img_a = bytes_to_image(&bytes_a);
    let img_b = bytes_to_image(&bytes_b);

    let kps_a = pf_core::FaceKeypoints {
        left_eye: (3860.8, 2364.9),
        right_eye: (4168.0, 2334.0),
        nose: (4037.4, 2495.6),
        left_mouth: (3907.6, 2621.1),
        right_mouth: (4202.6, 2581.0),
    };
    let kps_b = pf_core::FaceKeypoints {
        left_eye: (3869.1, 2981.0),
        right_eye: (4149.5, 2962.2),
        nose: (4031.5, 3133.7),
        left_mouth: (3876.5, 3270.4),
        right_mouth: (4149.5, 3234.3),
    };

    let aligned_a = aligner.align(&img_a, &kps_a).expect("align a");
    let aligned_b = aligner.align(&img_b, &kps_b).expect("align b");

    let rgb_a = aligned_a.image.as_rgb8();
    let rgb_b = aligned_b.image.as_rgb8();
    eprintln!("=== Aligner only ===");
    eprintln!("aligned_a dimensions: {}x{}", rgb_a.width(), rgb_a.height());
    eprintln!("aligned_b dimensions: {}x{}", rgb_b.width(), rgb_b.height());

    let bytes_a_aligned: Vec<u8> = rgb_a.as_raw().clone();
    let bytes_b_aligned: Vec<u8> = rgb_b.as_raw().clone();
    let hash_a = blake3_short(&bytes_a_aligned);
    let hash_b = blake3_short(&bytes_b_aligned);
    eprintln!("aligned_a hash: {} ({} bytes)", hash_a, bytes_a_aligned.len());
    eprintln!("aligned_b hash: {} ({} bytes)", hash_b, bytes_b_aligned.len());
    eprintln!("aligned bytes equal? {}", bytes_a_aligned == bytes_b_aligned);
}

/// Test 5: aligner 用 same image 但 different keypoints
#[test]
#[ignore]
fn debug_aligner_same_image_diff_kps() {
    let aligner = SimpleAligner::new();

    let bytes = std::fs::read("/Users/mac/Downloads/pexels-jaqor-33601831.jpg").unwrap();
    let img = bytes_to_image(&bytes);

    let kps_1 = pf_core::FaceKeypoints {
        left_eye: (3860.8, 2364.9),
        right_eye: (4168.0, 2334.0),
        nose: (4037.4, 2495.6),
        left_mouth: (3907.6, 2621.1),
        right_mouth: (4202.6, 2581.0),
    };
    let kps_2 = pf_core::FaceKeypoints {
        left_eye: (3869.1, 2981.0),
        right_eye: (4149.5, 2962.2),
        nose: (4031.5, 3133.7),
        left_mouth: (3876.5, 3270.4),
        right_mouth: (4149.5, 3234.3),
    };

    let aligned_1 = aligner.align(&img, &kps_1).expect("align 1");
    let aligned_2 = aligner.align(&img, &kps_2).expect("align 2");

    let rgb_1 = aligned_1.image.as_rgb8();
    let rgb_2 = aligned_2.image.as_rgb8();
    let bytes_1: Vec<u8> = rgb_1.as_raw().clone();
    let bytes_2: Vec<u8> = rgb_2.as_raw().clone();
    eprintln!("=== Aligner same-image diff-kps ===");
    eprintln!("aligned_1 first 32 bytes: {:?}", &bytes_1[..32]);
    eprintln!("aligned_2 first 32 bytes: {:?}", &bytes_2[..32]);
    eprintln!("aligned bytes equal? {}", bytes_1 == bytes_2);
    assert_ne!(bytes_1, bytes_2, "BUG: aligner produces same output for different keypoints");
}

/// Test 6: 手动逐级 detect → align → embed,看 aligned bytes hash
#[test]
#[ignore]
fn debug_manual_pipeline() {
    let scrfd = ScrfdDetector::load(&PathBuf::from(SCRFD_MODEL)).expect("load scrfd");
    let arcface = ArcFaceEmbedder::load(&PathBuf::from(FACE_MODEL)).expect("load arcface");
    let aligner = SimpleAligner::new();
    let rt = tokio::runtime::Runtime::new().unwrap();

    let bytes_a = std::fs::read("/Users/mac/Downloads/pexels-jaqor-33601831.jpg").unwrap();
    let bytes_b = std::fs::read("/Users/mac/Downloads/pexels-jaqor-33601835.jpg").unwrap();
    let img_a = bytes_to_image(&bytes_a);
    let img_b = bytes_to_image(&bytes_b);

    let det_a = rt.block_on(scrfd.detect(&img_a)).unwrap();
    let det_b = rt.block_on(scrfd.detect(&img_b)).unwrap();
    let kps_a = det_a[0].keypoints;
    let kps_b = det_b[0].keypoints;
    eprintln!("A kps = le=({:.1},{:.1}) re=({:.1},{:.1}) n=({:.1},{:.1}) lm=({:.1},{:.1}) rm=({:.1},{:.1})",
        kps_a.left_eye.0, kps_a.left_eye.1,
        kps_a.right_eye.0, kps_a.right_eye.1,
        kps_a.nose.0, kps_a.nose.1,
        kps_a.left_mouth.0, kps_a.left_mouth.1,
        kps_a.right_mouth.0, kps_a.right_mouth.1);
    eprintln!("B kps = le=({:.1},{:.1}) re=({:.1},{:.1}) n=({:.1},{:.1}) lm=({:.1},{:.1}) rm=({:.1},{:.1})",
        kps_b.left_eye.0, kps_b.left_eye.1,
        kps_b.right_eye.0, kps_b.right_eye.1,
        kps_b.nose.0, kps_b.nose.1,
        kps_b.left_mouth.0, kps_b.left_mouth.1,
        kps_b.right_mouth.0, kps_b.right_mouth.1);

    let aligned_a = aligner.align(&img_a, &kps_a).unwrap();
    let aligned_b = aligner.align(&img_b, &kps_b).unwrap();

    let rgb_a: image::RgbImage = aligned_a.image.as_rgb8();
    let rgb_b: image::RgbImage = aligned_b.image.as_rgb8();
    let raw_a = rgb_a.as_raw().clone();
    let raw_b = rgb_b.as_raw().clone();
    let hash_a = blake3_short(&raw_a);
    let hash_b = blake3_short(&raw_b);
    eprintln!("aligned_a hash: {} ({} bytes)", hash_a, raw_a.len());
    eprintln!("aligned_b hash: {} ({} bytes)", hash_b, raw_b.len());
    eprintln!("aligned bytes equal? {}", raw_a == raw_b);

    let emb_a = rt.block_on(arcface.embed(&aligned_a)).unwrap();
    let emb_b = rt.block_on(arcface.embed(&aligned_b)).unwrap();
    eprintln!("cos(A,B) from manual pipeline = {:.6}", cosine(&emb_a.values, &emb_b.values));
}

fn blake3_short(v: &[u8]) -> String {
    let h = blake3::hash(v);
    h.to_hex()[..16].to_string()
}

/// Test 7: 不走 quality filter，直接 detect → align → embed，并打印每个 face
/// 的所有质量指标。目的是诊断「为何 Image B 被过滤」「为何 Image A/C cosine 低」。
#[test]
#[ignore]
fn debug_unfiltered_quality_breakdown() {
    use pf_ai::face::pose::estimate_yaw_pitch_roll;
    use pf_ai::quality::{assess, eye_distance, pose_score};

    let scrfd = ScrfdDetector::load(&PathBuf::from(SCRFD_MODEL)).expect("load scrfd");
    // 生产默认已改 1-crop，这里显式开启 6-crop 作为 TTA 对比参照。
    let arcface = Arc::new(
        Arc::try_unwrap(ArcFaceEmbedder::load(&PathBuf::from(FACE_MODEL)).expect("load arcface"))
            .ok()
            .expect("arc has multiple owners")
            .with_flip(true)
            .with_rotation_tta(true),
    );
    let arcface_no_tta = ArcFaceEmbedder::load(&PathBuf::from(FACE_MODEL)).expect("load arcface2");
    let aligner = Arc::new(SimpleAligner::new());
    let rt = tokio::runtime::Runtime::new().unwrap();

    let paths = [
        ("A", "/Users/mac/Downloads/pexels-jaqor-33601831.jpg"),
        ("B", "/Users/mac/Downloads/pexels-jaqor-33601835.jpg"),
        ("C", "/Users/mac/Downloads/pexels-jaqor-33601811.jpg"),
    ];

    eprintln!("=== Unfiltered quality breakdown (no quality filter applied) ===");
    eprintln!("Thresholds: detector≥0.30 area≥0.15 eye_dist≥12 pose≥0.20 quality≥0.45");

    let mut all_embs: Vec<(char, Vec<f32>)> = Vec::new();
    let mut all_embs_no_tta: Vec<(char, Vec<f32>)> = Vec::new();

    for (name, p) in &paths {
        let bytes = std::fs::read(p).unwrap();
        let img = bytes_to_image(&bytes);
        let dets = rt.block_on(scrfd.detect(&img)).unwrap();
        eprintln!("\n--- Image {name} ({} faces) ---", dets.len());
        for (i, d) in dets.iter().enumerate() {
            let aligned = aligner.align(&img, &d.keypoints).unwrap();
            let gray = aligned.image.to_luma8();
            let blur = pf_ai::quality::blur_score_from_aligned(&gray);
            let q = assess(d, blur, estimate_yaw_pitch_roll(&d.keypoints));
            let ed = eye_distance(&d.keypoints);
            let ps = pose_score(&d.keypoints);

            eprintln!(
                "  [{name}#{i}] bbox=({:.0},{:.0},{}x{}) det_score={:.3}\n    \
                 eye_dist={:.1} pose={:.3} (tilt={:.3} eye_lvl={:.3}) face_area={:.3} blur={:.3} quality={:.3}\n    \
                 kps=le({:.0},{:.0}) re({:.0},{:.0}) n({:.0},{:.0}) lm({:.0},{:.0}) rm({:.0},{:.0})",
                d.bbox.x, d.bbox.y, d.bbox.w as i32, d.bbox.h as i32, d.score,
                ed, ps,
                (d.keypoints.right_eye.1 - d.keypoints.left_eye.1).abs() / (d.keypoints.right_eye.0 - d.keypoints.left_eye.0).abs().max(1.0),
                (d.keypoints.left_eye.1 - d.keypoints.right_eye.1).abs() / 50.0,
                q.face_area_score, q.blur_score, q.quality,
                d.keypoints.left_eye.0, d.keypoints.left_eye.1,
                d.keypoints.right_eye.0, d.keypoints.right_eye.1,
                d.keypoints.nose.0, d.keypoints.nose.1,
                d.keypoints.left_mouth.0, d.keypoints.left_mouth.1,
                d.keypoints.right_mouth.0, d.keypoints.right_mouth.1,
            );

            let reasons = [
                ("detector", q.detector_score, 0.30_f32),
                ("face_area", q.face_area_score, 0.15_f32),
                ("eye_dist ", ed, 12.0_f32),
                ("pose     ", ps, 0.20_f32),
                ("quality  ", q.quality, 0.45_f32),
            ];
            let fail: Vec<&str> = reasons.iter()
                .filter(|(_, v, t)| v < t)
                .map(|(n, _, _)| *n)
                .collect();
            if fail.is_empty() {
                eprintln!("    >>> PASS (would survive quality filter)");
            } else {
                eprintln!("    >>> FAIL ({})", fail.join(", "));
            }

            let emb = rt.block_on(arcface.embed(&aligned)).unwrap();
            all_embs.push((name.chars().next().unwrap(), emb.values));
            let emb_no_tta = rt.block_on(arcface_no_tta.embed(&aligned)).unwrap();
            all_embs_no_tta.push((name.chars().next().unwrap(), emb_no_tta.values));
        }
    }

    eprintln!("\n=== Cosine similarity (WITH rotation TTA, first face per image) ===");
    let names: Vec<char> = all_embs.iter().map(|(n, _)| *n).collect();
    for i in 0..all_embs.len() {
        for j in (i + 1)..all_embs.len() {
            eprintln!(
                "cos({}-{}) = {:.4}",
                names[i], names[j],
                cosine(&all_embs[i].1, &all_embs[j].1)
            );
        }
    }

    eprintln!("\n=== Cosine similarity (WITHOUT rotation TTA, first face per image) ===");
    let names_no_tta: Vec<char> = all_embs_no_tta.iter().map(|(n, _)| *n).collect();
    for i in 0..all_embs_no_tta.len() {
        for j in (i + 1)..all_embs_no_tta.len() {
            eprintln!(
                "cos({}-{}) = {:.4}",
                names_no_tta[i], names_no_tta[j],
                cosine(&all_embs_no_tta[i].1, &all_embs_no_tta[j].1)
            );
        }
    }
}

