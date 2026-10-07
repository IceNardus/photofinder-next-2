//! 把对齐后的人脸图 dump 到 /tmp/aligned_faces/，方便人工检查 alignment 质量。
//!
//! 用法:`cargo test --release -p pf_ai --test aligned_dump -- --ignored --nocapture`

use std::path::{Path, PathBuf};

use pf_ai::face::AlignmentConfig;
use pf_ai::image_data::ImageData;
use pf_ai::{ArcFaceEmbedder, FaceAligner, FaceDetector, FaceEmbedder, ScrfdDetector, SimpleAligner};
use pf_core::BBox;

const FACE_MODEL: &str = "/Users/mac/Library/Application Support/PhotoFinderNext/resources/models/w600k_r50.onnx";
const SCRFD_MODEL: &str = "/Users/mac/Library/Application Support/PhotoFinderNext/resources/models/scrfd_500m_bnkps.onnx";
const OUT_DIR: &str = "/tmp/aligned_faces";

fn bytes_to_image(bytes: &[u8]) -> ImageData {
    ImageData::from_bytes(bytes).expect("decode")
}

#[test]
#[ignore]
fn dump_aligned_faces() {
    std::fs::create_dir_all(OUT_DIR).expect("mkdir");
    let scrfd = ScrfdDetector::load(&PathBuf::from(SCRFD_MODEL)).expect("load scrfd");
    let aligner = SimpleAligner::new();
    let rt = tokio::runtime::Runtime::new().unwrap();

    let paths: Vec<(&str, &str)> = vec![
        ("jaqor-11", "/Users/mac/Downloads/pexels-jaqor-33601811.jpg"),
        ("jaqor-31", "/Users/mac/Downloads/pexels-jaqor-33601831.jpg"),
        ("jaqor-35", "/Users/mac/Downloads/pexels-jaqor-33601835.jpg"),
    ];

    for (name, p) in &paths {
        let bytes = std::fs::read(p).expect("read");
        let img = bytes_to_image(&bytes);
        let dets = rt.block_on(scrfd.detect(&img)).expect("detect");
        if dets.is_empty() {
            eprintln!("[{name}] no face");
            continue;
        }
        let det = &dets[0];
        // Dump 原始 bbox 区域(uncropped),看实际脸区域长什么样
        let bbox = det.bbox;
        let cropped = img
            .crop(pf_core::BBox::new(bbox.x, bbox.y, bbox.w, bbox.h))
            .expect("crop");
        let crop_path = Path::new(OUT_DIR).join(format!("{name}_bbox.png"));
        cropped.as_rgb8().save(&crop_path).expect("save bbox");
        let aligned = aligner.align(&img, &det.keypoints).expect("align");
        let rgb = aligned.image.as_rgb8();
        let out_path = Path::new(OUT_DIR).join(format!("{name}.png"));
        rgb.save(&out_path).expect("save");
        // 也保存未做 histogram_eq + ellipse_mask 的原始对齐图,用于 debug
        let mut raw_aligner = SimpleAligner::new();
        // 改 default config
        let _ = raw_aligner;
        let aligner_raw = SimpleAligner::with_config(AlignmentConfig {
            align_scale: 1.25,
            align_y_offset: -10.0,
            use_ellipse_mask: false,
            use_histogram_eq: false,
        });
        let aligned_raw = aligner_raw.align(&img, &det.keypoints).expect("align raw");
        let raw_path = Path::new(OUT_DIR).join(format!("{name}_raw.png"));
        aligned_raw.image.as_rgb8().save(&raw_path).expect("save raw");
        eprintln!(
            "[{name}] kps: le=({:.0},{:.0}) re=({:.0},{:.0}) n=({:.0},{:.0}) lm=({:.0},{:.0}) rm=({:.0},{:.0})",
            det.keypoints.left_eye.0, det.keypoints.left_eye.1,
            det.keypoints.right_eye.0, det.keypoints.right_eye.1,
            det.keypoints.nose.0, det.keypoints.nose.1,
            det.keypoints.left_mouth.0, det.keypoints.left_mouth.1,
            det.keypoints.right_mouth.0, det.keypoints.right_mouth.1,
        );
        eprintln!("    → saved {out_path:?} + {raw_path:?}");
    }

    // 也 dump 一张用于 sanity check 的嵌入,确认 aligned face 自相似度 ≈ 1.0
    let arcface = ArcFaceEmbedder::load(&PathBuf::from(FACE_MODEL)).expect("load arcface");
    let bytes = std::fs::read(paths[0].1).unwrap();
    let img = bytes_to_image(&bytes);
    let dets = rt.block_on(scrfd.detect(&img)).unwrap();
    let aligned = aligner.align(&img, &dets[0].keypoints).unwrap();
    let e1 = rt.block_on(arcface.embed(&aligned)).unwrap();
    let e2 = rt.block_on(arcface.embed(&aligned)).unwrap();
    let cos = {
        let dot: f32 = e1.values.iter().zip(&e2.values).map(|(a, b)| a * b).sum();
        let n1: f32 = e1.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n2: f32 = e2.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (n1 * n2)
    };
    eprintln!("\nSelf-similarity check (same aligned face, embed twice): cos = {cos:.6}");
}