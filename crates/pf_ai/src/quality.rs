//! 人脸质量评估。
//!
//! **算法与 ai-next/src-tauri/src/ai/face/quality.rs::FaceQuality::assess 对齐（formula v3）**：
//!
//! | 评分项 | 计算 | 权重 |
//! |---|---|---|
//! | `detector_score` | SCRFD score | `0.30` |
//! | `face_area_score` | min(face_w,face_h) / 150 | `0.25` |
//! | `blur_score` | Laplacian variance / 100（clamp 0..1） | `0.25` |
//! | `pose_score` | 由 5-KPS 估计的 yaw/pitch/roll 推导 | `0.20` |
//!
//! 总 `quality = 0.30*det + 0.25*area + 0.25*blur + 0.20*pose`（clamp 0..1）
//!
//! **过滤阈值**：
//! - `detector_score >= 0.30`
//! - `face_area_score >= 0.15`
//! - `eye_distance >= 12.0` 像素
//! - `pose_score >= 0.15`
//! - `quality >= 0.45`

use image::GrayImage;

use pf_core::FaceKeypoints;

use crate::face::traits::{FaceDetection, QualityFilter};

/// 最小 detector score（pipeline 内部，KPS-aware 放宽）。
pub const MIN_DETECTOR_SCORE: f32 = 0.30;
/// 最小 face area score。
pub const MIN_FACE_AREA_SCORE: f32 = 0.15;
/// 最小眼距（像素）。
pub const MIN_EYE_DISTANCE: f32 = 12.0;
/// 最小 pose score（已从 0.20 放宽到 0.15，因为 `pose_score` 现在用眼距归一化的
/// ratio，不会因高分辨率图片（眼距 400+px）出现 50px 硬切顶到 0）。
pub const MIN_POSE_SCORE: f32 = 0.15;
/// legacy `is_acceptable` 阈值。
pub const ACCEPTABLE_QUALITY: f32 = 0.45;

/// Face area 归一化分母（legacy 用 10000.0 像素²，new 改用 min 边长 150px）。
/// 保留 150px 以适应 SCRFD 输出 5-KPS 流程（legacy 用 yaw/pitch/roll 来自相同 KPS 估计）。
pub const FACE_AREA_REF_PX: f32 = 150.0;

/// Laplacian variance 归一化分母。
pub const BLUR_REF_VARIANCE: f32 = 100.0;

/// 质量详情（便于上层日志 / debug）。
#[derive(Debug, Clone)]
pub struct QualityBreakdown {
    pub detector_score: f32,
    pub blur_score: f32,
    pub pose_score: f32,
    pub face_area_score: f32,
    pub quality: f32,
    pub eye_distance: f32,
}

impl QualityBreakdown {
    /// 是否通过所有过滤（与 legacy `is_acceptable + passes` 对齐）。
    pub fn passes(&self) -> bool {
        self.detector_score >= MIN_DETECTOR_SCORE
            && self.face_area_score >= MIN_FACE_AREA_SCORE
            && self.eye_distance >= MIN_EYE_DISTANCE
            && self.pose_score >= MIN_POSE_SCORE
            && self.quality >= ACCEPTABLE_QUALITY
    }
}

/// 计算 blur_score = laplacian_variance / BLUR_REF_VARIANCE（clamp 0..1）。
pub fn blur_score_from_aligned(aligned_gray: &GrayImage) -> f32 {
    (laplacian_variance(aligned_gray) / BLUR_REF_VARIANCE).clamp(0.0, 1.0)
}

/// Laplacian variance（4-邻域）。
pub fn laplacian_variance(gray: &GrayImage) -> f32 {
    let mut sum = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    let mut count = 0.0_f32;
    let w = gray.width();
    let h = gray.height();
    if w < 3 || h < 3 {
        return 0.0;
    }
    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let center = gray.get_pixel(x, y)[0] as f32;
            let lap = 4.0 * center
                - gray.get_pixel(x - 1, y)[0] as f32
                - gray.get_pixel(x + 1, y)[0] as f32
                - gray.get_pixel(x, y - 1)[0] as f32
                - gray.get_pixel(x, y + 1)[0] as f32;
            sum += lap;
            sum_sq += lap * lap;
            count += 1.0;
        }
    }
    if count < 1.0 {
        return 0.0;
    }
    let mean = sum / count;
    (sum_sq / count) - (mean * mean)
}

/// 两眼距离（像素）。
pub fn eye_distance(kps: &FaceKeypoints) -> f32 {
    let dx = kps.right_eye.0 - kps.left_eye.0;
    let dy = kps.right_eye.1 - kps.left_eye.1;
    (dx * dx + dy * dy).sqrt()
}

/// pose_score：基于 5 个关键点的对称性。
///
/// 1. tilt_score = 1 - clamp(eye_tilt + mouth_tilt, 0, 1)
/// 2. eye_level_score = 1 - clamp(eye_dist_y / eye_dist_x, 0, 1)
/// 3. pose_score = (tilt + eye_level) / 2
///
/// **修复（vs ai-next）**：原本 `eye_level_diff / 50.0` 用绝对像素做归一化，对
/// 高分辨率图片（SCRFD 输出 4000+px 宽的脸，眼距 400+px）会过早归零 — 即使
/// 眼睛只是相对倾斜 5% 左右也会让 `eye_level_score` 直接到 0，进而把
/// `pose_score` 拖到 0 而被过滤（典型 case：pexels-jaqor-33601835 这种 27°
/// 倾斜的脸，眼 y-diff 216px / 眼 x-diff 420px，在 4K 图里是合理姿态但在旧
/// 公式里直接拒掉）。改为 ratio 后姿态评分与人脸像素尺度无关，对小图和大图
/// 一致。
pub fn pose_score(kps: &FaceKeypoints) -> f32 {
    let eye_dist_x = (kps.right_eye.0 - kps.left_eye.0).abs();
    let eye_dist_y = (kps.right_eye.1 - kps.left_eye.1).abs();
    let eye_tilt = eye_dist_y / eye_dist_x.max(1.0);

    let mouth_dist_x = (kps.right_mouth.0 - kps.left_mouth.0).abs();
    let mouth_dist_y = (kps.right_mouth.1 - kps.left_mouth.1).abs();
    let mouth_tilt = mouth_dist_y / mouth_dist_x.max(1.0);

    let tilt_score = 1.0 - (eye_tilt + mouth_tilt).min(1.0);

    // 关键修复：用 ratio 而非绝对像素做归一化
    let eye_level_ratio = eye_dist_y / eye_dist_x.max(1.0);
    let eye_level_score = 1.0 - eye_level_ratio.min(1.0);

    (tilt_score + eye_level_score) / 2.0
}

/// face_area_score = clamp(min(face_w, face_h) / FACE_AREA_REF_PX, 0, 1)
pub fn face_area_score(det: &FaceDetection) -> f32 {
    let min_dim = det.bbox.w.min(det.bbox.h);
    (min_dim / FACE_AREA_REF_PX).clamp(0.0, 1.0)
}

/// 综合 quality = 0.30*det + 0.25*area + 0.25*blur + 0.20*pose（formula v3）。
///
/// **与 ai-next `FaceQuality::assess` 权重一致**（detector=0.30, area=0.25, blur=0.25, pose=0.20）。
pub fn combined_quality(
    detector_score: f32,
    face_area_score: f32,
    blur_score: f32,
    pose_score: f32,
) -> f32 {
    (0.30 * detector_score
        + 0.25 * face_area_score
        + 0.25 * blur_score
        + 0.20 * pose_score)
        .clamp(0.0, 1.0)
}

/// 由 yaw/pitch/roll（度）推导 pose_score（ai-next v3）：
/// `1 - (yaw_dev + pitch_dev + roll_dev) / 3`，每个 `dev = (|angle| - 15).max(0) / 45`。
///
/// 完全正脸（|角度| ≤ 15°）→ 1.0；任一轴偏 60°+ 会把该轴 dev 顶到 1。
pub fn pose_score_from_ypr(yaw: f32, pitch: f32, roll: f32) -> f32 {
    let dev = |a: f32| ((a.abs() - 15.0).max(0.0) / 45.0).min(1.0);
    (1.0 - (dev(yaw) + dev(pitch) + dev(roll)) / 3.0).clamp(0.0, 1.0)
}

/// 评估一个 detection（blur_score 由调用方提供，从 aligned gray image 算出；
/// yaw/pitch/roll 由 5-KPS 姿态估计器给出，退化时传 `None`）。
pub fn assess(
    det: &FaceDetection,
    blur: f32,
    yaw_pitch_roll: Option<(f32, f32, f32)>,
) -> QualityBreakdown {
    let detector_score = det.score;
    let pose = match yaw_pitch_roll {
        Some((y, p, r)) => pose_score_from_ypr(y, p, r),
        None => 0.0,
    };
    let area = face_area_score(det);
    let eye = eye_distance(&det.keypoints);
    let quality = combined_quality(detector_score, area, blur, pose);
    QualityBreakdown {
        detector_score,
        blur_score: blur,
        pose_score: pose,
        face_area_score: area,
        quality,
        eye_distance: eye,
    }
}

/// QualityFilter 默认实现的扩展：是否通过（基于 ai-next 阈值）。
pub fn passes(q: &QualityBreakdown) -> bool {
    q.passes()
}

// 让 pf_ai::face::traits::QualityFilter 仍能 `passes(det, quality)` ——
/// 保留旧 API，但 deprecated，由 assess() 替代。
impl QualityFilter {
    /// 旧 API（仅做 detector + bbox + quality 检查，不含 eye/pose/blur）。
    /// 推荐用 `crate::quality::assess` + `passes` 走完整路径。
    pub fn passes_legacy(&self, det: &FaceDetection, quality: f32) -> bool {
        det.score >= self.min_detector_score
            && det.bbox.w >= self.min_face_size as f32
            && quality >= self.min_quality
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;
    use pf_core::{BBox, FaceKeypoints};
    use std::f32::consts::PI;

    fn det(score: f32, w: f32, h: f32) -> FaceDetection {
        FaceDetection {
            bbox: BBox::new(0.0, 0.0, w, h),
            score,
            keypoints: FaceKeypoints {
                left_eye: (30.0, 50.0),
                right_eye: (70.0, 50.0),
                nose: (50.0, 80.0),
                left_mouth: (35.0, 100.0),
                right_mouth: (65.0, 100.0),
            },
        }
    }

    #[test]
    fn constants_match_ai_next() {
        assert!((MIN_DETECTOR_SCORE - 0.30).abs() < 1e-6);
        assert!((MIN_FACE_AREA_SCORE - 0.15).abs() < 1e-6);
        assert!((MIN_EYE_DISTANCE - 12.0).abs() < 1e-6);
        assert!((MIN_POSE_SCORE - 0.15).abs() < 1e-6);
        assert!((ACCEPTABLE_QUALITY - 0.45).abs() < 1e-6);
        assert!((FACE_AREA_REF_PX - 150.0).abs() < 1e-6);
        assert!((BLUR_REF_VARIANCE - 100.0).abs() < 1e-6);
    }

    #[test]
    fn eye_distance_symmetric_face() {
        let kps = FaceKeypoints {
            left_eye: (0.0, 0.0),
            right_eye: (40.0, 0.0),
            nose: (20.0, 30.0),
            left_mouth: (5.0, 60.0),
            right_mouth: (35.0, 60.0),
        };
        assert!((eye_distance(&kps) - 40.0).abs() < 1e-6);
    }

    #[test]
    fn pose_score_straight_face_is_high() {
        let kps = FaceKeypoints {
            left_eye: (30.0, 50.0),
            right_eye: (70.0, 50.0),
            nose: (50.0, 80.0),
            left_mouth: (35.0, 100.0),
            right_mouth: (65.0, 100.0),
        };
        let s = pose_score(&kps);
        assert!(s > 0.9, "score = {s}");
    }

    #[test]
    fn pose_score_tilted_face_is_lower() {
        // 倾斜脸：眼睛 y 不一致
        let kps = FaceKeypoints {
            left_eye: (30.0, 60.0),
            right_eye: (70.0, 40.0),
            nose: (50.0, 80.0),
            left_mouth: (35.0, 100.0),
            right_mouth: (65.0, 100.0),
        };
        let s = pose_score(&kps);
        assert!(s < 0.9, "tilted score = {s}");
    }

    #[test]
    fn pose_score_moderately_tilted_4k_face_passes() {
        // 真实场景：pexels-jaqor-33601835 的 KPS。
        // 高分辨率图（4K+，眼距 420px），27° 倾斜。旧公式下
        // eye_level_diff/50.0 = 216/50 = 4.32 → eye_level_score = 0 → pose = 0，
        // 被质量过滤拒掉，导致这张图搜不到。新公式用 ratio 归一化后
        // 应该给出一个中等分（明显高于 0 但低于 0.5）。
        let kps = FaceKeypoints {
            left_eye: (3633.0, 2841.0),
            right_eye: (4053.0, 3057.0),
            nose: (3680.0, 3152.0),
            left_mouth: (3455.0, 3275.0),
            right_mouth: (3802.0, 3454.0),
        };
        let s = pose_score(&kps);
        assert!(
            s >= MIN_POSE_SCORE,
            "moderately tilted 4K face pose_score={s} < MIN_POSE_SCORE={MIN_POSE_SCORE}",
        );
        assert!(
            s < 0.5,
            "moderately tilted should not score too high (got {s})",
        );
    }

    #[test]
    fn pose_score_profile_face_fails() {
        // 极端情况：profile / 90° 侧脸，眼距几乎为 0 (kps 都在一条垂直线上)。
        // 这种脸确实不能识别，应该被拒掉。
        let kps = FaceKeypoints {
            left_eye: (50.0, 30.0),
            right_eye: (51.0, 70.0), // eye_dist_x = 1, eye_dist_y = 40
            nose: (50.0, 100.0),
            left_mouth: (50.0, 130.0),
            right_mouth: (50.0, 160.0), // mouth_dist_x ≈ 0
        };
        let s = pose_score(&kps);
        assert!(
            s < MIN_POSE_SCORE,
            "profile face should fail, got pose_score={s}",
        );
    }

    #[test]
    fn face_area_score_uses_min_dim() {
        let d = det(0.9, 100.0, 80.0);
        let s = face_area_score(&d);
        // min = 80 → 80 / 150 ≈ 0.533
        assert!((s - 0.533).abs() < 0.01);
    }

    #[test]
    fn face_area_score_clamps_at_one() {
        let d = det(0.9, 300.0, 300.0);
        let s = face_area_score(&d);
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn face_area_score_below_threshold() {
        let d = det(0.9, 10.0, 10.0);
        let s = face_area_score(&d);
        assert!(s < MIN_FACE_AREA_SCORE);
    }

    /// 权重与 ai-next v3 0.30/0.25/0.25/0.20 严格对齐。
    #[test]
    fn combined_quality_weighted_v3() {
        let q = combined_quality(1.0, 1.0, 1.0, 1.0);
        assert!((q - 1.0).abs() < 1e-6);
        let q = combined_quality(0.5, 0.5, 0.5, 0.5);
        // 全部 0.5 → 0.50（0.30*0.5 + 0.25*0.5 + 0.25*0.5 + 0.20*0.5）
        assert!((q - 0.50).abs() < 1e-6);
        // 仅 detector 满：0.30
        let q = combined_quality(1.0, 0.0, 0.0, 0.0);
        assert!((q - 0.30).abs() < 1e-6);
        // 仅 face_area 满：0.25
        let q = combined_quality(0.0, 1.0, 0.0, 0.0);
        assert!((q - 0.25).abs() < 1e-6);
        // 仅 blur 满：0.25
        let q = combined_quality(0.0, 0.0, 1.0, 0.0);
        assert!((q - 0.25).abs() < 1e-6);
        // 仅 pose 满：0.20
        let q = combined_quality(0.0, 0.0, 0.0, 1.0);
        assert!((q - 0.20).abs() < 1e-6);
    }

    /// 便捷：从 keypoints 估姿态后再 assess（与 pipeline 行为一致）。
    fn a(d: &FaceDetection, blur: f32) -> QualityBreakdown {
        assess(d, blur, crate::face::pose::estimate_yaw_pitch_roll(&d.keypoints))
    }

    /// pose_score_from_ypr：正脸 → 1.0，偏航/俯仰惩罚平滑。
    #[test]
    fn pose_score_from_ypr_frontal_is_one() {
        assert!((pose_score_from_ypr(0.0, 0.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((pose_score_from_ypr(10.0, -12.0, 8.0) - 1.0).abs() < 1e-6, "≤15° 无惩罚");
    }

    #[test]
    fn pose_score_from_ypr_penalizes_deviation() {
        // 单轴 30°：dev=(30-15)/45=1/3 → 1 - (1/3)/3 = 0.888…
        let s = pose_score_from_ypr(30.0, 0.0, 0.0);
        assert!((s - (1.0 - (1.0 / 3.0) / 3.0)).abs() < 1e-4, "s={s}");
        // 三轴都 60°+：dev 各顶到 1 → pose=0
        let s = pose_score_from_ypr(60.0, 60.0, 60.0);
        assert!((s - 0.0).abs() < 1e-4, "s={s}");
    }

    #[test]
    fn assess_returns_full_breakdown() {
        let d = det(0.85, 120.0, 120.0);
        let b = a(&d, 0.7);
        assert!((b.detector_score - 0.85).abs() < 1e-6);
        assert!((b.blur_score - 0.7).abs() < 1e-6);
        assert!(b.pose_score > 0.9);
        assert!(b.face_area_score > 0.5);
        assert!(b.passes());
    }

    #[test]
    fn breakdown_fails_on_low_detector() {
        let d = det(0.10, 120.0, 120.0);
        let b = a(&d, 0.7);
        assert!(!b.passes());
    }

    #[test]
    fn breakdown_fails_on_tiny_face() {
        let d = det(0.95, 5.0, 5.0);
        let b = a(&d, 0.7);
        assert!(!b.passes());
    }

    #[test]
    fn breakdown_fails_on_close_eyes() {
        let d = FaceDetection {
            bbox: BBox::new(0.0, 0.0, 100.0, 100.0),
            score: 0.9,
            keypoints: FaceKeypoints {
                left_eye: (40.0, 50.0),
                right_eye: (50.0, 50.0), // 距离 10px < 12
                nose: (45.0, 70.0),
                left_mouth: (35.0, 90.0),
                right_mouth: (55.0, 90.0),
            },
        };
        let b = a(&d, 0.7);
        assert!(b.eye_distance < MIN_EYE_DISTANCE);
        assert!(!b.passes());
    }

    #[test]
    fn breakdown_fails_on_low_quality() {
        // 全部很低 → quality < 0.45（v3 权重下仍被拒）
        let d = det(0.4, 30.0, 30.0); // face_area = 30/150 = 0.2
        let b = a(&d, 0.3);
        // 0.30*0.4 + 0.25*0.2 + 0.25*0.3 + 0.20*1.0(正脸 pose) = 0.445 < 0.45
        assert!(b.quality < ACCEPTABLE_QUALITY);
        assert!(!b.passes());
    }

    #[test]
    fn blur_score_on_blank_image_is_low() {
        let g = GrayImage::from_pixel(112, 112, Luma([128u8]));
        let s = blur_score_from_aligned(&g);
        assert!(s < 0.05);
    }

    #[test]
    fn laplacian_variance_on_white_noise_is_high() {
        use image::ImageBuffer;
        let buf: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_fn(112, 112, |x, y| {
            let v = ((x ^ y) & 0xff) as u8;
            Luma([v])
        });
        let v = laplacian_variance(&buf);
        assert!(v > 100.0);
    }

    #[test]
    fn laplacian_variance_on_tiny_image_is_zero() {
        let g = GrayImage::from_pixel(2, 2, Luma([0u8]));
        let v = laplacian_variance(&g);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn legacy_passes_works() {
        let q = QualityFilter::from_config(0.3, 20, 0.45, 30.0);
        let d = det(0.5, 25.0, 25.0);
        assert!(q.passes_legacy(&d, 0.5));
        let _ = PI;
    }
}
