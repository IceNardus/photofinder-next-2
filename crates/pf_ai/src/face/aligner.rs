//! Face alignment → 112×112 RGB。
//!
//! **算法与 ai-next/src-tauri/src/ai/face/align.rs **逐位对齐**：
//!
//! - 5 个关键点相似变换（Umeyama）到 112×112 参考坐标
//! - `align_scale=1.25`、`align_y_offset=-10.0`（放大并上移，更紧的裁剪）
//! - 可选椭圆 mask（去除衣服 / 肩膀）
//! - 可选直方图均衡化（光照鲁棒）
//! - 双线性插值采样

use image::{GenericImageView, Rgb, RgbImage};

use pf_core::{BBox, FaceKeypoints};

use crate::error::AIError;
use crate::face::traits::{AlignedFace, FaceAligner};
use crate::image_data::ImageData;

/// 输出尺寸。
pub const OUTPUT_SIZE: u32 = 112;

/// 当前对齐流程的版本标识（plan §18）。
///
/// 包含关键参数:
/// - `crop10` = FacePipeline::process 中 crop + 10% margin(本会话关键修复,58% → 91%)
/// - `align1.25` = `align_scale=1.25`
/// - `y-10` = `align_y_offset=-10.0`
///
/// 修改任何上述参数 → 同步更新此常量 + 写新行 face_embeddings(model_version+此版本)。
pub const ALIGNMENT_VERSION: &str = "crop10_align1.25_y-10_v1";

/// 5 个关键点在 112×112 参考图中的目标坐标（按 arcface-ref 公式）。
///
/// `(38.2946 * scale, 51.6968 * scale + y_off)` 等。
pub const REF_LANDMARKS_RAW: [(f32, f32); 5] = [
    (38.2946, 51.6968),  // 左眼
    (58.2946, 51.6968),  // 右眼
    (50.2946, 87.0256),  // 鼻
    (41.5493, 107.3550), // 左嘴角
    (61.7299, 107.3550), // 右嘴角
];

/// 对齐配置（与 ai-next FaceAlignmentConfig 一致）。
#[derive(Debug, Clone)]
pub struct AlignmentConfig {
    /// 缩放（>1 放大）
    pub align_scale: f32,
    /// Y 偏移（负值上移）
    pub align_y_offset: f32,
    /// 椭圆 mask
    pub use_ellipse_mask: bool,
    /// 直方图均衡
    pub use_histogram_eq: bool,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            align_scale: 1.25,
            align_y_offset: -10.0,
            use_ellipse_mask: true,
            use_histogram_eq: true,
        }
    }
}

/// 5 个关键点 → 112×112 对齐图。
pub struct SimpleAligner {
    config: AlignmentConfig,
}

impl Default for SimpleAligner {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleAligner {
    /// 构造（默认配置：mask on, hist-eq on）。
    pub fn new() -> Self {
        Self {
            config: AlignmentConfig::default(),
        }
    }

    /// 自定义配置。
    pub fn with_config(config: AlignmentConfig) -> Self {
        Self { config }
    }

    /// 当前配置。
    pub fn config(&self) -> &AlignmentConfig {
        &self.config
    }

    /// 计算目标参考点。
    fn ref_landmarks(&self) -> [(f32, f32); 5] {
        let s = self.config.align_scale;
        let y = self.config.align_y_offset;
        let mut out = REF_LANDMARKS_RAW;
        for p in &mut out {
            p.0 *= s;
            p.1 = p.1 * s + y;
        }
        out
    }

    /// Umeyama 相似变换：`dst = M * src + t`（M = scale * R）。
    pub fn compute_similarity_transform(
        src: &[(f32, f32); 5],
        dst: &[(f32, f32); 5],
    ) -> [f32; 6] {
        let mut src_center = (0.0f32, 0.0f32);
        let mut dst_center = (0.0f32, 0.0f32);
        for i in 0..5 {
            src_center.0 += src[i].0;
            src_center.1 += src[i].1;
            dst_center.0 += dst[i].0;
            dst_center.1 += dst[i].1;
        }
        src_center.0 /= 5.0;
        src_center.1 /= 5.0;
        dst_center.0 /= 5.0;
        dst_center.1 /= 5.0;

        let mut src_norm = [(0.0f32, 0.0f32); 5];
        let mut dst_norm = [(0.0f32, 0.0f32); 5];
        let mut src_scale = 0.0f32;
        let mut dst_scale = 0.0f32;
        for i in 0..5 {
            src_norm[i].0 = src[i].0 - src_center.0;
            src_norm[i].1 = src[i].1 - src_center.1;
            dst_norm[i].0 = dst[i].0 - dst_center.0;
            dst_norm[i].1 = dst[i].1 - dst_center.1;
            src_scale += src_norm[i].0 * src_norm[i].0 + src_norm[i].1 * src_norm[i].1;
            dst_scale += dst_norm[i].0 * dst_norm[i].0 + dst_norm[i].1 * dst_norm[i].1;
        }
        src_scale = (src_scale / 5.0).sqrt();
        dst_scale = (dst_scale / 5.0).sqrt();
        let scale = dst_scale / src_scale.max(1e-6);
        for i in 0..5 {
            src_norm[i].0 *= scale;
            src_norm[i].1 *= scale;
        }

        let mut a = 0.0f32;
        let mut b = 0.0f32;
        for i in 0..5 {
            a += src_norm[i].0 * dst_norm[i].0 + src_norm[i].1 * dst_norm[i].1;
            b += src_norm[i].0 * dst_norm[i].1 - src_norm[i].1 * dst_norm[i].0;
        }
        let norm = (a * a + b * b).sqrt().max(1e-6);
        let cos_theta = a / norm;
        let sin_theta = b / norm;

        let m00 = scale * cos_theta;
        let m01 = scale * sin_theta;
        let m10 = -scale * sin_theta;
        let m11 = scale * cos_theta;
        let t0 = dst_center.0 - (m00 * src_center.0 + m01 * src_center.1);
        let t1 = dst_center.1 - (m10 * src_center.0 + m11 * src_center.1);

        [m00, m01, t0, m10, m11, t1]
    }

    /// 椭圆 mask（保留脸部，剔除衣服）。
    fn apply_ellipse_mask(&self, img: &mut RgbImage) {
        let cx = OUTPUT_SIZE as f32 / 2.0;
        let cy = OUTPUT_SIZE as f32 / 2.0 - 3.0;
        let rx = OUTPUT_SIZE as f32 * 0.40;
        let ry = OUTPUT_SIZE as f32 * 0.48;
        for y in 0..OUTPUT_SIZE {
            for x in 0..OUTPUT_SIZE {
                let dx = (x as f32 - cx) / rx;
                let dy = (y as f32 - cy) / ry;
                if dx * dx + dy * dy > 1.0 {
                    img.put_pixel(x, y, Rgb([0, 0, 0]));
                }
            }
        }
    }

    /// 直方图均衡化（Y 通道 → 同比例缩放 R/G/B）。
    fn apply_histogram_equalization(&self, img: &mut RgbImage) {
        let n = (OUTPUT_SIZE * OUTPUT_SIZE) as usize;
        let mut gray = vec![0u32; n];
        for y in 0..OUTPUT_SIZE {
            for x in 0..OUTPUT_SIZE {
                let p = img.get_pixel(x, y);
                let lum = (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) as u32;
                gray[(y * OUTPUT_SIZE + x) as usize] = lum;
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
        let total = (OUTPUT_SIZE * OUTPUT_SIZE) as u32;
        let denom = total.saturating_sub(cdf_min);
        // 退化情况（denom=0 全部同色，或 cdf_min 覆盖了所有）→ 不修改
        if denom == 0 {
            return;
        }
        let mut lut = [0u8; 256];
        for i in 0..256 {
            lut[i] = ((cdf[i] - cdf_min) as f32 / denom as f32 * 255.0).round() as u8;
        }
        for y in 0..OUTPUT_SIZE {
            for x in 0..OUTPUT_SIZE {
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

    /// 从原图 + 5 个关键点 → 112×112 RGB。
    pub fn align_from_image(
        &self,
        img: &image::DynamicImage,
        kps: &[(f32, f32); 5],
    ) -> Result<RgbImage, AIError> {
        let ref_pts = self.ref_landmarks();
        let transform = Self::compute_similarity_transform(kps, &ref_pts);

        // Inverse transform: src = M^{-1} * (dst - t)
        let scale = (transform[0] * transform[0] + transform[1] * transform[1]).sqrt();
        let r00 = transform[0];
        let r01 = transform[1];
        let r10 = transform[3];
        let r11 = transform[4];
        let t0 = transform[2];
        let t1 = transform[5];
        let inv_scale = 1.0 / scale.max(1e-6);
        let inv_t0 = -inv_scale * (r00 * t0 + r10 * t1);
        let inv_t1 = -inv_scale * (r01 * t0 + r11 * t1);
        let inv_m00 = r00 * inv_scale;
        let inv_m01 = r10 * inv_scale;
        let inv_m10 = r01 * inv_scale;
        let inv_m11 = r11 * inv_scale;

        let mut output = RgbImage::new(OUTPUT_SIZE, OUTPUT_SIZE);
        let (w, h) = (img.width() as i32, img.height() as i32);
        for y in 0..OUTPUT_SIZE {
            for x in 0..OUTPUT_SIZE {
                let sx = inv_m00 * x as f32 + inv_m01 * y as f32 + inv_t0;
                let sy = inv_m10 * x as f32 + inv_m11 * y as f32 + inv_t1;
                if sx >= 0.0 && sx < w as f32 && sy >= 0.0 && sy < h as f32 {
                    let px = sx.floor() as i32;
                    let py = sy.floor() as i32;
                    let fx = sx - px as f32;
                    let fy = sy - py as f32;
                    let px0 = px.max(0).min(w - 1) as u32;
                    let px1 = (px + 1).max(0).min(w - 1) as u32;
                    let py0 = py.max(0).min(h - 1) as u32;
                    let py1 = (py + 1).max(0).min(h - 1) as u32;
                    let p00 = img.get_pixel(px0, py0);
                    let p10 = img.get_pixel(px1, py0);
                    let p01 = img.get_pixel(px0, py1);
                    let p11 = img.get_pixel(px1, py1);
                    let w00 = (1.0 - fx) * (1.0 - fy);
                    let w10 = fx * (1.0 - fy);
                    let w01 = (1.0 - fx) * fy;
                    let w11 = fx * fy;
                    let r = (p00[0] as f32 * w00
                        + p10[0] as f32 * w10
                        + p01[0] as f32 * w01
                        + p11[0] as f32 * w11) as u8;
                    let g = (p00[1] as f32 * w00
                        + p10[1] as f32 * w10
                        + p01[1] as f32 * w01
                        + p11[1] as f32 * w11) as u8;
                    let b = (p00[2] as f32 * w00
                        + p10[2] as f32 * w10
                        + p01[2] as f32 * w01
                        + p11[2] as f32 * w11) as u8;
                    output.put_pixel(x, y, Rgb([r, g, b]));
                } else {
                    output.put_pixel(x, y, Rgb([0, 0, 0]));
                }
            }
        }
        if self.config.use_ellipse_mask {
            self.apply_ellipse_mask(&mut output);
        }
        if self.config.use_histogram_eq {
            self.apply_histogram_equalization(&mut output);
        }
        Ok(output)
    }
}

impl FaceAligner for SimpleAligner {
    fn align(&self, image: &ImageData, keypoints: &FaceKeypoints) -> Result<AlignedFace, AIError> {
        let img = image.inner().clone();
        let kps_array = [
            keypoints.left_eye,
            keypoints.right_eye,
            keypoints.nose,
            keypoints.left_mouth,
            keypoints.right_mouth,
        ];
        let aligned_rgb = self.align_from_image(&img, &kps_array)?;
        // 转换 RgbImage → DynamicImage → ImageData
        let dyn_img = image::DynamicImage::from(aligned_rgb);
        let aligned_data = ImageData::from_dynamic(dyn_img);
        Ok(AlignedFace {
            image: aligned_data,
            original_bbox: BBox::new(0.0, 0.0, 0.0, 0.0), // 由调用方填
        })
    }

    fn output_size(&self) -> (u32, u32) {
        (OUTPUT_SIZE, OUTPUT_SIZE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn solid_img(w: u32, h: u32, c: [u8; 3]) -> image::DynamicImage {
        let buf: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(w, h, |_, _| Rgb(c));
        image::DynamicImage::from(buf)
    }

    #[test]
    fn output_is_112x112() {
        let aligner = SimpleAligner::new();
        let img = solid_img(200, 200, [128, 128, 128]);
        let kps = [(100.0, 100.0); 5];
        let out = aligner.align_from_image(&img, &kps).unwrap();
        assert_eq!(out.width(), 112);
        assert_eq!(out.height(), 112);
    }

    #[test]
    fn ref_landmarks_apply_scale_and_offset() {
        let aligner = SimpleAligner::new();
        let refs = aligner.ref_landmarks();
        // 左眼：(38.2946 * 1.25, 51.6968 * 1.25 + (-10)) = (47.8683, 54.621)
        let (lx, ly) = refs[0];
        assert!((lx - 47.8683).abs() < 0.01);
        assert!((ly - 54.621).abs() < 0.05);
    }

    #[test]
    fn identity_transform_when_src_eq_ref() {
        // 当 src = dst 时，变换应为恒等 (scale=1, theta=0)
        let kps = [(38.0, 51.0), (58.0, 51.0), (50.0, 87.0), (41.0, 107.0), (61.0, 107.0)];
        let t = SimpleAligner::compute_similarity_transform(&kps, &kps);
        // m00 ≈ 1, m01 ≈ 0, t0 ≈ 0, m10 ≈ 0, m11 ≈ 1, t1 ≈ 0
        assert!((t[0] - 1.0).abs() < 1e-4, "m00 = {}", t[0]);
        assert!(t[1].abs() < 1e-4, "m01 = {}", t[1]);
        assert!(t[2].abs() < 1e-3, "t0 = {}", t[2]);
        assert!(t[3].abs() < 1e-4, "m10 = {}", t[3]);
        assert!((t[4] - 1.0).abs() < 1e-4, "m11 = {}", t[4]);
        assert!(t[5].abs() < 1e-3, "t1 = {}", t[5]);
    }

    #[test]
    fn ellipse_mask_blackens_corners() {
        let aligner = SimpleAligner::new();
        let mut img = RgbImage::from_pixel(112, 112, Rgb([200, 100, 50]));
        aligner.apply_ellipse_mask(&mut img);
        // 4 个角应该在椭圆外 → 黑色
        assert_eq!(*img.get_pixel(0, 0), Rgb([0, 0, 0]));
        assert_eq!(*img.get_pixel(111, 0), Rgb([0, 0, 0]));
        assert_eq!(*img.get_pixel(0, 111), Rgb([0, 0, 0]));
        assert_eq!(*img.get_pixel(111, 111), Rgb([0, 0, 0]));
        // 中心应在椭圆内 → 保留原色
        assert_eq!(*img.get_pixel(56, 56), Rgb([200, 100, 50]));
    }

    #[test]
    fn align_with_face_at_center() {
        let aligner = SimpleAligner::with_config(AlignmentConfig {
            align_scale: 1.0,
            align_y_offset: 0.0,
            use_ellipse_mask: false,
            use_histogram_eq: false,
        });
        // 200x200 原图，5 个关键点放在中心
        let img = solid_img(200, 200, [200, 200, 200]);
        let kps = [(80.0, 80.0), (120.0, 80.0), (100.0, 110.0), (90.0, 130.0), (110.0, 130.0)];
        let out = aligner.align_from_image(&img, &kps).unwrap();
        // 应该输出 112x112（中心 200,200 不动，scale=1）
        assert_eq!(out.dimensions(), (112, 112));
        // 大部分像素应是非黑色（中心是灰色 200）
        let non_black = out
            .pixels()
            .filter(|p| p[0] > 50 || p[1] > 50 || p[2] > 50)
            .count();
        assert!(non_black > 1000);
    }

    #[test]
    fn histogram_equalization_uniform_output_for_solid_input() {
        let aligner = SimpleAligner::with_config(AlignmentConfig {
            align_scale: 1.0,
            align_y_offset: 0.0,
            use_ellipse_mask: false,
            use_histogram_eq: true,
        });
        let mut img = RgbImage::from_pixel(112, 112, Rgb([128, 128, 128]));
        aligner.apply_histogram_equalization(&mut img);
        // solid color 均衡后应近似原色（CDF 集中在该值）
        // 不严格测试值，因为均衡化行为依赖 CDF 形状
        let p = img.get_pixel(56, 56);
        assert!(p[0] >= 100 && p[0] <= 160);
    }
}