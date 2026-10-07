//! 索引期过滤：低色 / 字节每像素判断，决定一张图是否走人脸 / 对象 pipeline。
//!
//! **算法参数与 ai-next/src-tauri/src/core/scanner.rs::should_skip_for_face_pipeline 逐位对齐**：
//!
//! | 条件 | 阈值 |
//! |---|---|
//! | `bytes_per_pixel < 0.05` | 直接跳过（认为是向量图 / 图标） |
//! | `0.05 ≤ bpp < 0.15` 且 unique colors < 100 | 跳过（认为是低色图） |
//!
//! unique colors 通过 50×50 nearest-neighbor 重采样后统计。

#![allow(dead_code)]

use image::GenericImageView;
use std::collections::HashSet;

use crate::error::AIError;

/// bytes-per-pixel 下限：低于此值视为非照片。
pub const BPP_HARD_SKIP: f32 = 0.05;
/// bytes-per-pixel 软下限：低于此值且低色则跳过。
pub const BPP_SOFT_SKIP: f32 = 0.15;
/// unique colors 下限（soft skip 阈值）。
pub const UNIQUE_COLOR_THRESHOLD: usize = 100;
/// 提前退出阈值：超过这个 unique colors 一定不是低色图。
pub const UNIQUE_COLOR_EARLY_EXIT: usize = 10_000;
/// unique colors 采样尺寸（缩放后再统计）。
pub const UNIQUE_COLOR_SAMPLE_DIM: u32 = 50;

/// 是否应跳过此图（不进入人脸 / 对象 pipeline）。
///
/// 失败（无法 decode）→ 不跳过（让上层报更具体的错）。
pub fn should_skip_for_face_pipeline(path: &std::path::Path) -> bool {
    let (bpp, colors) = match inspect(path) {
        Ok(v) => v,
        Err(_) => return false,
    };
    should_skip(bpp, colors)
}

/// 内部判断（已拿到 bpp + colors 后）。
pub fn should_skip(bytes_per_pixel: f32, unique_colors: Option<usize>) -> bool {
    if bytes_per_pixel < BPP_HARD_SKIP {
        return true;
    }
    if bytes_per_pixel < BPP_SOFT_SKIP {
        if let Some(c) = unique_colors {
            if c < UNIQUE_COLOR_THRESHOLD {
                return true;
            }
        }
    }
    false
}

/// 解码图片并计算 (bytes_per_pixel, unique_colors)。
pub fn inspect(path: &std::path::Path) -> Result<(f32, Option<usize>), AIError> {
    let file_size = std::fs::metadata(path)
        .map_err(|e| AIError::ImageDecode(e.to_string()))?
        .len();
    let img = image::open(path).map_err(|e| AIError::ImageDecode(e.to_string()))?;
    let (w, h) = img.dimensions();
    let pixels = (w as f64) * (h as f64);
    if pixels <= 0.0 {
        return Ok((0.0, Some(0)));
    }
    let bpp = (file_size as f64 / pixels) as f32;
    let colors = count_unique_colors(&img);
    Ok((bpp, Some(colors)))
}

/// 采样 unique colors（50×50 nearest-neighbor）。
pub fn count_unique_colors(img: &image::DynamicImage) -> usize {
    let img = img.resize(
        UNIQUE_COLOR_SAMPLE_DIM,
        UNIQUE_COLOR_SAMPLE_DIM,
        image::imageops::FilterType::Nearest,
    );
    let rgba = img.to_rgba8();
    let mut unique: HashSet<u32> = HashSet::new();
    for pixel in rgba.pixels() {
        let rgb = ((pixel[0] as u32) << 16) | ((pixel[1] as u32) << 8) | (pixel[2] as u32);
        unique.insert(rgb);
        if unique.len() > UNIQUE_COLOR_EARLY_EXIT {
            return unique.len();
        }
    }
    unique.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn solid_image(w: u32, h: u32, color: [u8; 3]) -> image::DynamicImage {
        let buf: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(w, h, |_x, _y| Rgb(color));
        image::DynamicImage::from(buf)
    }

    #[test]
    fn hard_skip_for_very_low_bpp() {
        // bpp = 0.0 → 直接跳过
        assert!(should_skip(0.0, Some(500)));
        assert!(should_skip(0.04, Some(500)));
    }

    #[test]
    fn no_skip_for_high_bpp_even_if_low_color() {
        assert!(!should_skip(1.0, Some(5)));
    }

    #[test]
    fn soft_skip_for_low_bpp_and_low_color() {
        // bpp < 0.15 且 colors < 100 → 跳过
        assert!(should_skip(0.10, Some(50)));
    }

    #[test]
    fn no_soft_skip_for_low_bpp_but_high_color() {
        // bpp < 0.15 但 colors >= 100 → 不跳过（照片）
        assert!(!should_skip(0.10, Some(200)));
    }

    #[test]
    fn border_bpp_at_hard_skip_threshold_falls_through_to_soft() {
        // bpp = 0.05 不满足 < 0.05（hard skip），落入 soft skip 分支：colors=50 < 100 → skip
        assert!(should_skip(0.05, Some(50)));
        // 但 high color 时不 skip
        assert!(!should_skip(0.05, Some(500)));
    }

    #[test]
    fn count_unique_colors_on_solid_image_is_one() {
        let img = solid_image(100, 100, [255, 0, 0]);
        assert_eq!(count_unique_colors(&img), 1);
    }

    #[test]
    fn count_unique_colors_on_wide_gradient_is_large() {
        // 200x100 original → 50x50 sample → 每个采样像素 x 唯一（50 distinct），
        // 但 y 也参与 → (x*200 + y) → 2500 distinct values > 100
        let img = image::DynamicImage::from(
            ImageBuffer::from_fn(200, 100, |x, y| {
                Rgb([((x + y * 7) % 256) as u8, (y % 256) as u8, 0])
            }),
        );
        let n = count_unique_colors(&img);
        assert!(n >= 100, "n = {n}");
    }
}