//! `ImageData` 抽象。
//!
//! 跨 crate 传递图片，避免暴露 `image::DynamicImage`。
//! 内部用 `Arc<DynamicImage>` 共享，clone 便宜。

#![allow(dead_code)]

use std::path::Path;
use std::sync::Arc;

use image::{DynamicImage, GrayImage, RgbImage};

use pf_core::BBox;

use crate::error::AIError;

/// 图片数据（共享所有权）。
#[derive(Debug, Clone)]
pub struct ImageData {
    inner: Arc<DynamicImage>,
}

impl ImageData {
    /// 从文件加载。
    pub fn from_file(path: &Path) -> Result<Self, AIError> {
        let img = image::open(path).map_err(|e| AIError::ImageDecode(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(img),
        })
    }

    /// 从 bytes 加载（自动识别格式）。
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AIError> {
        let img = image::load_from_memory(bytes).map_err(|e| AIError::ImageDecode(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(img),
        })
    }

    /// 从内部 DynamicImage 直接构造（仅 crate 内）。
    pub(crate) fn from_dynamic(inner: DynamicImage) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    /// 宽（像素）。
    pub fn width(&self) -> u32 {
        self.inner.width()
    }

    /// 高（像素）。
    pub fn height(&self) -> u32 {
        self.inner.height()
    }

    /// 按 BBox 裁剪（pixel coordinates，bbox 越界会被 clamp 到图片范围内）。
    pub fn crop(&self, bbox: BBox) -> Result<ImageData, AIError> {
        let w = self.inner.width() as f32;
        let h = self.inner.height() as f32;
        let x = bbox.x.max(0.0).min(w - 1.0) as u32;
        let y = bbox.y.max(0.0).min(h - 1.0) as u32;
        let x2 = (bbox.x + bbox.w).max(0.0).min(w) as u32;
        let y2 = (bbox.y + bbox.h).max(0.0).min(h) as u32;

        if x2 <= x || y2 <= y {
            return Err(AIError::InvalidInput(format!(
                "crop bbox out of range: ({x},{y}) - ({x2},{y2}) in {w}x{h}"
            )));
        }

        // Arc<DynamicImage> 不能 DerefMut，需要 clone 出 owned DynamicImage 再 crop
        let mut owned: DynamicImage = (*self.inner).clone();
        let cropped = owned.crop(x, y, x2 - x, y2 - y);
        Ok(ImageData {
            inner: Arc::new(cropped),
        })
    }

    /// 缩放到目标尺寸。
    pub fn resize(&self, width: u32, height: u32) -> ImageData {
        let owned: DynamicImage = (*self.inner).clone();
        let resized = owned.resize(width, height, image::imageops::FilterType::Lanczos3);
        ImageData {
            inner: Arc::new(resized),
        }
    }

    /// 转成 8-bit RGB。
    pub fn as_rgb8(&self) -> RgbImage {
        self.inner.to_rgb8()
    }

    /// 转成 8-bit luma（灰度）。
    pub fn to_luma8(&self) -> GrayImage {
        self.inner.to_luma8()
    }

    /// 内部 DynamicImage 引用（仅 crate 内 ONNX 适配使用）。
    pub(crate) fn inner(&self) -> &DynamicImage {
        &self.inner
    }

    /// 编码为 JPEG bytes（用于 IPC 传输缩略图）。
    pub fn to_jpeg_bytes(&self, quality: u8) -> Result<Vec<u8>, AIError> {
        let mut buf = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buf);
        self.inner
            .to_rgb8()
            .write_to(&mut cursor, image::ImageFormat::Jpeg)
            .map_err(|e| AIError::ImageDecode(e.to_string()))?;
        // quality 暂未生效（image 0.25 的 JPEG encoder 暂不暴露 quality 参数；Phase 2 替换 encoder）
        let _ = quality;
        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn make_test_image(w: u32, h: u32) -> ImageData {
        let buf: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(w, h, |x, y| Rgb([(x % 256) as u8, (y % 256) as u8, 128]));
        ImageData::from_dynamic(DynamicImage::from(buf))
    }

    #[test]
    fn from_dynamic_dimensions() {
        let img = make_test_image(100, 50);
        assert_eq!(img.width(), 100);
        assert_eq!(img.height(), 50);
    }

    #[test]
    fn crop_clamps_to_bounds() {
        let img = make_test_image(100, 100);
        let cropped = img.crop(BBox::new(-10.0, -10.0, 200.0, 200.0)).unwrap();
        assert_eq!(cropped.width(), 100);
        assert_eq!(cropped.height(), 100);
    }

    #[test]
    fn crop_invalid_bbox_errors() {
        let img = make_test_image(100, 100);
        let r = img.crop(BBox::new(50.0, 50.0, 0.0, 0.0));
        assert!(r.is_err());
    }

    #[test]
    fn resize_works() {
        let img = make_test_image(100, 100);
        let r = img.resize(50, 50);
        assert_eq!(r.width(), 50);
        assert_eq!(r.height(), 50);
    }
}