//! 图像类型分类器（对齐 ai-next `ai/image_classifier.rs`）。
//!
//! 将图片分类为：Photo / Screenshot / Document / Poster / Meme / Anime / Unknown。
//!
//! | 类型 | 判断逻辑 |
//! |---|---|
//! | Screenshot | aspect < 0.4 或 > 2.5 |
//! | Anime | 熵 < 4.0 |
//! | Poster | 平均饱和度 > 0.65 |
//! | Meme | 颜色种类 < 100 |
//! | Screenshot | 亮度方差 < 0.01 且 极端宽高比 |
//! | Photo | 其他 |

use image::GenericImageView;
use std::path::Path;

/// 图像类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageType {
    /// 真实照片。
    Photo,
    /// 截屏。
    Screenshot,
    /// 文档（扫描件）。
    Document,
    /// 海报/设计图。
    Poster,
    /// 表情包/文字图。
    Meme,
    /// 动漫/手绘。
    Anime,
    /// 未知。
    Unknown,
}

impl std::fmt::Display for ImageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Photo => write!(f, "Photo"),
            Self::Screenshot => write!(f, "Screenshot"),
            Self::Document => write!(f, "Document"),
            Self::Poster => write!(f, "Poster"),
            Self::Meme => write!(f, "Meme"),
            Self::Anime => write!(f, "Anime"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// 图像类型分类器。
#[derive(Debug, Clone)]
pub struct ImageTypeClassifier {
    /// 最小宽/高。
    pub min_width_height: u32,
    /// 最大宽高比。
    pub max_aspect_ratio: f32,
    /// 最小熵。
    pub min_entropy: f32,
    /// 最大饱和度（超过认为是 Poster）。
    pub max_saturation: f32,
}

impl Default for ImageTypeClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageTypeClassifier {
    /// 用默认阈值构造。
    pub fn new() -> Self {
        Self {
            min_width_height: 256,
            max_aspect_ratio: 2.5,
            min_entropy: 4.0,
            max_saturation: 0.65,
        }
    }

    /// 从路径分类。
    pub fn classify_path(&self, path: &Path) -> (ImageType, String) {
        let img = match image::open(path) {
            Ok(i) => i,
            Err(e) => return (ImageType::Unknown, format!("decode error: {e}")),
        };
        self.classify_image(&img)
    }

    /// 对已解码的图片分类。
    pub fn classify_image(&self, img: &image::DynamicImage) -> (ImageType, String) {
        let (w, h) = img.dimensions();

        // 1. 尺寸检查
        if w < self.min_width_height || h < self.min_width_height {
            return (ImageType::Unknown, format!("too small: {}x{}", w, h));
        }

        // 2. 宽高比检查 → Screenshot
        let aspect = w as f32 / h as f32;
        if aspect < 0.4 || aspect > self.max_aspect_ratio {
            return (ImageType::Screenshot, format!("aspect ratio {:.2}", aspect));
        }

        let rgba = img.to_rgba8();

        // 3. 熵检查 → Anime
        let entropy = self.compute_entropy(&rgba);
        if entropy < self.min_entropy {
            return (ImageType::Anime, format!("low entropy {:.2}", entropy));
        }

        // 4. 饱和度检查 → Poster
        let sat = self.compute_avg_saturation(&rgba);
        if sat > self.max_saturation {
            return (ImageType::Poster, format!("high saturation {:.2}", sat));
        }

        // 5. 颜色数量检查 → Meme
        let color_count = self.compute_color_count(&rgba);
        if color_count < 100 {
            return (ImageType::Meme, format!("few colors {}", color_count));
        }

        // 6. 亮度方差 + 极端宽高比 → Screenshot
        let brightness_var = self.compute_brightness_variance(&rgba);
        if brightness_var < 0.01 && (aspect < 0.6 || aspect > 1.8) {
            return (ImageType::Screenshot, format!("likely screenshot, var={:.3}", brightness_var));
        }

        // 7. 默认 → Photo
        (ImageType::Photo, format!("ok, entropy={:.1}, sat={:.2}, colors={}", entropy, sat, color_count))
    }

    /// 判断是否为可接受的图像类型（对齐 ai-next：只接受 Photo）。
    pub fn is_acceptable(&self, img: &image::DynamicImage) -> bool {
        let (t, _) = self.classify_image(img);
        t == ImageType::Photo
    }

    /// 同 is_acceptable（别名，对齐 ai-next 的命名）。
    pub fn should_process(&self, img: &image::DynamicImage) -> bool {
        self.is_acceptable(img)
    }

    /// 计算灰度熵（对齐 ai-next：基于直方图的标准熵）。
    fn compute_entropy(&self, rgba: &image::RgbaImage) -> f32 {
        let mut hist = vec![0usize; 256];
        for pixel in rgba.pixels() {
            let gray = (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32) as u8;
            hist[gray as usize] += 1;
        }
        let total = rgba.pixels().count() as f32;
        let mut entropy = 0.0_f32;
        for &count in &hist {
            if count == 0 {
                continue;
            }
            let p = count as f32 / total;
            entropy -= p * p.log2();
        }
        entropy
    }

    /// 计算平均饱和度。
    fn compute_avg_saturation(&self, rgba: &image::RgbaImage) -> f32 {
        let mut total_sat = 0.0_f32;
        let mut count = 0u32;

        for pixel in rgba.pixels() {
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;

            let max_c = r.max(g).max(b);
            let min_c = r.min(g).min(b);
            let delta = max_c - min_c;

            if delta < 1e-6 {
                continue;
            }

            let l = (max_c + min_c) / 2.0;

            let s = if l < 0.5 {
                delta / (max_c + min_c)
            } else {
                delta / (2.0 - max_c - min_c)
            };

            total_sat += s;
            count += 1;
        }

        if count == 0 {
            return 0.0;
        }
        total_sat / count as f32
    }

    /// 计算颜色种类数。
    fn compute_color_count(&self, rgba: &image::RgbaImage) -> usize {
        use std::collections::HashSet;
        let mut set: HashSet<u32> = HashSet::with_capacity(10_000);
        for p in rgba.pixels() {
            let rgb = ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | (p[2] as u32);
            set.insert(rgb);
            if set.len() > 200_000 {
                break;
            }
        }
        set.len()
    }

    /// 计算亮度方差。
    fn compute_brightness_variance(&self, rgba: &image::RgbaImage) -> f32 {
        let n = rgba.pixels().len() as f32;
        if n == 0.0 {
            return 0.0;
        }
        let mean: f32 = rgba
            .pixels()
            .map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)
            .sum::<f32>()
            / n;
        let var: f32 = rgba
            .pixels()
            .map(|p| {
                let v = 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
                (v - mean).powi(2)
            })
            .sum::<f32>()
            / n;
        var
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn photo_type_has_acceptable_entropy() {
        // 模拟真实照片：灰度基底 + 小色彩扰动 → 高熵 + 低饱和 → Photo
        let img = image::DynamicImage::from(image::RgbaImage::from_fn(400, 300, |x, y| {
            let base = ((x + y * 2) % 256) as u8;
            let noise = ((x * 7 + y * 3) % 30) as u8;
            let r = base.saturating_add(noise);
            let g = base;
            let b = base.saturating_sub(noise);
            image::Rgba([r, g, b, 255])
        }));
        let classifier = ImageTypeClassifier::new();
        let (t, reason) = classifier.classify_image(&img);
        assert_eq!(t, ImageType::Photo, "got {t}: {reason}");
    }

    #[test]
    fn low_entropy_is_anime() {
        // 几乎纯色 → 低熵 → Anime
        let img = image::DynamicImage::from(image::RgbaImage::from_pixel(400, 300, image::Rgba([200, 100, 100, 255])));
        let classifier = ImageTypeClassifier::new();
        let (t, _) = classifier.classify_image(&img);
        assert_eq!(t, ImageType::Anime, "got {t}");
    }

    #[test]
    fn extreme_aspect_is_screenshot() {
        // 超宽图 → Screenshot（宽高都要 >=256 才能过第一个检查）
        let img = image::DynamicImage::from(image::RgbaImage::from_fn(2560, 256, |_x, _y| {
            image::Rgba([128; 4])
        }));
        let classifier = ImageTypeClassifier::new();
        let (t, _) = classifier.classify_image(&img);
        assert_eq!(t, ImageType::Screenshot, "got {t}");
    }

    #[test]
    fn too_small_is_unknown() {
        let img = image::DynamicImage::from(image::RgbaImage::from_pixel(100, 100, image::Rgba([128; 4])));
        let classifier = ImageTypeClassifier::new();
        let (t, _) = classifier.classify_image(&img);
        assert_eq!(t, ImageType::Unknown, "got {t}");
    }
}
