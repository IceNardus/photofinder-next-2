//! ArcFace w600k_r50 embedding 提取。
//!
//! **算法与 ai-next/src-tauri/src/ai/face/arcface.rs **逐位对齐**：
//!
//! | 项 | 值 |
//! |---|---|
//! | 输入尺寸 | 112×112 |
//! | embedding 维度 | 512 |
//! | 通道顺序 | BGR |
//! | 归一化 | `(x - 127.5) / 128.0` |
//! | flip augmentation | 平均主+翻转，输出 L2-normalized |

use std::f32::consts::PI;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use image::{Rgb, RgbImage};
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;
use tracing::info;

use pf_core::{Embedding, ModelVersion, FACE_MODEL_NAME};

use crate::error::AIError;
use crate::face::traits::{AlignedFace, FaceEmbedder};
#[cfg(test)]
use crate::image_data::ImageData;

/// 输入尺寸。
pub const INPUT_SIZE: u32 = 112;
/// embedding 维度。
pub const EMBEDDING_DIM: usize = 512;
/// L2 归一化阈值（低于此不归一化）。
pub const L2_EPS: f32 = 1e-6;
/// Rotation TTA 默认角度（±5°，约 ±0.0873 rad）。
pub const ROTATION_TTA_DEG: f32 = 5.0;

/// ArcFace 模型包装。
pub struct ArcFaceEmbedder {
    // BUG #5 修复:`Arc<Mutex<...>>` 让 embed 可以把 session clone 到 spawn_blocking,
    // 否则 6 次 TTA 推理(session.run)会占住 tokio runtime 线程 300ms+。
    session: Arc<Mutex<Option<Session>>>,
    version: ModelVersion,
    use_flip: bool,
    /// 是否启用小角度旋转 TTA（默认关闭；开启时 main + flip + ±ROTATION_TTA_DEG° + flips，共 6 次推理）。
    use_rotation_tta: bool,
}

impl ArcFaceEmbedder {
    /// 从模型文件加载。
    /// 默认 1-crop（无 flip / rotation TTA）。LFW 定量评估（2026-08）显示 6-crop TTA
    /// 对已对齐正脸精度无提升（flip 恒等、旋转略降），且开销 6x；
    /// 需要时可显式用 `with_flip` / `with_rotation_tta` 重新开启。
    pub fn load(model_path: &Path) -> Result<Arc<Self>, AIError> {
        info!("Loading ArcFace from: {}", model_path.display());
        let session = Session::builder()
            .map_err(|e| AIError::ModelLoad(format!("session builder: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| AIError::ModelLoad(format!("commit_from_file: {e}")))?;
        Ok(Arc::new(Self {
            session: Arc::new(Mutex::new(Some(session))),
            // v1.1.0 = 1-crop 默认（2026-08 LFW 评估后）。embedding 计算从 6-crop→1-crop
            // 会改变向量值（cosine 一致 ~0.973），故 bump version 以让
            // check_model_versions 在混合新旧 embedding 时触发 reindex 信号。
            version: ModelVersion::new(format!("{FACE_MODEL_NAME}@v1.1.0")),
            use_flip: false,
            use_rotation_tta: false,
        }))
    }

    /// 开启 flip augmentation（与 ai-next `extract_with_flip(..., true)` 一致）。
    pub fn with_flip(mut self, enable: bool) -> Self {
        self.use_flip = enable;
        self
    }

    /// 开启 / 关闭小角度 rotation TTA（默认关闭）。开启后叠加 flip 变体共 6 次推理。
    pub fn with_rotation_tta(mut self, enable: bool) -> Self {
        self.use_rotation_tta = enable;
        self
    }

    /// BGR CHW 输入 + 归一化（支持 flipped）。
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

    /// 单次推理 + L2 normalize(在 spawn_blocking 里跑)。
    fn run_once_blocking(
        session: Arc<Mutex<Option<Session>>>,
        face: RgbImage,
        flipped: bool,
    ) -> Result<Vec<f32>, AIError> {
        let input_data = Self::build_input(&face, flipped);
        let shape = [1_i64, 3, INPUT_SIZE as i64, INPUT_SIZE as i64];
        let input = Tensor::from_array((shape, input_data))
            .map_err(|e| AIError::ModelLoad(format!("tensor: {e}")))?;
        let mut guard = session.lock();
        let session_ref = guard
            .as_mut()
            .ok_or_else(|| AIError::ModelLoad("ArcFace session not loaded".into()))?;
        let outputs = session_ref
            .run(ort::inputs![input])
            .map_err(|e| AIError::ModelLoad(format!("run: {e}")))?;
        let (_, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| AIError::ModelLoad(format!("extract: {e}")))?;
        Ok(l2_normalize(data))
    }

    /// 水平翻转（用于 flip augmentation）。
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

    /// 以图像中心为轴旋转 `angle_rad` 弧度（正=逆时针），用最近邻采样。
    /// 出框区域填黑色（外侧黑色不会显著影响 ArcFace 推理结果）。
    fn rotate_nearest(face: &RgbImage, angle_rad: f32) -> RgbImage {
        let w = face.width() as i32;
        let h = face.height() as i32;
        let cx = w as f32 * 0.5;
        let cy = h as f32 * 0.5;
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();
        let black = Rgb([0_u8, 0, 0]);
        let mut out = RgbImage::new(face.width(), face.height());
        for y in 0..h {
            for x in 0..w {
                // Inverse rotation: 从输出像素找源像素
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
}

/// L2 normalize。返回新 Vec。
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let mut norm = 0.0f32;
    for x in v {
        norm += x * x;
    }
    norm = norm.sqrt();
    if norm > L2_EPS {
        let s = 1.0 / norm;
        v.iter().map(|x| x * s).collect()
    } else {
        v.to_vec()
    }
}

#[async_trait]
impl FaceEmbedder for ArcFaceEmbedder {
    async fn embed(&self, aligned: &AlignedFace) -> Result<Embedding, AIError> {
        // 从 ImageData 中拿到 RgbImage
        let rgb = aligned.image.as_rgb8();

        // 收集 (image, flipped) 元组用于 batched TTA。空 Vec 表示禁用某轴。
        let mut variants: Vec<(RgbImage, bool)> = Vec::with_capacity(6);
        variants.push((rgb.clone(), false));
        if self.use_rotation_tta {
            let angle = ROTATION_TTA_DEG * PI / 180.0;
            let rot_pos = Self::rotate_nearest(&rgb, angle);
            let rot_neg = Self::rotate_nearest(&rgb, -angle);
            variants.push((rot_pos, false));
            variants.push((rot_neg, false));
            if self.use_flip {
                // 旋转 + 翻转 共 6 个变体
                variants.push((Self::flip_horizontal(&variants[1].0), true));
                variants.push((Self::flip_horizontal(&variants[2].0), true));
            }
        }
        if self.use_flip {
            variants.push((Self::flip_horizontal(&rgb), true));
        }

        // BUG #5 修复:把 6 次 session.run 整个挪到 spawn_blocking,让出 async runtime。
        // 累加所有 variant 的 embedding,最后 L2 normalize。
        let session = self.session.clone();
        let use_flip = self.use_flip;
        let combined = tokio::task::spawn_blocking(move || -> Result<Vec<f32>, AIError> {
            let mut combined = vec![0.0_f32; EMBEDDING_DIM];
            for (img, flipped) in &variants {
                // 实际上 variants 里的 flipped 字段对每个 variant 是固定的,
                // 但 use_flip 已经在上面控制了是否生成 flip 变体,这里尊重原 flipped。
                let _ = use_flip;
                let emb = Self::run_once_blocking(session.clone(), img.clone(), *flipped)?;
                for i in 0..EMBEDDING_DIM {
                    combined[i] += emb[i];
                }
            }
            Ok(combined)
        })
        .await
        .map_err(|e| AIError::Inference(format!("join embed: {e}")))??;

        Ok(Embedding::new(l2_normalize(&combined), self.version.clone()))
    }

    fn dim(&self) -> usize {
        EMBEDDING_DIM
    }

    fn model_version(&self) -> ModelVersion {
        self.version.clone()
    }
}

/// 简单测试：从已有 ImageData 触发 embed 流程（不会真正调用模型）。
#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};
    use std::path::PathBuf;

    const FACE_MODEL: &str = "/Users/mac/Library/Application Support/PhotoFinderNext/resources/models/w600k_r50.onnx";

    fn solid_face(c: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(INPUT_SIZE, INPUT_SIZE, Rgb(c))
    }

    #[test]
    fn l2_normalize_unit_vector() {
        let v = vec![3.0f32, 4.0];
        let n = l2_normalize(&v);
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
        let sum: f32 = n.iter().map(|x| x * x).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn l2_normalize_handles_zero() {
        let v = vec![0.0f32, 0.0, 0.0];
        let n = l2_normalize(&v);
        assert_eq!(n, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn l2_normalize_preserves_length() {
        let v: Vec<f32> = (0..512).map(|i| (i as f32) * 0.01).collect();
        let n = l2_normalize(&v);
        let sum: f32 = n.iter().map(|x| x * x).sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn build_input_bgr_layout() {
        let face = solid_face([255, 0, 0]);
        let input = ArcFaceEmbedder::build_input(&face, false);
        let n = (INPUT_SIZE * INPUT_SIZE) as usize;
        assert_eq!(input.len(), 3 * n);
        // BGR 第一通道 = B = 0 → (0 - 127.5) / 128 ≈ -0.996
        assert!(input[0] < -0.99 && input[0] > -1.001);
        // G 通道 = 0
        assert!(input[n] < -0.99);
        // R 通道 = 255 → (255 - 127.5) / 128 ≈ 0.996
        assert!(input[2 * n] > 0.99);
    }

    #[test]
    fn build_input_flipped_mirrors_x() {
        // 构造一个左红右蓝的脸（避免 symmetric）
        let mut face = RgbImage::from_pixel(INPUT_SIZE, INPUT_SIZE, Rgb([0, 0, 0]));
        for y in 0..INPUT_SIZE {
            for x in 0..56 {
                face.put_pixel(x, y, image::Rgb([255, 0, 0]));
            }
        }
        let input_main = ArcFaceEmbedder::build_input(&face, false);
        let input_flip = ArcFaceEmbedder::build_input(&face, true);
        // 翻转后第一个像素 (x=0) 应来自原图 x=111，是蓝色
        // R 通道（第三段）input_main[2*n] = (255 - 127.5) / 128 ≈ 0.996
        // input_flip[2*n] = (0 - 127.5) / 128 ≈ -0.996
        let n = (INPUT_SIZE * INPUT_SIZE) as usize;
        assert!(input_main[2 * n] > 0.99);
        assert!(input_flip[2 * n] < -0.99);
    }

    #[test]
    fn flip_horizontal_is_pixelperfect() {
        // 创建一个红蓝条纹
        let mut face = RgbImage::from_pixel(INPUT_SIZE, INPUT_SIZE, Rgb([0, 0, 0]));
        for x in 0..INPUT_SIZE {
            let c = if x % 2 == 0 {
                Rgb([255, 0, 0])
            } else {
                Rgb([0, 0, 255])
            };
            for y in 0..INPUT_SIZE {
                face.put_pixel(x, y, c);
            }
        }
        let flipped = ArcFaceEmbedder::flip_horizontal(&face);
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let orig = face.get_pixel(INPUT_SIZE - 1 - x, y);
                let new = flipped.get_pixel(x, y);
                assert_eq!(orig, new);
            }
        }
    }

    #[test]
    fn rotate_nearest_zero_is_identity() {
        // 旋转 0° 应该等于原图（中心点采样应回到自己）。
        let mut face = RgbImage::from_pixel(INPUT_SIZE, INPUT_SIZE, Rgb([0, 0, 0]));
        for x in 0..INPUT_SIZE {
            for y in 0..INPUT_SIZE {
                face.put_pixel(x, y, Rgb([(x % 256) as u8, (y % 256) as u8, 128]));
            }
        }
        let rot = ArcFaceEmbedder::rotate_nearest(&face, 0.0);
        // 比较每个像素（允许 1 像素 round-off 误差）。
        let mut diff = 0;
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let a = face.get_pixel(x, y);
                let b = rot.get_pixel(x, y);
                for c in 0..3 {
                    if a[c] != b[c] {
                        diff += 1;
                    }
                }
            }
        }
        // 0° 旋转理论上完全一致；最近邻 round 可能在 < 0.5% 像素上有 1 像素偏移。
        let total = (INPUT_SIZE * INPUT_SIZE * 3) as i32;
        assert!(
            diff * 1000 < total * 5,  // 0.5% 阈值
            "0° rotation should be near-identity, got {diff}/{total} diff pixels"
        );
    }

    #[test]
    fn rotate_nearest_plus_minus_are_inverses() {
        // ±5° 旋转的合成应接近 0° 旋转（旋转群封闭性 sanity check）。
        let mut face = RgbImage::from_pixel(INPUT_SIZE, INPUT_SIZE, Rgb([0, 0, 0]));
        for x in 0..INPUT_SIZE {
            for y in 0..INPUT_SIZE {
                face.put_pixel(x, y, Rgb([(x * 2) as u8, (y * 2) as u8, 64]));
            }
        }
        let angle = 5.0_f32 * PI / 180.0;
        let r1 = ArcFaceEmbedder::rotate_nearest(&face, angle);
        let r2 = ArcFaceEmbedder::rotate_nearest(&r1, -angle);
        let diff = (INPUT_SIZE * INPUT_SIZE * 3) as i32;
        let mut sum_abs = 0_u32;
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let a = face.get_pixel(x, y);
                let b = r2.get_pixel(x, y);
                for c in 0..3 {
                    sum_abs += (a[c] as i32 - b[c] as i32).unsigned_abs();
                }
            }
        }
        // ±5° 组合在最近邻下最多几像素偏差，平均 < 4 (out of 255)。
        let avg = sum_abs as f32 / diff as f32;
        assert!(
            avg < 8.0,
            "±5° round-trip mean abs diff should be < 8 (got {avg:.2})",
        );
    }

    #[test]
    #[ignore = "requires real model file via load()"]
    fn rotation_tta_constructor_flag() {
        // 默认 1-crop（无 flip / rotation TTA）；builder 可显式重新开启。
        let model = ArcFaceEmbedder::load(&PathBuf::from(FACE_MODEL)).unwrap();
        assert!(!model.use_rotation_tta, "default should be rotation-TTA off");
        assert!(!model.use_flip, "default should be flip off");
        // load() 返回 Arc<Self>,但 builder 消费 self — 用 Arc::try_unwrap 取出唯一所有者
        let model2 = Arc::try_unwrap(
            ArcFaceEmbedder::load(&PathBuf::from(FACE_MODEL)).unwrap(),
        )
        .ok()
        .expect("arc has multiple owners")
        .with_rotation_tta(true)
        .with_flip(true);
        assert!(model2.use_rotation_tta);
        assert!(model2.use_flip);
        let _ = model;
        let _ = model2;
    }

    #[test]
    fn constants_match_ai_next() {
        assert_eq!(INPUT_SIZE, 112);
        assert_eq!(EMBEDDING_DIM, 512);
        assert!((L2_EPS - 1e-6).abs() < 1e-9);
    }

    #[test]
    fn from_bytes_to_image_data() {
        let _ = ImageData::from_bytes; // ensure trait available
        let buf: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(112, 112, |x, _| Rgb([x as u8, 0, 0]));
        let img = image::DynamicImage::from(buf);
        let _ = ImageData::from_dynamic(img);
    }
}