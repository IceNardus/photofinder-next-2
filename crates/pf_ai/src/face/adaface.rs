//! AdaFace IR101 embedding extraction.
//!
//! Based on paper "AdaFace: Quality-Aware Face Recognition" (CVPR 2022).
//! Extension of ArcFace with adaptive margin based on image quality.
//!
//! | 项 | 值 |
//! |---|---|
//! | 输入尺寸 | 112×112 |
//! | embedding 维度 | 512 |
//! | 通道顺序 | BGR |
//! | 归一化 | `(x - 127.5) / 127.5` (different from ArcFace's / 128.0) |
//! | flip augmentation | 平均主+翻转，输出 L2-normalized |

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use image::RgbImage;
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;
use tracing::info;

use pf_core::{Embedding, ModelVersion, FACE_MODEL_NAME};

use crate::error::AIError;
use crate::face::traits::{AlignedFace, FaceEmbedder};

/// 输入尺寸。
pub const INPUT_SIZE: u32 = 112;
/// embedding 维度。
pub const EMBEDDING_DIM: usize = 512;
/// L2 归一化阈值。
pub const L2_EPS: f32 = 1e-6;

/// AdaFace 模型包装。
pub struct AdaFaceEmbedder {
    session: Arc<Mutex<Option<Session>>>,
    version: ModelVersion,
    use_flip: bool,
}

impl AdaFaceEmbedder {
    /// 从模型文件加载。
    pub fn load(model_path: &Path) -> Result<Arc<Self>, AIError> {
        info!("Loading AdaFace from: {}", model_path.display());
        let session = Session::builder()
            .map_err(|e| AIError::ModelLoad(format!("session builder: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| AIError::ModelLoad(format!("commit_from_file: {e}")))?;
        Ok(Arc::new(Self {
            session: Arc::new(Mutex::new(Some(session))),
            version: ModelVersion::new(format!("{}@adaface-v1", FACE_MODEL_NAME)),
            use_flip: false,
        }))
    }

    /// 开启 flip augmentation。
    pub fn with_flip(mut self, enable: bool) -> Self {
        self.use_flip = enable;
        self
    }

    /// BGR CHW 输入 + 归一化。
    /// **关键差异**: AdaFace 使用 `/ 127.5`，而 ArcFace 使用 `/ 128.0`。
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
                // AdaFace normalization: / 127.5
                input.push((p[2] as f32 - 127.5) / 127.5);
            }
        }
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let p = if flipped {
                    face.get_pixel(INPUT_SIZE - 1 - x, y)
                } else {
                    face.get_pixel(x, y)
                };
                input.push((p[1] as f32 - 127.5) / 127.5);
            }
        }
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let p = if flipped {
                    face.get_pixel(INPUT_SIZE - 1 - x, y)
                } else {
                    face.get_pixel(x, y)
                };
                input.push((p[0] as f32 - 127.5) / 127.5);
            }
        }
        input
    }

    /// 单次推理 + L2 normalize。
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
            .ok_or_else(|| AIError::ModelLoad("AdaFace session not loaded".into()))?;
        let outputs = session_ref
            .run(ort::inputs![input])
            .map_err(|e| AIError::ModelLoad(format!("run: {e}")))?;
        let (_, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| AIError::ModelLoad(format!("extract: {e}")))?;
        Ok(l2_normalize(data))
    }

    /// 水平翻转。
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
}

/// L2 normalize。
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
impl FaceEmbedder for AdaFaceEmbedder {
    async fn embed(&self, aligned: &AlignedFace) -> Result<Embedding, AIError> {
        let rgb = aligned.image.as_rgb8();

        let mut variants: Vec<(RgbImage, bool)> = Vec::with_capacity(2);
        variants.push((rgb.clone(), false));
        if self.use_flip {
            variants.push((Self::flip_horizontal(&rgb), true));
        }

        let session = self.session.clone();
        let combined = tokio::task::spawn_blocking(move || -> Result<Vec<f32>, AIError> {
            let mut combined = vec![0.0_f32; EMBEDDING_DIM];
            for (img, flipped) in &variants {
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn solid_face(c: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(INPUT_SIZE, INPUT_SIZE, Rgb(c))
    }

    #[test]
    fn l2_normalize_unit_vector() {
        let v = vec![3.0f32, 4.0];
        let n = l2_normalize(&v);
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn build_input_bgr_layout() {
        let face = solid_face([255, 0, 0]);
        let input = AdaFaceEmbedder::build_input(&face, false);
        let n = (INPUT_SIZE * INPUT_SIZE) as usize;
        assert_eq!(input.len(), 3 * n);
        // AdaFace normalization: (0 - 127.5) / 127.5 ≈ -1.0
        assert!(input[0] < -0.99 && input[0] > -1.001);
        // G channel
        assert!(input[n] < -0.99);
        // R channel: (255 - 127.5) / 127.5 ≈ 1.0
        assert!(input[2 * n] > 0.99);
    }

    #[test]
    fn constants_match_arcface() {
        // Same dimensions as ArcFace for compatibility
        assert_eq!(INPUT_SIZE, 112);
        assert_eq!(EMBEDDING_DIM, 512);
    }
}