//! MobileCLIP-s2 全图 / 裁剪 embedder。
//!
//! **算法与 ai-next/src-tauri/src/ai/clip/mobileclip_s2.py **逐位对齐**：
//!
//! | 项 | 值 |
//! |---|---|
//! | 输入尺寸 | 224×224 |
//! | embedding 维度 | 512 |
//! | 通道顺序 | RGB |
//! | 归一化 | `imageNetMean/Std`（mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]） |
//! | 重采样 | `Triangle`（双线性） |
//!
//! 用途：object full-image embedding（512-d），进入 HNSW 做粗筛；之后由 LightGlue 精排。

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use image::imageops::FilterType;
use image::RgbImage;
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;
use tracing::info;

use pf_core::{BBox, Embedding, ModelVersion};

use crate::error::AIError;
use crate::image_data::ImageData;
use crate::object::traits::{ObjectDetection, ObjectEmbedder};

/// 输入尺寸。
pub const INPUT_SIZE: u32 = 224;
/// embedding 维度。
pub const EMBEDDING_DIM: usize = 512;
/// L2 归一化阈值。
pub const L2_EPS: f32 = 1e-6;

/// ImageNet 归一化均值（RGB 顺序）。
const IMAGE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// ImageNet 归一化标准差（RGB 顺序）。
const IMAGE_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// MobileCLIP-s2 embedder。
pub struct MobileClipEmbedder {
    session: Mutex<Option<Session>>,
    version: ModelVersion,
}

impl MobileClipEmbedder {
    /// 从模型文件加载。
    pub fn load(model_path: &Path) -> Result<Arc<Self>, AIError> {
        info!("Loading MobileCLIP-s2 from: {}", model_path.display());
        let session = Session::builder()
            .map_err(|e| AIError::ModelLoad(format!("session builder: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| AIError::ModelLoad(format!("commit_from_file: {e}")))?;
        Ok(Arc::new(Self {
            session: Mutex::new(Some(session)),
            version: ModelVersion::new("mobileclip-s2@v1.0.0"),
        }))
    }

    /// RGB CHW 输入 + ImageNet 归一化。
    fn build_input(image: &RgbImage) -> Vec<f32> {
        let resized = image::imageops::resize(image, INPUT_SIZE, INPUT_SIZE, FilterType::Triangle);
        let n = (INPUT_SIZE * INPUT_SIZE) as usize;
        let mut input = Vec::with_capacity(3 * n);
        for c in 0..3 {
            for y in 0..INPUT_SIZE {
                for x in 0..INPUT_SIZE {
                    let p = resized.get_pixel(x, y)[c] as f32 / 255.0;
                    input.push((p - IMAGE_MEAN[c]) / IMAGE_STD[c]);
                }
            }
        }
        input
    }

    /// L2 归一化。
    fn l2_normalize(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(L2_EPS);
        for x in v.iter_mut() {
            *x /= norm;
        }
    }

    /// Run 全图 embedding。
    fn embed_internal(&self, image: &RgbImage) -> Result<Vec<f32>, AIError> {
        let mut session_guard = self.session.lock();
        let session = session_guard
            .as_mut()
            .ok_or_else(|| AIError::ModelLoad("session not loaded".into()))?;
        let input = Self::build_input(image);
        let tensor = Tensor::from_array(([1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize], input))
            .map_err(|e| AIError::Inference(format!("tensor: {e}")))?;
        let outputs = session
            .run(ort::inputs![tensor])
            .map_err(|e| AIError::Inference(format!("run: {e}")))?;
        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| AIError::Inference(format!("extract: {e}")))?;
        let mut vec: Vec<f32> = data[..EMBEDDING_DIM.min(data.len())].to_vec();
        Self::l2_normalize(&mut vec);
        Ok(vec)
    }
}

#[async_trait]
impl ObjectEmbedder for MobileClipEmbedder {
    async fn embed(&self, image: &ImageData) -> Result<Embedding, AIError> {
        let rgb = image.as_rgb8();
        let vec = self.embed_internal(&rgb)?;
        Ok(Embedding {
            model: self.version.clone(),
            values: vec.clone(),
            dim: vec.len(),
        })
    }

    async fn embed_crop(
        &self,
        image: &ImageData,
        bbox: BBox,
    ) -> Result<Embedding, AIError> {
        let crop = image.crop(bbox)?;
        let rgb = crop.as_rgb8();
        let vec = self.embed_internal(&rgb)?;
        Ok(Embedding {
            model: self.version.clone(),
            values: vec.clone(),
            dim: vec.len(),
        })
    }

    fn dim(&self) -> usize {
        EMBEDDING_DIM
    }

    fn model_version(&self) -> ModelVersion {
        self.version.clone()
    }
}

/// 辅助：从一组 ObjectDetection 跑全图 embedding（用于 `search_by_object_image`）。
pub async fn embed_detections(
    embedder: &dyn ObjectEmbedder,
    image: &ImageData,
    detections: &[ObjectDetection],
) -> Vec<(ObjectDetection, Embedding)> {
    let mut out = Vec::with_capacity(detections.len());
    for d in detections {
        match embedder.embed_crop(image, d.bbox).await {
            Ok(e) => out.push((d.clone(), e)),
            Err(e) => {
                tracing::warn!(class = %d.class_name, error = %e, "embed_crop failed");
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn input_size_is_224() {
        assert_eq!(INPUT_SIZE, 224);
    }

    #[test]
    fn embedding_dim_is_512() {
        assert_eq!(EMBEDDING_DIM, 512);
    }

    #[test]
    fn l2_normalize_unit_length() {
        let mut v = vec![3.0_f32, 4.0];
        MobileClipEmbedder::l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn build_input_shape_is_chw() {
        let img = RgbImage::from_pixel(64, 64, Rgb([128, 128, 128]));
        let data = MobileClipEmbedder::build_input(&img);
        // 3 * INPUT_SIZE * INPUT_SIZE
        assert_eq!(data.len(), 3 * (INPUT_SIZE as usize) * (INPUT_SIZE as usize));
    }
}
