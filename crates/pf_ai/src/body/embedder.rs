//! YouTu Re-ID ONNX embedder.
//!
//! Model: person_reid_youtu_2021nov.onnx
//! Input: 256×128 RGB (BGR order for ONNX)
//! Output: 768D embedding, L2-normalized

use std::path::Path;
use std::sync::Arc;

use image::{Rgb, RgbImage};
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;
use tracing::info;

use pf_core::{Embedding, ModelVersion};

use crate::body::{BODY_EMBEDDING_DIM, BODY_INPUT_SIZE, MEAN, STD};
use crate::error::AIError;

/// YouTu Re-ID embedder using ONNX Runtime.
pub struct YouTuReIdEmbedder {
    session: Arc<Mutex<Option<Session>>>,
    version: ModelVersion,
}

impl YouTuReIdEmbedder {
    /// Load from YouTu Re-ID ONNX model file.
    pub fn load(model_path: &Path) -> Result<Arc<Self>, AIError> {
        info!("Loading YouTu Re-ID from: {}", model_path.display());
        let session = Session::builder()
            .map_err(|e| AIError::ModelLoad(format!("session builder: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| AIError::ModelLoad(format!("commit_from_file: {e}")))?;

        Ok(Arc::new(Self {
            session: Arc::new(Mutex::new(Some(session))),
            version: ModelVersion::new("ytu_reid@v1".to_string()),
        }))
    }

    /// Embed a body crop image (256×128 RGB).
    pub fn embed(&self, crop: &RgbImage) -> Result<Embedding, AIError> {
        let (target_h, target_w) = BODY_INPUT_SIZE;

        // Build CHW input tensor with ImageNet normalization
        let mut input = Vec::with_capacity(3 * target_h as usize * target_w as usize);

        // Process each channel (BGR order for ONNX compatibility)
        for channel_offset in 0..3 {
            // Target size 256×128
            let mut channel_data = Vec::with_capacity((target_h * target_w) as usize);
            for y in 0..target_h {
                for x in 0..target_w {
                    // Bilinear-like resize: map to crop coordinates
                    let src_x = (x as f32 * crop.width() as f32 / target_w as f32) as u32;
                    let src_y = (y as f32 * crop.height() as f32 / target_h as f32) as u32;
                    let px = src_x.min(crop.width() - 1);
                    let py = src_y.min(crop.height() - 1);
                    let pixel = crop.get_pixel(px, py);

                    // Get channel value (BGR order for ONNX)
                    let channel_idx = 2 - channel_offset; // RGB→BGR
                    let value = match channel_idx {
                        0 => pixel[2], // B
                        1 => pixel[1], // G
                        _ => pixel[0], // R
                    };

                    // Normalize: (x - mean) / std
                    let normalized = (value as f32 / 255.0 - MEAN[channel_offset]) / STD[channel_offset];
                    channel_data.push(normalized);
                }
            }
            input.extend(channel_data);
        }

        let shape = [1_i64, 3, target_h as i64, target_w as i64];
        let tensor = Tensor::from_array((shape, input)).map_err(|e| AIError::Inference(e.to_string()))?;

        let mut session_guard = self.session.lock();
        let session = session_guard.as_mut().ok_or_else(|| {
            AIError::Inference("session not loaded".to_string())
        })?;

        let outputs = session
            .run(ort::inputs![tensor])
            .map_err(|e| AIError::Inference(format!("run failed: {e}")))?;

        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| AIError::Inference(format!("extract tensor: {e}")))?;

        // L2 normalize
        let embedding = l2_normalize(data);

        Ok(Embedding::new(embedding, self.version.clone()))
    }

    /// Get embedding dimension.
    pub fn dim(&self) -> usize {
        BODY_EMBEDDING_DIM
    }

    /// Get model version.
    pub fn version(&self) -> &ModelVersion {
        &self.version
    }
}

/// L2 normalize a vector.
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-8 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedder_dim() {
        // Just verify the embedder can be constructed (actual embedding requires model file)
        // YouTuReIdEmbedder::load would need the model file
        assert_eq!(BODY_EMBEDDING_DIM, 768);
        assert_eq!(BODY_INPUT_SIZE, (256, 128));
    }
}
