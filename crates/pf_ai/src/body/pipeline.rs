//! Body pipeline: face bbox → body crop → embedding.
//!
//! Takes a face bounding box and expands it to a body region using a configurable
//! crop strategy, then extracts YouTu Re-ID embeddings.

use std::sync::Arc;

use image::RgbImage;
use tracing::debug;

use pf_core::{BBox, Embedding};

use crate::body::{YouTuReIdEmbedder, BODY_INPUT_SIZE};
use crate::error::AIError;
use crate::image_data::ImageData;

/// Body crop strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyCropStrategy {
    /// Tight bounding box around face (face only)
    BBoxTight,
    /// Face + 20% margin on each side
    BBoxP20,
    /// Head and upper body: 2x face height, 1.5x face width
    HeadBody,
    /// Head and full upper body: 2.5x face height, 1.3x face width
    UpperBody,
}

impl Default for BodyCropStrategy {
    fn default() -> Self {
        // Production default: head_body
        BodyCropStrategy::HeadBody
    }
}

impl BodyCropStrategy {
    /// Expand face bbox to body bbox using this strategy.
    pub fn expand(&self, face: &BBox, img_h: f32, img_w: f32) -> BBox {
        let cx = face.x + face.w / 2.0;
        let cy = face.y + face.h / 2.0;

        match self {
            BodyCropStrategy::BBoxTight => *face,
            BodyCropStrategy::BBoxP20 => {
                let m = 0.20;
                BBox::new(
                    (face.x - face.w * m).max(0.0),
                    (face.y - face.h * m).max(0.0),
                    (face.w * (1.0 + 2.0 * m)).min(img_w),
                    (face.h * (1.0 + 2.0 * m)).min(img_h),
                )
            }
            BodyCropStrategy::HeadBody => {
                let new_h = face.h * 2.0;
                let new_w = face.w * 1.5;
                let nx = (cx - new_w / 2.0).max(0.0);
                let ny = (cy - new_h * 0.5).max(0.0);
                BBox::new(nx, ny, new_w.min(img_w - nx), new_h.min(img_h - ny))
            }
            BodyCropStrategy::UpperBody => {
                let new_h = face.h * 2.5;
                let new_w = face.w * 1.3;
                let nx = (cx - new_w / 2.0).max(0.0);
                let ny = (cy - new_h * 0.3).max(0.0);
                BBox::new(nx, ny, new_w.min(img_w - nx), new_h.min(img_h - ny))
            }
        }
    }
}

/// A single body feature extracted from an image.
#[derive(Debug, Clone)]
pub struct BodyFeature {
    /// Body bounding box (in image coordinates)
    pub bbox: BBox,
    /// 768D body embedding
    pub embedding: Embedding,
    /// Crop strategy used
    pub crop_strategy: BodyCropStrategy,
    /// Quality score (based on face detection score)
    pub quality_score: f32,
}

/// Body pipeline: face bbox → body crop → YouTu Re-ID embedding.
pub struct BodyPipeline {
    embedder: Arc<YouTuReIdEmbedder>,
    crop_strategy: BodyCropStrategy,
}

impl BodyPipeline {
    /// Construct a new BodyPipeline.
    pub fn new(embedder: Arc<YouTuReIdEmbedder>, crop_strategy: BodyCropStrategy) -> Self {
        Self {
            embedder,
            crop_strategy,
        }
    }

    /// Process an image given a face detection, returning body features.
    ///
    /// Takes the face bounding box, expands it to a body region using the
    /// configured crop strategy, extracts the crop, resizes to 256×128,
    /// and runs YouTu Re-ID embedding.
    pub fn process(
        &self,
        image: &ImageData,
        face_bbox: &BBox,
        face_quality: f32,
    ) -> Result<BodyFeature, AIError> {
        let img_h = image.height() as f32;
        let img_w = image.width() as f32;

        // Expand face bbox to body bbox
        let body_bbox = self.crop_strategy.expand(face_bbox, img_h, img_w);

        // Crop body region from image
        let crop = image.crop(body_bbox).map_err(|e| AIError::Preprocess(e.to_string()))?;

        // Convert to RgbImage
        let rgb = crop.as_rgb8();

        // Embed via YouTu Re-ID (handles resize to 256×128 internally)
        let embedding = self.embedder.embed(&rgb)?;

        Ok(BodyFeature {
            bbox: body_bbox,
            embedding,
            crop_strategy: self.crop_strategy,
            quality_score: face_quality,
        })
    }

    /// Get crop strategy.
    pub fn crop_strategy(&self) -> BodyCropStrategy {
        self.crop_strategy
    }

    /// Get embedding dimension.
    pub fn dim(&self) -> usize {
        self.embedder.dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crop_strategy_expand() {
        let face = BBox::new(100.0, 100.0, 80.0, 80.0);

        // BBoxTight: unchanged
        let result = BodyCropStrategy::BBoxTight.expand(&face, 600.0, 800.0);
        assert!((result.x - 100.0).abs() < 0.1);
        assert!((result.y - 100.0).abs() < 0.1);

        // HeadBody: 2x height, 1.5x width
        let result = BodyCropStrategy::HeadBody.expand(&face, 600.0, 800.0);
        assert!((result.w - 80.0 * 1.5).abs() < 0.1);
        assert!((result.h - 80.0 * 2.0).abs() < 0.1);
    }
}
