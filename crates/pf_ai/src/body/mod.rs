//! Body embedding module for person re-identification.
//!
//! Uses YouTu Re-ID ONNX model to extract 768D body appearance embeddings.
//! Input: 256×128 RGB body crop
//! Output: 768D L2-normalized embedding

mod embedder;
mod pipeline;

pub use embedder::YouTuReIdEmbedder;
pub use pipeline::{BodyCropStrategy, BodyFeature, BodyPipeline};

// Re-export Embedding type
use pf_core::Embedding;

/// Body embedding dimension for YouTu Re-ID
pub const BODY_EMBEDDING_DIM: usize = 768;

/// Body embedding model name for database queries
pub const BODY_MODEL_NAME: &str = "ytu_reid";

/// YouTu Re-ID input size (height, width)
pub const BODY_INPUT_SIZE: (u32, u32) = (256, 128);

/// Preprocessing mean values (ImageNet statistics)
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// Preprocessing std values (ImageNet statistics)
const STD: [f32; 3] = [0.229, 0.224, 0.225];
