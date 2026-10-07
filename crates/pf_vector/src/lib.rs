//! # pf_vector — 向量索引引擎
//!
//! 不依赖任何 AI / DB / 业务类型。看到的只是 `&[f32]` 和 `id: i64`。
//!
//! 实现路线：
//! - Phase 1：`HnswIndex`（基于 `hnsw-rs` 真 HNSW）
//! - Phase 4：`QuantizedHnswIndex`（PQ 量化）+ `MmapVectorStorage`

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod hnsw_index;
pub mod traits;

pub use error::VectorError;
pub use hnsw_index::{HnswIndex, DEFAULT_EF_CONSTRUCTION, DEFAULT_EF_SEARCH, DEFAULT_MAX_LAYER, DEFAULT_MAX_NB_CONNECTION};
pub use traits::{Metric, SearchHit, VectorIndex};