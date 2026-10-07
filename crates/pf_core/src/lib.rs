//! # pf_core — 业务核心 domain types
//!
//! 不依赖 ONNX / SQLite / HNSW / Tauri / 任何平台。
//! 不知道 image::DynamicImage、tokio、tauri。
//!
//! 职责：
//! - 定义领域类型（Image / Face / Person / Object / Embedding / Detection）
//! - 定义业务规则（BBox / ClusterPolicy）
//! - 提供纯函数工具

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod bbox;
pub mod clustering;
pub mod domain;
pub mod error;

pub use bbox::BBox;
pub use clustering::{Clusterer, PersonAssignment};
pub use domain::{
    Detection, Embedding, Face, FaceKeypoints, FACE_MODEL_NAME, Image, ImageFilter, Metadata,
    ModelVersion, Object, ObjectSearchResult, Person, SearchResult,
};
pub use error::CoreError;