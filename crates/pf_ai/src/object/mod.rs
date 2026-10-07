//! 对象检测 / embedding / 关键点模块。
//!
//! Phase 3 新增：
//! - `mobileclip`: MobileCLIP-s2 全图 / 裁剪 embedder（512-d）
//! - `superpoint`: SuperPoint 关键点 + 256-d 描述子
//! - `lightglue`: LightGlue mutual-NN + RANSAC 匹配
//! - `roi`: multi-scale sliding window ROI 提取

pub mod category_search;
pub mod homography;
pub mod lightglue;
pub mod mobileclip;
pub mod roi;
pub mod selective_search;
pub mod superpoint;
pub mod traits;
pub mod yolo;

pub use category_search::{CategoryConfig, CategorySearch, CategorySearchResult};
pub use homography::{estimate_homography_dlt, ransac_homography, Homography, RansacConfig, RansacResult};
pub use lightglue::{l2_distance, LightGlueMatcher, DESCRIPTOR_DIM as LIGHTGLUE_DESC_DIM};
pub use mobileclip::{
    MobileClipEmbedder, EMBEDDING_DIM as MOBILECLIP_DIM, INPUT_SIZE as MOBILECLIP_INPUT_SIZE,
};
pub use roi::{crop_roi, extract_rois, Roi};
pub use selective_search::{selective_search, SelectiveRoi};
pub use superpoint::{SuperPointExtractor, DESCRIPTOR_DIM as SUPERPOINT_DESC_DIM};
pub use traits::{ObjectDetection, ObjectDetector, ObjectEmbedder, ObjectPipeline};
pub use yolo::YoloV8Detector;
