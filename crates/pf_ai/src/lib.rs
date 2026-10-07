//! # pf_ai — AI 推理层
//!
//! Phase 0 / Phase 1 脚手架：仅 trait + 类型 + `ImageData` 抽象。
//! 真正的 `ScrfdDetector` / `ArcFaceEmbedder` / `YoloV8Detector` 在后续 Phase 实现。
//!
//! 不知道 SQLite / Tauri / 任何业务。

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod body;
pub mod error;
pub mod face;
pub mod filter;
pub mod image_classifier;
pub mod image_data;
pub mod object;
pub mod quality;
pub mod similarity;

pub use body::{YouTuReIdEmbedder, BodyCropStrategy, BodyFeature, BodyPipeline, BODY_EMBEDDING_DIM, BODY_INPUT_SIZE};
pub use error::AIError;
pub use face::{
    AlignedFace, ArcFaceEmbedder, DetectorOrigin, FaceAligner, FaceDetection, FaceDetector,
    FaceEmbedder, FaceFeature, FacePipeline, Keypoints, QualityFilter, ScrfdDetector,
    SimpleAligner,
};
pub use image_classifier::{ImageType, ImageTypeClassifier};
pub use image_data::ImageData;
pub use object::{
    CategoryConfig, CategorySearch, CategorySearchResult, LightGlueMatcher, MobileClipEmbedder,
    ObjectDetection, ObjectDetector, ObjectEmbedder, ObjectPipeline, Roi, SuperPointExtractor,
    YoloV8Detector, LIGHTGLUE_DESC_DIM, MOBILECLIP_DIM, MOBILECLIP_INPUT_SIZE,
    SUPERPOINT_DESC_DIM,
};
pub use similarity::{FeatureMatcher, Keypoint, KeypointExtractor, KeypointSet, MatchResult, VladAggregator, PatchConfig, PatchExtractor, PatchFeatures, PatchSearchResult, PatchSearchService, Patch};