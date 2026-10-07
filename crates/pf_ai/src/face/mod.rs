//! 人脸 pipeline trait + 类型 + ONNX 实现。

pub mod aligner;
pub mod adaface;
pub mod arcface;
pub mod pose;
pub mod scrfd;
pub mod traits;

pub use adaface::{
    AdaFaceEmbedder, EMBEDDING_DIM as ADAFACE_EMBEDDING_DIM, INPUT_SIZE as ADAFACE_INPUT_SIZE,
};
pub use aligner::{AlignmentConfig, ALIGNMENT_VERSION, REF_LANDMARKS_RAW, SimpleAligner};
pub use arcface::{ArcFaceEmbedder, EMBEDDING_DIM, INPUT_SIZE as ARCFACE_INPUT_SIZE};
pub use pose::estimate_yaw_pitch_roll;
pub use scrfd::{
    INPUT_SIZE as SCRFD_INPUT_SIZE, MIN_CONFIDENCE, MIN_RAW_SCORE, NMS_IOU_THRESHOLD,
    STRIDES as SCRFD_STRIDES, ScrfdDetector,
};
pub use traits::{
    AlignedFace, DetectorOrigin, FaceAligner, FaceDetection, FaceDetector, FaceEmbedder,
    FaceFeature, FacePipeline, Keypoints, QualityFilter,
};