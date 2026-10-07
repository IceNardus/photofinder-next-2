//! Visual Scan V2 - DINOv2 feature extraction
//!
//! Provides DINOv2 ViT-S/14 feature extraction for visual search.
//! This module is platform-agnostic and can be used by both desktop and iOS.

pub mod types;

pub use types::{
    RoiFeature, ScanStatus,
    VisualFeatureV2, VisualRoi, VisualScanResult, VisualScanStatusV2, VisualScanSummary,
};
pub use types::dino::{DinoExtractor, DinoError, DINO_INPUT, DINO_MEAN, DINO_STD, PATCH_DIM};
