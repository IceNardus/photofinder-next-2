//! 相似图搜索 trait（Phase 3）。
//!
//! 用 SuperPoint 提取关键点，LightGlue 做匹配，VLAD 聚合到 global descriptor。

pub mod traits;
pub mod vlad;
pub mod patch_search;

pub use traits::{FeatureMatcher, Keypoint, KeypointExtractor, KeypointSet, MatchResult};
pub use vlad::VladAggregator;
pub use patch_search::{PatchConfig, PatchExtractor, PatchFeatures, PatchSearchResult, PatchSearchService, Patch};