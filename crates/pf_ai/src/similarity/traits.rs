//! 相似图相关 trait。

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use pf_core::Embedding;

use crate::error::AIError;
use crate::image_data::ImageData;

/// 单个关键点。
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Keypoint {
    /// x 坐标
    pub x: f32,
    /// y 坐标
    pub y: f32,
    /// 描述子起始索引（指向全局 descriptor buffer）
    pub desc_offset: usize,
}

/// 一张图的全部关键点。
#[derive(Debug, Clone)]
pub struct KeypointSet {
    /// 关键点列表
    pub keypoints: Vec<Keypoint>,
    /// 描述子（所有 keypoint 描述子连续存储）
    pub descriptors: Vec<f32>,
    /// 描述子维度（SuperPoint = 256）
    pub descriptor_dim: usize,
}

impl KeypointSet {
    /// 关键点数。
    pub fn len(&self) -> usize {
        self.keypoints.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.keypoints.is_empty()
    }
}

/// 匹配结果。
#[derive(Debug, Clone)]
pub struct MatchResult {
    /// 匹配对：`(query_idx, candidate_idx)`，按 score 降序
    pub matches: Vec<(usize, usize)>,
    /// RANSAC 内点数量
    pub num_inliers: usize,
    /// 内点率（num_inliers / matches.len()）
    pub confidence: f32,
}

/// 关键点提取器（SuperPoint）。
#[async_trait]
pub trait KeypointExtractor: Send + Sync {
    /// 提取关键点 + 描述子。
    async fn extract(&self, image: &ImageData) -> Result<KeypointSet, AIError>;

    /// 描述子维度。
    fn descriptor_dim(&self) -> usize;
}

/// 特征匹配器（LightGlue）。
#[async_trait]
pub trait FeatureMatcher: Send + Sync {
    /// 匹配两组关键点。
    async fn match_features(
        &self,
        query: &KeypointSet,
        candidate: &KeypointSet,
    ) -> Result<MatchResult, AIError>;

    /// 最小匹配阈值（少于这个数视为不匹配）。
    fn min_matches(&self) -> usize;
}

/// 把 KeypointSet 聚合成一个全局 embedding（VLAD / NetVLAD）。
///
/// 注意：这个 trait 不在 `pf_ai` 而是在 `pf_application`，因为它跨模块。
/// 这里只暴露占位 trait name。
pub trait GlobalDescriptorAggregator: Send + Sync {
    /// 把关键点集合聚合成单个 embedding。
    fn aggregate(&self, kp_set: &KeypointSet) -> Embedding;
}