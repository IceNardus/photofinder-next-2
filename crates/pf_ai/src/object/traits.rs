//! 对象相关 trait。

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use pf_core::{BBox, Embedding, ModelVersion};

use crate::error::AIError;
use crate::image_data::ImageData;

/// 单次对象检测。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectDetection {
    /// 类别 ID
    pub class_id: i32,
    /// 类别名
    pub class_name: String,
    /// 置信度
    pub confidence: f32,
    /// 边界框
    pub bbox: BBox,
}

/// 对象检测器。
#[async_trait]
pub trait ObjectDetector: Send + Sync {
    /// 检测。
    async fn detect(&self, image: &ImageData) -> Result<Vec<ObjectDetection>, AIError>;

    /// 类别名称表。
    fn class_names(&self) -> &[String];

    /// 模型版本。
    fn model_version(&self) -> ModelVersion;
}

/// 对象 embedder。
#[async_trait]
pub trait ObjectEmbedder: Send + Sync {
    /// 整图 embedding。
    async fn embed(&self, image: &ImageData) -> Result<Embedding, AIError>;

    /// 裁剪区域 embedding（用于按类别细搜）。
    async fn embed_crop(
        &self,
        image: &ImageData,
        bbox: BBox,
    ) -> Result<Embedding, AIError>;

    /// 维度。
    fn dim(&self) -> usize;

    /// 模型版本。
    fn model_version(&self) -> ModelVersion;
}

/// 对象 pipeline：detect → embed（裁剪）。
pub struct ObjectPipeline {
    detector: Arc<dyn ObjectDetector>,
    embedder: Arc<dyn ObjectEmbedder>,
}

impl ObjectPipeline {
    /// 构造。
    pub fn new(detector: Arc<dyn ObjectDetector>, embedder: Arc<dyn ObjectEmbedder>) -> Self {
        Self { detector, embedder }
    }

    /// 获取 embedder（供 SearchService 使用）。
    pub fn embedder(&self) -> Arc<dyn ObjectEmbedder> {
        self.embedder.clone()
    }

    /// 获取 detector（供 CategorySearch 使用）。
    pub fn detector(&self) -> Arc<dyn ObjectDetector> {
        self.detector.clone()
    }

    /// 处理一张图片，返回 (类别ID, bbox, embedding)。
    pub async fn process(
        &self,
        image: &ImageData,
    ) -> Result<Vec<(ObjectDetection, Embedding)>, AIError> {
        let detections = self.detector.detect(image).await?;
        let mut results = Vec::new();
        for det in detections {
            let embedding = self.embedder.embed_crop(image, det.bbox).await?;
            results.push((det, embedding));
        }
        Ok(results)
    }
}