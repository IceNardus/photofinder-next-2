//! CategorySearch — YOLOv8 检测 + MobileCLIP embedding + prototype averaging.
//!
//! **算法与 ai-next `category_search/detector.py` 逐位对齐**：
//!
//! | 项 | 值 |
//! |---|---|
//! | 检测器 | YOLOv8n |
//! | Embedder | MobileCLIP-s2（512-d） |
//! | Index | HNSW（cosine） |
//! | Prototype | 检测类别的 cropped embeddings 平均 + L2 归一化 |
//! | 搜索 | cosine similarity on HNSW |

use std::sync::Arc;

use crate::error::AIError;
use crate::image_data::ImageData;
use crate::object::traits::ObjectDetector;
use crate::object::traits::ObjectEmbedder;

/// Category 配置。
#[derive(Debug, Clone)]
pub struct CategoryConfig {
    /// MobileCLIP embedding 维度（512）
    pub embedding_dim: usize,
    /// 最小检测置信度
    pub min_confidence: f32,
    /// Prototype 平均时最小样本数
    pub min_prototype_samples: usize,
    /// 搜索返回的最大结果数
    pub max_results: usize,
}

impl Default for CategoryConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 512,
            min_confidence: 0.3,
            min_prototype_samples: 1,
            max_results: 50,
        }
    }
}

/// CategorySearch 结果。
#[derive(Debug, Clone)]
pub struct CategorySearchResult {
    /// 图片 id
    pub image_id: i64,
    /// 类别名
    pub category: String,
    /// MobileCLIP cosine 相似度
    pub similarity: f32,
    /// 检测置信度
    pub detection_confidence: f32,
    /// bbox
    pub bbox: pf_core::BBox,
}

/// CategorySearch — YOLOv8 检测 + MobileCLIP embedding + HNSW prototype search。
pub struct CategorySearch {
    detector: Arc<dyn ObjectDetector>,
    embedder: Arc<dyn ObjectEmbedder>,
    config: CategoryConfig,
}

impl CategorySearch {
    /// 构造。
    pub fn new(
        detector: Arc<dyn ObjectDetector>,
        embedder: Arc<dyn ObjectEmbedder>,
    ) -> Self {
        Self {
            detector,
            embedder,
            config: CategoryConfig::default(),
        }
    }

    /// 从单张图片提取类别 prototype embedding。
    ///
    /// 1. YOLOv8 检测所有对象
    /// 2. 取最高置信度的检测 crop
    /// 3. MobileCLIP embed
    /// 4. L2 归一化
    pub async fn extract_prototype_from_image(
        &self,
        image: &ImageData,
    ) -> Result<Vec<f32>, AIError> {
        let detections = self.detector.detect(image).await?;

        if detections.is_empty() {
            return Err(AIError::InvalidInput("no objects detected".into()));
        }

        // 取最高置信度检测
        let best = detections
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .cloned()
            .ok_or_else(|| AIError::InvalidInput("no best detection".into()))?;

        // Crop
        let cropped = image.crop(best.bbox)?;

        // MobileCLIP embed
        let embedding = self.embedder.embed(&cropped).await?;

        // L2 normalize
        let mut emb = embedding.as_slice().to_vec();
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for x in emb.iter_mut() {
            *x /= norm;
        }

        Ok(emb)
    }

    /// 从多张图片创建类别 prototype（平均 + 归一化）。
    ///
    /// `image_paths`: 示例图片路径列表
    pub async fn create_prototype(
        &self,
        images: &[ImageData],
    ) -> Result<Vec<f32>, AIError> {
        if images.len() < self.config.min_prototype_samples {
            return Err(AIError::InvalidInput(format!(
                "need at least {} samples for prototype",
                self.config.min_prototype_samples
            )));
        }

        let mut sum = vec![0.0_f32; self.config.embedding_dim];
        let mut count = 0usize;

        for img in images {
            match self.extract_prototype_from_image(img).await {
                Ok(emb) => {
                    for i in 0..self.config.embedding_dim {
                        sum[i] += emb[i];
                    }
                    count += 1;
                }
                Err(_) => continue,
            }
        }

        if count < self.config.min_prototype_samples {
            return Err(AIError::InvalidInput(format!(
                "only {} valid detections, need {}",
                count, self.config.min_prototype_samples
            )));
        }

        // 平均
        let n = count as f32;
        for i in 0..self.config.embedding_dim {
            sum[i] /= n;
        }

        // L2 归一化
        let norm: f32 = sum.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for x in sum.iter_mut() {
            *x /= norm;
        }

        Ok(sum)
    }

    /// 搜索与 prototype 最相似的已索引图片。
    ///
    /// `prototype`: L2 归一化 prototype 向量
    /// `index`: HNSW vector index（512-d MobileCLIP embeddings）
    /// `top_k`: 返回数量
    ///
    /// 返回: (image_id, similarity) 列表
    pub fn search_by_prototype(
        &self,
        prototype: &[f32],
        index: &dyn pf_vector::VectorIndex,
        top_k: usize,
    ) -> Result<Vec<(i64, f32)>, AIError> {
        let hits = index
            .search(prototype, top_k)
            .map_err(|e| AIError::Vector(e.to_string()))?;
        Ok(hits.into_iter().map(|h| (h.id, h.score)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn category_config_defaults() {
        let cfg = CategoryConfig::default();
        assert_eq!(cfg.embedding_dim, 512);
        assert_eq!(cfg.min_prototype_samples, 1);
    }
}
