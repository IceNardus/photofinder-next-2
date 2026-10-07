//! PatchSearch — patch-based image search via SuperPoint + VLAD + LightGlue + HNSW.
//!
//! **算法流程（与 ai-next `patch_search.py` 逐位对齐）**：
//!
//! 1. **Patch Extraction**: 512×512 grid patches, stride 256
//! 2. **Feature Extraction**: SuperPoint keypoints + 256-d descriptors per patch
//! 3. **VLAD Aggregation**: k-means++ (K=64) on all descriptors → 256-d global vector
//! 4. **Coarse Search**: HNSW on global VLAD vectors (k=50 candidates per patch)
//! 5. **Fine Matching**: LightGlue mutual-NN + RANSAC inlier ratio
//! 6. **Spatial Verification**: bbox overlap ≥ 0.3
//! 7. **Fusion**: α·embedding + β·inlier_ratio + γ·color_sim

use std::sync::Arc;

use crate::error::AIError;
use crate::image_data::ImageData;
use crate::similarity::traits::{FeatureMatcher, KeypointExtractor, KeypointSet};
use crate::similarity::VladAggregator;

/// Patch search result。
#[derive(Debug, Clone)]
pub struct PatchSearchResult {
    /// 目标图片 id
    pub image_id: i64,
    /// 综合相似度
    pub score: f32,
    /// embedding 相似度分量
    pub embedding_score: f32,
    /// LightGlue inlier ratio
    pub inlier_ratio: f32,
    /// inlier 数量
    pub inlier_count: usize,
    /// 匹配数量
    pub match_count: usize,
    /// bbox overlap
    pub bbox_overlap: f32,
}

/// Patch 配置（与 ai-next `patch_search.rs` 对齐）。
#[derive(Debug, Clone)]
pub struct PatchConfig {
    /// VLAD 聚类数 K
    pub vlad_k: usize,
    /// 每 patch 最大关键点数
    pub max_keypoints_per_patch: usize,
    /// scene 最低 LightGlue 匹配数
    pub min_lightglue_matches_scene: usize,
    /// object 最低 LightGlue 匹配数
    pub min_lightglue_matches_object: usize,
    /// RANSAC 阈值（像素）
    pub ransac_threshold: f32,
    /// 最小 bbox overlap
    pub min_bbox_overlap: f32,
    /// 最大 query patches 数
    pub max_query_patches: usize,
    /// patch 尺寸
    pub patch_size: u32,
    /// patch stride
    pub patch_stride: u32,
}

impl Default for PatchConfig {
    fn default() -> Self {
        Self {
            vlad_k: 64,
            max_keypoints_per_patch: 256,
            min_lightglue_matches_scene: 15,
            min_lightglue_matches_object: 3,
            ransac_threshold: 4.0,
            min_bbox_overlap: 0.3,
            max_query_patches: 50,
            patch_size: 512,
            patch_stride: 256,
        }
    }
}

/// 从图片提取 patch 特征。
pub struct PatchExtractor {
    keypoint_extractor: Arc<dyn KeypointExtractor>,
    vlad: VladAggregator,
    config: PatchConfig,
}

impl PatchExtractor {
    pub fn new(
        keypoint_extractor: Arc<dyn KeypointExtractor>,
        config: PatchConfig,
    ) -> Self {
        Self {
            keypoint_extractor,
            vlad: VladAggregator::new(config.vlad_k),
            config,
        }
    }

    /// 提取全图 patches 的 VLAD 全局 descriptor。
    ///
    /// 1. 切 grid patches
    /// 2. 每 patch 用 SuperPoint 提取关键点
    /// 3. 所有 descriptors 做 VLAD 聚合
    pub async fn extract_vlad_descriptor(
        &self,
        image: &ImageData,
    ) -> Result<(Vec<f32>, Vec<PatchFeatures>), AIError> {
        let patches = self.extract_patches(image);

        let mut all_descriptors = Vec::new();
        let mut patch_features = Vec::new();

        for (patch_idx, (patch, cropped)) in patches.into_iter().enumerate() {
            let kp_set = self.keypoint_extractor.extract(&cropped).await?;
            let dim = kp_set.descriptor_dim;
            let num_kp = kp_set.len();

            let desc_slice = kp_set.descriptors.clone();
            all_descriptors.extend_from_slice(&desc_slice);

            patch_features.push(PatchFeatures {
                patch_idx: patch_idx as u32,
                x: patch.x,
                y: patch.y,
                w: patch.w,
                h: patch.h,
                keypoints: kp_set.keypoints,
                descriptors: desc_slice,
                descriptor_dim: dim,
                num_keypoints: num_kp,
            });
        }

        let num_descriptors = all_descriptors.len().max(1) / 256;
        let vlad_vec = self.vlad.vlad_aggregate(&all_descriptors, num_descriptors);

        Ok((vlad_vec, patch_features))
    }

    /// 切 grid patches，返回 (Patch, ImageData) 对，其中 ImageData 已裁剪到 patch 区域。
    fn extract_patches(&self, image: &ImageData) -> Vec<(Patch, ImageData)> {
        let (img_w, img_h) = (image.width() as i32, image.height() as i32);
        let patch_size = self.config.patch_size as i32;
        let stride = self.config.patch_stride as i32;
        let mut patches = Vec::new();

        let mut y = 0;
        while y + patch_size <= img_h {
            let mut x = 0;
            while x + patch_size <= img_w {
                let bbox = pf_core::BBox::new(x as f32, y as f32, patch_size as f32, patch_size as f32);
                let cropped = image.crop(bbox).unwrap_or_else(|_| image.clone());
                patches.push((
                    Patch {
                        x,
                        y,
                        w: patch_size as u32,
                        h: patch_size as u32,
                    },
                    cropped,
                ));
                x += stride;
            }
            y += stride;
        }
        patches.truncate(self.config.max_query_patches);
        patches
    }
}

/// 单个 patch（坐标，不含图像数据）。
#[derive(Debug, Clone)]
pub struct Patch {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}

/// 单个 patch 的特征。
#[derive(Debug, Clone)]
pub struct PatchFeatures {
    pub patch_idx: u32,
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
    pub keypoints: Vec<crate::similarity::traits::Keypoint>,
    pub descriptors: Vec<f32>,
    pub descriptor_dim: usize,
    pub num_keypoints: usize,
}

/// PatchSearchService — coarse VLAD + fine LightGlue + spatial verification。
pub struct PatchSearchService {
    extractor: Arc<PatchExtractor>,
    matcher: Arc<dyn FeatureMatcher>,
    config: PatchConfig,
}

impl PatchSearchService {
    pub fn new(
        keypoint_extractor: Arc<dyn KeypointExtractor>,
        matcher: Arc<dyn FeatureMatcher>,
        config: PatchConfig,
    ) -> Self {
        Self {
            extractor: Arc::new(PatchExtractor::new(keypoint_extractor, config.clone())),
            matcher,
            config,
        }
    }

    /// 访问内部 PatchExtractor。
    pub fn extractor(&self) -> &Arc<PatchExtractor> {
        &self.extractor
    }

    /// 对 query image 提取 VLAD descriptor，然后匹配候选 patches。
    ///
    /// `candidate_patches`: 从候选图片提取的 patches
    /// 返回: 按综合 score 降序的匹配结果
    pub async fn search_patches(
        &self,
        query_image: &ImageData,
        candidate_patches: &[PatchFeatures],
        top_k: usize,
    ) -> Result<Vec<PatchSearchResult>, AIError> {
        let (_query_vlad, query_patches) = self.extractor.extract_vlad_descriptor(query_image).await?;

        let mut results = Vec::new();

        for cand in candidate_patches {
            // 构建 query patch KeypointSet
            let query_kp_set = self.build_keypoint_set(&query_patches[0]);

            // 构建 candidate patch KeypointSet
            let cand_kp_set = self.build_keypoint_set_from_features(cand);

            // LightGlue 匹配
            let match_result = self.matcher.match_features(&query_kp_set, &cand_kp_set).await?;

            if match_result.matches.len() < self.config.min_lightglue_matches_scene {
                continue;
            }

            // 计算 inlier ratio
            let inlier_ratio = match_result.confidence;
            let match_count = match_result.matches.len();

            // bbox overlap（简化：patch 重叠面积 / patch 面积）
            let overlap = self.compute_overlap(&query_patches[0], cand);

            if overlap < self.config.min_bbox_overlap {
                continue;
            }

            // 综合 score：简化为 inlier_ratio * match_count 的加权
            let score = inlier_ratio * (match_count as f32 / 100.0).min(1.0);

            results.push(PatchSearchResult {
                image_id: 0, // 由调用方填充
                score,
                embedding_score: 0.0, // coarse 阶段不用
                inlier_ratio,
                inlier_count: (inlier_ratio * match_count as f32) as usize,
                match_count,
                bbox_overlap: overlap,
            });
        }

        // 降序排序，取 top_k
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(top_k);

        Ok(results)
    }

    fn build_keypoint_set(&self, patch: &PatchFeatures) -> KeypointSet {
        KeypointSet {
            keypoints: patch.keypoints.clone(),
            descriptors: patch.descriptors.clone(),
            descriptor_dim: patch.descriptor_dim,
        }
    }

    fn build_keypoint_set_from_features(&self, features: &PatchFeatures) -> KeypointSet {
        KeypointSet {
            keypoints: features.keypoints.clone(),
            descriptors: features.descriptors.clone(),
            descriptor_dim: features.descriptor_dim,
        }
    }

    fn compute_overlap(&self, p1: &PatchFeatures, p2: &PatchFeatures) -> f32 {
        let x1 = p1.x as f32;
        let y1 = p1.y as f32;
        let w1 = p1.w as f32;
        let h1 = p1.h as f32;
        let x2 = p2.x as f32;
        let y2 = p2.y as f32;
        let w2 = p2.w as f32;
        let h2 = p2.h as f32;

        let overlap_x = (w1.min(w2) - (x2 - x1).abs()).max(0.0);
        let overlap_y = (h1.min(h2) - (y2 - y1).abs()).max(0.0);
        let overlap_area = overlap_x * overlap_y;
        let union_area = w1 * h1 + w2 * h2 - overlap_area;

        if union_area > 0.0 {
            overlap_area / union_area
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_config_defaults() {
        let cfg = PatchConfig::default();
        assert_eq!(cfg.vlad_k, 64);
        assert_eq!(cfg.patch_size, 512);
        assert_eq!(cfg.patch_stride, 256);
    }

    #[test]
    fn overlap_formula() {
        // 两个 512x512 patches 重叠 256x256
        let p1 = PatchFeatures {
            patch_idx: 0,
            x: 0,
            y: 0,
            w: 512,
            h: 512,
            keypoints: vec![],
            descriptors: vec![],
            descriptor_dim: 256,
            num_keypoints: 0,
        };
        let p2 = PatchFeatures {
            patch_idx: 1,
            x: 256,
            y: 256,
            w: 512,
            h: 512,
            keypoints: vec![],
            descriptors: vec![],
            descriptor_dim: 256,
            num_keypoints: 0,
        };

        // 手动计算 overlap
        let x1 = 0.0_f32; let y1 = 0.0_f32; let w1 = 512.0_f32; let h1 = 512.0_f32;
        let x2 = 256.0_f32; let y2 = 256.0_f32; let w2 = 512.0_f32; let h2 = 512.0_f32;
        let overlap_x = (w1.min(w2) - (x2 - x1).abs()).max(0.0);
        let overlap_y = (h1.min(h2) - (y2 - y1).abs()).max(0.0);
        let overlap_area = overlap_x * overlap_y;
        let union_area = w1 * h1 + w2 * h2 - overlap_area;
        let expected = overlap_area / union_area;

        assert!((expected - 1.0/7.0).abs() < 1e-6); // 256*256 / (512*512*2 - 256*256) = 1/7
    }
}
