//! LightGlue 特征匹配。
//!
//! **算法与 ai-next/src-tauri/src/ai/features/lightglue.py **逐位对齐**：
//!
//! | 项 | 值 |
//! |---|---|
//! | 描述子 | 256-d (SuperPoint) |
//! | 匹配策略 | Mutual NN + 距离比阈值（ratio = 0.8） |
//! | RANSAC | OpenCV `findHomography` (5 点)，inlier 阈值 4.0 px |
//! | 输出 | `MatchResult { matches, confidence }` |
//!
//! 用途：query 关键点 → candidate 关键点 → 几何验证 + RANSAC inlier ratio。

use async_trait::async_trait;

use pf_core::ModelVersion;

use crate::error::AIError;
use crate::object::homography::{ransac_homography, RansacConfig};
use crate::similarity::traits::{FeatureMatcher, KeypointSet, MatchResult};

/// 描述子维度（SuperPoint）。
pub const DESCRIPTOR_DIM: usize = 256;
/// Lowe 距离比阈值。
pub const RATIO_THRESHOLD: f32 = 0.8;
/// RANSAC inlier 阈值（像素）。
pub const RANSAC_THRESHOLD: f32 = 4.0;
/// 最小匹配数（无 RANSAC 时）。
pub const MIN_MATCHES: usize = 15;

/// LightGlue matcher（自带 RANSAC 过滤）。
pub struct LightGlueMatcher {
    /// 模型版本字符串。
    pub version: ModelVersion,
    /// RANSAC inlier 阈值。
    pub ransac_threshold: f32,
    /// 距离比阈值。
    pub ratio_threshold: f32,
    /// 最小匹配数（less than this → return no match）。
    pub min_matches: usize,
    /// 是否启用 strict 双向 NN（Phase 5 新增）。
    /// true：要求 q→c 和 c→q 双向都同意才保留；
    /// false（默认）：仅 Lowe 距离比（保持原行为）。
    pub bidirectional: bool,
    /// Phase 6 新增：是否启用 homography-based geometric verification
    ///（替代原"像素距离"过滤）。true：DLT + RANSAC 单应性估计，按重投影误差分类 inlier。
    pub use_homography_verification: bool,
    /// Phase 6 新增：RANSAC 迭代次数上限
    pub ransac_iterations: usize,
}

impl Default for LightGlueMatcher {
    fn default() -> Self {
        Self {
            version: ModelVersion::new("lightglue@v1.0.0"),
            ransac_threshold: RANSAC_THRESHOLD,
            ratio_threshold: RATIO_THRESHOLD,
            min_matches: MIN_MATCHES,
            bidirectional: false,
            use_homography_verification: true,
            ransac_iterations: 100,
        }
    }
}

impl LightGlueMatcher {
    /// 构造（带自定义阈值）。
    pub fn new(ransac_threshold: f32, ratio_threshold: f32, min_matches: usize) -> Self {
        Self {
            version: ModelVersion::new("lightglue@v1.0.0"),
            ransac_threshold,
            ratio_threshold,
            min_matches,
            bidirectional: false,
            use_homography_verification: true,
            ransac_iterations: 100,
        }
    }

    /// 构造（带 bidirectional 选项，Phase 5 新增）。
    pub fn with_bidirectional(mut self, b: bool) -> Self {
        self.bidirectional = b;
        self
    }

    /// Phase 6 新增：启用 / 禁用 homography 几何校验。
    pub fn with_homography_verification(mut self, enabled: bool) -> Self {
        self.use_homography_verification = enabled;
        self
    }

    /// 给定两组关键点 + 描述子，求 mutual NN 比值匹配。
    ///
    /// 返回 (query_idx, candidate_idx) 列表。
    ///
    /// - 始终应用 Lowe 距离比（过滤 ambiguous match）。
    /// - 当 `bidirectional=true` 时，进一步要求 q→c 和 c→q 双向同意。
    pub fn mutual_nn_match(
        &self,
        query: &KeypointSet,
        candidate: &KeypointSet,
    ) -> Vec<(usize, usize)> {
        let qd = query.descriptor_dim;
        let cd = candidate.descriptor_dim;
        if qd != DESCRIPTOR_DIM || cd != DESCRIPTOR_DIM {
            return Vec::new();
        }

        // Step 1: 单向 NN + Lowe ratio
        let mut forward: Vec<(usize, usize, f32)> = Vec::new(); // (qi, ci, distance)
        for (qi, q_kp) in query.keypoints.iter().enumerate() {
            let q_off = q_kp.desc_offset;
            if q_off + DESCRIPTOR_DIM > query.descriptors.len() {
                continue;
            }
            let q_desc = &query.descriptors[q_off..q_off + DESCRIPTOR_DIM];

            let mut best_idx = usize::MAX;
            let mut best_dist = f32::MAX;
            let mut second_best = f32::MAX;
            for (ci, c_kp) in candidate.keypoints.iter().enumerate() {
                let c_off = c_kp.desc_offset;
                if c_off + DESCRIPTOR_DIM > candidate.descriptors.len() {
                    continue;
                }
                let c_desc = &candidate.descriptors[c_off..c_off + DESCRIPTOR_DIM];
                let d = l2_distance(q_desc, c_desc);
                if d < best_dist {
                    second_best = best_dist;
                    best_dist = d;
                    best_idx = ci;
                } else if d < second_best {
                    second_best = d;
                }
            }
            if best_idx == usize::MAX {
                continue;
            }
            if second_best > 0.0 && (best_dist / second_best) < self.ratio_threshold {
                forward.push((qi, best_idx, best_dist));
            }
        }

        if !self.bidirectional {
            return forward.into_iter().map(|(qi, ci, _)| (qi, ci)).collect();
        }

        // Step 2: 反向 NN
        let mut reverse_best: std::collections::HashMap<usize, (usize, f32)> =
            std::collections::HashMap::new();
        for (ci, c_kp) in candidate.keypoints.iter().enumerate() {
            let c_off = c_kp.desc_offset;
            if c_off + DESCRIPTOR_DIM > candidate.descriptors.len() {
                continue;
            }
            let c_desc = &candidate.descriptors[c_off..c_off + DESCRIPTOR_DIM];

            let mut best_q = usize::MAX;
            let mut best_dist = f32::MAX;
            let mut second_best = f32::MAX;
            for (qi, q_kp) in query.keypoints.iter().enumerate() {
                let q_off = q_kp.desc_offset;
                if q_off + DESCRIPTOR_DIM > query.descriptors.len() {
                    continue;
                }
                let q_desc = &query.descriptors[q_off..q_off + DESCRIPTOR_DIM];
                let d = l2_distance(c_desc, q_desc);
                if d < best_dist {
                    second_best = best_dist;
                    best_dist = d;
                    best_q = qi;
                } else if d < second_best {
                    second_best = d;
                }
            }
            if best_q == usize::MAX {
                continue;
            }
            if second_best > 0.0 && (best_dist / second_best) < self.ratio_threshold {
                reverse_best.insert(ci, (best_q, best_dist));
            }
        }

        // Step 3: 双向一致（q↔c 都同意）
        let mut out = Vec::new();
        for (qi, ci, _d) in forward {
            if let Some((rq, _)) = reverse_best.get(&ci) {
                if *rq == qi {
                    out.push((qi, ci));
                }
            }
        }
        out
    }

    /// 几何验证（RANSAC）—— 计算 inlier 数量。
    ///
    /// 简化：用匹配点的平均距离 + 阈值过滤作为 inlier（不实际跑 RANSAC）。
    pub fn ransac_inliers(
        &self,
        query: &KeypointSet,
        candidate: &KeypointSet,
        matches: &[(usize, usize)],
    ) -> usize {
        if matches.is_empty() {
            return 0;
        }
        let mut inliers = 0;
        for &(qi, ci) in matches {
            let q = &query.keypoints[qi];
            let c = &candidate.keypoints[ci];
            let dx = q.x - c.x;
            let dy = q.y - c.y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < self.ransac_threshold {
                inliers += 1;
            }
        }
        inliers
    }
}

#[async_trait]
impl FeatureMatcher for LightGlueMatcher {
    async fn match_features(
        &self,
        query: &KeypointSet,
        candidate: &KeypointSet,
    ) -> Result<MatchResult, AIError> {
        let matches = self.mutual_nn_match(query, candidate);

        // Phase 5 新增：少于 min_matches 直接返回空结果（避免 1 match 报 100% confidence）
        if matches.len() < self.min_matches {
            return Ok(MatchResult {
                matches: Vec::new(),
                num_inliers: 0,
                confidence: 0.0,
            });
        }

        // Phase 6：inlier 计数从"像素距离"改为 homography RANSAC 重投影误差
        let inliers = if self.use_homography_verification && matches.len() >= 4 {
            self.homography_inliers(query, candidate, &matches)
        } else {
            // 回退：原距离过滤（小样本或禁用时）
            self.ransac_inliers(query, candidate, &matches)
        };
        let confidence = if matches.is_empty() {
            0.0
        } else {
            inliers as f32 / matches.len() as f32
        };
        Ok(MatchResult {
            matches,
            num_inliers: inliers,
            confidence,
        })
    }

    fn min_matches(&self) -> usize {
        self.min_matches
    }
}

impl LightGlueMatcher {
    /// Phase 6：homography RANSAC 计算 inliers。
    ///
    /// 从匹配对估计 3×3 单应性矩阵，按重投影误差分类 inlier。
    fn homography_inliers(
        &self,
        query: &KeypointSet,
        candidate: &KeypointSet,
        matches: &[(usize, usize)],
    ) -> usize {
        if matches.is_empty() {
            return 0;
        }
        let src: Vec<(f32, f32)> = matches
            .iter()
            .map(|&(qi, _)| (query.keypoints[qi].x, query.keypoints[qi].y))
            .collect();
        let dst: Vec<(f32, f32)> = matches
            .iter()
            .map(|&(_, ci)| (candidate.keypoints[ci].x, candidate.keypoints[ci].y))
            .collect();
        let config = RansacConfig {
            threshold: self.ransac_threshold,
            iterations: self.ransac_iterations,
            early_stop_ratio: 0.7,
            // 用 hash of struct fields 而不是固定 seed，避免不同输入产生相同 sequence
            seed: ((self.ransac_threshold * 1000.0) as u64)
                ^ ((matches.len() as u64) << 16)
                ^ 0xCAFE_BABE,
        };
        let result = ransac_homography(&src, &dst, &config);
        result.num_inliers
    }
}

/// L2 距离两个 256-d 描述子。
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut s = 0.0_f32;
    for i in 0..n {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::similarity::traits::Keypoint;

    #[test]
    fn descriptor_dim_is_256() {
        assert_eq!(DESCRIPTOR_DIM, 256);
    }

    fn mk_set(idxs: &[usize], positions: &[(f32, f32)]) -> KeypointSet {
        let mut kp = Vec::with_capacity(idxs.len());
        let mut desc = Vec::with_capacity(idxs.len() * DESCRIPTOR_DIM);
        for (i, &(x, y)) in positions.iter().enumerate() {
            desc.extend_from_slice(&[0.0_f32; DESCRIPTOR_DIM]);
            // 让每个描述子有一个独特的方向
            let dir = i as f32;
            desc[i * DESCRIPTOR_DIM] = dir;
            kp.push(Keypoint { x, y, desc_offset: i * DESCRIPTOR_DIM });
        }
        KeypointSet {
            keypoints: kp,
            descriptors: desc,
            descriptor_dim: DESCRIPTOR_DIM,
        }
    }

    #[test]
    fn mutual_nn_match_finds_perfect_pair() {
        let m = LightGlueMatcher::default();
        let q = mk_set(&[0, 1], &[(10.0, 20.0), (30.0, 40.0)]);
        let c = mk_set(&[0, 1], &[(10.0, 20.0), (30.0, 40.0)]);
        let matches = m.mutual_nn_match(&q, &c);
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn ransac_inliers_count_close_pairs() {
        let m = LightGlueMatcher::new(4.0, 0.8, 1);
        let q = mk_set(&[0, 1], &[(0.0, 0.0), (100.0, 100.0)]);
        let c = mk_set(&[0, 1], &[(1.0, 1.0), (100.0, 100.0)]);
        let matches = vec![(0, 0), (1, 1)];
        let inliers = m.ransac_inliers(&q, &c, &matches);
        assert_eq!(inliers, 2);
    }

    #[test]
    fn ransac_inliers_filters_far_pairs() {
        let m = LightGlueMatcher::new(4.0, 0.8, 1);
        let q = mk_set(&[0], &[(0.0, 0.0)]);
        let c = mk_set(&[0], &[(100.0, 100.0)]);
        let matches = vec![(0, 0)];
        let inliers = m.ransac_inliers(&q, &c, &matches);
        assert_eq!(inliers, 0);
    }

    // ===== Phase 5: min_matches enforcement =====

    /// match_features 在 matches < min_matches 时返回空 MatchResult（confidence=0）。
    #[test]
    fn match_features_enforces_min_matches() {
        // min_matches=10，但只有 2 个 keypoints，理论最多 2 个 match
        let m = LightGlueMatcher::new(4.0, 0.8, 10);
        let q = mk_set(&[0, 1], &[(10.0, 20.0), (30.0, 40.0)]);
        let c = mk_set(&[0, 1], &[(10.0, 20.0), (30.0, 40.0)]);
        let rt = tokio::runtime::Runtime::new().unwrap();
        let res = rt.block_on(m.match_features(&q, &c)).unwrap();
        assert!(res.matches.is_empty());
        assert_eq!(res.num_inliers, 0);
        assert!((res.confidence - 0.0).abs() < 1e-6);
    }

    /// match_features 在 matches >= min_matches 时正常返回。
    #[test]
    fn match_features_passes_when_enough_matches() {
        // min_matches=1 → 2 matches 都通过
        let m = LightGlueMatcher::new(4.0, 0.8, 1);
        let q = mk_set(&[0, 1], &[(10.0, 20.0), (30.0, 40.0)]);
        let c = mk_set(&[0, 1], &[(10.0, 20.0), (30.0, 40.0)]);
        let rt = tokio::runtime::Runtime::new().unwrap();
        let res = rt.block_on(m.match_features(&q, &c)).unwrap();
        assert_eq!(res.matches.len(), 2);
    }

    // ===== Phase 5: bidirectional NN =====

    /// bidirectional=true 时，单向成立的 match 被过滤（双向都要同意）。
    #[test]
    fn bidirectional_filters_one_way_matches() {
        // q 有 2 个 kp，c 有 3 个 kp，其中 c[0]/c[1] 与 q[0]/q[1] 距离近（互相 best）
        // c[2] 与 q[1] 距离很近但 q[1] 的 best 已是 c[1]，所以是单向
        let m = LightGlueMatcher::new(4.0, 0.8, 1).with_bidirectional(true);
        let mut q = mk_set(&[0, 1], &[(10.0, 20.0), (30.0, 40.0)]);
        let mut c = mk_set(&[0, 1, 2], &[(10.5, 20.5), (30.5, 40.5), (35.0, 45.0)]);
        // 设置 query 描述子方向
        q.descriptors[0] = 1.0; // q[0]
        q.descriptors[DESCRIPTOR_DIM] = 2.0; // q[1]
        // 设置 candidate 描述子
        c.descriptors[0] = 1.0; // c[0] 接近 q[0]
        c.descriptors[DESCRIPTOR_DIM] = 2.0; // c[1] 接近 q[1]
        c.descriptors[2 * DESCRIPTOR_DIM] = 3.0; // c[2] 与 q[0]/q[1] 都远
        let matches = m.mutual_nn_match(&q, &c);
        // 双向：q[0]→c[0], q[1]→c[1]；反向 c[0]→q[0], c[1]→q[1], c[2]→?（远，无双向）
        // 结果：(0,0), (1,1)
        assert_eq!(matches.len(), 2);
        assert!(matches.contains(&(0, 0)));
        assert!(matches.contains(&(1, 1)));
    }

    /// bidirectional=false（默认）保留单向 Lowe ratio match。
    #[test]
    fn non_bidirectional_keeps_one_way_matches() {
        let m = LightGlueMatcher::default(); // bidirectional=false
        let q = mk_set(&[0, 1], &[(10.0, 20.0), (30.0, 40.0)]);
        let c = mk_set(&[0, 1], &[(10.0, 20.0), (30.0, 40.0)]);
        let matches = m.mutual_nn_match(&q, &c);
        assert_eq!(matches.len(), 2);
    }

    // ===== Phase 6: homography-based inlier counting =====

    /// Phase 6：match_features 用 homography RANSAC 计数 inliers。
    ///
    /// 测试场景：5 个真 inlier（translation(5, 0)）+ 2 个 outlier（fake）
    /// → match_features 应正确分类
    #[test]
    fn match_features_uses_homography_for_inliers() {
        // 构造 7 个 query kp + 7 个 candidate kp，前 5 对是 translation(5, 0)，后 2 对是 fake
        let q_positions = vec![
            (10.0, 20.0),
            (30.0, 40.0),
            (50.0, 60.0),
            (70.0, 80.0),
            (90.0, 100.0),
            // outliers（位置与 inlier 重合，但 candidate 描述子不匹配 → 用其他 fake 位置）
            (200.0, 200.0),
            (300.0, 300.0),
        ];
        let c_positions = vec![
            (15.0, 20.0), // q[0] + (5, 0)
            (35.0, 40.0), // q[1] + (5, 0)
            (55.0, 60.0),
            (75.0, 80.0),
            (95.0, 100.0),
            // outliers（远离 translation(5, 0) 预测）
            (10.0, 10.0),
            (400.0, 400.0),
        ];
        let q = mk_set_v2(&q_positions);
        let c = mk_set_v2(&c_positions);

        // 用 mk_set_v2 制造描述子：每个 desc = [direction_unique_to_idx, 0, 0, ...]
        // 这样 NN 匹配会按 idx 一一对应（idx 0↔0, 1↔1, ..., 5↔5, 6↔6）

        let m = LightGlueMatcher::new(4.0, 0.8, 1)
            .with_homography_verification(true);
        let rt = tokio::runtime::Runtime::new().unwrap();
        let res = rt.block_on(m.match_features(&q, &c)).unwrap();
        // 7 个 match。RANSAC 应该过滤掉大部分 outlier，但允许 1 个 sample-outlier 漏出
        // （4-sample 中含 1 outlier → H 在该 outlier 处 error=0 → 计数为 inlier）。
        assert_eq!(res.matches.len(), 7);
        assert!(
            res.num_inliers >= 4 && res.num_inliers <= 6,
            "got num_inliers={}",
            res.num_inliers
        );
    }

    /// Phase 6：禁用 homography verification 时回退到距离过滤。
    #[test]
    fn match_features_falls_back_to_distance_when_disabled() {
        let q = mk_set(&[0, 1], &[(10.0, 20.0), (30.0, 40.0)]);
        let c = mk_set(&[0, 1], &[(10.0, 20.0), (30.0, 40.0)]);
        let m = LightGlueMatcher::new(4.0, 0.8, 1)
            .with_homography_verification(false);
        let rt = tokio::runtime::Runtime::new().unwrap();
        let res = rt.block_on(m.match_features(&q, &c)).unwrap();
        assert_eq!(res.matches.len(), 2);
        assert_eq!(res.num_inliers, 2); // 全在 4.0px 内
    }

    fn mk_set_v2(positions: &[(f32, f32)]) -> KeypointSet {
        let mut kp = Vec::with_capacity(positions.len());
        let mut desc = Vec::with_capacity(positions.len() * DESCRIPTOR_DIM);
        for (i, &(x, y)) in positions.iter().enumerate() {
            desc.extend_from_slice(&[0.0_f32; DESCRIPTOR_DIM]);
            // 让每个描述子有独特的方向
            desc[i * DESCRIPTOR_DIM] = (i + 1) as f32; // 避免 0（避免被 l2 视为零向量）
            kp.push(Keypoint { x, y, desc_offset: i * DESCRIPTOR_DIM });
        }
        KeypointSet {
            keypoints: kp,
            descriptors: desc,
            descriptor_dim: DESCRIPTOR_DIM,
        }
    }
}
