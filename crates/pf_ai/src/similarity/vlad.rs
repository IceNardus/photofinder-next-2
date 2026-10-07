//! VLAD（Vector of Locally Aggregated Descriptors）聚合。
//!
//! **算法与 ai-next/src-tauri/src/core/features/aggregation.rs 逐位对齐**：
//!
//! | 项 | 值 |
//! |---|---|
//! | 描述子维度 | 256（SuperPoint） |
//! | 默认聚类数 K | 64 |
//! | 最大关键点 | 256（单 patch） |
//! | 归一化 | L2 |

use crate::similarity::Keypoint;

/// VLAD 聚合器。
pub struct VladAggregator {
    /// 聚类数 K
    pub k: usize,
    /// 聚类中心 [K, 256]，按列存储
    centroids: Vec<f32>,
    /// 描述子维度（256）
    dim: usize,
}

impl VladAggregator {
    /// 从已有聚类中心构造。
    pub fn from_centroids(k: usize, centroids: Vec<f32>, dim: usize) -> Self {
        Self { k, centroids, dim }
    }

    /// 默认 K=64 构造（需要随后 load_centroids）。
    pub fn new(k: usize) -> Self {
        Self {
            k,
            centroids: vec![0.0_f32; k * 256],
            dim: 256,
        }
    }

    /// 用 k-means++ 从描述子集合初始化聚类中心。
    ///
    /// `descriptors`: 扁平 [N*256], `num_descriptors`: N
    pub fn initialize_centroids(
        &mut self,
        descriptors: &[f32],
        num_descriptors: usize,
        max_iterations: usize,
    ) {
        if num_descriptors == 0 {
            return;
        }
        self.dim = 256; // SuperPoint descriptor dim
        let dim = self.dim;
        self.centroids = vec![0.0_f32; self.k * dim];

        // k-means++ 初始化
        let mut rng = rand_simple(descriptors.len() as u64);
        // 第一个中心：随机选一个点
        let first = (rng.next().unwrap() as usize) % num_descriptors;
        for d in 0..dim {
            self.centroids[d] = descriptors[first * dim + d];
        }

        // 剩余 k-1 个中心
        let _center_dim = dim;
        for c in 1..self.k {
            let mut best_dist_sum = 0.0_f32;
            let mut dists = vec![0.0_f32; num_descriptors];

            for i in 0..num_descriptors {
                // 找离这个点最近的已选中心
                let mut min_dist = f32::INFINITY;
                for j in 0..c {
                    let dist = l2_sq(&descriptors[i * dim..], &self.centroids[j * dim..], dim);
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }
                dists[i] = min_dist;
                best_dist_sum += min_dist;
            }

            // 按概率选下一个中心
            let r = rng.next().unwrap();
            let threshold = (r % best_dist_sum as u64) as f32;
            let mut cum = 0.0_f32;
            let mut chosen = 0;
            for i in 0..num_descriptors {
                cum += dists[i];
                if cum >= threshold {
                    chosen = i;
                    break;
                }
            }
            for d in 0..dim {
                self.centroids[c * dim + d] = descriptors[chosen * dim + d];
            }
        }

        // 标准 k-means 迭代
        let mut assignments = vec![0usize; num_descriptors];
        for _iter in 0..max_iterations {
            // E 步：分配到最近中心
            let mut changed = false;
            for i in 0..num_descriptors {
                let mut best_c = 0usize;
                let mut best_dist = f32::INFINITY;
                for c in 0..self.k {
                    let dist = l2_sq(
                        &descriptors[i * dim..],
                        &self.centroids[c * dim..],
                        dim,
                    );
                    if dist < best_dist {
                        best_dist = dist;
                        best_c = c;
                    }
                }
                if assignments[i] != best_c {
                    assignments[i] = best_c;
                    changed = true;
                }
            }
            if !changed {
                break;
            }

            // M 步：重新计算中心
            let mut sums = vec![0.0_f32; self.k * dim];
            let mut counts = vec![0usize; self.k];
            for i in 0..num_descriptors {
                let c = assignments[i];
                counts[c] += 1;
                for d in 0..dim {
                    sums[c * dim + d] += descriptors[i * dim + d];
                }
            }
            for c in 0..self.k {
                if counts[c] > 0 {
                    for d in 0..dim {
                        self.centroids[c * dim + d] = sums[c * dim + d] / counts[c] as f32;
                    }
                }
            }
        }
    }

    /// VLAD 聚合。
    ///
    /// `descriptors`: 扁平 [N*256]
    /// `num_descriptors`: N
    /// 返回: 256-d L2 归一化向量
    pub fn vlad_aggregate(
        &self,
        descriptors: &[f32],
        num_descriptors: usize,
    ) -> Vec<f32> {
        let dim = self.dim;
        let mut residuals = vec![0.0_f32; dim];

        for i in 0..num_descriptors {
            // 找最近中心
            let desc_start = i * dim;
            let mut best_c = 0usize;
            let mut best_dist = f32::INFINITY;
            for c in 0..self.k {
                let dist = l2_sq(
                    &descriptors[desc_start..desc_start + dim],
                    &self.centroids[c * dim..],
                    dim,
                );
                if dist < best_dist {
                    best_dist = dist;
                    best_c = c;
                }
            }
            // 残差累加
            for d in 0..dim {
                residuals[d] += descriptors[desc_start + d] - self.centroids[best_c * dim + d];
            }
        }

        // L2 归一化
        let norm: f32 = residuals.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for d in 0..dim {
            residuals[d] /= norm;
        }
        residuals
    }

    /// Mean 聚合（简单平均 + L2 归一化）。
    pub fn mean_aggregate(&self, descriptors: &[f32], num_descriptors: usize) -> Vec<f32> {
        let dim = self.dim;
        let mut mean = vec![0.0_f32; dim];
        for i in 0..num_descriptors {
            for d in 0..dim {
                mean[d] += descriptors[i * dim + d];
            }
        }
        let n = num_descriptors as f32;
        for d in 0..dim {
            mean[d] /= n;
        }
        let norm: f32 = mean.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for d in 0..dim {
            mean[d] /= norm;
        }
        mean
    }

    /// 取 top-N 关键点（按 score 降序）。
    pub fn select_top_keypoints(
        keypoints: &[Keypoint],
        scores: &[f32],
        descriptors: &[f32],
        max_keypoints: usize,
    ) -> (Vec<Keypoint>, Vec<f32>, Vec<f32>) {
        let n = keypoints.len().min(scores.len());
        if n == 0 {
            return (Vec::new(), Vec::new(), Vec::new());
        }
        let dim = 256;
        let mut indices: Vec<usize> = (0..n).collect();
        // 降序排序
        indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());

        let taken = indices.into_iter().take(max_keypoints).collect::<Vec<_>>();
        let mut out_kps = Vec::with_capacity(taken.len());
        let mut out_scores = Vec::with_capacity(taken.len());
        let mut out_descs = Vec::with_capacity(taken.len() * dim);

        for &i in &taken {
            out_kps.push(keypoints[i]);
            out_scores.push(scores[i]);
            out_descs.extend_from_slice(&descriptors[i * dim..i * dim + dim]);
        }
        (out_kps, out_scores, out_descs)
    }

    /// 归一化关键点坐标到 [-1, 1]。
    pub fn normalize_keypoints(keypoints: &[f32], width: u32, height: u32) -> Vec<f32> {
        let w = width as f32;
        let h = height as f32;
        let mut out = Vec::with_capacity(keypoints.len());
        for i in 0..keypoints.len() / 2 {
            let x = keypoints[i * 2];
            let y = keypoints[i * 2 + 1];
            out.push((x / w) * 2.0 - 1.0);
            out.push((y / h) * 2.0 - 1.0);
        }
        out
    }
}

/// L2 距离平方（256-d）。
fn l2_sq(a: &[f32], b: &[f32], dim: usize) -> f32 {
    let mut s = 0.0_f32;
    for i in 0..dim {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// 简单 PRNG（线性同余）。
fn rand_simple(seed: u64) -> impl Iterator<Item = u64> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    std::iter::from_fn(move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        Some(state)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vlad_dim_is_256() {
        let mut v = VladAggregator::new(64);
        v.centroids = vec![0.0; 64 * 256];
        let desc = vec![0.5_f32; 10 * 256]; // 10 descriptors
        let result = v.vlad_aggregate(&desc, 10);
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn mean_aggregate_is_unit_length() {
        let v = VladAggregator::new(64);
        let desc = vec![1.0_f32; 5 * 256];
        let result = v.mean_aggregate(&desc, 5);
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm = {}", norm);
    }

    #[test]
    fn vlad_output_is_unit_length() {
        let mut v = VladAggregator::new(2);
        // k=2 中心，各 256 维
        v.centroids = vec![0.5_f32; 2 * 256];
        let desc = vec![1.0_f32; 5 * 256];
        let result = v.vlad_aggregate(&desc, 5);
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm = {}", norm);
    }

    #[test]
    fn select_top_truncates() {
        let kps: Vec<Keypoint> = (0..100)
            .map(|i| Keypoint {
                x: i as f32,
                y: 0.0,
                desc_offset: i * 256,
            })
            .collect();
        let scores: Vec<f32> = (0..100).map(|i| i as f32).collect(); // 0..99
        let desc = vec![1.0_f32; 100 * 256];

        let (out_kps, out_scores, _) = VladAggregator::select_top_keypoints(&kps, &scores, &desc, 10);
        assert_eq!(out_kps.len(), 10);
        assert_eq!(out_scores[0], 99.0); // highest score
    }

    #[test]
    fn normalize_keypoints() {
        let raw = vec![100.0, 200.0]; // 100x200 image
        let norm = VladAggregator::normalize_keypoints(&raw, 200, 400);
        assert!((norm[0] - 0.0).abs() < 1e-6); // 100/200*2-1=0
        assert!((norm[1] - 0.0).abs() < 1e-6); // 200/400*2-1=0
    }
}
