//! Homography estimation + RANSAC geometric verification.
//!
//! **Phase 6**：替换 LightGlue 中基于像素距离的简单 inlier 过滤，
//! 改用基于单应性矩阵（homography）的真 RANSAC 几何校验。
//!
//! 算法：
//! - DLT（Direct Linear Transform）从 ≥4 个匹配对估计 3×3 单应性矩阵
//! - RANSAC 迭代：随机采样 4 对 → 估计 H → 重投影误差 < 阈值 视为 inlier
//! - 返回 inliers 最多的 H + per-pair inlier mask
//!
//! 用途：query ROI ↔ candidate ROI 关键点匹配 → 验证是否几何一致（同一物体/视角）。
//!
//! **简化点**：用 Normal Equations（`A^T A x = A^T b`）+ 高斯消元求解 8×8 线性系统，
//! 不引入 SVD/nlalgebra 依赖。精度足够用于物体检索场景。

use std::collections::HashSet;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// 3×3 单应性矩阵（row-major：`[h00, h01, h02, h10, h11, h12, h20, h21, h22]`）。
///
/// 约束 `h22 = 1`（normalize），所以实际自由度 8。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Homography {
    /// 9 个元素
    pub m: [f32; 9],
}

impl Homography {
    /// 单位矩阵。
    pub fn identity() -> Self {
        Self {
            m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// 平移矩阵 `tx, ty`。
    pub fn translation(tx: f32, ty: f32) -> Self {
        Self {
            m: [1.0, 0.0, tx, 0.0, 1.0, ty, 0.0, 0.0, 1.0],
        }
    }

    /// 应用 H 到点 `(x, y)`，返回 `(x', y')`。
    ///
    /// 当 `H * (x, y, 1)^T = (a, b, w)` 时，返回 `(a/w, b/w)`。
    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        let m = &self.m;
        let a = m[0] * x + m[1] * y + m[2];
        let b = m[3] * x + m[4] * y + m[5];
        let w = m[6] * x + m[7] * y + m[8];
        if w.abs() < 1e-8 {
            (a, b)
        } else {
            (a / w, b / w)
        }
    }

    /// 重投影误差：源点经 H 变换后与目标点的欧氏距离。
    pub fn reprojection_error(&self, src: (f32, f32), dst: (f32, f32)) -> f32 {
        let (u, v) = self.apply(src.0, src.1);
        let du = u - dst.0;
        let dv = v - dst.1;
        (du * du + dv * dv).sqrt()
    }
}

/// DLT 估计结果：None 表示解算失败（退化或欠定）。
pub type DltResult = Result<Homography, &'static str>;

/// 从 `src → dst` 估计单应性矩阵（DLT，最小二乘）。
///
/// 输入 `src.len() == dst.len() >= 4`。
/// 用 Normal Equations `A^T A h = A^T b` + 高斯消元解 8×8 系统（`h22` 固定为 1）。
pub fn estimate_homography_dlt(src: &[(f32, f32)], dst: &[(f32, f32)]) -> DltResult {
    let n = src.len();
    if n < 4 || dst.len() != n {
        return Err("need >= 4 point pairs");
    }

    // Build 2N x 8 matrix A and 2N vector b
    // For pair (x, y) → (u, v):
    //   Row 0: [x, y, 1, 0, 0, 0, -x*u, -y*u] = u
    //   Row 1: [0, 0, 0, x, y, 1, -x*v, -y*v] = v
    let mut ata = [0.0_f32; 64]; // 8x8
    let mut atb = [0.0_f32; 8];

    for i in 0..n {
        let (x, y) = src[i];
        let (u, v) = dst[i];

        // 2 rows for this pair
        let row_a = [x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u];
        let row_b = [0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v];
        let targets = [u, v];

        for r in 0..2 {
            let row = if r == 0 { &row_a } else { &row_b };
            let target = targets[r];
            // A^T A += outer(row, row); A^T b += row * target
            for i_idx in 0..8 {
                for j_idx in 0..8 {
                    ata[i_idx * 8 + j_idx] += row[i_idx] * row[j_idx];
                }
                atb[i_idx] += row[i_idx] * target;
            }
        }
    }

    // Solve 8x8 linear system via Gaussian elimination
    let h = match solve_8x8(&mut ata, &mut atb) {
        Some(x) => x,
        None => return Err("singular system"),
    };
    Ok(Homography {
        m: [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0],
    })
}

/// 8×8 线性系统求解（in-place Gaussian elimination with partial pivoting）。
///
/// `ata`: 8×8 系数矩阵（row-major），调用后会被破坏。
/// `atb`: 8 维右手向量，调用后会被解覆盖。
///
/// 返回 `Some([x0, x1, ..., x7])` 或 `None`（奇异）。
fn solve_8x8(ata: &mut [f32; 64], atb: &mut [f32; 8]) -> Option<[f32; 8]> {
    // Forward elimination with partial pivoting
    for col in 0..8 {
        // Find pivot
        let mut pivot_row = col;
        let mut pivot_val = ata[col * 8 + col].abs();
        for row in (col + 1)..8 {
            let v = ata[row * 8 + col].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = row;
            }
        }
        if pivot_val < 1e-8 {
            return None; // Singular
        }
        // Swap rows
        if pivot_row != col {
            for j in 0..8 {
                ata.swap(col * 8 + j, pivot_row * 8 + j);
            }
            atb.swap(col, pivot_row);
        }
        // Eliminate below
        let pivot = ata[col * 8 + col];
        for row in (col + 1)..8 {
            let factor = ata[row * 8 + col] / pivot;
            for j in col..8 {
                ata[row * 8 + j] -= factor * ata[col * 8 + j];
            }
            atb[row] -= factor * atb[col];
        }
    }

    // Back substitution
    let mut x = [0.0_f32; 8];
    for i in (0..8).rev() {
        let mut s = atb[i];
        for j in (i + 1)..8 {
            s -= ata[i * 8 + j] * x[j];
        }
        x[i] = s / ata[i * 8 + i];
    }
    Some(x)
}

/// RANSAC 配置。
#[derive(Debug, Clone, Copy)]
pub struct RansacConfig {
    /// 重投影误差阈值（像素）；低于此视为 inlier。
    pub threshold: f32,
    /// 最大迭代次数。
    pub iterations: usize,
    /// 早停：inlier 比例超过此值时停止（避免过度迭代）。
    pub early_stop_ratio: f32,
    /// 随机种子（保证可复现）。
    pub seed: u64,
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            threshold: 4.0,
            iterations: 100,
            early_stop_ratio: 0.7,
            seed: 42,
        }
    }
}

/// RANSAC 结果。
#[derive(Debug, Clone)]
pub struct RansacResult {
    /// 估计出的单应性矩阵（若 RANSAC 失败则为 `None`）。
    pub homography: Option<Homography>,
    /// 每个匹配对是否为 inlier（长度 = matches 数量）。
    pub inlier_mask: Vec<bool>,
    /// inlier 总数。
    pub num_inliers: usize,
    /// 实际迭代次数。
    pub iterations_used: usize,
}

/// RANSAC homography 估计。
///
/// 输入 `src.len() == dst.len() < 4` 时直接返回（homography=None, all outlier）。
pub fn ransac_homography(
    src: &[(f32, f32)],
    dst: &[(f32, f32)],
    config: &RansacConfig,
) -> RansacResult {
    let n = src.len();
    if n < 4 || dst.len() != n {
        return RansacResult {
            homography: None,
            inlier_mask: vec![false; n],
            num_inliers: 0,
            iterations_used: 0,
        };
    }

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut best_inliers = 0usize;
    let mut best_h: Option<Homography> = None;
    let mut best_mask: Vec<bool> = vec![false; n];

    let mut iter_used = 0usize;
    for _ in 0..config.iterations {
        iter_used += 1;
        // Sample 4 distinct indices
        let sample = sample_4_distinct(n, &mut rng);
        let s_sample: Vec<(f32, f32)> = sample.iter().map(|&i| src[i]).collect();
        let d_sample: Vec<(f32, f32)> = sample.iter().map(|&i| dst[i]).collect();

        let h = match estimate_homography_dlt(&s_sample, &d_sample) {
            Ok(h) => h,
            Err(_) => continue, // Degenerate sample, try again
        };

        // Count inliers against all matches
        let mut inlier_mask = vec![false; n];
        let mut inliers = 0usize;
        for i in 0..n {
            if h.reprojection_error(src[i], dst[i]) < config.threshold {
                inlier_mask[i] = true;
                inliers += 1;
            }
        }

        if inliers > best_inliers {
            best_inliers = inliers;
            best_h = Some(h);
            best_mask = inlier_mask;

            // Early stop if "good enough"
            if inliers as f32 / n as f32 >= config.early_stop_ratio {
                break;
            }
        }
    }

    // Refit H using only best inliers（标准 RANSAC 后处理）
    if best_inliers >= 4 {
        let inlier_src: Vec<(f32, f32)> = (0..n)
            .filter(|&i| best_mask[i])
            .map(|i| src[i])
            .collect();
        let inlier_dst: Vec<(f32, f32)> = (0..n)
            .filter(|&i| best_mask[i])
            .map(|i| dst[i])
            .collect();
        if let Ok(refit_h) = estimate_homography_dlt(&inlier_src, &inlier_dst) {
            // Re-evaluate inliers with refit H
            let mut new_mask = vec![false; n];
            let mut new_inliers = 0usize;
            for i in 0..n {
                if refit_h.reprojection_error(src[i], dst[i]) < config.threshold {
                    new_mask[i] = true;
                    new_inliers += 1;
                }
            }
            // Accept refit if it doesn't reduce inliers
            if new_inliers >= best_inliers {
                best_h = Some(refit_h);
                best_mask = new_mask;
                best_inliers = new_inliers;
            }
        }
    }

    RansacResult {
        homography: best_h,
        inlier_mask: best_mask,
        num_inliers: best_inliers,
        iterations_used: iter_used,
    }
}

/// 部分采样 4 个不同的索引（适用于任意 n ≥ 4）。
///
/// Bug 修复（Audit A1）：原实现 `[bool; 64]` 只在 `n <= 64` 时去重；n > 64 时
/// 静默跳过去重 → 可能产出含重复的 4-tuple → DLT 喂 3 个或更少 unique 点对
/// → `A` 矩阵秩 < 8 → 解出偏置 H → RANSAC 误判 inlier。
fn sample_4_distinct(n: usize, rng: &mut StdRng) -> [usize; 4] {
    // Caller (ransac_homography) 仅在 n >= 4 时调用,所以此处可以安全假设 n >= 4。
    let mut seen: HashSet<usize> = HashSet::with_capacity(4);
    let mut chosen = [0usize; 4];
    let mut count = 0;
    while count < 4 {
        let idx = rng.gen_range(0..n);
        if seen.insert(idx) {
            chosen[count] = idx;
            count += 1;
        }
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-2
    }

    #[test]
    fn identity_apply() {
        let h = Homography::identity();
        let (u, v) = h.apply(10.0, 20.0);
        assert!(approx(u, 10.0));
        assert!(approx(v, 20.0));
    }

    #[test]
    fn translation_apply() {
        let h = Homography::translation(5.0, -3.0);
        let (u, v) = h.apply(10.0, 20.0);
        assert!(approx(u, 15.0));
        assert!(approx(v, 17.0));
    }

    /// DLT 应该能恢复出准确的 translation。
    #[test]
    fn dlt_recovers_translation() {
        let src: Vec<(f32, f32)> = vec![(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)];
        let dst: Vec<(f32, f32)> = src.iter().map(|&(x, y)| (x + 3.0, y - 7.0)).collect();
        let h = estimate_homography_dlt(&src, &dst).unwrap();
        let (u, v) = h.apply(0.0, 0.0);
        assert!(approx(u, 3.0));
        assert!(approx(v, -7.0));
        let (u, v) = h.apply(10.0, 10.0);
        assert!(approx(u, 13.0));
        assert!(approx(v, 3.0));
    }

    /// DLT 应该能恢复出 90° rotation。
    #[test]
    fn dlt_recovers_rotation() {
        // Rotation 90° clockwise: (x, y) → (y, -x)
        let src: Vec<(f32, f32)> = vec![(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
        let dst: Vec<(f32, f32)> = src.iter().map(|&(x, y)| (y, -x)).collect();
        let h = estimate_homography_dlt(&src, &dst).unwrap();
        // Test: (1, 0) → (0, -1)
        let (u, v) = h.apply(1.0, 0.0);
        assert!(approx(u, 0.0));
        assert!(approx(v, -1.0));
    }

    /// 欠定 (<4 对) 应该返回 Err。
    #[test]
    fn dlt_rejects_too_few_pairs() {
        let src = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
        let dst = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
        assert!(estimate_homography_dlt(&src, &dst).is_err());
    }

    /// RANSAC 应该能在有 outliers 的情况下识别出 inliers 并恢复出正确的 H。
    ///
    /// 测试场景：6 个真 inliers（translation(5, 3)）+ 2 个 outliers（33% 异常率）
    /// 33% 是真实场景下的常见 outlier 比例。
    #[test]
    fn ransac_handles_outliers() {
        // 6 个真 inliers（translation(5, 3)）
        let src: Vec<(f32, f32)> = vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (0.0, 10.0),
            (10.0, 10.0),
            (5.0, 5.0),
            (15.0, 7.0),
        ];
        let dst: Vec<(f32, f32)> = src.iter().map(|&(x, y)| (x + 5.0, y + 3.0)).collect();
        // 2 个真 outliers
        let mut src2 = src.clone();
        let mut dst2 = dst.clone();
        src2.extend_from_slice(&[(1000.0, 1000.0), (2000.0, 500.0)]);
        dst2.extend_from_slice(&[(0.0, 0.0), (1.0, 1.0)]);

        let config = RansacConfig {
            threshold: 4.0,
            iterations: 200,
            early_stop_ratio: 0.7,
            seed: 42,
        };
        let result = ransac_homography(&src2, &dst2, &config);
        assert!(result.homography.is_some(), "RANSAC should find H");
        assert!(
            result.num_inliers >= 6,
            "got inliers={}, mask={:?}",
            result.num_inliers,
            result.inlier_mask
        );
        // True inliers 一定是 idx 0..5
        for i in 0..6 {
            assert!(result.inlier_mask[i], "true inlier {i} not classified");
        }
        for i in 6..8 {
            assert!(!result.inlier_mask[i], "true outlier {i} classified as inlier");
        }
        // H 应该近似 translation(5, 3)
        let h = result.homography.unwrap();
        let (u, v) = h.apply(0.0, 0.0);
        assert!((u - 5.0).abs() < 0.5, "u={}", u);
        assert!((v - 3.0).abs() < 0.5, "v={}", v);
    }

    /// <4 对时 RANSAC 直接返回 None。
    #[test]
    fn ransac_rejects_too_few() {
        let src = vec![(0.0, 0.0), (1.0, 1.0)];
        let dst = vec![(1.0, 1.0), (2.0, 2.0)];
        let result = ransac_homography(&src, &dst, &RansacConfig::default());
        assert!(result.homography.is_none());
        assert_eq!(result.num_inliers, 0);
    }

    /// 退化采样（4 个共线点）应被 RANSAC 容错处理（DLT 返回 Err → skip）。
    #[test]
    fn ransac_handles_degenerate_samples() {
        let src = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (5.0, 5.0), (10.0, 10.0)];
        let dst = vec![(0.0, 5.0), (1.0, 5.0), (2.0, 5.0), (3.0, 5.0), (5.0, 10.0), (10.0, 15.0)];
        let config = RansacConfig {
            threshold: 4.0,
            iterations: 200,
            early_stop_ratio: 0.5,
            seed: 42,
        };
        let result = ransac_homography(&src, &dst, &config);
        // 退化 sample（4 个共线）应被 DLT 拒绝，RANSAC 继续找好 sample
        assert!(result.homography.is_some());
    }

    /// Audit A1 回归：sample_4_distinct 在任意 n 下都应返回 4 个**互不相同**的索引。
    ///
    /// 修复前：`n > 64` 时跳过 dedup,可能产出含重复的 4-tuple,导致 DLT 喂 <4 unique 点对。
    /// 修复后：用 HashSet 始终去重 → 严格 4 distinct,任意 n。
    #[test]
    fn sample_4_distinct_returns_4_distinct_for_arbitrary_n() {
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(42);

        for n in [4_usize, 5, 10, 50, 100, 500, 1000] {
            for trial in 0..200 {
                let s = sample_4_distinct(n, &mut rng);
                // 1) 4 个索引各不相同
                let mut sorted = s;
                sorted.sort();
                for i in 0..3 {
                    assert!(
                        sorted[i] < sorted[i + 1],
                        "n={n} trial={trial}: duplicate index in {s:?}"
                    );
                }
                // 2) 所有索引均在 [0, n) 内
                for &idx in &s {
                    assert!(idx < n, "n={n} trial={trial}: idx={idx} out of bounds");
                }
            }
        }
    }
}