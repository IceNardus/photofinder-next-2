//! Phase 8 — object AI module micro-benchmarks.
//!
//! 所有 benchmark 都是 `#[ignore]` 标记的，需要显式调用：
//! `cargo test --release -p pf_ai -- --ignored --nocapture bench`
//!
//! 范围：纯算法 helper（DLT、RANSAC、mutual NN、L2 距离）。
//! 模型相关推理（SuperPoint/LightGlue ONNX）不在此范围。

use std::time::{Duration, Instant};

use pf_ai::object::{
    estimate_homography_dlt, l2_distance, ransac_homography, Homography, LightGlueMatcher,
    RansacConfig,
};
use pf_ai::{Keypoint, KeypointSet};

/// `per_op` 默认回归上界。
const PER_OP_BUDGET_US: u64 = 100;
/// `ransac_homography` 默认 100 iter × 8x8 solve = ~500µs，上界 5ms。
const PER_OP_BUDGET_RANSAC_US: u64 = 5_000;
/// `mutual_nn_match` 100×100 = 10k pair L2 + sort → ~500µs，上界 5ms。
const PER_OP_BUDGET_NN_US: u64 = 5_000;
/// `l2_distance` 256-d → ~100ns，上界 1µs。
const PER_OP_BUDGET_L2_US: u64 = 1;

/// Phase 8：测 `estimate_homography_dlt` 在 1k 4-point DLT solve 上的 latency。
///
/// 4-point DLT = 8x8 Normal Equations solve + 8x8 Gaussian elimination。
/// 每次调用输入 4 个 (src, dst) 点对（合成 translation + rotation）。
#[test]
#[ignore]
fn bench_estimate_homography_dlt() {
    // 合成 4 对：translation(3, -7) + rotation 15°
    let src: Vec<(f32, f32)> = vec![(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)];
    // 简化：固定 translation（rotation 算 H 时由 DLT 自动恢复）
    let dst: Vec<(f32, f32)> = vec![(3.0, -7.0), (13.0, -7.0), (3.0, 3.0), (13.0, 3.0)];
    // warmup
    let mut sink = 0.0_f32;
    for _ in 0..10 {
        let h = estimate_homography_dlt(&src, &dst).unwrap();
        sink += h.m[0];
    }
    let n = 1_000;
    let start = Instant::now();
    for _ in 0..n {
        let h = estimate_homography_dlt(&src, &dst).unwrap();
        sink += h.m[0];
    }
    let elapsed = start.elapsed();
    let per_op = elapsed / n;
    eprintln!(
        "bench_estimate_homography_dlt n={n} total={:?} per_op={:?} sink={sink}",
        elapsed, per_op
    );
    assert!(
        per_op < Duration::from_micros(PER_OP_BUDGET_US),
        "regression: per_op={:?} > budget={}µs",
        per_op,
        PER_OP_BUDGET_US
    );
}

/// Phase 8：测 `ransac_homography` 在 100 RANSAC run 上的 latency。
///
/// 每次 RANSAC：100 iter × DLT 4-pt solve + per-iter 7-point inlier check + refit。
#[test]
#[ignore]
fn bench_ransac_homography() {
    // 5 个真 inliers（translation(5, 3)）+ 2 outliers
    let mut src: Vec<(f32, f32)> = vec![
        (0.0, 0.0),
        (10.0, 0.0),
        (0.0, 10.0),
        (10.0, 10.0),
        (5.0, 5.0),
    ];
    let mut dst: Vec<(f32, f32)> = src.iter().map(|&(x, y)| (x + 5.0, y + 3.0)).collect();
    src.extend_from_slice(&[(1000.0, 1000.0), (2000.0, 500.0)]);
    dst.extend_from_slice(&[(0.0, 0.0), (1.0, 1.0)]);

    let config = RansacConfig {
        threshold: 4.0,
        iterations: 100,
        early_stop_ratio: 0.7,
        seed: 42,
    };

    // warmup
    let mut sink = 0usize;
    for _ in 0..5 {
        let r = ransac_homography(&src, &dst, &config);
        sink += r.num_inliers;
    }
    let n = 100;
    let start = Instant::now();
    for _ in 0..n {
        let r = ransac_homography(&src, &dst, &config);
        sink += r.num_inliers;
    }
    let elapsed = start.elapsed();
    let per_op = elapsed / n;
    eprintln!(
        "bench_ransac_homography n={n} iters/config=100 total={:?} per_op={:?} sink={sink}",
        elapsed, per_op
    );
    assert!(
        per_op < Duration::from_micros(PER_OP_BUDGET_RANSAC_US),
        "regression: per_op={:?} > budget={}µs",
        per_op,
        PER_OP_BUDGET_RANSAC_US
    );
}

/// Phase 8：测 `LightGlueMatcher::mutual_nn_match` 在 100×100 关键点上的 latency。
///
/// Forward + reverse NN：~10k L2 distance + 100 Lowe ratio checks + ~100 reverse NN。
#[test]
#[ignore]
fn bench_mutual_nn_match() {
    let mk = |positions: &[(f32, f32)]| -> KeypointSet {
        let mut kp = Vec::with_capacity(positions.len());
        let mut desc = Vec::with_capacity(positions.len() * 256);
        for (i, &(x, y)) in positions.iter().enumerate() {
            desc.extend_from_slice(&[0.0_f32; 256]);
            desc[i * 256] = (i + 1) as f32;
            kp.push(Keypoint {
                x,
                y,
                desc_offset: i * 256,
            });
        }
        KeypointSet {
            keypoints: kp,
            descriptors: desc,
            descriptor_dim: 256,
        }
    };
    let positions: Vec<(f32, f32)> = (0..100).map(|i| (i as f32, (i * 2) as f32)).collect();
    let q = mk(&positions);
    let c = mk(&positions);

    let matcher = LightGlueMatcher::default();
    // warmup
    let mut sink = 0_usize;
    for _ in 0..5 {
        sink += matcher.mutual_nn_match(&q, &c).len();
    }
    let n = 100;
    let start = Instant::now();
    for _ in 0..n {
        sink += matcher.mutual_nn_match(&q, &c).len();
    }
    let elapsed = start.elapsed();
    let per_op = elapsed / n;
    eprintln!(
        "bench_mutual_nn_match q=100 c=100 iterations={n} total={:?} per_op={:?} sink={sink}",
        elapsed, per_op
    );
    assert!(
        per_op < Duration::from_micros(PER_OP_BUDGET_NN_US),
        "regression: per_op={:?} > budget={}µs",
        per_op,
        PER_OP_BUDGET_NN_US
    );
}

/// Phase 8：测 `l2_distance` 在 100k 256-d L2 上的 latency（trivial sanity）。
#[test]
#[ignore]
fn bench_l2_distance() {
    let a: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
    let b: Vec<f32> = (0..256).map(|i| (255 - i) as f32 * 0.01).collect();
    // warmup
    let mut sink = 0.0_f32;
    for _ in 0..1_000 {
        sink += l2_distance(&a, &b);
    }
    let n = 100;
    let start = Instant::now();
    for _ in 0..n {
        for _ in 0..1_000 {
            sink += l2_distance(&a, &b);
        }
    }
    let elapsed = start.elapsed();
    let per_op = elapsed / (n * 1_000) as u32;
    eprintln!(
        "bench_l2_distance n=100k iterations={n} total={:?} per_op={:?} sink={sink}",
        elapsed, per_op
    );
    assert!(
        per_op < Duration::from_micros(PER_OP_BUDGET_L2_US),
        "regression: per_op={:?} > budget={}µs",
        per_op,
        PER_OP_BUDGET_L2_US
    );
}

/// Phase 8：测 `Homography::apply` + `reprojection_error` 在 10k apply 上的 latency。
///
/// 实用路径：RANSAC 后对每个 inlier 重投影。
#[test]
#[ignore]
fn bench_homography_apply() {
    let h = Homography::translation(5.0, 3.0);
    let points: Vec<(f32, f32)> = (0..10_000).map(|i| (i as f32 * 0.1, i as f32 * 0.2)).collect();
    let dst: Vec<(f32, f32)> = points.iter().map(|&(x, y)| (x + 5.0, y + 3.0)).collect();
    // warmup
    let mut sink = 0.0_f32;
    for (p, d) in points.iter().zip(dst.iter()) {
        sink += h.reprojection_error(*p, *d);
    }
    let n = 100;
    let start = Instant::now();
    for _ in 0..n {
        for (p, d) in points.iter().zip(dst.iter()) {
            sink += h.reprojection_error(*p, *d);
        }
    }
    let elapsed = start.elapsed();
    let per_op = elapsed / (n * points.len()) as u32;
    eprintln!(
        "bench_homography_apply n=10k iterations={n} total={:?} per_op={:?} sink={sink}",
        elapsed, per_op
    );
    assert!(
        per_op < Duration::from_micros(PER_OP_BUDGET_L2_US),
        "regression: per_op={:?} > budget={}µs",
        per_op,
        PER_OP_BUDGET_L2_US
    );
}