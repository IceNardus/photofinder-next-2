//! Phase 8 — object-search pipeline micro-benchmarks.
//!
//! 所有 benchmark 都是 `#[ignore]` 标记的，需要显式调用：
//! `cargo test --release -p pf_application -- --ignored --nocapture bench`
//!
//! 设计：
//! - 纯合成数据（无 model、无 I/O）
//! - `std::time::Instant` 测时
//! - `eprintln!` 输出 per-op latency
//! - 上界 assertion（防止 10x 级别的回归）
//!
//! Pipeline 阶段（按调用顺序）：
//! 1. `build_query_embedding`（MobileCLIP，模型相关，跳过）
//! 2. `HNSW coarse search`（`pf_vector`，跳过 — 不在本 crate 范围）
//! 3. **per-image max-pool aggregation**（算法，纯 Rust）
//! 4. **`select_fine_candidates`**（算法，纯 Rust — Phase 7 重点）
//! 5. `LightGlue fine` + RANSAC（模型相关，跳过 — 在 `pf_ai`）
//! 6. **fusion scoring**（trivially fast，跳过）
//!
//! 范围：本文件测阶段 3 + 4 + ROI expansion（BBox math）+ 一些辅助 helper。

use std::collections::HashMap;
use std::time::{Duration, Instant};

use pf_application::{
    clamp_roi_to_image, coarse_fetch_k, expand_roi_for_matching, select_fine_candidates,
    SearchService,
};
use pf_core::BBox;
use pf_vector::SearchHit;

/// `per_op` 默认回归上界：100µs（适合 HashMap sort、small linear algebra）。
const PER_OP_BUDGET_US: u64 = 100;
/// ROI expansion 稍慢（clamp + scale），上界 200µs。
const PER_OP_BUDGET_EXPAND_US: u64 = 200;
/// `create_prototype` 100 个 512-d embedding 约 50µs，上界 500µs。
const PER_OP_BUDGET_PROTOTYPE_US: u64 = 500;
/// `coarse_fetch_k` 极简算术（仅 3 个 cmp + 1 个 mul），上界 1µs。
const PER_OP_BUDGET_FETCH_US: u64 = 1;

/// 构造 N 个候选 `(image_id, embedding_score, weighted_score)`。
fn make_best_per_image(n: usize) -> HashMap<i64, (SearchHit, i64, BBox, f32)> {
    (0..n)
        .map(|i| {
            let img_id = (i % 200) as i64; // 200 个 unique image
            let score = (i as f32 / n as f32) * 0.95 + 0.05; // 0.05–1.0
            let weighted = score * 0.9;
            (
                img_id,
                (
                    SearchHit {
                        id: img_id,
                        score,
                    },
                    img_id * 10 + i as i64,
                    BBox::new(
                        (i % 100) as f32,
                        (i % 100) as f32,
                        100.0,
                        100.0,
                    ),
                    weighted,
                ),
            )
        })
        .collect()
}

/// Phase 8：测 `select_fine_candidates` 在不同 cap 下的 latency。
///
/// 1k candidates 是真实场景下的典型值（200 unique images × ~5 ROIs each）。
#[test]
#[ignore]
fn bench_select_fine_candidates() {
    for &cap in &[10_usize, 50, 200, 1000] {
        let data = make_best_per_image(1000);
        // warmup
        let _ = select_fine_candidates(data.clone(), 0.3, cap);
        let n = 100;
        let start = Instant::now();
        for _ in 0..n {
            let data = make_best_per_image(1000);
            let _ = select_fine_candidates(data, 0.3, cap);
        }
        let elapsed = start.elapsed();
        let per_op = elapsed / n as u32;
        eprintln!(
            "bench_select_fine_candidates cap={cap:>4} n={n} total={:?} per_op={:?}",
            elapsed, per_op
        );
        assert!(
            per_op < Duration::from_micros(PER_OP_BUDGET_US),
            "regression: per_op={:?} > budget={}µs",
            per_op,
            PER_OP_BUDGET_US
        );
    }
}

/// Phase 8：测 `create_prototype` 在 100 个 512-d embedding 上的 latency。
///
/// MobileCLIP-s2 输出 512-d，所以 prototype 平均是 100 个 512-d 向量的 L2-normalized mean。
#[test]
#[ignore]
fn bench_create_prototype() {
    // 100 个 512-d embedding（合成随机单位向量）
    let mut rng_state = 12345_u64;
    let mut next = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        rng_state as f32 / u64::MAX as f32 - 0.5
    };
    let embeddings: Vec<Vec<f32>> = (0..100)
        .map(|_| (0..512).map(|_| next()).collect())
        .collect();
    // warmup
    let _ = SearchService::create_prototype(&embeddings);
    let n = 200;
    let start = Instant::now();
    for _ in 0..n {
        let _ = SearchService::create_prototype(&embeddings);
    }
    let elapsed = start.elapsed();
    let per_op = elapsed / n as u32;
    eprintln!(
        "bench_create_prototype n_rois=100 dim=512 iterations={n} total={:?} per_op={:?}",
        elapsed, per_op
    );
    assert!(
        per_op < Duration::from_micros(PER_OP_BUDGET_PROTOTYPE_US),
        "regression: per_op={:?} > budget={}µs",
        per_op,
        PER_OP_BUDGET_PROTOTYPE_US
    );
}

/// Phase 8：测 `BBox::iou` 在 10k 随机 pair 上的 latency。
#[test]
#[ignore]
fn bench_iou() {
    let mut rng_state = 99_u64;
    let mut next = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        rng_state as f32 / u64::MAX as f32
    };
    let pairs: Vec<(BBox, BBox)> = (0..10_000)
        .map(|_| {
            (
                BBox::new(next() * 1000.0, next() * 1000.0, next() * 100.0 + 1.0, next() * 100.0 + 1.0),
                BBox::new(next() * 1000.0, next() * 1000.0, next() * 100.0 + 1.0, next() * 100.0 + 1.0),
            )
        })
        .collect();
    // warmup
    let mut sink = 0.0_f32;
    for (a, b) in &pairs {
        sink += a.iou(b);
    }
    let n = 10;
    let start = Instant::now();
    for _ in 0..n {
        let mut s = 0.0_f32;
        for (a, b) in &pairs {
            s += a.iou(b);
        }
        sink += s;
    }
    let elapsed = start.elapsed();
    let per_op = elapsed / (n * pairs.len()) as u32;
    eprintln!(
        "bench_iou n_pairs={} iterations={n} total={:?} per_op={:?} sink={sink}",
        pairs.len(),
        elapsed,
        per_op
    );
    assert!(
        per_op < Duration::from_micros(PER_OP_BUDGET_US),
        "regression: per_op={:?} > budget={}µs",
        per_op,
        PER_OP_BUDGET_US
    );
}

/// Phase 8：测 `expand_roi_for_matching` + `clamp_roi_to_image` 在 10k ROI 上的 latency。
///
/// 覆盖 3 个 scale bin：
/// - < 0.2（small，2x expand）
/// - 0.2–0.5（moderate，1.5x expand）
/// - ≥ 0.5（large，1x no expand）
#[test]
#[ignore]
fn bench_expand_roi_for_matching() {
    let mut rng_state = 7_u64;
    let mut next = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        rng_state as f32 / u64::MAX as f32
    };
    // 10k ROI，scale 分布均匀（覆盖所有 3 个 bin）
    let rois: Vec<BBox> = (0..10_000)
        .map(|i| {
            let scale = match i % 3 {
                0 => 0.1,   // small
                1 => 0.3,   // moderate
                _ => 0.7,   // large
            };
            let w = scale * 1000.0;
            let h = scale * 1000.0;
            let x = next() * (1000.0 - w);
            let y = next() * (1000.0 - h);
            BBox::new(x, y, w, h)
        })
        .collect();
    let img = (1000.0, 1000.0);
    // warmup
    let mut sink = 0.0_f32;
    for r in &rois {
        let out = expand_roi_for_matching(r, img.0, img.1);
        sink += out.w;
    }
    let n = 10;
    let start = Instant::now();
    for _ in 0..n {
        let mut s = 0.0_f32;
        for r in &rois {
            let out = expand_roi_for_matching(r, img.0, img.1);
            s += out.w;
        }
        sink += s;
    }
    let elapsed = start.elapsed();
    let per_op = elapsed / (n * rois.len()) as u32;
    eprintln!(
        "bench_expand_roi_for_matching n_rois={} iterations={n} total={:?} per_op={:?} sink={sink}",
        rois.len(),
        elapsed,
        per_op
    );
    assert!(
        per_op < Duration::from_micros(PER_OP_BUDGET_EXPAND_US),
        "regression: per_op={:?} > budget={}µs",
        per_op,
        PER_OP_BUDGET_EXPAND_US
    );
}

/// Phase 8：测 `coarse_fetch_k` 在 1k 次调用上的 latency（trivial sanity）。
#[test]
#[ignore]
fn bench_coarse_fetch_k() {
    // warmup
    let mut sink = 0_usize;
    for i in 0..1000 {
        sink += coarse_fetch_k(i, 20, 200);
    }
    let n = 100;
    let start = Instant::now();
    for _ in 0..n {
        for i in 0..1000 {
            sink += coarse_fetch_k(i, 20, 200);
        }
    }
    let elapsed = start.elapsed();
    let per_op = elapsed / (n * 1000) as u32;
    eprintln!(
        "bench_coarse_fetch_k n=1000 iterations={n} total={:?} per_op={:?} sink={sink}",
        elapsed, per_op
    );
    assert!(
        per_op < Duration::from_micros(PER_OP_BUDGET_FETCH_US),
        "regression: per_op={:?} > budget={}µs",
        per_op,
        PER_OP_BUDGET_FETCH_US
    );
}

/// Phase 8：测 `clamp_roi_to_image` 在 10k ROI 上的 latency。
#[test]
#[ignore]
fn bench_clamp_roi_to_image() {
    let mut rng_state = 13_u64;
    let mut next = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        rng_state as f32 / u64::MAX as f32
    };
    // 一些故意越界的 ROI
    let rois: Vec<BBox> = (0..10_000)
        .map(|_| BBox::new(
            next() * 2000.0 - 500.0,  // 可能为负
            next() * 2000.0 - 500.0,
            next() * 1500.0 + 1.0,
            next() * 1500.0 + 1.0,
        ))
        .collect();
    let img = (1000.0, 1000.0);
    // warmup
    let mut sink = 0.0_f32;
    for r in &rois {
        let out = clamp_roi_to_image(r, img.0, img.1);
        sink += out.w;
    }
    let n = 20;
    let start = Instant::now();
    for _ in 0..n {
        let mut s = 0.0_f32;
        for r in &rois {
            let out = clamp_roi_to_image(r, img.0, img.1);
            s += out.w;
        }
        sink += s;
    }
    let elapsed = start.elapsed();
    let per_op = elapsed / (n * rois.len()) as u32;
    eprintln!(
        "bench_clamp_roi_to_image n_rois={} iterations={n} total={:?} per_op={:?} sink={sink}",
        rois.len(),
        elapsed,
        per_op
    );
    assert!(
        per_op < Duration::from_micros(PER_OP_BUDGET_US),
        "regression: per_op={:?} > budget={}µs",
        per_op,
        PER_OP_BUDGET_US
    );
}