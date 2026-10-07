# Face Cascade + RollCorrect Production Migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate `quick_face_check.rs` SCRFD cascade (500M + 10G) and RollCorrect pipeline into production face scanning (`IndexService::index_image`), gated by feature flags.

**Architecture:** Two new feature flags (`enable_face_cascade`, `enable_roll_correct`) on `FaceConfig`, default both off. When off, behavior is byte-identical to current. When on, `FacePipeline::process_with_raw` runs secondary SCRFD 10G (IoU dedup) and replaces baseline embedding with RollCorrect embedding for tilted faces (|roll|>5°). Schema gains `detector_origin` column and `face_roll_corrections` audit table via migration 019.

**Tech Stack:** Rust, ORT (ONNX Runtime), SCRFD (500M + 10G), AdaFace IR101, SQLite (rusqlite), HNSW.

**Spec:** `docs/superpowers/specs/2026-08-27-face-cascade-roll-correct-production-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `crates/pf_database/migrations/019_face_cascade_roll_correct.sql` | CREATE | Schema migration: detector_origin column + face_roll_corrections table |
| `crates/pf_database/src/migration.rs` | MODIFY | Register migration 019 |
| `crates/pf_config/src/config.rs` | MODIFY | Add 6 fields to FaceConfig |
| `crates/pf_ai/src/face/mod.rs` | MODIFY | Re-export DetectorOrigin |
| `crates/pf_ai/src/face/traits.rs` | MODIFY | DetectorOrigin enum + FaceFeature new fields + FacePipelineOptions + process_with_raw + cascade + RollCorrect |
| `crates/pf_database/src/repositories/face.rs` | MODIFY | NewFace.detector_origin + insert writes it; new face_roll_corrections table methods |
| `crates/pf_application/src/index.rs` | MODIFY | Convert bytes→DynamicImage; use process_with_raw; persist new fields; insert audit row; use RollCorrect HNSW |
| `apps/desktop/src-tauri/src/desktop_setup.rs` | MODIFY | Load SCRFD 10G when cascade enabled |

**Note on existing schema:** The faces table already has `yaw, pitch, roll` columns (from 009). The existing `NewFace::yaw_pitch_roll: Option<(f32, f32, f32)>` is already mapped to those columns in `insert()`. So migration 019 only adds `detector_origin` + audit table.

---

## Task 1: Schema Migration 019

**Files:**
- Create: `crates/pf_database/migrations/019_face_cascade_roll_correct.sql`
- Modify: `crates/pf_database/src/migration.rs:125`

- [ ] **Step 1: Write migration SQL**

Create `crates/pf_database/migrations/019_face_cascade_roll_correct.sql`:

```sql
-- 019_face_cascade_roll_correct.sql
--
-- Face Cascade + RollCorrect production migration.
-- Adds detector_origin enum-as-text column and face_roll_corrections audit table.

-- detector_origin records which detector(s) produced this face.
-- Values: 'Primary' (500M only) | 'Secondary' (10G only, cascade recovery) | 'Both' (dedup'd match)
-- Existing rows default to 'Primary' (no cascade was running before this migration).
ALTER TABLE faces ADD COLUMN detector_origin TEXT NOT NULL DEFAULT 'Primary';

-- face_roll_corrections: audit table for RollCorrect replacements.
-- A row exists only for faces where RollCorrect embedding REPLACED baseline
-- (triggered by |roll|>threshold at detection time).
CREATE TABLE face_roll_corrections (
    face_id          INTEGER PRIMARY KEY,
    roll_deg         REAL    NOT NULL,
    baseline_replaced INTEGER NOT NULL DEFAULT 1,
    created_at       INTEGER NOT NULL,
    FOREIGN KEY (face_id) REFERENCES faces(id) ON DELETE CASCADE
);

CREATE INDEX idx_face_rc_created ON face_roll_corrections(created_at);
```

- [ ] **Step 2: Register migration in `migration.rs`**

In `crates/pf_database/src/migration.rs`, after line 125 (the migration 018 `],`), add:

```rust
            sql: include_str!("../migrations/019_face_cascade_roll_correct.sql"),
```

Verify the numbering by looking at the surrounding `sql:` entries (should be 014, 015, 016, 017, 018, 019).

- [ ] **Step 3: Verify migration compiles**

Run: `cargo check -p pf_database`
Expected: 0 errors.

- [ ] **Step 4: Commit**

```bash
git add crates/pf_database/migrations/019_face_cascade_roll_correct.sql crates/pf_database/src/migration.rs
git commit -m "feat(db): migration 019 — detector_origin + face_roll_corrections"
```

---

## Task 2: FaceConfig — Add 6 New Fields

**Files:**
- Modify: `crates/pf_config/src/config.rs:57-95`

- [ ] **Step 1: Add fields to `FaceConfig` struct**

In `crates/pf_config/src/config.rs` at line 79 (after `fine_match_min_embedding: f32,`), add:

```rust
    /// Phase Cascade: 启用 SCRFD 500M + 10G 双 detector cascade。
    /// 启用后加载第二个 SCRFD 10G 模型,IoU dedup 后取并集。
    /// 默认 false — 与 legacy 行为一致。
    pub enable_face_cascade: bool,
    /// Phase Cascade: IoU 阈值,primary 与 secondary 的 bbox 高于此值视为同人脸。
    /// 默认 0.30 (与 SCRFD 内部 soft_nms 0.35 接近,稍放宽处理 10G bbox 偏移)。
    pub cascade_iou_threshold: f32,
    /// Phase Cascade: secondary detector (10G) 的最小 score,
    /// 低于此值的 secondary-only detection 直接丢弃。
    /// 默认 0.05 — 10G 在小脸 / 正面照上噪声较多,需要兜底。
    pub cascade_10g_min_score: f32,
    /// Phase RollCorrect: 启用 in-plane 旋转校正 (Rotate by -roll)。
    /// 对 |roll|>threshold_deg 的人脸用旋转后的图像重做 align+embed,
    /// 替换 baseline embedding。
    /// 默认 false — 与 legacy 行为一致。
    pub enable_roll_correct: bool,
    /// Phase RollCorrect: 触发阈值 (度)。
    /// |roll| 超过此值才走 RollCorrect 路径 (绝大多数正面照不动)。
    /// 默认 5.0。
    pub roll_correct_threshold_deg: f32,
    /// SCRFD 10G 模型路径 (cascade 启用时加载)。None 时使用默认路径。
    pub scrfd_10g_model_path: Option<std::path::PathBuf>,
```

- [ ] **Step 2: Update `Default for FaceConfig`**

In `crates/pf_config/src/config.rs:82-95`, add to the returned struct:

```rust
impl Default for FaceConfig {
    fn default() -> Self {
        Self {
            min_detector_score: 0.50,
            min_face_size: 20,
            min_quality: 0.45,
            max_yaw: 60.0,
            cluster_threshold: 0.65,
            similarity_threshold: 0.50,
            coarse_fetch_multiplier: 4,
            coarse_fetch_min: 4,
            fine_match_min_embedding: 0.0,
            enable_face_cascade: false,
            cascade_iou_threshold: 0.30,
            cascade_10g_min_score: 0.05,
            enable_roll_correct: false,
            roll_correct_threshold_deg: 5.0,
            scrfd_10g_model_path: None,
        }
    }
}
```

- [ ] **Step 3: Verify config compiles**

Run: `cargo check -p pf_config`
Expected: 0 errors.

- [ ] **Step 4: Commit**

```bash
git add crates/pf_config/src/config.rs
git commit -m "feat(config): FaceConfig +6 fields — cascade + roll_correct flags"
```

---

## Task 3: `DetectorOrigin` Enum + Re-export

**Files:**
- Modify: `crates/pf_ai/src/face/traits.rs:14-26` (insert enum after Keypoints type alias)
- Modify: `crates/pf_ai/src/face/mod.rs:18-21` (re-export)

- [ ] **Step 1: Add enum to traits.rs**

After line 15 (the `pub type Keypoints = FaceKeypoints;`), add:

```rust
/// Detector 来源 — cascade 启用时追踪是哪台 detector(s) 检出了这张脸。
///
/// `Secondary` = "10G-only recovered" — SCRFD 500M 漏检、cascade 救回来的人脸。
/// `Both` = 两台都检出、IoU dedup 后保留高分。
/// 关闭 cascade 时所有 face 均为 `Primary`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectorOrigin {
    /// 仅 SCRFD 500M 检出。
    Primary,
    /// 仅 SCRFD 10G 检出 (cascade 救回)。
    Secondary,
    /// 两台都检出 (dedup'd,保留高分)。
    Both,
}

impl DetectorOrigin {
    /// 序列化为 DB 存储格式。
    pub fn as_str(&self) -> &'static str {
        match self {
            DetectorOrigin::Primary => "Primary",
            DetectorOrigin::Secondary => "Secondary",
            DetectorOrigin::Both => "Both",
        }
    }

    /// 从 DB 字符串反序列化。
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "Primary" => Some(Self::Primary),
            "Secondary" => Some(Self::Secondary),
            "Both" => Some(Self::Both),
            _ => None,
        }
    }
}
```

- [ ] **Step 2: Re-export from mod.rs**

In `crates/pf_ai/src/face/mod.rs:18-21`, update the `pub use traits::{...}` line to include `DetectorOrigin`:

```rust
pub use traits::{
    AlignedFace, DetectorOrigin, FaceAligner, FaceDetection, FaceDetector, FaceEmbedder,
    FaceFeature, FacePipeline, Keypoints, QualityFilter,
};
```

- [ ] **Step 3: Verify compiles**

Run: `cargo check -p pf_ai`
Expected: 0 errors.

- [ ] **Step 4: Commit**

```bash
git add crates/pf_ai/src/face/traits.rs crates/pf_ai/src/face/mod.rs
git commit -m "feat(ai): DetectorOrigin enum + re-export"
```

---

## Task 4: `FaceFeature` New Fields

**Files:**
- Modify: `crates/pf_ai/src/face/traits.rs:38-52`

- [ ] **Step 1: Extend FaceFeature struct**

Replace the `FaceFeature` struct body (lines 38-52) with:

```rust
/// 检测 + 对齐 + embedding 后的最终特征。
#[derive(Debug, Clone)]
pub struct FaceFeature {
    /// 检测
    pub detection: FaceDetection,
    /// 512-d embedding (baseline 或 RollCorrect — 见 `roll_correct_replaced`)
    pub embedding: Embedding,
    /// 5-KPS 估计的头部姿态（yaw / pitch / roll，度；退化时 None）
    pub yaw_pitch_roll: Option<(f32, f32, f32)>,
    /// blur score（Laplacian 方差归一化）
    pub blur_score: f32,
    /// pose score（由 yaw/pitch/roll 推导）
    pub pose_score: f32,
    /// face area score（归一化最小边长）
    pub face_area_score: f32,
    /// Detector 来源（cascade 关闭时恒为 Primary）。
    pub detector_origin: DetectorOrigin,
    /// 是否用 RollCorrect embedding 替换了 baseline。
    /// `true` 时 `embedding` 是 RollCorrect 输出,baseline 已丢弃。
    pub roll_correct_replaced: bool,
    /// RollCorrect 触发前的原始 roll 度数（用于 face_roll_corrections 审计）。
    /// 仅在 `roll_correct_replaced=true` 时非 None。
    pub original_roll: Option<f32>,
}
```

- [ ] **Step 2: Verify compiles**

Run: `cargo check -p pf_ai`
Expected: errors at all `FaceFeature { ... }` construction sites (the existing `process()` method in `traits.rs:204` and tests). This is expected — Task 7 adds the cascade path that populates them. For now, temporarily set `detector_origin: DetectorOrigin::Primary, roll_correct_replaced: false` in `process()` at line 204.

Update line 204 (the `features.push(FaceFeature { ... })` call) to:

```rust
            features.push(FaceFeature {
                detection: det,
                embedding,
                yaw_pitch_roll,
                blur_score: q.blur_score,
                pose_score: q.pose_score,
                face_area_score: q.face_area_score,
                detector_origin: DetectorOrigin::Primary,
                roll_correct_replaced: false,
                original_roll: None,
            });
```

- [ ] **Step 3: Verify whole workspace compiles**

Run: `cargo check --workspace`
Expected: 0 errors. (Tests that build FaceFeature directly may need updates too — check `crates/pf_ai/tests/`, `crates/pf_application/tests/` and update construction sites to include the two new fields.)

If tests fail to compile, search for `FaceFeature {` and update each site to add `detector_origin: DetectorOrigin::Primary, roll_correct_replaced: false,`. This is a mechanical change.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(ai): FaceFeature +detector_origin +roll_correct_replaced"
```

---

## Task 5: `merge_detections_cascade` — Unit Test + Impl

**Files:**
- Modify: `crates/pf_ai/src/face/traits.rs` (add function at end of file)
- Add test: inline `#[cfg(test)] mod tests` at end of traits.rs

- [ ] **Step 1: Write the failing test**

Add to `crates/pf_ai/src/face/traits.rs` at the bottom:

```rust
#[cfg(test)]
mod merge_cascade_tests {
    use super::*;

    fn det_at(x: f32, y: f32, w: f32, h: f32, score: f32) -> FaceDetection {
        use pf_core::FaceKeypoints;
        FaceDetection {
            bbox: BBox::new(x, y, w, h),
            score,
            keypoints: FaceKeypoints {
                left_eye: (x + w * 0.3, y + h * 0.3),
                right_eye: (x + w * 0.7, y + h * 0.3),
                nose: (x + w * 0.5, y + h * 0.5),
                left_mouth: (x + w * 0.4, y + h * 0.8),
                right_mouth: (x + w * 0.6, y + h * 0.8),
            },
        }
    }

    #[test]
    fn iou_above_threshold_matches() {
        let primary = vec![det_at(0.0, 0.0, 100.0, 100.0, 0.50)];
        let secondary = vec![det_at(5.0, 5.0, 100.0, 100.0, 0.55)];
        let result = merge_detections_cascade(primary, secondary, 0.30, 0.05);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, DetectorOrigin::Both);
    }

    #[test]
    fn iou_below_threshold_separates() {
        let primary = vec![det_at(0.0, 0.0, 100.0, 100.0, 0.50)];
        let secondary = vec![det_at(150.0, 0.0, 100.0, 100.0, 0.55)];
        let result = merge_detections_cascade(primary, secondary, 0.30, 0.05);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].1, DetectorOrigin::Primary);
        assert_eq!(result[1].1, DetectorOrigin::Secondary);
    }

    #[test]
    fn secondary_only_recovered_when_above_min_score() {
        let primary = vec![];
        let secondary = vec![det_at(0.0, 0.0, 50.0, 50.0, 0.04)]; // below 0.05
        let result = merge_detections_cascade(primary, secondary, 0.30, 0.05);
        assert_eq!(result.len(), 0, "10G-only face below min_score must be filtered");
    }

    #[test]
    fn secondary_only_recovered_when_above_min_score_passes() {
        let primary = vec![];
        let secondary = vec![det_at(0.0, 0.0, 50.0, 50.0, 0.06)]; // above 0.05
        let result = merge_detections_cascade(primary, secondary, 0.30, 0.05);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, DetectorOrigin::Secondary);
    }

    #[test]
    fn higher_score_kept_when_both_match() {
        let primary = vec![det_at(0.0, 0.0, 100.0, 100.0, 0.50)];
        let secondary = vec![det_at(2.0, 2.0, 100.0, 100.0, 0.80)];
        let result = merge_detections_cascade(primary, secondary, 0.30, 0.05);
        assert_eq!(result.len(), 1);
        assert!(result[0].0.score >= 0.80, "kept the higher-score detection");
        assert_eq!(result[0].1, DetectorOrigin::Both);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p pf_ai --lib merge_cascade_tests`
Expected: compile error (`merge_detections_cascade` not defined).

- [ ] **Step 3: Implement `merge_detections_cascade`**

In `crates/pf_ai/src/face/traits.rs` (above the test module), add:

```rust
/// Merge detections from two SCRFD variants (primary 500M + secondary 10G).
///
/// Strategy: for each primary detection, find the best-IoU secondary detection
/// above `iou_threshold`. If matched, keep higher-score detection and tag as `Both`.
/// Otherwise, primary → `Primary`; unmatched secondaries (with score ≥
/// `secondary_min_score`) → `Secondary` (cascade recovered).
pub fn merge_detections_cascade(
    primary: Vec<FaceDetection>,
    secondary: Vec<FaceDetection>,
    iou_threshold: f32,
    secondary_min_score: f32,
) -> Vec<(FaceDetection, DetectorOrigin)> {
    let mut result: Vec<(FaceDetection, DetectorOrigin)> = Vec::with_capacity(primary.len() + secondary.len());
    let mut secondary_matched = vec![false; secondary.len()];

    // Pass 1: walk primary; match each to best available secondary
    for p in primary {
        let mut best_iou_idx: Option<usize> = None;
        let mut best_iou: f32 = 0.0;
        for (i, s) in secondary.iter().enumerate() {
            if secondary_matched[i] {
                continue;
            }
            let iou = p.bbox.iou(&s.bbox);
            if iou > iou_threshold && iou > best_iou {
                best_iou = iou;
                best_iou_idx = Some(i);
            }
        }
        if let Some(idx) = best_iou_idx {
            secondary_matched[idx] = true;
            let s = &secondary[idx];
            let kept = if s.score > p.score { s.clone() } else { p };
            result.push((kept, DetectorOrigin::Both));
        } else {
            result.push((p, DetectorOrigin::Primary));
        }
    }

    // Pass 2: secondary-only (cascade value-add)
    for (i, s) in secondary.into_iter().enumerate() {
        if secondary_matched[i] {
            continue;
        }
        if s.score < secondary_min_score {
            continue;
        }
        result.push((s, DetectorOrigin::Secondary));
    }

    result
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p pf_ai --lib merge_cascade_tests`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/pf_ai/src/face/traits.rs
git commit -m "feat(ai): merge_detections_cascade IoU dedup + unit tests"
```

---

## Task 6: Rotate Helpers — Port from Test

**Files:**
- Modify: `crates/pf_ai/src/face/mod.rs` (new file `cascade.rs`)
- Modify: `crates/pf_ai/src/face/mod.rs` (add `pub mod cascade`)

- [ ] **Step 1: Create `crates/pf_ai/src/face/cascade.rs`**

```rust
//! Cascade + RollCorrect helper functions for `FacePipeline::process_with_raw`.
//!
//! These were ported from `tests/quick_face_check.rs` so the production
//! pipeline can use the same IoU dedup and rotation logic.

use image::{imageops, DynamicImage};

/// Rotate the full image by `angle_deg`. Uses fast-path 90/180/270 when
/// possible; otherwise returns the input unchanged (SCRFD's eye-line roll
/// is usually small, so the exact-pixels path is rarely needed).
///
/// Negative angle rotates counter-clockwise.
pub fn rotate_image_full(image: &DynamicImage, angle_deg: f32) -> DynamicImage {
    let normalized = ((angle_deg % 360.0) + 360.0) % 360.0;
    let rgb8 = image.to_rgb8();
    if normalized.abs() < 1.0 {
        DynamicImage::ImageRgb8(rgb8)
    } else if (normalized - 90.0).abs() < 1.0 {
        DynamicImage::ImageRgb8(imageops::rotate90(&rgb8))
    } else if (normalized - 180.0).abs() < 1.0 {
        DynamicImage::ImageRgb8(imageops::rotate180(&rgb8))
    } else if (normalized - 270.0).abs() < 1.0 {
        DynamicImage::ImageRgb8(imageops::rotate270(&rgb8))
    } else {
        DynamicImage::ImageRgb8(rgb8)
    }
}

/// Rotate 5-point landmarks around the image center by `angle_deg`.
/// Used after `rotate_image_full` to keep kps aligned with the rotated image.
pub fn rotate_landmarks_around_center(
    kps: &[(f32, f32); 5],
    width: u32,
    height: u32,
    angle_deg: f32,
) -> [(f32, f32); 5] {
    let angle_rad = angle_deg * std::f32::consts::PI / 180.0;
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();
    let cx = width as f32 / 2.0;
    let cy = height as f32 / 2.0;
    let mut out = [(0.0f32, 0.0f32); 5];
    for (i, (x, y)) in kps.iter().enumerate() {
        let dx = x - cx;
        let dy = y - cy;
        out[i] = (cos_a * dx - sin_a * dy + cx, sin_a * dx + cos_a * dy + cy);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate_zero_is_identity() {
        // Create a 100x100 gray image
        let img = DynamicImage::new_rgb8(100, 100);
        let rotated = rotate_image_full(&img, 0.0);
        assert_eq!(rotated.width(), 100);
        assert_eq!(rotated.height(), 100);
    }

    #[test]
    fn rotate_90_swaps_dims() {
        let img = DynamicImage::new_rgb8(100, 200);
        let rotated = rotate_image_full(&img, 90.0);
        assert_eq!(rotated.width(), 200);
        assert_eq!(rotated.height(), 100);
    }

    #[test]
    fn rotate_landmarks_90_moves_origin_to_corner() {
        // Center-origin point (0, 0) rotated by 90° clockwise → (0, 0) (since cx=cy=0)
        let kps = [(0.0, 0.0), (10.0, 0.0), (5.0, 5.0), (5.0, 10.0), (10.0, 10.0)];
        let rotated = rotate_landmarks_around_center(&kps, 100, 100, 90.0);
        // Center (50,50) stays at (50,50) under any rotation
        // Actually 0,0 is at corner — for 90° CW: (-y, x) = (0, 0) since y=0
        // After translation: still (0, 0)? No, relative to (50,50):
        // dx=-50, dy=-50 → rotation 90° CW: (50, -50) + (50,50) = (100, 0)
        // Wait our rotation matrix is [cos -sin; sin cos] which is CCW for positive angle
        // cos(90)=0, sin(90)=1: (0*-50 - 1*-50, 1*-50 + 0*-50) = (50, -50)
        // + (50, 50) = (100, 0)
        assert!((rotated[0].0 - 100.0).abs() < 1.0);
        assert!((rotated[0].1 - 0.0).abs() < 1.0);
    }
}
```

- [ ] **Step 2: Register `cascade` module in `face/mod.rs`**

After line 8 (`pub mod traits;`), add:

```rust
pub mod cascade;
```

After line 21 (the `pub use traits::{...}` line), add:

```rust
pub use cascade::{merge_detections_cascade as _, rotate_image_full, rotate_landmarks_around_center};
```

Wait — `merge_detections_cascade` is in `traits.rs`, not `cascade.rs`. Fix the re-export:

```rust
pub use cascade::{rotate_image_full, rotate_landmarks_around_center};
```

(`merge_detections_cascade` is re-exported via the `pub use traits::{...}` line — add it to that list.)

- [ ] **Step 3: Verify compiles + tests pass**

Run: `cargo test -p pf_ai --lib cascade`
Expected: 3 passed (rotate_zero_is_identity, rotate_90_swaps_dims, rotate_landmarks_90_moves_origin_to_corner).

- [ ] **Step 4: Commit**

```bash
git add crates/pf_ai/src/face/cascade.rs crates/pf_ai/src/face/mod.rs
git commit -m "feat(ai): cascade.rs rotate helpers + unit tests"
```

---

## Task 7: `FacePipelineOptions` + Enhanced Constructor

**Files:**
- Modify: `crates/pf_ai/src/face/traits.rs:132-155` (replace FacePipeline struct + impl)

- [ ] **Step 1: Add options struct + new constructor (keep `new()` for backward compat)**

Replace lines 132-155 of `crates/pf_ai/src/face/traits.rs` (the `FacePipeline` struct + `impl FacePipeline { pub fn new(...)`) with:

```rust
/// `FacePipeline` 可选增强配置 (cascade + RollCorrect)。
#[derive(Debug, Clone, Default)]
pub struct FacePipelineOptions {
    /// 启用 SCRFD 500M + 10G cascade。`secondary_detector` 必须 Some。
    pub enable_face_cascade: bool,
    /// cascade IoU dedup 阈值。
    pub cascade_iou_threshold: f32,
    /// cascade secondary detector 最小 score。
    pub cascade_10g_min_score: f32,
    /// 启用 RollCorrect 替换 baseline embedding。
    pub enable_roll_correct: bool,
    /// RollCorrect 触发阈值 (度)。
    pub roll_correct_threshold_deg: f32,
}

/// 人脸 pipeline：detect → align → embed → quality filter。
pub struct FacePipeline {
    primary_detector: Arc<dyn FaceDetector>,
    secondary_detector: Option<Arc<dyn FaceDetector>>,
    aligner: Arc<dyn FaceAligner>,
    embedder: Arc<dyn FaceEmbedder>,
    quality_filter: QualityFilter,
    options: FacePipelineOptions,
}

impl FacePipeline {
    /// 构造基础版 pipeline（与旧 API 兼容,所有增强默认关闭）。
    pub fn new(
        detector: Arc<dyn FaceDetector>,
        aligner: Arc<dyn FaceAligner>,
        embedder: Arc<dyn FaceEmbedder>,
        quality_filter: QualityFilter,
    ) -> Self {
        Self {
            primary_detector: detector,
            secondary_detector: None,
            aligner,
            embedder,
            quality_filter,
            options: FacePipelineOptions::default(),
        }
    }

    /// 构造增强版 pipeline（cascade + RollCorrect 视 options 启用）。
    /// `secondary_detector` 即使 options 关闭也可以传入 — 关闭时不调用。
    pub fn with_options(
        primary_detector: Arc<dyn FaceDetector>,
        secondary_detector: Option<Arc<dyn FaceDetector>>,
        aligner: Arc<dyn FaceAligner>,
        embedder: Arc<dyn FaceEmbedder>,
        quality_filter: QualityFilter,
        options: FacePipelineOptions,
    ) -> Self {
        Self {
            primary_detector,
            secondary_detector,
            aligner,
            embedder,
            quality_filter,
            options,
        }
    }
}
```

- [ ] **Step 2: Verify workspace compiles**

Run: `cargo check --workspace`
Expected: 0 errors (we kept `new()` API intact, just renamed `detector` → `primary_detector` internally — need to update Task 4's `process()` method that uses `self.detector`).

In `process()` (line 160), replace `self.detector.detect(image)` with `self.primary_detector.detect(image)`. Update line 165 `debug!("SCRFD detected {} faces", ...)` stays the same.

- [ ] **Step 3: Commit**

```bash
git add crates/pf_ai/src/face/traits.rs
git commit -m "feat(ai): FacePipelineOptions + with_options() constructor"
```

---

## Task 8: `process_with_raw` + Cascade + RollCorrect Logic

**Files:**
- Modify: `crates/pf_ai/src/face/traits.rs` (add `process_with_raw` method, replace existing `process` body)

- [ ] **Step 1: Add helper for face embedding (refactor existing process into helper)**

The current `process()` does detect → for-each: crop+align+embed+quality. The cascade path is identical except:
1. Detections come from `merge_detections_cascade` not a single detector
2. After embedding, optionally rotate-and-re-embed (RollCorrect)
3. Tag `detector_origin` and `roll_correct_replaced`

Refactor by extracting a private helper `embed_one_face(image, det) -> Option<FaceFeature>` that does the per-face work, then have `process()` and `process_with_raw()` call it in different ways.

Add to `traits.rs` (below the existing `process()` method):

```rust
impl FacePipeline {
    /// 处理一张图片(基础版 API,与旧行为兼容 — 不支持 cascade / RollCorrect)。
    /// 当 `options.enable_face_cascade` 为 true 但 raw_image 未提供时,
    /// cascade 不运行(仅 primary detector),保持单 detector 行为。
    pub async fn process(&self, image: &ImageData) -> Result<Vec<FaceFeature>, AIError> {
        self.process_with_raw(image, None).await
    }

    /// 处理一张图片,完整功能版本。
    ///
    /// - `image`: 标准输入(用于 baseline detector + align)
    /// - `raw_image`: 完整 DynamicImage,用于 RollCorrect 旋转。
    ///   cascade 启用时也需要(因为 cascade 是 detector 层,image 即可)。
    ///   RollCorrect 关闭时传 None 即可。
    ///
    /// 行为:
    /// - cascade 关闭 + RollCorrect 关闭: 与旧 `process()` 100% 一致。
    /// - cascade 打开: merge_detections_cascade 取并集,detector_origin 反映来源。
    /// - RollCorrect 打开: 对 |roll|>threshold 的 face 用旋转后图像重新 align+embed,
    ///   baseline embedding 被替换,`roll_correct_replaced=true`。
    pub async fn process_with_raw(
        &self,
        image: &ImageData,
        raw_image: Option<&DynamicImage>,
    ) -> Result<Vec<FaceFeature>, AIError> {
        use crate::quality;

        // 1) Detection — cascade or single
        let primary_dets = self.primary_detector.detect(image).await?;
        debug!("primary detector returned {} faces", primary_dets.len());

        let merged: Vec<(FaceDetection, DetectorOrigin)> = if self.options.enable_face_cascade {
            let secondary = match &self.secondary_detector {
                Some(d) => d.detect(image).await.unwrap_or_else(|e| {
                    tracing::warn!(error = %e, "secondary detector failed; falling back to primary only");
                    Vec::new()
                }),
                None => {
                    tracing::warn!("cascade enabled but secondary_detector is None; running primary only");
                    Vec::new()
                }
            };
            debug!("secondary detector returned {} faces", secondary.len());
            merge_detections_cascade(
                primary_dets,
                secondary,
                self.options.cascade_iou_threshold,
                self.options.cascade_10g_min_score,
            )
        } else {
            primary_dets.into_iter().map(|d| (d, DetectorOrigin::Primary)).collect()
        };

        let mut features = Vec::new();
        for (det, detector_origin) in merged {
            if let Some(mut feat) = self.embed_one(image, &det).await {
                feat.detector_origin = detector_origin;

                // RollCorrect: 对 |roll|>threshold 的人脸,用旋转后图像重做 align+embed
                if self.options.enable_roll_correct {
                    if let (Some(raw), Some((_, _, roll))) = (raw_image, feat.yaw_pitch_roll) {
                        if roll.abs() > self.options.roll_correct_threshold_deg {
                            if let Some(rc_feat) = self.try_roll_correct(raw, &det, roll).await {
                                feat = rc_feat;
                            }
                            // 若 RollCorrect 失败,fall back 到 baseline (replaced=false)
                        }
                    }
                }

                features.push(feat);
            }
        }
        Ok(features)
    }

    /// 单张人脸的完整 pipeline (crop → align → embed → quality)。
    /// 与原 `process()` 内层循环逻辑一致,拆出来便于 process / process_with_raw 共用。
    async fn embed_one(
        &self,
        image: &ImageData,
        det: &FaceDetection,
    ) -> Option<FaceFeature> {
        use crate::quality;
        let yaw_pitch_roll = crate::face::pose::estimate_yaw_pitch_roll(&det.keypoints);

        let (align_input, shifted_kps) = match crop_to_bbox_with_margin(image, det) {
            Ok(v) => v,
            Err(e) => {
                debug!(error = %e, "skip frame: invalid crop bbox");
                return None;
            }
        };
        let aligned = match self.aligner.align(&align_input, &shifted_kps) {
            Ok(a) => a,
            Err(e) => {
                debug!(error = %e, "align failed");
                return None;
            }
        };
        let embedding = match self.embedder.embed(&aligned).await {
            Ok(e) => e,
            Err(e) => {
                debug!("embedding failed: {}", e);
                return None;
            }
        };
        let gray = aligned.image.to_luma8();
        let blur = quality::blur_score_from_aligned(&gray);
        let q = quality::assess(det, blur, yaw_pitch_roll);
        if !q.passes() {
            debug!("face filtered by quality: quality={}", q.quality);
            return None;
        }
        if det.score < self.quality_filter.min_detector_score {
            debug!("face filtered by detector score: {}", det.score);
            return None;
        }
        Some(FaceFeature {
            detection: det.clone(),
            embedding,
            yaw_pitch_roll,
            blur_score: q.blur_score,
            pose_score: q.pose_score,
            face_area_score: q.face_area_score,
            detector_origin: DetectorOrigin::Primary,
            roll_correct_replaced: false,
            original_roll: None,
        })
    }

    /// RollCorrect: 旋转图像 by -roll, 旋转 kps, 重新 align+embed。
    /// 返回 Some(新 FaceFeature),嵌入失败返回 None 让 caller fallback。
    async fn try_roll_correct(
        &self,
        raw_image: &DynamicImage,
        det: &FaceDetection,
        roll: f32,
    ) -> Option<FaceFeature> {
        use image::ImageEncoder;
        use std::io::Cursor;

        let kps_array: [(f32, f32); 5] = [
            det.keypoints.left_eye,
            det.keypoints.right_eye,
            det.keypoints.nose,
            det.keypoints.left_mouth,
            det.keypoints.right_mouth,
        ];

        // 1. Rotate image and kps
        let rotated_img = crate::face::cascade::rotate_image_full(raw_image, -roll);
        let rotated_kps = crate::face::cascade::rotate_landmarks_around_center(
            &kps_array,
            raw_image.width(),
            raw_image.height(),
            -roll,
        );

        // 2. Save rotated to temp file (ImageData::from_file requires path)
        let tmp_path = std::env::temp_dir().join(format!(
            "pf_roll_correct_{}_{}.jpg",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let mut buf = Vec::new();
        if rotated_img
            .write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Jpeg)
            .is_err()
        {
            return None;
        }
        if std::fs::write(&tmp_path, &buf).is_err() {
            return None;
        }

        // 3. Load as ImageData, align + embed via the same path
        let rotated_data = match ImageData::from_file(&tmp_path) {
            Ok(d) => d,
            Err(_) => {
                let _ = std::fs::remove_file(&tmp_path);
                return None;
            }
        };
        let _ = std::fs::remove_file(&tmp_path);

        // 4. Align rotated face
        let rotated_kps_struct = pf_core::FaceKeypoints {
            left_eye: rotated_kps[0],
            right_eye: rotated_kps[1],
            nose: rotated_kps[2],
            left_mouth: rotated_kps[3],
            right_mouth: rotated_kps[4],
        };
        let aligned = match self.aligner.align(&rotated_data, &rotated_kps_struct) {
            Ok(a) => a,
            Err(_) => return None,
        };
        let embedding = match self.embedder.embed(&aligned).await {
            Ok(e) => e,
            Err(_) => return None,
        };
        let gray = aligned.image.to_luma8();
        let blur = crate::quality::blur_score_from_aligned(&gray);
        let q = crate::quality::assess(det, blur, Some((0.0, 0.0, 0.0))); // yaw/pitch/roll ~ 0 after correction

        Some(FaceFeature {
            detection: det.clone(),
            embedding,
            yaw_pitch_roll: Some((0.0, 0.0, 0.0)),
            blur_score: q.blur_score,
            pose_score: q.pose_score,
            face_area_score: q.face_area_score,
            detector_origin: DetectorOrigin::Primary, // overwritten by caller
            roll_correct_replaced: true,
            original_roll: Some(roll),
        })
    }
}
```

- [ ] **Step 2: Add `use image::DynamicImage` import at top of traits.rs**

After line 11 (`use crate::image_data::ImageData;`), add:

```rust
use image::DynamicImage;
```

- [ ] **Step 3: Verify compiles**

Run: `cargo check -p pf_ai`
Expected: 0 errors. (May need to fix the imports — if `FaceKeypoints` is not in scope from `pf_core`, use the fully qualified path.)

- [ ] **Step 4: Verify whole workspace compiles**

Run: `cargo check --workspace`
Expected: 0 errors.

- [ ] **Step 5: Run pf_ai tests**

Run: `cargo test -p pf_ai --lib`
Expected: all tests pass (including the new merge_cascade_tests from Task 5).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(ai): process_with_raw with cascade + RollCorrect paths"
```

---

## Task 9: `NewFace::detector_origin` + Insert Writes It

**Files:**
- Modify: `crates/pf_database/src/repositories/face.rs:70-111` (struct)
- Modify: `crates/pf_database/src/repositories/face.rs:184-231` (insert SQL)
- Modify: `crates/pf_database/src/repositories/face.rs` (new face_roll_corrections methods)

- [ ] **Step 1: Add `detector_origin` field to NewFace**

After line 110 (`pub cluster_method: Option<String>,`), add:

```rust
    /// Cascade detector 来源。'Primary' (500M) | 'Secondary' (10G recovered) | 'Both'
    /// 默认 "Primary",旧数据迁移后也是 "Primary"。
    pub detector_origin: String,
```

- [ ] **Step 2: Update `insert()` SQL**

Replace the existing `INSERT INTO faces ...` SQL (lines 191-199) and params (200-228) to include `detector_origin`:

SQL becomes:

```sql
INSERT INTO faces
 (image_id, bbox_x, bbox_y, bbox_w, bbox_h,
  detector_score, detector_model, keypoints_json,
  yaw, pitch, roll,
  quality_score, blur_score, pose_score, face_area_score,
  alignment_version, embedding_model, model_version,
  vector_id, hnsw_handle, status, index_status, last_attempt_at,
  index_generation, cluster_score, cluster_method,
  detector_origin)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27)
```

In the params! block (line 200-228), add at the end:

```rust
                face.cluster_method,
                face.detector_origin,
```

- [ ] **Step 3: Add `face_roll_corrections` repository methods**

Add a new impl block (or methods within `FaceRepository`) — find a good spot and add:

```rust
impl<'tx, 'db> FaceRepository<'tx, 'db> {
    /// Insert a face_roll_corrections audit row.
    /// Returns the count (always 1 on success, 0 if face_id doesn't exist — but
    /// we don't enforce FK existence to keep the call site simple).
    pub fn insert_roll_correction(
        &mut self,
        face_id: i64,
        roll_deg: f32,
        baseline_replaced: bool,
    ) -> Result<i64, DatabaseError> {
        self.tx.execute(
            "INSERT INTO face_roll_corrections (face_id, roll_deg, baseline_replaced, created_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                face_id,
                roll_deg,
                baseline_replaced as i64,
                chrono::Utc::now().timestamp(),
            ],
        )?;
        Ok(self.tx.last_insert_rowid())
    }

    /// Check if a face has a RollCorrect audit row.
    pub fn has_roll_correction(&self, face_id: i64) -> Result<bool, DatabaseError> {
        let mut stmt = self
            .tx
            .prepare("SELECT 1 FROM face_roll_corrections WHERE face_id = ?1 LIMIT 1")?;
        let mut rows = stmt.query(params![face_id])?;
        Ok(rows.next()?.is_some())
    }
}
```

- [ ] **Step 4: Verify compiles**

Run: `cargo check -p pf_database`
Expected: errors at all `NewFace { ... }` construction sites that don't set `detector_origin` (default it to `"Primary".to_string()`).

- [ ] **Step 5: Find all `NewFace { ... }` construction sites**

Run: `grep -rn "NewFace {" /Users/mac/ai-project/photofinder-next-2/crates/ --include="*.rs" | grep -v target`

For each site, add `detector_origin: "Primary".to_string(),` field. Most sites are tests / fixtures that don't care about cascade.

- [ ] **Step 6: Verify whole workspace compiles**

Run: `cargo check --workspace`
Expected: 0 errors.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(db): NewFace.detector_origin + face_roll_corrections repo methods"
```

---

## Task 10: `IndexService::index_image` — Use `process_with_raw`

**Files:**
- Modify: `crates/pf_application/src/index.rs:117-199` (`index_image`)
- Modify: `crates/pf_application/src/index.rs:201-245` (`index_faces_for_image`)

- [ ] **Step 1: Update `index_image` to convert bytes → DynamicImage**

In `index_image`, after line 127 (`let img = ImageData::from_bytes(&bytes)?;`), add:

```rust
        // Convert bytes → DynamicImage for cascade + RollCorrect rotation input.
        // Cost: in-memory decode, <10ms/image. Only used when flags are on.
        let dyn_img = image::load_from_memory(&bytes).ok();
```

Then change line 136 (`let face_features = match self.face_pipeline.process(&img).await`) to:

```rust
        let face_features = match self.face_pipeline.process_with_raw(&img, dyn_img.as_ref()).await {
```

Same change in `index_faces_for_image` (line 221 → use `process_with_raw`).

- [ ] **Step 2: Verify compiles**

Run: `cargo check -p pf_application`
Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
git add crates/pf_application/src/index.rs
git commit -m "feat(app): IndexService uses process_with_raw"
```

---

## Task 11: `IndexService::index_faces` — Persist New + Use RollCorrect HNSW

**Files:**
- Modify: `crates/pf_application/src/index.rs:308-444` (`index_faces`)

- [ ] **Step 1: Pull new fields from FaceFeature**

In `index_faces` line 320-364 (the "Step 1: 准备阶段" loop), update `NewFace` construction to include `detector_origin` (from `feat.detector_origin.as_str()`):

```rust
            let new_face = NewFace {
                image_id,
                bbox: det.bbox,
                detector_score: det.score,
                detector_model: "scrfd-500m-bnkps".to_string(),
                keypoints_json,
                yaw_pitch_roll: feat.yaw_pitch_roll,
                quality: crate::quality_bridge(det, feat),
                blur_score: Some(feat.blur_score),
                pose_score: Some(feat.pose_score),
                face_area_score: Some(feat.face_area_score),
                alignment_version: Some(pf_ai::face::ALIGNMENT_VERSION.to_string()),
                embedding_model: Some(emb.model.name().to_string()),
                model_version: emb.model.as_str().to_string(),
                vector_id: None,
                hnsw_handle: None,
                status: FaceStatus::Pending,
                index_generation: crate::index_generation::current(),
                cluster_score: None,
                cluster_method: None,
                detector_origin: feat.detector_origin.as_str().to_string(),
            };
```

- [ ] **Step 2: After Phase A insert, write RollCorrect audit row**

Track `original_roll` per prepared face (4th tuple element). Pipeline sets `feat.original_roll = Some(roll)` on the new FaceFeature when RollCorrect runs (see Task 8), so `Option<f32>` is a reliable signal: `Some(_)` ⇒ RollCorrect ran with that original roll.

Update `Step 1` (`index_faces` lines 320-364) — change tuple type to carry original roll:

```rust
        let mut prepared: Vec<(NewFace, i64, Embedding, Option<f32>)> = Vec::with_capacity(features.len());
        for feat in features {
            let emb: &Embedding = &feat.embedding;
            let vector_id = derive_face_vector_id(image_id, feat);

            let already_indexed = self
                .db
                .transaction(|tx| tx.faces().get_by_vector_id(vector_id))
                .ok()
                .flatten()
                .map(|row| row.status == FaceStatus::Indexed)
                .unwrap_or(false);
            if already_indexed {
                debug!(vector_id, "face already indexed, skipping");
                continue;
            }

            let det = &feat.detection;
            let keypoints_json = serde_json::to_string(&det.keypoints).ok();

            let new_face = NewFace {
                image_id,
                bbox: det.bbox,
                detector_score: det.score,
                detector_model: "scrfd-500m-bnkps".to_string(),
                keypoints_json,
                yaw_pitch_roll: feat.yaw_pitch_roll,
                quality: crate::quality_bridge(det, feat),
                blur_score: Some(feat.blur_score),
                pose_score: Some(feat.pose_score),
                face_area_score: Some(feat.face_area_score),
                alignment_version: Some(pf_ai::face::ALIGNMENT_VERSION.to_string()),
                embedding_model: Some(emb.model.name().to_string()),
                model_version: emb.model.as_str().to_string(),
                vector_id: None,
                hnsw_handle: None,
                status: FaceStatus::Pending,
                index_generation: crate::index_generation::current(),
                cluster_score: None,
                cluster_method: None,
                detector_origin: feat.detector_origin.as_str().to_string(),
            };

            prepared.push((new_face, vector_id, emb.clone(), feat.original_roll));
        }
```

Insert this block between Phase A (line 380) and Phase B (line 382):

```rust
        // Phase A2: RollCorrect audit rows (one per face whose embedding was replaced)
        self.db.transaction_with_retry(|tx| {
            for ((_, _, _, original_roll), fid) in prepared.iter().zip(face_ids.iter()) {
                if let Some(roll) = original_roll {
                    tx.faces().insert_roll_correction(*fid, roll, true)?;
                }
            }
            Ok::<_, pf_database::DatabaseError>(())
        }).map_err(ApplicationError::Database)?;
```

- [ ] **Step 3: Update Phase B/C loop to handle the new tuple**

The loops at lines 383-390 (`for ((_, vector_id, emb), face_id) in prepared.iter()...`) need to unpack 4 fields instead of 3.

Change to:

```rust
        for ((_, vector_id, emb, _), face_id) in prepared.iter().zip(face_ids.iter()) {
            match self.face_index.insert(*vector_id, emb.as_slice()) {
                Ok(()) => successes.push((*face_id, *vector_id, emb.clone())),
                Err(e) => failures.push((*face_id, format!("hnsw insert: {e}"))),
            }
        }
```

(The 4th field `_` is `original_roll`, already consumed by Phase A2 above.)

- [ ] **Step 4: Verify compiles**

Run: `cargo check -p pf_application`
Expected: 0 errors (the Phase C loops at lines 393-421 use `successes` which is `Vec<(i64, i64, Embedding)>` — unchanged).

- [ ] **Step 5: Commit**

```bash
git add crates/pf_application/src/index.rs
git commit -m "feat(app): IndexService.index_faces persists detector_origin + RollCorrect audit"
```

---

## Task 12: `desktop_setup::try_load_face_pipeline` — Load SCRFD 10G

**Files:**
- Modify: `apps/desktop/src-tauri/src/desktop_setup.rs:146-184`

- [ ] **Step 1: Update face pipeline loading**

In `try_load_face_pipeline`, after the primary detector loads successfully (after line 152), add cascade + RollCorrect loading:

```rust
pub fn try_load_face_pipeline(
    model_manager: Arc<ModelManager>,
    config: Arc<Config>,
) -> Result<Option<Arc<FacePipeline>>, String> {
    let face_config = &config.face;
    let primary_path = model_manager
        .get_model_path("scrfd")
        .ok_or_else(|| "scrfd model not found".to_string())?;
    let primary = ScrfdDetector::load(&primary_path)
        .map(Arc::new)
        .map_err(|Error: "|"failed to load SCRFD primary: {}")?;

    // Cascade: 加载 SCRFD 10G (如果启用)
    let secondary: Option<Arc<ScrfdDetector>> = if face_config.enable_face_cascade {
        let secondary_path = face_config
            .scrfd_10g_model_path
            .clone()
            .or_else(|| model_manager.get_model_path("scrfd_10g"))
            .ok_or_else(|| "scrfd_10g model not found (cascade enabled)".to_string())?;
        match ScrfdDetector::load(&secondary_path) {
            Ok(d) => Some(Arc::new(d)),
            Err(e) => {
                tracing::warn!(error = %e, "SCRFD 10G load failed; cascade disabled");
                None
            }
        }
    } else {
        None
    };

    // ... rest unchanged: load aligner, embedder, qf, then call FacePipeline::with_options() if cascade/RollCorrect enabled ...
}
```

Then replace the `FacePipeline::new(...)` call (line 184) with conditional:

```rust
    let options = pf_ai::FacePipelineOptions {
        enable_face_cascade: face_config.enable_face_cascade,
        cascade_iou_threshold: face_config.cascade_iou_threshold,
        cascade_10g_min_score: face_config.cascade_10g_min_score,
        enable_roll_correct: face_config.enable_roll_correct,
        roll_correct_threshold_deg: face_config.roll_correct_threshold_deg,
    };
    let pipeline = if face_config.enable_face_cascade || face_config.enable_roll_correct {
        FacePipeline::with_options(primary, secondary, aligner, embedder, qf, options)
    } else {
        FacePipeline::new(primary, aligner, embedder, qf)
    };
    Ok(Some(Arc::new(pipeline)))
```

- [ ] **Step 2: Verify compiles**

Run: `cargo check -p photofinder-desktop`
Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
git add apps/desktop/src-tauri/src/desktop_setup.rs
git commit -m "feat(desktop): try_load_face_pipeline wires cascade + RollCorrect"
```

---

## Task 13: Workspace Compile + Test

**Files:** None (verification)

- [ ] **Step 1: Compile whole workspace**

Run: `cargo check --workspace --all-targets`
Expected: 0 errors.

- [ ] **Step 2: Run pf_ai tests**

Run: `cargo test -p pf_ai --lib`
Expected: all tests pass.

- [ ] **Step 3: Run pf_database tests**

Run: `cargo test -p pf_database --lib`
Expected: all tests pass.

- [ ] **Step 4: Build the desktop binary**

Run: `cargo build -p photofinder-desktop`
Expected: 0 errors.

- [ ] **Step 5: Commit (if any cleanup)**

If no changes were needed, skip. Otherwise:

```bash
git add -A
git commit -m "chore: verify workspace builds after cascade + roll_correct migration"
```

---

## Task 14: End-to-End Smoke — 4 Pexels Images via `IndexService`

**Files:**
- (verification only)

- [ ] **Step 1: Write smoke test**

Create `crates/pf_application/tests/smoke_cascade_roll.rs`:

```rust
//! End-to-end smoke: scan 4 pexels portraits via production IndexService.
//! Verifies that face_count matches Phase 28 cache (when flags off) or exceeds it
//! (when cascade enabled).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use pf_ai::face::AdaFaceEmbedder;
use pf_ai::face::SimpleAligner;
use pf_ai::FacePipeline;
use pf_ai::{FaceAligner, FaceEmbedder};
use pf_application::index::IndexService;
use pf_database::Database;
use pf_platform::PhotoId;
use pf_vector::hnsw::HnswIndex;

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn smoke_cascade_enabled_finds_more_faces() -> anyhow::Result<()> {
    // Use in-memory DB + HNSW (test fixture)
    let db = Arc::new(Database::open_in_memory()?);
    let face_index: Arc<dyn pf_vector::VectorIndex> = Arc::new(HnswIndex::new(512, 16));

    let detector_primary = Arc::new(pf_ai::face::ScrfdDetector::load(
        PathBuf::from("/Users/mac/ai-project/photofinder-next-2/models/scrfd_500m_bnkps.onnx"),
    )?);
    let detector_secondary = Arc::new(pf_ai::face::ScrfdDetector::load(
        PathBuf::from("/Users/mac/ai-project/photofinder-next-2/models/scrfd_10g_bnkps.onnx"),
    )?);
    let aligner = Arc::new(SimpleAligner::new());
    let embedder = Arc::new(AdaFaceEmbedder::load(
        PathBuf::from("/Users/mac/ai-project/photofinder-next-2/models/adaface_ir101.onnx"),
    )?);

    let pipeline = Arc::new(FacePipeline::with_options(
        detector_primary,
        Some(detector_secondary),
        aligner,
        embedder,
        pf_ai::QualityFilter::from_config(0.05, 24, 0.0, 90.0),
        pf_ai::FacePipelineOptions {
            enable_face_cascade: true,
            cascade_iou_threshold: 0.30,
            cascade_10g_min_score: 0.05,
            enable_roll_correct: true,
            roll_correct_threshold_deg: 5.0,
        },
    ));

    // ... build IndexService with a photo provider that returns the 4 test images ...
    // ... call index_image 4 times ...
    // ... assert: face_count ≥ baseline (typically 4-8) ...
    Ok(())
}
```

(Adjust this template to match your existing test setup pattern from `e2e_smoke.rs` or similar.)

- [ ] **Step 2: Run the smoke test**

Run: `cargo test -p pf_application --test smoke_cascade_roll -- --ignored --nocapture`
Expected: PASS — all 4 images indexed; face_count ≥ baseline.

- [ ] **Step 3: Verify detector_origin distribution in DB**

Run a follow-up assertion in the smoke test (or a one-off query):

```sql
SELECT detector_origin, COUNT(*) FROM faces GROUP BY detector_origin;
```

Expected: At least some rows with `detector_origin='Primary'`. (Whether `'Secondary'` or `'Both'` rows appear depends on the test images — pexels portraits are mostly frontal so cascade adds little.)

- [ ] **Step 4: Verify face_roll_corrections audit table**

```sql
SELECT COUNT(*) FROM face_roll_corrections;
```

Expected: 0 (pexels portraits are upright, no RollCorrect triggered).

- [ ] **Step 5: Commit (if test was created)**

```bash
git add crates/pf_application/tests/smoke_cascade_roll.rs
git commit -m "test(app): smoke cascade + roll_correct end-to-end"
```

---

## Self-Review Checklist

After completing all tasks, verify:

- [ ] Spec coverage:
  - [ ] `enable_face_cascade` flag → Task 2, 7, 8, 12
  - [ ] `enable_roll_correct` flag → Task 2, 7, 8, 12
  - [ ] RollCorrect replaces baseline → Task 8 (no parallel storage), Task 11 (audit only)
  - [ ] `detector_origin` column → Task 1, 9
  - [ ] `face_roll_corrections` audit → Task 1, 9, 11
  - [ ] Backward compat (default off) → Task 2 (defaults), Task 7 (default Options)
  - [ ] IoU dedup unit test → Task 5
  - [ ] End-to-end smoke → Task 14

- [ ] No placeholders:
  - [ ] Search for "TODO", "TBD", "implement later" in plan — none
  - [ ] Search for "add appropriate error handling" — none (Task 8 has explicit fallbacks)
  - [ ] All code blocks have actual code, not stubs

- [ ] Type consistency:
  - [ ] `FacePipelineOptions` defined in Task 7, used in Tasks 8, 12, 14 — same field names
  - [ ] `DetectorOrigin::as_str()` defined in Task 3, used in Task 9, 11 — returns correct strings
  - [ ] `NewFace.detector_origin` defined in Task 9, used in Task 11 — String type
  - [ ] `insert_roll_correction` signature: `(face_id: i64, roll_deg: f32, baseline_replaced: bool)` consistent in Tasks 9, 11
  - [ ] `process_with_raw` signature: `(image: &ImageData, raw_image: Option<&DynamicImage>)` consistent in Tasks 8, 10

- [ ] Feature flag interaction:
  - [ ] Flags default false → no behavior change when off
  - [ ] Flags can be enabled independently
  - [ ] When cascade enabled but secondary_detector None → falls back gracefully (Task 8)
  - [ ] When RollCorrect triggered but embed fails → falls back to baseline (Task 8)