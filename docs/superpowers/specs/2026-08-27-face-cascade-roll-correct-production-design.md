# Face Cascade + RollCorrect Production Migration

**Date:** 2026-08-27
**Status:** Approved (user pre-approved design in chat)
**Scope:** Migrate `quick_face_check.rs` cascade + RollCorrect + pose tracking into production face scan pipeline, gated by feature flags.

## Goal

Bring the multi-pose (SCRFD 500M + 10G cascade) and RollCorrect (tilt correction) pipelines that we validated in `quick_face_check.rs` into the actual production scan path (`IndexService::index_image → FacePipeline::process`), behind two independent feature flags.

This closes the gap where big-yaw profile faces are missed by SCRFD 500M and tilted faces produce worse embeddings due to in-plane rotation.

## Non-Goals

- Full reindex of existing indexed faces (deferred; only new scans get cascade/RollCorrect)
- Production-grade face detection with RetinaFace/YuNet (only SCRFD 10G, which is already downloaded)
- Migration of `quick_face_check.rs` ablation report (it stays as a test-only benchmark)
- Migration of PoseBucket persistence (computed on-demand from yaw/pitch/roll instead)

## Architecture

```
                       ┌────────────────────────────────────┐
                       │   FaceConfig (pf_config)           │
                       │   enable_face_cascade   (bool, F)  │
                       │   enable_roll_correct   (bool, F)  │
                       │   cascade_iou_threshold (0.30)     │
                       │   cascade_10g_min_score (0.05)     │
                       │   roll_correct_threshold_deg (5.0) │
                       │   scrfd_10g_model_path   (Option)  │
                       └────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────┐
│  FacePipeline                                            │
│  primary_detector (500M, always)                         │
│  secondary_detector (10G, only when cascade enabled)     │
│  ┌─ detect (single or cascade)                           │
│  ├─ merge_detections_cascade(IoU≥0.30) [if cascade]      │
│  ├─ for each detection:                                  │
│  │  ├─ crop + align + embed (baseline)                   │
│  │  ├─ if roll_correct && |roll|>5°:                     │
│  │  │  ├─ rotate raw_image by -roll                     │
│  │  │  ├─ rotate landmarks around center                │
│  │  │  ├─ re-align + re-embed (RollCorrect)             │
│  │  │  └─ replace baseline embedding with RollCorrect   │
│  │  └─ return FaceFeature {                              │
│  │     detector_origin, roll, embedding, replaced }    │
│  └──────────────────────────────────────────────────────-┘
└──────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────┐
│  IndexService::index_faces                               │
│  ├─ pull detector_origin / yaw / pitch from feature     │
│  ├─ if roll_correct_replaced:                            │
│  │  ├─ insert face_roll_corrections row (audit)         │
│  │  └─ HNSW entry = RollCorrect embedding              │
│  ├─ else: HNSW entry = baseline embedding               │
│  └─ insert face row with origin / yaw / pitch columns   │
└──────────────────────────────────────────────────────────┘
```

### Key API Changes

- `FacePipeline::process(image)` stays unchanged (backward compatible when flags off)
- **New**: `FacePipeline::process_with_raw(image, raw_image: Option<DynamicImage>)` — required when cascade or RollCorrect enabled, since image rotation needs the original pixels
- `IndexService` always converts bytes → `DynamicImage` (in-memory, <10ms/image, negligible cost)

## Files Modified / Created

| File | Action | Change |
|------|--------|--------|
| `crates/pf_config/src/config.rs` | Modify | `FaceConfig` +6 fields (see below) |
| `crates/pf_ai/src/face/traits.rs` | Modify | `FacePipeline` adds secondary + RollCorrect paths; `FaceFeature` adds fields |
| `crates/pf_ai/src/face/mod.rs` | Modify | Add `DetectorOrigin` enum |
| `crates/pf_application/src/index.rs` | Modify | `index_faces` persists new fields; writes `face_roll_corrections` audit row |
| `crates/pf_database/src/repositories/faces.rs` | Modify | `NewFace` adds `detector_origin`, `yaw`, `pitch`; `insert()` writes them |
| `crates/pf_database/src/migration.rs` | Modify | Register migration 019 |
| `crates/pf_database/migrations/019_face_cascade_roll_correct.sql` | **Create** | Schema migration (below) |
| `apps/desktop/src-tauri/src/desktop_setup.rs` | Modify | Load SCRFD 10G if cascade enabled; build enhanced pipeline |

## Schema Migration (019)

```sql
-- 019_face_cascade_roll_correct.sql
--
-- Face Cascade + RollCorrect production migration.
-- Adds detector_origin, yaw/pitch split, and face_roll_corrections audit table.

-- detector_origin: enum-as-text. Existing rows get 'Primary' (backward compat).
ALTER TABLE faces ADD COLUMN detector_origin TEXT NOT NULL DEFAULT 'Primary';

-- Split yaw_pitch_roll JSON into queryable columns for trend analysis.
-- Existing rows: backfill from JSON (best-effort; NULL-safe).
ALTER TABLE faces ADD COLUMN yaw REAL;
ALTER TABLE faces ADD COLUMN pitch REAL;

UPDATE faces
SET yaw = json_extract(yaw_pitch_roll, '$[0]'),
    pitch = json_extract(yaw_pitch_roll, '$[1]')
WHERE yaw_pitch_roll IS NOT NULL
  AND json_extract(yaw_pitch_roll, '$[0]') IS NOT NULL;

-- RollCorrect audit table. Records face_id where baseline was replaced
-- by RollCorrect embedding due to |roll|>threshold.
CREATE TABLE face_roll_corrections (
    face_id          INTEGER PRIMARY KEY,
    roll_deg         REAL    NOT NULL,
    baseline_replaced INTEGER NOT NULL DEFAULT 1,
    created_at       INTEGER NOT NULL,
    FOREIGN KEY (face_id) REFERENCES faces(id) ON DELETE CASCADE
);

CREATE INDEX idx_face_rc_created ON face_roll_corrections(created_at);
```

**Backward compat:** Migration is idempotent (ADD COLUMN with default; new table). Old DBs migrate cleanly; new DBs (test setup) include the columns from the start.

## Feature Flag Semantics

| Flag | Default | When enabled |
|------|---------|--------------|
| `enable_face_cascade` | `false` | Load SCRFD 10G additionally; IoU-merge detections; populate `detector_origin` |
| `enable_roll_correct` | `false` | For `\|roll\|>threshold_deg` faces, replace baseline embedding with RollCorrect; write audit row |

Flags are independent:
- Both off → exact current behavior (no code path changes)
- Only cascade → more detections found, embeddings still baseline
- Only RollCorrect → single detector, but tilted faces use RollCorrect
- Both on → full functionality

**Backstop:** When flags are off, the existing `FacePipeline::process(image)` API path is taken — `process_with_raw` is not called.

## Backward Compatibility & Reindex

**No full reindex required.**
- Existing faces: `detector_origin='Primary'` (default), `yaw/pitch` backfilled from JSON
- `face_roll_corrections` empty for old data
- HNSW old embeddings unchanged; search works as before

**Optional upgrade** (deferred, out of scope):
- Future CLI `--upgrade-faces` could re-detect existing images with new pipeline
- Existing indexed images do NOT auto re-detect (would invalidate cluster / person_id)

## Data Flow

### Index time

1. `IndexService::index_image()` reads bytes via `photo_provider.get_image()`
2. Calls `face_pipeline.process_with_raw(&img, Some(&dyn_img))` instead of `face_pipeline.process(&img)`
3. Each `FaceFeature` carries: `detector_origin`, `yaw`, `pitch` (split out), `roll`, `roll_correct_replaced: bool`
4. `index_faces` constructs `NewFace` with new fields populated
5. If `roll_correct_replaced`: insert into `face_roll_corrections`, use RollCorrect embedding as HNSW entry
6. Otherwise: use baseline embedding as HNSW entry (unchanged behavior)

### Search time

No changes. `FaceFeature` is the input to HNSW search; new fields are diagnostic only.

## Testing Strategy

### 1. Unit tests (new)

- `merge_detections_cascade()` IoU dedup boundaries:
  - IoU = 0.31 → matched (above threshold)
  - IoU = 0.29 → not matched (below threshold)
  - Partial overlap (sharing corners) → matched
  - No overlap → not matched
- Higher-score detection kept when both match

### 2. Regression on `quick_face_check` images

Run the 4 pexels jaqor+peter portraits through `IndexService::index_image()`:
- With flags OFF: same `face_count` as today
- With cascade ON: more faces found on profile shots
- Jaqor → P_jaqor test: still PASSES (no false merge from cascade)
- Image6-9 vs P_jaqor: still <0.55 (no false merge)

### 3. Shadow comparison (Phase 30)

Use existing `shadow_records` to run old vs new paths side-by-side:
- `detector_origin='Primary'` paths should match baseline exactly (no behavior change for 500M-only faces)
- `detector_origin='Secondary'` (10G recovery) should add NEW faces only, never change existing matches
- `face_roll_corrections` growth rate tracks RollCorrect adoption

### 4. Migration upgrade test

- Pre-migration DB (synthetic) → run migration 019 → verify columns present
- Existing `yaw_pitch_roll` JSON → verify `yaw`/`pitch` backfill correct
- `face_roll_corrections` empty for old data

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| SCRFD 10G false positives on small/frontal faces | `cascade_10g_min_score=0.05` + IoU dedup; 10G only contributes when 500M has no IoU match |
| RollCorrect breaks same-face matching | Trigger threshold is conservative (5°); upright faces (the majority) untouched; baseline discarded only when RollCorrect runs |
| `ImageData → DynamicImage` conversion cost | In-memory decode, <10ms/image; `IndexService` already has bytes |
| Migration 019 fails on production DB | Idempotent ADD COLUMN with default; new table has no external dependencies |
| vector_id collision from same face detected by both detectors | Cascade dedup produces ONE `FaceFeature` per face; `derive_face_vector_id` hash unchanged |

## Open Questions

None. Feature flags gate all new behavior; default off means zero risk on rollout.

## Verification Plan

1. Compile: `cargo check -p pf_application -p pf_ai -p pf_config -p pf_database -p photofinder-desktop`
2. Unit tests: `cargo test -p pf_ai --lib merge_detections_cascade`
3. End-to-end: re-run `quick_face_check.rs` on 4 pexels images; compare counts with previous baseline
4. Migration test: create synthetic old DB, apply migration 019, verify schema
5. Shadow verification: enable cascade in dev, run scan, check `shadow_records` for parity