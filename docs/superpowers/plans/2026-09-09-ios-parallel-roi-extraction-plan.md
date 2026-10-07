# iOS Parallel ROI Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize ROI extraction in `process_visual_sync()` using `std::thread::scope`

**Architecture:** Use scoped threads to parallelize DINO inference across ROIs. Each thread uses thread-local model cache (already implemented).

**Tech Stack:** Rust, std::thread::scope, Arc<RgbImage>

---

## File Map

| File | Responsibility |
|------|----------------|
| `apps/flutter_ios/rust/src/ffi.rs` | Parallel ROI extraction in process_visual_sync() |

---

## Tasks

### Task 1: Modify process_visual_sync for parallel ROI extraction

**Files:**
- Modify: `apps/flutter_ios/rust/src/ffi.rs`

- [ ] **Step 1: Find the ROI extraction loop**

Find around line 867-877 in `process_visual_sync()`:
```rust
// Extract DINO features for each ROI
for roi in &rois {
    match dino.extract(roi, rgb_image) {
        Ok(embedding) => {
            all_roi_features.push(RoiFeature::new(*roi, embedding));
        }
        Err(e) => {
            eprintln!("[VISUAL] DINO extraction failed for ROI: {}", e);
        }
    }
}
```

- [ ] **Step 2: Replace with parallel version**

Replace the sequential ROI extraction with parallel version:

```rust
// Parallel ROI extraction using thread::scope
// Each thread uses thread-local model cache (dino_thread_cache)
eprintln!("[VISUAL] Starting parallel ROI extraction with {} ROIs", rois.len());

// Wrap image in Arc for shared access
let rgb_arc = Arc::new(rgb_image.clone());

let all_embeddings: Vec<Result<(VisualRoi, Vec<f32>), DinoError>> =
    std::thread::scope(|s| {
        let handles: Vec<_> = rois.iter().map(|roi| {
            let roi = *roi;
            let rgb = Arc::clone(&rgb_arc);
            s.spawn(move || {
                dino.extract(&roi, &rgb)
                    .map(|embedding| (roi, embedding))
            })
        }).collect();

        // Collect results
        handles.into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });

eprintln!("[VISUAL] Parallel extraction complete, collecting results");

// Filter successful extractions and collect features
for result in all_embeddings {
    match result {
        Ok((roi, embedding)) => all_roi_features.push(RoiFeature::new(roi, embedding)),
        Err(e) => eprintln!("[VISUAL] ROI extraction failed: {}", e),
    }
}
```

- [ ] **Step 3: Verify file compiles**

Run: `cargo check -p photofinder_flutter_ffi --target aarch64-apple-ios-sim 2>&1 | grep -E "^error" | head -5`
Expected: No errors

---

### Task 2: Build and verify

- None

- [ ] **Step 1: Build the library**

Run: `cargo build -p photofinder_flutter_ffi --target aarch64-apple-ios-sim 2>&1 | tail -5`
Expected: `Finished`

- [ ] **Step 2: Copy to iOS project**

Run: `cp /Users/mac/ai-project/photofinder-next-2/target/aarch64-apple-ios-sim/debug/libphotofinder_flutter_ffi.a /Users/mac/ai-project/photofinder-next-2/apps/flutter_ios/ios/LocalPods/PhotoFinderFFI/lib/ && echo "Copied"`

- [ ] **Step 3: Build Flutter iOS**

Run: `cd /Users/mac/ai-project/photofinder-next-2/apps/flutter_ios && flutter build ios --simulator --no-codesign 2>&1 | tail -5`
Expected: `✓ Built build/ios/iphonesimulator/Runner.app`

---

## Verification

After build, run on iOS simulator and check logs:
- Should see `[iOS DINO] Loading model from disk (thread first call)` once per thread
- Multiple threads will load their own model copies
- Expect ~2-4x speedup depending on CPU cores

## Spec Coverage Check

| Design Requirement | Task |
|-------------------|------|
| Parallel ROI extraction with thread::scope | Task 1 |
| Arc<RgbImage> for shared image | Task 1 |
| Thread-local model cache (existing) | Already implemented |
| Build and verify | Task 2 |

All design requirements covered.
