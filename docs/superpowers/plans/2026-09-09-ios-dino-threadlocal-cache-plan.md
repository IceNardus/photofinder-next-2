# iOS DINO Thread-Local Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add thread-local model caching to DinoExtractor to eliminate repeated disk loads and compilations per image.

**Architecture:** Use `thread_local!` static to cache the compiled ONNX model per thread. First call on each thread loads from disk and compiles; subsequent calls reuse cached model.

**Tech Stack:** Rust, tract-onnx, thread_local, RefCell

---

## File Map

| File | Responsibility |
|------|----------------|
| `crates/pf_ai/src/visual/types.rs` | iOS DinoExtractor with thread-local model cache |

---

## Tasks

### Task 1: Add thread_local cache module

**Files:**
- Modify: `crates/pf_ai/src/visual/types.rs`

- [ ] **Step 1: Add thread_local cache module before the iOS impl block**

Find line ~238 where `#[cfg(target_os = "ios")] impl DinoExtractor {` begins and add the cache module above it:

```rust
// Thread-local model cache for iOS - avoids repeated disk loads
#[cfg(target_os = "ios")]
mod dino_thread_cache {
    use std::cell::RefCell;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tract_onnx::prelude::*;

    thread_local! {
        static CACHED_MODEL: RefCell<Option<(PathBuf, Arc<TypedModel>)>> = RefCell::new(None);
    }

    /// Get model from cache, loading if not present
    pub fn get_or_load(path: &PathBuf) -> Result<Arc<TypedModel>, super::DinoError> {
        CACHED_MODEL.with(|cell| {
            let mut cached = cell.borrow_mut();
            if let Some((cached_path, model)) = cached.as_ref() {
                if cached_path == path {
                    return Ok(model.clone());
                }
            }
            // Load and cache
            eprintln!("[iOS DINO] Loading model from disk (thread first call)");
            let model = tract_onnx::onnx()
                .model_for_path(path)
                .map_err(|e| super::DinoError::Load(format!("tract: {}", e)))?;
            let model = Arc::new(model);
            *cached = Some((path.clone(), model.clone()));
            Ok(model)
        })
    }
}
```

- [ ] **Step 2: Verify file compiles**

Run: `cargo check -p pf_ai --target aarch64-apple-ios-sim 2>&1 | grep -E "^error" | head -5`
Expected: No errors

---

### Task 2: Modify extract() to use cache

**Files:**
- Modify: `crates/pf_ai/src/visual/types.rs`

- [ ] **Step 1: Find the extract method and replace model loading**

Find the iOS `extract` method (around line 385). Replace the direct model loading:

Old code (around line 404):
```rust
// Load model and run inference
let model = tract_onnx::onnx()
    .model_for_path(&self.model_path)
    .map_err(|e| DinoError::Load(format!("tract: {}", e)))?;
let runnable = model.into_runnable()
    .map_err(|e| DinoError::Load(format!("into_runnable: {}", e)))?;
```

New code:
```rust
// Use thread-local cached model for inference
let model = dino_thread_cache::get_or_load(&self.model_path)?;
let runnable = model.clone().into_runnable()
    .map_err(|e| DinoError::Load(format!("into_runnable: {}", e)))?;
```

- [ ] **Step 2: Verify file compiles**

Run: `cargo check -p pf_ai --target aarch64-apple-ios-sim 2>&1 | grep -E "^error" | head -5`
Expected: No errors

---

### Task 3: Remove redundant logging

**Files:**
- Modify: `crates/pf_ai/src/visual/types.rs`

- [ ] **Step 1: Remove or reduce verbose logging**

Find and remove or reduce these log lines that fire every ROI:
- Line ~404: Remove `eprintln!("[iOS DINO] Loading model from disk...");` (now in cache)
- Keep only: `eprintln!("[iOS DINO] tract output shape: {:?}", shape);`

- [ ] **Step 2: Verify file compiles**

Run: `cargo check -p pf_ai --target aarch64-apple-ios-sim 2>&1 | grep -E "^error" | head -5`
Expected: No errors

---

### Task 4: Build and copy to iOS project

**Files:**
- None (build and deploy)

- [ ] **Step 1: Build the iOS library**

Run: `cargo build -p photofinder_flutter_ffi --target aarch64-apple-ios-sim 2>&1 | tail -5`
Expected: `Finished` or `error` count = 0

- [ ] **Step 2: Copy to iOS project**

Run: `cp /Users/mac/ai-project/photofinder-next-2/target/aarch64-apple-ios-sim/debug/libphotofinder_flutter_ffi.a /Users/mac/ai-project/photofinder-next-2/apps/flutter_ios/ios/LocalPods/PhotoFinderFFI/lib/ && echo "Copied"`
Expected: `Copied`

---

### Task 5: Build Flutter iOS app

**Files:**
- None (build only)

- [ ] **Step 1: Build Flutter iOS for simulator**

Run: `cd /Users/mac/ai-project/photofinder-next-2/apps/flutter_ios && flutter build ios --simulator --no-codesign 2>&1 | tail -10`
Expected: `✓ Built build/ios/iphonesimulator/Runner.app`

---

### Task 6: Commit changes

**Files:**
- Modify: `crates/pf_ai/src/visual/types.rs`

- [ ] **Step 1: Stage and commit**

Run:
```bash
git add crates/pf_ai/src/visual/types.rs
git commit -m "perf(ios): add thread-local model cache for DINO extraction

- Add dino_thread_cache module with thread_local! static
- Modify extract() to use cached model instead of reloading
- Remove redundant per-ROI logging

Expected ~5-10x speedup for visual feature extraction"
```
Expected: `1 file changed`

---

## Verification

After build, run on iOS simulator and check logs:
- Should see `[iOS DINO] Loading model from disk (thread first call)` only **once** per scan
- Should NOT see it repeated for each ROI

## Spec Coverage Check

| Design Requirement | Task |
|-------------------|------|
| Thread-local cache with `thread_local!` | Task 1 |
| Cache check before loading | Task 1, Task 2 |
| Use cached model in extract() | Task 2 |
| Remove redundant logging | Task 3 |
| Build and verify | Task 4, Task 5 |

All design requirements covered.
