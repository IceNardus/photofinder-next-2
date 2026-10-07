# iOS Parallel ROI Extraction Design

## Status: Draft

## Context

After implementing thread-local model caching (dino_thread_cache), the next optimization is parallel ROI extraction in `process_visual_sync()`.

**Current bottleneck**: ROIs are processed sequentially in a for loop (~16-32 ROIs per image, each requiring DINO inference).

## Problem

**Current code** (`apps/flutter_ios/rust/src/ffi.rs` ~line 868):
```rust
for roi in &rois {
    match dino.extract(roi, rgb_image) {
        Ok(embedding) => all_roi_features.push(RoiFeature::new(*roi, embedding)),
        Err(e) => eprintln!("[VISUAL] DINO extraction failed for ROI: {}", e),
    }
}
```

**Issues**:
- Sequential processing: each ROI waits for the previous one
- CPU cores underutilized on multi-core iOS devices
- Thread-local cache helps but doesn't parallelize

## Solution: Parallel ROI Extraction with thread::scope

### Architecture

```
process_visual_sync()
├── Generate ROIs (sequential - fast)
├── Parallel ROI extraction
│   ├── Thread 1: dino.extract(roi[0]) → embedding[0]
│   ├── Thread 2: dino.extract(roi[1]) → embedding[1]
│   └── Thread N: dino.extract(roi[N]) → embedding[N]
├── Collect results
├── FPS selection (sequential - fast)
└── Write to DB (sequential - I/O bound)
```

### Key Design Decisions

1. **Thread pool via `std::thread::scope`**:
   - Scoped threads - no lifetime issues
   - All threads complete before function returns
   - Safe borrowing of `dino` and `rgb_image`

2. **Thread-local model cache** (already implemented):
   - Each thread caches its own model copy
   - No cross-thread synchronization needed
   - First call per thread loads model, subsequent calls use cache

3. **`Arc<RgbImage>` for shared image**:
   - Image data is read-only, safe to share
   - Avoid cloning large data

4. **Results collected via `JoinHandle`**:
   - `spawn` returns `JoinHandle<Result<...>>`
   - Collect all handles, then collect results
   - Maintain ROI order for FPS selection

### Implementation

```rust
// Parallel ROI extraction using thread::scope
let all_embeddings: Vec<Result<(VisualRoi, Vec<f32>), DinoError>> =
    std::thread::scope(|s| {
        let handles: Vec<_> = rois.iter().map(|roi| {
            // Clone arc for each thread
            let roi = *roi;
            let rgb = Arc::clone(&rgb_image);
            s.spawn(move || {
                dino.extract(&roi, &rgb)
                    .map(|embedding| (roi, embedding))
            })
        }).collect();

        handles.into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });

// Filter successful extractions
for result in all_embeddings {
    match result {
        Ok((roi, embedding)) => all_roi_features.push(RoiFeature::new(roi, embedding)),
        Err(e) => eprintln!("[VISUAL] ROI extraction failed: {}", e),
    }
}
```

### Thread Safety Analysis

| Component | Thread Safe? | Notes |
|-----------|-------------|-------|
| `DinoExtractor::extract()` | Yes | Uses thread-local cache, no shared state |
| `RgbImage` (image crate) | Yes | Read-only access, no mutation |
| `Arc<RgbImage>` | Yes | Atomic reference counting |
| `dino_thread_cache` | Yes | thread_local, per-thread storage |

### Files to Modify

| File | Change |
|------|--------|
| `apps/flutter_ios/rust/src/ffi.rs` | Modify `process_visual_sync()` to use parallel ROI extraction |

### Performance Expectations

| Scenario | Before | After |
|---------|--------|-------|
| 16 ROIs, 4 cores | ~4s (sequential) | ~1.5s (4x parallel) |
| 32 ROIs, 4 cores | ~8s (sequential) | ~2.5s (4x parallel) |

### Error Handling

- Each ROI extraction is independent
- Failed extractions are logged but don't stop processing
- If all extractions fail, return 0 (no features)
- Thread panic: `spawn` result includes panic payload

## Scope

This is a focused optimization. No changes to:
- Database schema
- FFI interface
- Flutter-side code
- ROI generation logic
- FPS selection algorithm
- Thread-local cache (already implemented)

## Alternatives Considered

**Approach A (Rayon)**: Use `rayon` crate for data parallelism
- More ergonomic `par_iter()`
- Requires adding rayon dependency
- Overkill for this use case

**Approach C (Tokio spawn)**: Use async threads
- Already using tokio runtime elsewhere
- More complex lifetime management
- Not needed for CPU-bound image processing
