//! DINO Parallel Pool - Multiple Session Parallel Processing
//!
//! Architecture: Worker-per-Session (1:1), one channel per worker, round-robin dispatch

use crate::visual_scan_v2::dino::{DinoError, DinoExtractor};
use crate::visual_scan_v2::types::Roi;
use image::RgbImage;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

/// Task sent to a worker
#[derive(Clone)]
struct DinoTask {
    roi_index: usize,
    roi: Roi,
    image: Arc<RgbImage>,
}

/// Result returned from a worker
struct DinoResult {
    roi_index: usize,
    embedding: Option<Vec<f32>>,
    preprocess_ms: u64,
    inference_ms: u64,
    postprocess_ms: u64,
    error: Option<String>,
}

/// A worker thread that owns one DINO session
struct WorkerHandle {
    handle: thread::JoinHandle<()>,
}

/// DINO Parallel Pool - manages multiple worker sessions
pub struct DinoPool {
    workers: Vec<WorkerHandle>,
    task_txs: Vec<mpsc::Sender<DinoTask>>,
    result_rx: mpsc::Receiver<DinoResult>,
    session_count: usize,
}

impl DinoPool {
    /// Create a new pool with `session_count` workers (1 worker per session)
    pub fn new(model_path: String, session_count: usize) -> Result<Self, DinoError> {
        assert!(session_count >= 1 && session_count <= 4);

        tracing::info!(
            "[DINO][INIT] session_count={} batch_size=1 intra_threads=1 inter_threads=1 memory_pattern=false",
            session_count
        );

        let mut workers = Vec::with_capacity(session_count);
        let mut task_txs = Vec::with_capacity(session_count);
        let (result_tx, result_rx) = mpsc::channel();

        for i in 0..session_count {
            let model_path = model_path.clone();
            let (task_tx, task_rx): (mpsc::Sender<DinoTask>, mpsc::Receiver<DinoTask>) = mpsc::channel();
            let result_tx_clone = result_tx.clone();

            let handle = thread::spawn(move || {
                // Each worker loads its own session
                match DinoExtractor::load(model_path) {
                    Ok(mut session) => {
                        tracing::info!("[DINO][WORKER] worker={} session={} loaded", i, i);
                        // Worker loop
                        while let Ok(task) = task_rx.recv() {
                            let overall_start = Instant::now();

                            // Use the single-Roi extract method (includes all preprocessing/inference/postprocessing)
                            let extract_result = session.extract(&task.roi, &task.image);

                            let total_elapsed = overall_start.elapsed();

                            match extract_result {
                                Ok(embedding) => {
                                    tracing::debug!(
                                        "[DINO][PERF] roi={} worker={} duration_ms={} OK",
                                        task.roi_index,
                                        i,
                                        total_elapsed.as_millis()
                                    );
                                    result_tx_clone.send(DinoResult {
                                        roi_index: task.roi_index,
                                        embedding: Some(embedding),
                                        preprocess_ms: 0,
                                        inference_ms: total_elapsed.as_millis() as u64,
                                        postprocess_ms: 0,
                                        error: None,
                                    }).ok();
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "[DINO][PERF] roi={} worker={} duration_ms={} FAILED: {}",
                                        task.roi_index,
                                        i,
                                        total_elapsed.as_millis(),
                                        e
                                    );
                                    result_tx_clone.send(DinoResult {
                                        roi_index: task.roi_index,
                                        embedding: None,
                                        preprocess_ms: 0,
                                        inference_ms: total_elapsed.as_millis() as u64,
                                        postprocess_ms: 0,
                                        error: Some(e.to_string()),
                                    }).ok();
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("[DINO][WORKER] worker={} failed to load session: {}", i, e);
                    }
                }
                tracing::info!("[DINO][WORKER] worker={} exiting", i);
            });

            workers.push(WorkerHandle { handle });
            task_txs.push(task_tx);
        }

        Ok(Self {
            workers,
            task_txs,
            result_rx,
            session_count,
        })
    }

    /// Process multiple ROIs in parallel across workers
    /// Returns results in original ROI order
    pub fn extract_batch(&mut self, rois: &[Roi], image: &RgbImage) -> Vec<Result<Vec<f32>, String>> {
        if rois.is_empty() {
            return Vec::new();
        }

        let roi_count = rois.len();
        let start = Instant::now();
        let image_arc = Arc::new(image.clone());

        // Distribute tasks round-robin to workers
        let dispatch_start = Instant::now();
        for (i, roi) in rois.iter().enumerate() {
            let worker_id = i % self.session_count;
            tracing::info!(
                "[DINO][DISPATCH] roi={} worker={} dispatch_ms={}",
                i,
                worker_id,
                dispatch_start.elapsed().as_millis()
            );
            let task = DinoTask {
                roi_index: i,
                roi: roi.clone(),
                image: image_arc.clone(),
            };
            if let Err(e) = self.task_txs[worker_id].send(task) {
                tracing::error!("[DINO][ERROR] failed to send task roi={} to worker={}: {}", i, worker_id, e);
            }
        }
        tracing::info!("[DINO][DISPATCH] all {} ROIs dispatched in {}ms", roi_count, dispatch_start.elapsed().as_millis());

        // Collect results
        let mut results: Vec<(usize, Result<Vec<f32>, String>)> = Vec::with_capacity(roi_count);
        let mut success_count = 0;
        let mut failed_count = 0;
        let mut total_inference_ms: u64 = 0;

        for _ in 0..roi_count {
            match self.result_rx.recv() {
                Ok(result) => {
                    let worker_id = result.roi_index % self.session_count;
                    total_inference_ms += result.inference_ms;

                    if let Some(ref err) = result.error {
                        failed_count += 1;
                        tracing::warn!(
                            "[DINO][RESULT] roi={} worker={} duration_ms={} FAILED: {}",
                            result.roi_index, worker_id, result.inference_ms, err
                        );
                    } else {
                        success_count += 1;
                        tracing::info!(
                            "[DINO][RESULT] roi={} worker={} duration_ms={} OK",
                            result.roi_index, worker_id, result.inference_ms
                        );
                    }
                    results.push((
                        result.roi_index,
                        result.embedding.ok_or_else(|| result.error.unwrap_or_default()),
                    ));
                }
                Err(e) => {
                    tracing::error!("[DINO][ERROR] failed to receive result: {}", e);
                    break;
                }
            }
        }

        // Sort by roi_index to restore original order
        results.sort_by_key(|(idx, _)| *idx);

        let elapsed = start.elapsed();
        let avg_roi_ms = if roi_count > 0 { total_inference_ms / roi_count as u64 } else { 0 };

        tracing::info!(
            "[DINO][PERF][SUMMARY] roi_count={} session_count={} elapsed_ms={} roi_inference_total_ms={} avg_roi_ms={} success={} failed={}",
            roi_count,
            self.session_count,
            elapsed.as_millis(),
            total_inference_ms,
            avg_roi_ms,
            success_count,
            failed_count
        );

        results.into_iter().map(|(_, r)| r).collect()
    }

    /// Get number of sessions in this pool
    pub fn session_count(&self) -> usize {
        self.session_count
    }
}

impl Drop for DinoPool {
    fn drop(&mut self) {
        tracing::info!("[DINO][CLEANUP] dropping pool with {} sessions", self.session_count);
        // Drop task senders to signal workers to exit
        self.task_txs.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_clone() {
        let task = DinoTask {
            roi_index: 0,
            roi: Roi::new(0.0, 0.0, 100.0, 100.0, 0.5),
            image: Arc::new(RgbImage::new(1, 1)),
        };
        let _ = task.clone();
    }
}
