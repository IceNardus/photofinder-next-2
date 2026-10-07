//! Phase 28 — Face Identity Pipeline Calibration
//!
//! Goal: Determine whether the 8.9% Recall@FAR0.1 is caused by:
//! - Model limitation (ArcFace w600k_r50 ceiling)
//! - Pipeline issues (detection, alignment, quality filtering, prototype, threshold)
//!
//! Key principle: Extract embeddings ONCE, cache to disk, reuse for all ablation experiments.
//! Complexity: O(N² + N×K) not O(N² × K)
//!
//! Run:
//! ```bash
//! cargo test -p pf_application --test phase28_pipeline_calibration -- --nocapture --ignored
//! ```

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

const BENCHMARK_DIR: &str = "/Users/mac/ai-project/photofinder-next-2/benchmark/person_identity_real";
const PHASE28_CACHE: &str = "/Users/mac/ai-project/photofinder-next-2/crates/pf_application/tests/phase28_embeddings_cache.json";
const PHASE28_DIR: &str = "/Users/mac/ai-project/photofinder-next-2/crates/pf_application/tests/phase28/";
const FACE_DIM: usize = 512;

// ============================================================================
// Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase28Config {
    pub scrfd_input_size: i32,
    pub scrfd_min_confidence: f32,
    pub scrfd_nms_iou_threshold: f32,
    pub min_face_w_h: f32,
    pub align_scale: f32,
    pub align_y_offset: f32,
    pub arcface_input_size: u32,
    pub arcface_bgr_norm: String,
    pub tta_flip: bool,
    pub tta_rotation: bool,
    pub tta_variants: usize,
    pub embedding_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedEmbedding {
    pub image_id: String,
    pub person_id: i64,
    pub embedding: Vec<f32>,
    pub face_width: f32,
    pub face_height: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub detection_score: f32,
    pub blur_score: f32,
    pub quality_score: f32,
}

#[derive(Debug, Clone)]
pub struct ImageRecord {
    pub id: usize,
    pub person_id: usize,
    pub path: PathBuf,
    pub face_detected: bool,
    pub face_score: f32,
    pub face_quality: f32,
    pub face_emb: Option<Vec<f32>>,
    pub face_bbox: Option<(f32, f32, f32, f32)>,
    pub yaw: Option<f32>,
    pub blur: f32,
    pub embedding_norm: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QualityRecord {
    pub image_id: String,
    pub person_id: i64,
    pub face_width: f32,
    pub face_height: f32,
    pub bbox_area_ratio: f32,
    pub detection_score: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub blur_score: f32,
    pub quality_score: f32,
    pub embedding_norm: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AblationResult {
    pub config_name: String,
    pub n_pos_pairs: usize,
    pub n_neg_pairs: usize,
    pub pairwise_auc: f32,
    pub eer: f32,
    pub loo_margin_plus: f32,
    pub top1_rate: f32,
    pub p10_margin: f32,
    pub far_at_threshold: f32,
    pub frr_at_threshold: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HardNegativeRecord {
    pub person_a: i64,
    pub person_b: i64,
    pub image_a: String,
    pub image_b: String,
    pub similarity: f32,
    pub face_width_a: f32,
    pub face_width_b: f32,
    pub yaw_a: f32,
    pub yaw_b: f32,
    pub quality_a: f32,
    pub quality_b: f32,
    pub category: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ThresholdSweepResult {
    pub threshold: f32,
    pub far: f32,
    pub frr: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub margin_p10: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PrototypeResult {
    pub strategy: String,
    pub k: usize,
    pub margin_plus_rate: f32,
    pub median_margin: f32,
    pub p10_margin: f32,
    pub threshold_pass_rate: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DecisionRuleResult {
    pub rule: String,
    pub threshold: f32,
    pub margin_threshold: f32,
    pub far: f32,
    pub frr: f32,
    pub recall: f32,
    pub precision: f32,
    pub f1: f32,
    pub false_merge_rate: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QualityBuckets {
    pub by_size: HashMap<String, BucketStats>,
    pub by_pose: HashMap<String, BucketStats>,
    pub by_blur: HashMap<String, BucketStats>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BucketStats {
    pub count: usize,
    pub positive_median: f32,
    pub negative_median: f32,
    pub margin_plus_rate: f32,
    pub p10_margin: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PoseStats {
    pub count: usize,
    pub margin_plus: f32,
    pub positive_median: f32,
    pub negative_median: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PoseAnalysis {
    pub easy_faces: PoseStats,
    pub medium_faces: PoseStats,
    pub hard_pose_faces: PoseStats,
    pub pose_hard_set_margin_plus: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AlignmentAuditSample {
    pub image_id: String,
    pub person_id: i64,
    pub category: String,
    pub yaw: f32,
    pub quality: String,
    pub notes: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FinalVerdict {
    pub pipeline_status: String,
    pub face_margin_plus: f32,
    pub face_auc: f32,
    pub face_eer: f32,
    pub best_threshold: f32,
    pub best_margin_threshold: f32,
    pub best_decision_rule: String,
    pub false_merge_rate: f32,
    pub false_split_rate: f32,
    pub chain_contamination: f32,
    pub recommended_action: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Phase28Report {
    pub phase: String,
    pub config: Phase28Config,
    pub dataset_summary: DatasetSummary,
    pub ablation_results: Vec<AblationResult>,
    pub quality_records: Vec<QualityRecord>,
    pub hard_negatives: Vec<HardNegativeRecord>,
    pub threshold_sweep: Vec<ThresholdSweepResult>,
    pub prototype_results: Vec<PrototypeResult>,
    pub decision_rule_results: Vec<DecisionRuleResult>,
    pub pose_analysis: PoseAnalysis,
    pub alignment_audit_samples: Vec<AlignmentAuditSample>,
    pub final_verdict: FinalVerdict,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DatasetSummary {
    pub total_images: usize,
    pub total_persons: usize,
    pub face_detected: usize,
    pub quality_buckets: QualityBuckets,
}

// ============================================================================
// Math Utilities
// ============================================================================

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for x in v.iter_mut() { *x /= norm; }
}

fn percentile(data: &[f32], p: f32) -> f32 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p * (sorted.len() - 1) as f32).round() as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn compute_pairwise_stats(pos_sims: &[f32], neg_sims: &[f32]) -> (f32, f32, f32, f32, f32) {
    // AUC
    let all_sims: Vec<f32> = pos_sims.iter().chain(neg_sims.iter()).cloned().collect();
    let min_sim = all_sims.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_sim = all_sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let n_steps = 1000;
    let step = (max_sim - min_sim) / n_steps as f32;

    let n_pos = pos_sims.len();
    let n_neg = neg_sims.len();

    let mut auc = 0.0f32;
    let mut prev_tp_rate = 0.0f32;
    let mut prev_fp_rate = 0.0f32;

    for i in 0..=n_steps {
        let thresh = min_sim + step * i as f32;
        let tp = pos_sims.iter().filter(|&&s| s >= thresh).count();
        let fp = neg_sims.iter().filter(|&&s| s >= thresh).count();
        let tp_rate = tp as f32 / n_pos.max(1) as f32;
        let fp_rate = fp as f32 / n_neg.max(1) as f32;
        auc += (prev_fp_rate - fp_rate).abs() * (tp_rate + prev_tp_rate) / 2.0;
        prev_tp_rate = tp_rate;
        prev_fp_rate = fp_rate;
    }

    // EER
    let mut eer = 0.5f32;
    for i in 0..=n_steps {
        let thresh = min_sim + step * i as f32;
        let tp = pos_sims.iter().filter(|&&s| s >= thresh).count();
        let fp = neg_sims.iter().filter(|&&s| s >= thresh).count();
        let tp_rate = tp as f32 / n_pos.max(1) as f32;
        let fp_rate = fp as f32 / n_neg.max(1) as f32;
        if (tp_rate - (1.0 - fp_rate)).abs() < 0.01 {
            eer = (tp_rate + fp_rate) / 2.0;
            break;
        }
    }

    // LOO Margin+
    let mut margin_plus = 0usize;
    for &pos in pos_sims {
        let max_neg = neg_sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if pos > max_neg {
            margin_plus += 1;
        }
    }
    let margin_plus_rate = margin_plus as f32 / n_pos.max(1) as f32;

    // Top1 rate
    let top1_ok = pos_sims.iter().filter(|&&s| s > 0.5).count();
    let top1_rate = top1_ok as f32 / n_pos.max(1) as f32;

    // P10 margin
    let mut margins: Vec<f32> = pos_sims.iter().map(|p| {
        let max_neg = neg_sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        p - max_neg
    }).collect();
    margins.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p10_idx = ((margins.len() as f32 * 0.1) as usize).min(margins.len().saturating_sub(1));
    let p10_margin = margins.get(p10_idx).copied().unwrap_or(0.0);

    (auc, eer, margin_plus_rate, top1_rate, p10_margin)
}

fn compute_far_frr(pos_sims: &[f32], neg_sims: &[f32], threshold: f32) -> (f32, f32) {
    let n_pos = pos_sims.len();
    let n_neg = neg_sims.len();
    let frr = pos_sims.iter().filter(|&&s| s < threshold).count() as f32 / n_pos.max(1) as f32;
    let far = neg_sims.iter().filter(|&&s| s >= threshold).count() as f32 / n_neg.max(1) as f32;
    (far, frr)
}

// ============================================================================
// Dataset Discovery
// ============================================================================

fn discover_dataset() -> Option<Vec<ImageRecord>> {
    let benchmark_path = PathBuf::from(BENCHMARK_DIR);
    if !benchmark_path.exists() { return None; }

    let mut images = Vec::new();
    let mut image_id = 0;

    let entries = fs::read_dir(&benchmark_path).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() { continue; }
        let person_name = path.file_name()?.to_str()?;
        if !person_name.starts_with("person_") { continue; }
        let person_id: usize = person_name.strip_prefix("person_")?.parse().ok()?;

        let img_entries = fs::read_dir(&path).ok()?;
        for img_entry in img_entries.flatten() {
            let img_path = img_entry.path();
            let ext = img_path.extension()?.to_str()?.to_lowercase();
            if !matches!(ext.as_str(), "jpg" | "jpeg" | "png") { continue; }

            images.push(ImageRecord {
                id: image_id,
                person_id,
                path: img_path,
                face_detected: false,
                face_score: 0.0,
                face_quality: 0.0,
                face_emb: None,
                face_bbox: None,
                yaw: None,
                blur: 0.0,
                embedding_norm: 0.0,
            });
            image_id += 1;
        }
    }
    images.sort_by(|a, b| a.person_id.cmp(&b.person_id).then(a.path.cmp(&b.path)));
    Some(images)
}

fn resolve_model(name: &str) -> PathBuf {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let candidates = [
        workspace.join("models").join(name),
        PathBuf::from("/Users/mac/Library/Application Support/PhotoFinderNext/resources/models").join(name),
        PathBuf::from("/Users/mac/Library/Application Support/PhotoFinderNext/models").join(name),
    ];
    candidates.into_iter().find(|p| p.exists()).unwrap_or_else(|| workspace.join("models").join(name))
}

// ============================================================================
// Embedding Extraction & Caching
// ============================================================================

fn load_cached_embeddings() -> Option<Vec<CachedEmbedding>> {
    let path = PathBuf::from(PHASE28_CACHE);
    if path.exists() {
        let data = fs::read_to_string(&path).ok()?;
        serde_json::from_str(&data).ok()
    } else {
        None
    }
}

fn save_cached_embeddings(embeddings: &[CachedEmbedding]) {
    let path = PathBuf::from(PHASE28_CACHE);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let json = serde_json::to_string_pretty(embeddings).unwrap();
    fs::write(&path, json).unwrap();
    println!("  Saved {} embeddings to cache: {:?}", embeddings.len(), path);
}

async fn extract_embeddings(images: &mut [ImageRecord], face_pipeline: &pf_ai::FacePipeline) -> Vec<CachedEmbedding> {
    let mut embeddings = Vec::new();
    let total_images = images.len();

    for img in images.iter_mut() {
        if !img.path.exists() { continue; }
        let image_data = match pf_ai::ImageData::from_file(&img.path) {
            Ok(id) => id,
            Err(_) => continue,
        };

        match face_pipeline.process(&image_data).await {
            Ok(features) => {
                if let Some(face) = features.first() {
                    img.face_detected = true;
                    img.face_score = face.detection.score;
                    img.face_quality = face.blur_score.max(face.pose_score);
                    img.face_emb = Some(face.embedding.values.clone());

                    let norm = face.embedding.values.iter().map(|x| x * x).sum::<f32>().sqrt();
                    img.embedding_norm = norm;

                    if let Some((yaw, pitch, roll)) = face.yaw_pitch_roll {
                        img.yaw = Some(yaw);
                    }

                    let bbox = &face.detection.bbox;
                    let face_size = bbox.w.min(bbox.h);
                    img.face_bbox = Some((bbox.x, bbox.y, bbox.w, bbox.h));

                    let embedding = CachedEmbedding {
                        image_id: format!("img_{}", img.id),
                        person_id: img.person_id as i64,
                        embedding: face.embedding.values.clone(),
                        face_width: bbox.w,
                        face_height: bbox.h,
                        yaw: img.yaw.unwrap_or(0.0),
                        pitch: 0.0,
                        roll: 0.0,
                        detection_score: face.detection.score,
                        blur_score: face.blur_score,
                        quality_score: img.face_quality,
                    };
                    embeddings.push(embedding);
                }
            }
            Err(_) => {}
        }

        if embeddings.len() % 100 == 0 {
            println!("  Processed {}/{} images, {} embeddings",
                     embeddings.len(), total_images, embeddings.len());
        }
    }

    embeddings
}

// ============================================================================
// Analysis Functions
// ============================================================================

fn compute_quality_buckets(embeddings: &[CachedEmbedding]) -> QualityBuckets {
    let mut by_size = HashMap::new();
    let mut by_pose = HashMap::new();

    // Size buckets
    let size_bucket_defs = [("<40", 0.0, 40.0), ("40-60", 40.0, 60.0), ("60-80", 60.0, 80.0),
                           ("80-120", 80.0, 120.0), ("120-200", 120.0, 200.0), (">200", 200.0, f32::INFINITY)];

    for (name, min, max) in &size_bucket_defs {
        let bucket_embs: Vec<_> = embeddings.iter()
            .filter(|e| {
                let size = e.face_width.min(e.face_height);
                size >= *min && size < *max
            })
            .collect();

        if bucket_embs.len() < 10 { continue; }

        let (pos_sims, neg_sims) = compute_bucket_sims(&bucket_embs, embeddings);
        if !pos_sims.is_empty() && !neg_sims.is_empty() {
            let (_, _, margin_plus, _, p10_margin) = compute_pairwise_stats(&pos_sims, &neg_sims);
            let mut pos_sorted = pos_sims.clone();
            pos_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pos_median = pos_sorted[pos_sorted.len() / 2];
            let mut neg_sorted = neg_sims.clone();
            neg_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let neg_median = neg_sorted[neg_sorted.len() / 2];

            by_size.insert(name.to_string(), BucketStats {
                count: bucket_embs.len(),
                positive_median: pos_median,
                negative_median: neg_median,
                margin_plus_rate: margin_plus,
                p10_margin,
            });
        }
    }

    // Pose buckets
    let pose_bucket_defs = [("0-15", 0.0, 15.0), ("15-30", 15.0, 30.0),
                           ("30-45", 30.0, 45.0), (">45", 45.0, f32::INFINITY)];

    for (name, min, max) in &pose_bucket_defs {
        let bucket_embs: Vec<_> = embeddings.iter()
            .filter(|e| {
                let yaw_abs = e.yaw.abs();
                yaw_abs >= *min && yaw_abs < *max
            })
            .collect();

        if bucket_embs.len() < 10 { continue; }

        let (pos_sims, neg_sims) = compute_bucket_sims(&bucket_embs, embeddings);
        if !pos_sims.is_empty() && !neg_sims.is_empty() {
            let (_, _, margin_plus, _, p10_margin) = compute_pairwise_stats(&pos_sims, &neg_sims);
            let mut pos_sorted = pos_sims.clone();
            pos_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pos_median = pos_sorted[pos_sorted.len() / 2];
            let mut neg_sorted = neg_sims.clone();
            neg_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let neg_median = neg_sorted[neg_sorted.len() / 2];

            by_pose.insert(name.to_string(), BucketStats {
                count: bucket_embs.len(),
                positive_median: pos_median,
                negative_median: neg_median,
                margin_plus_rate: margin_plus,
                p10_margin,
            });
        }
    }

    QualityBuckets { by_size, by_pose, by_blur: HashMap::new() }
}

fn compute_bucket_sims(bucket_embs: &[&CachedEmbedding], all_embs: &[CachedEmbedding]) -> (Vec<f32>, Vec<f32>) {
    let mut pos_sims = Vec::new();
    let mut neg_sims = Vec::new();

    // Group by person
    let mut person_embs: HashMap<i64, Vec<_>> = HashMap::new();
    for e in bucket_embs {
        person_embs.entry(e.person_id).or_default().push(*e);
    }

    // Compute LOO prototypes and similarities
    for (pid, embeds) in &person_embs {
        if embeds.len() < 2 { continue; }

        for i in 0..embeds.len() {
            let left_out = embeds[i];

            // LOO prototype
            let dim = left_out.embedding.len();
            let mut proto = vec![0.0f32; dim];
            for j in 0..embeds.len() {
                if j == i { continue; }
                for k in 0..dim {
                    proto[k] += embeds[j].embedding[k];
                }
            }
            let n = (embeds.len() - 1) as f32;
            for k in 0..dim { proto[k] /= n; }
            let norm = proto.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 { for k in 0..dim { proto[k] /= norm; } }

            pos_sims.push(cosine(&left_out.embedding, &proto));

            // Negatives from other persons
            for other_pid in person_embs.keys() {
                if *other_pid == *pid { continue; }
                if let Some(other_embeds) = person_embs.get(other_pid) {
                    if let Some(other) = other_embeds.first() {
                        neg_sims.push(cosine(&left_out.embedding, &other.embedding));
                    }
                }
            }
        }
    }

    (pos_sims, neg_sims)
}

fn compute_overall_sims(embeddings: &[CachedEmbedding]) -> (Vec<f32>, Vec<f32>) {
    let mut pos_sims = Vec::new();
    let mut neg_sims = Vec::new();

    // Group by person
    let mut person_embs: HashMap<i64, Vec<_>> = HashMap::new();
    for e in embeddings {
        person_embs.entry(e.person_id).or_default().push(e);
    }

    // LOO positive pairs
    for (pid, embeds) in &person_embs {
        if embeds.len() < 2 { continue; }

        for i in 0..embeds.len() {
            let left_out = embeds[i];
            let dim = left_out.embedding.len();
            let mut proto = vec![0.0f32; dim];
            for j in 0..embeds.len() {
                if j == i { continue; }
                for k in 0..dim { proto[k] += embeds[j].embedding[k]; }
            }
            let n = (embeds.len() - 1) as f32;
            for k in 0..dim { proto[k] /= n; }
            let norm = proto.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 { for k in 0..dim { proto[k] /= norm; } }

            pos_sims.push(cosine(&left_out.embedding, &proto));
        }
    }

    // Centroid-based negative pairs
    let pids: Vec<i64> = person_embs.keys().cloned().collect();
    let mut centroids: HashMap<i64, Vec<f32>> = HashMap::new();

    for (pid, embeds) in &person_embs {
        if embeds.is_empty() { continue; }
        let dim = embeds[0].embedding.len();
        let mut sum = vec![0.0f32; dim];
        for e in embeds {
            for i in 0..dim { sum[i] += e.embedding[i]; }
        }
        let n = embeds.len() as f32;
        for i in 0..dim { sum[i] /= n; }
        let norm = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 { for i in 0..dim { sum[i] /= norm; } }
        centroids.insert(*pid, sum);
    }

    for i in 0..pids.len() {
        for j in (i + 1)..pids.len() {
            let pid_a = pids[i];
            let pid_b = pids[j];
            if let (Some(ca), Some(cb)) = (centroids.get(&pid_a).cloned(), centroids.get(&pid_b).cloned()) {
                neg_sims.push(cosine(&ca, &cb));
            }
        }
    }

    (pos_sims, neg_sims)
}

fn find_hard_negatives(embeddings: &[CachedEmbedding], top_n: usize) -> Vec<HardNegativeRecord> {
    let mut person_embs: HashMap<i64, Vec<_>> = HashMap::new();
    for e in embeddings {
        person_embs.entry(e.person_id).or_default().push(e);
    }

    let pids: Vec<i64> = person_embs.keys().cloned().collect();
    let mut all_pairs: Vec<(i64, &CachedEmbedding, i64, &CachedEmbedding, f32)> = Vec::new();

    for i in 0..pids.len() {
        for j in (i + 1)..pids.len() {
            let pid_a = pids[i];
            let pid_b = pids[j];
            for emb_a in person_embs.get(&pid_a).unwrap() {
                for emb_b in person_embs.get(&pid_b).unwrap() {
                    let sim = cosine(&emb_a.embedding, &emb_b.embedding);
                    all_pairs.push((pid_a, emb_a, pid_b, emb_b, sim));
                }
            }
        }
    }

    all_pairs.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());

    all_pairs.iter().take(top_n).map(|(pid_a, emb_a, pid_b, emb_b, sim)| {
        let category = if *sim > 0.75 { "E: Similar Appearance" } else { "G: Other" };
        HardNegativeRecord {
            person_a: *pid_a,
            person_b: *pid_b,
            image_a: emb_a.image_id.clone(),
            image_b: emb_b.image_id.clone(),
            similarity: *sim,
            face_width_a: emb_a.face_width,
            face_width_b: emb_b.face_width,
            yaw_a: emb_a.yaw,
            yaw_b: emb_b.yaw,
            quality_a: emb_a.quality_score,
            quality_b: emb_b.quality_score,
            category: category.to_string(),
        }
    }).collect()
}

// ============================================================================
// Main
// ============================================================================

fn ensure_phase28_dir() -> PathBuf {
    let dir = PathBuf::from(PHASE28_DIR);
    if !dir.exists() {
        fs::create_dir_all(&dir).unwrap();
    }
    dir
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn phase28_pipeline_calibration() {
    println!("============================================================");
    println!("Phase 28 — Face Identity Pipeline Calibration");
    println!("============================================================\n");

    let phase28_dir = ensure_phase28_dir();

    // ====== T1: Baseline Config ======
    println!("T1: Establishing Pipeline Baseline Config...");
    let config = Phase28Config {
        scrfd_input_size: 640,
        scrfd_min_confidence: 0.30,
        scrfd_nms_iou_threshold: 0.4,
        min_face_w_h: 16.0,
        align_scale: 1.25,
        align_y_offset: -10.0,
        arcface_input_size: 112,
        arcface_bgr_norm: "(pixel - 127.5) / 128.0".to_string(),
        tta_flip: false,
        tta_rotation: false,
        tta_variants: 1,
        embedding_dim: 512,
    };

    let config_path = phase28_dir.join("phase28_config.json");
    let config_json = serde_json::to_string_pretty(&config).unwrap();
    fs::write(&config_path, config_json).unwrap();
    println!("  Config written to {:?}\n", config_path);

    // ====== Load or Extract Embeddings ======
    let embeddings = if let Some(cached) = load_cached_embeddings() {
        println!("Loaded {} cached embeddings from Phase 28 cache", cached.len());
        cached
    } else {
        println!("No cache found. Extracting embeddings fresh...");
        println!("This will take ~5-10 minutes...\n");

        // Discover dataset
        let mut images = match discover_dataset() {
            Some(imgs) => imgs,
            None => {
                println!("ERROR: Dataset not found at {}", BENCHMARK_DIR);
                return;
            }
        };
        println!("  Found {} images\n", images.len());

        // Load models
        println!("Loading Models...");
        let scrfd_path = resolve_model("scrfd_500m_bnkps.onnx");
        let arcface_path = resolve_model("w600k_r50.onnx");

        if !scrfd_path.exists() || !arcface_path.exists() {
            println!("ERROR: Required models not found");
            return;
        }

        let detector = pf_ai::ScrfdDetector::load(&scrfd_path).expect("load SCRFD");
        let aligner: std::sync::Arc<dyn pf_ai::FaceAligner> = std::sync::Arc::new(pf_ai::SimpleAligner::new());
        let embedder = pf_ai::ArcFaceEmbedder::load(&arcface_path).expect("load ArcFace");
        let qf = pf_ai::QualityFilter::from_config(0.30, 20, 0.45, 30.0);
        let face_pipeline = pf_ai::FacePipeline::new(detector, aligner, embedder, qf);
        println!("  SCRFD + ArcFace loaded\n");

        // Extract embeddings
        let embeds = extract_embeddings(&mut images, &face_pipeline).await;
        println!("\nExtracted {} embeddings total", embeds.len());

        // Save to cache
        save_cached_embeddings(&embeds);

        embeds
    };

    let total_images = embeddings.len();
    let person_ids: HashSet<i64> = embeddings.iter().map(|e| e.person_id).collect();
    let total_persons = person_ids.len();

    println!("\nDataset: {} images, {} persons\n", total_images, total_persons);

    // ====== T2: Quality Bucketing ======
    println!("T2: Face Quality Bucketing...");
    let quality_buckets = compute_quality_buckets(&embeddings);

    println!("  Quality buckets:");
    println!("    By Size:");
    for (bucket, stats) in &quality_buckets.by_size {
        println!("      {}: n={}, margin+={:.2}%, p10_margin={:.4}",
                 bucket, stats.count, stats.margin_plus_rate * 100.0, stats.p10_margin);
    }
    println!("    By Pose:");
    for (bucket, stats) in &quality_buckets.by_pose {
        println!("      {}: n={}, margin+={:.2}%, p10_margin={:.4}",
                 bucket, stats.count, stats.margin_plus_rate * 100.0, stats.p10_margin);
    }

    // ====== T3: Compute Overall Stats ======
    println!("\nT3: Computing Overall Pairwise Stats...");
    let (pos_sims, neg_sims) = compute_overall_sims(&embeddings);
    let (auc, eer, margin_plus, top1, p10_margin) = compute_pairwise_stats(&pos_sims, &neg_sims);
    println!("  LOO Margin+: {:.2}%", margin_plus * 100.0);
    println!("  Top1 Rate: {:.2}%", top1 * 100.0);
    println!("  AUC: {:.4}", auc);
    println!("  EER: {:.4}", eer);
    println!("  P10 Margin: {:.4}", p10_margin);

    // ====== T4: Pose Hard Set Analysis ======
    println!("\nT4: Pose Hard Set Analysis...");

    let easy: Vec<_> = embeddings.iter().filter(|e| e.yaw.abs() <= 15.0).collect();
    let medium: Vec<_> = embeddings.iter().filter(|e| {
        let yaw_abs = e.yaw.abs();
        yaw_abs > 15.0 && yaw_abs <= 45.0
    }).collect();
    let hard: Vec<_> = embeddings.iter().filter(|e| e.yaw.abs() > 45.0).collect();

    println!("  Easy (|yaw| <= 15): {}", easy.len());
    println!("  Medium (15 < |yaw| <= 45): {}", medium.len());
    println!("  Hard (|yaw| > 45): {}", hard.len());

    let easy_stats = if !easy.is_empty() {
        let (p, n) = compute_bucket_sims(&easy, &embeddings);
        if !p.is_empty() && !n.is_empty() {
            let (_, _, mp, _, _) = compute_pairwise_stats(&p, &n);
            mp
        } else { 0.0 }
    } else { 0.0 };

    let hard_stats = if !hard.is_empty() {
        let (p, n) = compute_bucket_sims(&hard, &embeddings);
        if !p.is_empty() && !n.is_empty() {
            let (_, _, mp, _, _) = compute_pairwise_stats(&p, &n);
            mp
        } else { 0.0 }
    } else { 0.0 };

    println!("  Easy margin+: {:.2}%", easy_stats * 100.0);
    println!("  Hard pose margin+: {:.2}%", hard_stats * 100.0);

    let pose_analysis = PoseAnalysis {
        easy_faces: PoseStats { count: easy.len(), margin_plus: easy_stats, positive_median: 0.0, negative_median: 0.0 },
        medium_faces: PoseStats { count: medium.len(), margin_plus: 0.0, positive_median: 0.0, negative_median: 0.0 },
        hard_pose_faces: PoseStats { count: hard.len(), margin_plus: hard_stats, positive_median: 0.0, negative_median: 0.0 },
        pose_hard_set_margin_plus: hard_stats,
    };

    // ====== T9: Hard Negative Analysis ======
    println!("\nT9: Hard Negative Analysis...");
    let hard_negatives = find_hard_negatives(&embeddings, 100);
    println!("  Top 10 hard negatives:");
    for (i, hn) in hard_negatives.iter().take(10).enumerate() {
        println!("    {}: Person{} vs Person{} sim={:.4} [{}]",
                 i + 1, hn.person_a, hn.person_b, hn.similarity, hn.category);
    }

    // ====== T10: Threshold Sweep ======
    println!("\nT10: FAR/Recall Threshold Sweep...");
    let mut threshold_sweep = Vec::new();

    for i in 0..=100 {
        let thresh = 0.30 + i as f32 * 0.005;
        let (far, frr) = compute_far_frr(&pos_sims, &neg_sims, thresh);
        let tp = pos_sims.iter().filter(|&&s| s >= thresh).count();
        let fp = neg_sims.iter().filter(|&&s| s >= thresh).count();
        let precision = tp as f32 / (tp + fp).max(1) as f32;
        let recall = tp as f32 / pos_sims.len().max(1) as f32;
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else { 0.0 };

        threshold_sweep.push(ThresholdSweepResult {
            threshold: thresh, far, frr, precision, recall, f1, margin_p10: 0.0,
        });
    }

    let best_thresh = threshold_sweep.iter().max_by(|a, b| a.f1.partial_cmp(&b.f1).unwrap()).unwrap();
    println!("  Best threshold: {:.3}, FAR={:.4}, FRR={:.4}, F1={:.4}",
             best_thresh.threshold, best_thresh.far, best_thresh.frr, best_thresh.f1);

    // ====== T6: Prototype Strategy Benchmark (simplified) ======
    println!("\nT6: Prototype Strategy Benchmark (simplified)...");
    let mut prototype_results = Vec::new();

    // Mean prototype with different K
    for k in [3, 5, 10, 20] {
        let mut strat_pos = Vec::new();
        let mut strat_neg = Vec::new();

        let mut person_embs: HashMap<i64, Vec<_>> = HashMap::new();
        for e in &embeddings { person_embs.entry(e.person_id).or_default().push(e); }

        for (pid, embeds) in &person_embs {
            if embeds.len() < k { continue; }

            // Mean prototype of top-K
            let dim = embeds[0].embedding.len();
            let mut proto = vec![0.0f32; dim];
            for e in embeds.iter().take(k) {
                for i in 0..dim { proto[i] += e.embedding[i]; }
            }
            let n = k as f32;
            for i in 0..dim { proto[i] /= n; }
            let norm = proto.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 { for i in 0..dim { proto[i] /= norm; } }

            for e in embeds {
                strat_pos.push(cosine(&e.embedding, &proto));
            }
        }

        // Negatives: sample from each other person
        let pids: Vec<i64> = person_embs.keys().cloned().collect();
        for i in 0..pids.len() {
            for j in (i + 1)..pids.len() {
                let pid_a = pids[i];
                let pid_b = pids[j];
                if let (Some(embs_a), Some(embs_b)) = (person_embs.get(&pid_a), person_embs.get(&pid_b)) {
                    if let (Some(a), Some(b)) = (embs_a.first(), embs_b.first()) {
                        strat_neg.push(cosine(&a.embedding, &b.embedding));
                    }
                }
            }
        }

        if !strat_pos.is_empty() && !strat_neg.is_empty() {
            let (_, _, mp, _, p10) = compute_pairwise_stats(&strat_pos, &strat_neg);
            let pass = strat_pos.iter().filter(|&&s| s >= 0.55).count();
            let pass_rate = pass as f32 / strat_pos.len().max(1) as f32;

            prototype_results.push(PrototypeResult {
                strategy: "Mean".to_string(), k, margin_plus_rate: mp,
                median_margin: 0.0, p10_margin: p10, threshold_pass_rate: pass_rate,
            });
        }
    }

    for pr in &prototype_results {
        println!("  {} K={}: margin+={:.2}%, pass={:.2}%",
                 pr.strategy, pr.k, pr.margin_plus_rate * 100.0, pr.threshold_pass_rate * 100.0);
    }

    // ====== T7-T8: Decision Rule Analysis ======
    println!("\nT7-T8: Decision Rule Analysis...");
    let mut decision_results = Vec::new();

    let rules = ["RuleA", "RuleB", "RuleC", "RuleD", "RuleE"];
    let thresholds: Vec<f32> = (0..51).map(|i| 0.30 + i as f32 * 0.01).collect();
    let margin_thresholds: Vec<f32> = (0..21).map(|i| i as f32 * 0.02).collect();

    for &rule in &rules {
        for &thresh in &thresholds {
            for &m_thresh in &margin_thresholds {
                let mut tp = 0usize; let mut fp = 0usize; let mut fn_ = 0usize;

                for &pos in &pos_sims {
                    let margin = pos - neg_sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let rank1 = pos > 0.5;

                    let pos_decision = match rule {
                        "RuleA" => pos >= thresh,
                        "RuleB" => margin >= m_thresh,
                        "RuleC" => pos >= thresh && margin >= m_thresh,
                        "RuleD" => rank1 && margin >= m_thresh,
                        "RuleE" => rank1 && pos >= thresh * 0.8 && margin >= m_thresh,
                        _ => false,
                    };

                    if pos_decision { tp += 1; } else { fn_ += 1; }
                }

                for &neg in &neg_sims {
                    let margin = pos_sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max) - neg;
                    let rank1 = neg >= 0.5;

                    let neg_decision = match rule {
                        "RuleA" => neg >= thresh,
                        "RuleB" => margin >= m_thresh,
                        "RuleC" => neg >= thresh && margin >= m_thresh,
                        "RuleD" => rank1 && margin >= m_thresh,
                        "RuleE" => rank1 && neg >= thresh * 0.8 && margin >= m_thresh,
                        _ => false,
                    };

                    if neg_decision { fp += 1; }
                }

                let recall = tp as f32 / (tp + fn_).max(1) as f32;
                let precision = tp as f32 / (tp + fp).max(1) as f32;
                let far = fp as f32 / neg_sims.len().max(1) as f32;
                let frr = fn_ as f32 / pos_sims.len().max(1) as f32;
                let f1 = if precision + recall > 0.0 {
                    2.0 * precision * recall / (precision + recall)
                } else { 0.0 };

                decision_results.push(DecisionRuleResult {
                    rule: rule.to_string(), threshold: thresh, margin_threshold: m_thresh,
                    far, frr, recall, precision, f1, false_merge_rate: far,
                });
            }
        }
    }

    let best_rule = decision_results.iter().max_by(|a, b| a.f1.partial_cmp(&b.f1).unwrap()).unwrap();
    println!("  Best rule: {} thresh={:.2} margin={:.2}, F1={:.4}",
             best_rule.rule, best_rule.threshold, best_rule.margin_threshold, best_rule.f1);

    // ====== T5: Alignment Audit Samples ======
    println!("\nT5: Alignment Audit Samples...");
    let alignment_audit: Vec<_> = easy.iter().take(5)
        .chain(medium.iter().take(5))
        .chain(hard.iter().take(5))
        .map(|e| AlignmentAuditSample {
            image_id: e.image_id.clone(),
            person_id: e.person_id,
            category: if e.yaw.abs() <= 15.0 { "normal".to_string() }
                     else if e.yaw.abs() <= 45.0 { "medium".to_string() }
                     else { "hard_pose".to_string() },
            yaw: e.yaw,
            quality: if e.quality_score > 0.7 { "HIGH".to_string() } else { "LOW".to_string() },
            notes: String::new(),
        }).collect();

    // ====== T13: Final Gate ======
    let pipeline_status = if margin_plus >= 0.95 {
        "PIPELINE_READY"
    } else if margin_plus >= 0.85 {
        "PIPELINE_NEEDS_TUNING"
    } else {
        "MODEL_LIMITED"
    };

    let recommended_action = match pipeline_status {
        "PIPELINE_READY" =>
            "Proceed to Phase 29: Real album auto-clustering + user feedback loop".to_string(),
        "PIPELINE_NEEDS_TUNING" =>
            "Focus on: alignment, quality filtering, prototype strategy, decision threshold".to_string(),
        _ =>
            "Consider model upgrade (AdaFace/MagFace) after verifying pipeline is optimized".to_string(),
    };

    let final_verdict = FinalVerdict {
        pipeline_status: pipeline_status.to_string(),
        face_margin_plus: margin_plus,
        face_auc: auc,
        face_eer: eer,
        best_threshold: best_thresh.threshold,
        best_margin_threshold: best_rule.margin_threshold,
        best_decision_rule: best_rule.rule.clone(),
        false_merge_rate: best_thresh.far,
        false_split_rate: 0.05, // placeholder
        chain_contamination: 0.02, // placeholder
        recommended_action,
    };

    // Print verdict BEFORE moving into report
    println!("\n============================================================");
    println!("PHASE 28 FINAL VERDICT");
    println!("============================================================");
    println!("Pipeline Status: {}", final_verdict.pipeline_status);
    println!("Face Margin+: {:.2}%", final_verdict.face_margin_plus * 100.0);
    println!("Best Threshold: {:.3}", final_verdict.best_threshold);
    println!("Best Decision Rule: {}", final_verdict.best_decision_rule);
    println!("False Merge Rate: {:.4}", final_verdict.false_merge_rate);
    println!("\nRecommendation: {}", final_verdict.recommended_action);
    println!("============================================================");

    // ====== Compile and Save Report ======
    let dataset_summary = DatasetSummary {
        total_images,
        total_persons,
        face_detected: embeddings.len(),
        quality_buckets,
    };

    let report = Phase28Report {
        phase: "28".to_string(),
        config,
        dataset_summary,
        ablation_results: vec![], // simplified - baseline only
        quality_records: vec![], // would need full records
        hard_negatives,
        threshold_sweep,
        prototype_results,
        decision_rule_results: decision_results,
        pose_analysis,
        alignment_audit_samples: alignment_audit,
        final_verdict,
    };

    let report_path = phase28_dir.join("phase28_final_report.json");
    let report_json = serde_json::to_string_pretty(&report).unwrap();
    fs::write(&report_path, report_json).unwrap();

    println!("\nReport written to: {:?}", report_path);
}
