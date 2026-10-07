//! Phase 27 — Face Embedding Model Capability Benchmark
//!
//! Phase 26 revealed:
//!   Face LOO Margin+ = 90.95% (embeddings separate correctly)
//!   But Recall@FAR0.1 = 8.9% (can't safely auto-merge)
//!
//! Phase 27's goal: Determine if failure is from Pipeline, Prototype, or Model ceiling.
//!
//! Run:
//! ```bash
//! cargo test -p pf_application --test phase27_model_capability -- --nocapture --ignored
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use serde::{Serialize, Deserialize};

// ============================================================================
// Constants
// ============================================================================

const BENCHMARK_DIR: &str = "/Users/mac/ai-project/photofinder-next-2/benchmark/person_identity_real";
const CACHE_DIR: &str = "/Users/mac/ai-project/photofinder-next-2/benchmark/person_identity_v3/cache";
const FACE_DIM: usize = 512;

// ============================================================================
// Report Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase27Report {
    pub phase: String,
    pub dataset_summary: DatasetSummaryV3,
    pub pipeline_audit: PipelineAuditReport,
    pub quality_bucket_analysis: QualityBucketReport,
    pub hard_negative_analysis: HardNegativeReport,
    pub prototype_quality_experiment: PrototypeQualityReport,
    pub model_ceiling: ModelCeilingVerdict,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSummaryV3 {
    pub total_images: usize,
    pub total_persons: usize,
    pub face_detected: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineAuditReport {
    pub face_size_distribution: HashMap<String, usize>,
    pub pose_distribution: HashMap<String, usize>,
    pub quality_distribution: HashMap<String, usize>,
    pub detection_score_stats: ScoreStats,
    pub blur_score_stats: ScoreStats,
    pub embedding_norm_stats: ScoreStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreStats {
    pub min: f64, pub p10: f64, pub p50: f64, pub p90: f64, pub max: f64, pub mean: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityBucketReport {
    pub by_face_size: Vec<QualityBucket>,
    pub by_pose: Vec<QualityBucket>,
    pub by_detection_confidence: Vec<QualityBucket>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityBucket {
    pub bucket_name: String,
    pub count: usize,
    pub positive_median: f64,
    pub negative_median: f64,
    pub margin_plus_rate: f64,
    pub margin_p10: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardNegativeReport {
    pub top_hard_negatives: Vec<HardNegativeCase>,
    pub categories: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardNegativeCase {
    pub person_a: usize,
    pub image_a: usize,
    pub person_b: usize,
    pub image_b: usize,
    pub similarity: f64,
    pub face_size_a: f32,
    pub face_size_b: f32,
    pub yaw_a: Option<f32>,
    pub yaw_b: Option<f32>,
    pub blur_a: f32,
    pub blur_b: f32,
    pub likely_category: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrototypeQualityReport {
    pub results: Vec<PrototypeQualityResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrototypeQualityResult {
    pub filter: String,
    pub margin_plus_rate: f64,
    pub median_margin: f64,
    pub p10_margin: f64,
    pub threshold_pass_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCeilingVerdict {
    pub verdict: String,
    pub confidence: String,
    pub recall_at_far01: f64,
    pub margin_plus_rate: f64,
    pub evidence: Vec<String>,
    pub recommendation: String,
}

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone)]
struct ImageRecordV3 {
    id: usize,
    person_id: usize,
    path: PathBuf,
    face_detected: bool,
    face_score: f32,
    face_quality: f32,
    face_emb: Option<Vec<f32>>,
    face_bbox: Option<(f32, f32, f32, f32)>,
    yaw: Option<f32>,
    blur: f32,
    embedding_norm: f32,
}

#[derive(Debug, Clone)]
struct QueryGallerySplit {
    gallery: Vec<usize>,
    query: Vec<usize>,
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

fn compute_score_stats(scores: &[f32]) -> ScoreStats {
    if scores.is_empty() {
        return ScoreStats { min: 0.0, p10: 0.0, p50: 0.0, p90: 0.0, max: 0.0, mean: 0.0 };
    }
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum: f64 = scores.iter().map(|&x| x as f64).sum();
    ScoreStats {
        min: sorted.first().copied().unwrap_or(0.0) as f64,
        p10: percentile(&sorted, 0.10) as f64,
        p50: percentile(&sorted, 0.50) as f64,
        p90: percentile(&sorted, 0.90) as f64,
        max: sorted.last().copied().unwrap_or(0.0) as f64,
        mean: sum / scores.len() as f64,
    }
}

// ============================================================================
// Dataset Discovery
// ============================================================================

fn discover_dataset() -> Option<Vec<ImageRecordV3>> {
    let benchmark_path = PathBuf::from(BENCHMARK_DIR);
    if !benchmark_path.exists() { return None; }

    fs::create_dir_all(CACHE_DIR).ok();

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

            images.push(ImageRecordV3 {
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
// Query/Gallery Split
// ============================================================================

fn create_query_gallery_split(images: &[ImageRecordV3], seed: u64) -> HashMap<usize, QueryGallerySplit> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut person_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for img in images {
        person_groups.entry(img.person_id).or_default().push(img.id);
    }

    let mut splits = HashMap::new();
    for (person_id, img_count) in person_groups.iter().map(|(k, v)| (k, v.len())) {
        let gallery_size = (img_count as f32 * 0.6) as usize;
        let remaining: Vec<usize> = person_groups.get(person_id).cloned().unwrap_or_default();

        let mut hash_input = format!("{}:{}", person_id, seed);
        let mut hasher = DefaultHasher::new();
        hash_input.hash(&mut hasher);
        let hash_val = hasher.finish();
        let mut rng_state = hash_val;

        let mut gallery = Vec::with_capacity(gallery_size);
        let mut remaining_copy = remaining.clone();

        for _ in 0..gallery_size {
            if remaining_copy.is_empty() { break; }
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let idx = (rng_state as usize) % remaining_copy.len();
            gallery.push(remaining_copy.remove(idx));
        }

        let query: Vec<usize> = remaining_copy;
        splits.insert(*person_id, QueryGallerySplit { gallery, query });
    }
    splits
}

// ============================================================================
// T27.1: Pipeline Audit
// ============================================================================

async fn audit_face_pipeline_async(
    images: &mut [ImageRecordV3],
    face_pipeline: &pf_ai::FacePipeline,
) -> PipelineAuditReport {
    let mut face_size_dist: HashMap<String, usize> = HashMap::new();
    let mut pose_dist: HashMap<String, usize> = HashMap::new();
    let mut quality_dist: HashMap<String, usize> = HashMap::new();
    let mut all_detection_scores = Vec::new();
    let mut all_blur_scores = Vec::new();
    let mut all_embedding_norms = Vec::new();

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
                    all_detection_scores.push(face.detection.score);
                    all_blur_scores.push(face.blur_score);

                    if let Some((yaw, _, _)) = face.yaw_pitch_roll {
                        img.yaw = Some(yaw);
                    }

                    let bbox = &face.detection.bbox;
                    let face_size = bbox.w.min(bbox.h);
                    img.face_bbox = Some((bbox.x, bbox.y, bbox.w, bbox.h));

                    let size_bucket = if face_size < 40.0 { "<40".to_string() }
                        else if face_size < 60.0 { "40-60".to_string() }
                        else if face_size < 80.0 { "60-80".to_string() }
                        else if face_size < 120.0 { "80-120".to_string() }
                        else if face_size < 200.0 { "120-200".to_string() }
                        else { ">200".to_string() };
                    *face_size_dist.entry(size_bucket).or_insert(0) += 1;

                    if let Some(yaw) = img.yaw {
                        let pose_bucket = if yaw.abs() < 15.0 { "0-15".to_string() }
                            else if yaw.abs() < 30.0 { "15-30".to_string() }
                            else if yaw.abs() < 45.0 { "30-45".to_string() }
                            else { ">45".to_string() };
                        *pose_dist.entry(pose_bucket).or_insert(0) += 1;
                    }

                    let quality_bucket = if img.face_quality < 0.3 { "low".to_string() }
                        else if img.face_quality < 0.5 { "medium".to_string() }
                        else if img.face_quality < 0.7 { "good".to_string() }
                        else { "high".to_string() };
                    *quality_dist.entry(quality_bucket).or_insert(0) += 1;

                    all_embedding_norms.push(norm);
                }
            }
            Err(_) => {}
        }
    }

    PipelineAuditReport {
        face_size_distribution: face_size_dist,
        pose_distribution: pose_dist,
        quality_distribution: quality_dist,
        detection_score_stats: compute_score_stats(&all_detection_scores),
        blur_score_stats: compute_score_stats(&all_blur_scores),
        embedding_norm_stats: compute_score_stats(&all_embedding_norms),
    }
}

// ============================================================================
// T27.2: Quality Bucket Analysis
// ============================================================================

fn analyze_by_quality_bucket(
    images: &[ImageRecordV3],
    splits: &HashMap<usize, QueryGallerySplit>,
) -> QualityBucketReport {
    let face_images: Vec<&ImageRecordV3> = images.iter()
        .filter(|i| i.face_detected && i.face_emb.is_some())
        .collect();

    let id_to_img: HashMap<usize, &ImageRecordV3> = face_images.iter()
        .map(|i| (i.id, *i))
        .collect();

    let mut person_embs: HashMap<usize, Vec<&Vec<f32>>> = HashMap::new();
    for img in &face_images {
        if let Some(ref emb) = img.face_emb {
            person_embs.entry(img.person_id).or_default().push(emb);
        }
    }

    // Size buckets
    let size_buckets = ["<40", "40-60", "60-80", "80-120", "120-200", ">200"];
    let mut size_bucket_results = Vec::new();

    for bucket in &size_buckets {
        let mut bucket_positive = Vec::new();
        let mut bucket_negative = Vec::new();

        for (&pid, split) in splits {
            for &q_id in &split.query {
                let Some(q_img) = id_to_img.get(&q_id) else { continue; };
                let Some(q_emb) = &q_img.face_emb else { continue; };

                let Some((_, _, w, h)) = q_img.face_bbox else { continue; };
                let face_size = w.min(h);
                let img_bucket = if face_size < 40.0 { "<40" }
                    else if face_size < 60.0 { "40-60" }
                    else if face_size < 80.0 { "60-80" }
                    else if face_size < 120.0 { "80-120" }
                    else if face_size < 200.0 { "120-200" }
                    else { ">200" };

                if img_bucket != *bucket { continue; }

                for &g_id in &split.gallery {
                    if g_id == q_id { continue; }
                    let Some(g_img) = id_to_img.get(&g_id) else { continue; };
                    if g_img.person_id != pid { continue; }
                    let Some(g_emb) = &g_img.face_emb else { continue; };
                    bucket_positive.push(cosine(q_emb, g_emb));
                }

                for (other_pid, other_split) in splits {
                    if *other_pid == pid { continue; }
                    for &g_id in &other_split.gallery {
                        let Some(g_img) = id_to_img.get(&g_id) else { continue; };
                        let Some(g_emb) = &g_img.face_emb else { continue; };
                        bucket_negative.push(cosine(q_emb, g_emb));
                    }
                }
            }
        }

        if !bucket_positive.is_empty() && !bucket_negative.is_empty() {
            let mut pos_sorted = bucket_positive.clone();
            pos_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pos_median = pos_sorted[pos_sorted.len()/2];

            let mut neg_sorted = bucket_negative.clone();
            neg_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let neg_median = neg_sorted[neg_sorted.len()/2];

            let margin_plus = bucket_positive.iter().filter(|&&s| s > neg_median).count() as f64 / bucket_positive.len() as f64;

            let mut margins: Vec<f32> = bucket_positive.iter().map(|&s| s - neg_median).collect();
            margins.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p10_idx = ((margins.len() as f32 * 0.1) as usize).min(margins.len() - 1);
            let p10_margin = margins[p10_idx];

            size_bucket_results.push(QualityBucket {
                bucket_name: bucket.to_string(),
                count: bucket_positive.len(),
                positive_median: pos_median as f64,
                negative_median: neg_median as f64,
                margin_plus_rate: margin_plus,
                margin_p10: p10_margin as f64,
            });
        }
    }

    // Pose buckets
    let pose_buckets = ["0-15", "15-30", "30-45", ">45"];
    let mut pose_bucket_results = Vec::new();

    for bucket in &pose_buckets {
        let mut bucket_positive = Vec::new();
        let mut bucket_negative = Vec::new();

        for (&pid, split) in splits {
            for &q_id in &split.query {
                let Some(q_img) = id_to_img.get(&q_id) else { continue; };
                let Some(q_emb) = &q_img.face_emb else { continue; };
                let Some(yaw) = q_img.yaw else { continue; };

                let img_bucket = if yaw.abs() < 15.0 { "0-15" }
                    else if yaw.abs() < 30.0 { "15-30" }
                    else if yaw.abs() < 45.0 { "30-45" }
                    else { ">45" };

                if img_bucket != *bucket { continue; }

                for &g_id in &split.gallery {
                    if g_id == q_id { continue; }
                    let Some(g_img) = id_to_img.get(&g_id) else { continue; };
                    if g_img.person_id != pid { continue; }
                    let Some(g_emb) = &g_img.face_emb else { continue; };
                    bucket_positive.push(cosine(q_emb, g_emb));
                }

                for (other_pid, other_split) in splits {
                    if *other_pid == pid { continue; }
                    for &g_id in &other_split.gallery {
                        let Some(g_img) = id_to_img.get(&g_id) else { continue; };
                        let Some(g_emb) = &g_img.face_emb else { continue; };
                        bucket_negative.push(cosine(q_emb, g_emb));
                    }
                }
            }
        }

        if !bucket_positive.is_empty() && !bucket_negative.is_empty() {
            let mut pos_sorted = bucket_positive.clone();
            pos_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pos_median = pos_sorted[pos_sorted.len()/2];

            let mut neg_sorted = bucket_negative.clone();
            neg_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let neg_median = neg_sorted[neg_sorted.len()/2];

            let margin_plus = bucket_positive.iter().filter(|&&s| s > neg_median).count() as f64 / bucket_positive.len() as f64;

            let mut margins: Vec<f32> = bucket_positive.iter().map(|&s| s - neg_median).collect();
            margins.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p10_idx = ((margins.len() as f32 * 0.1) as usize).min(margins.len() - 1);
            let p10_margin = margins[p10_idx];

            pose_bucket_results.push(QualityBucket {
                bucket_name: bucket.to_string(),
                count: bucket_positive.len(),
                positive_median: pos_median as f64,
                negative_median: neg_median as f64,
                margin_plus_rate: margin_plus,
                margin_p10: p10_margin as f64,
            });
        }
    }

    // Detection confidence buckets
    let det_buckets = ["low(<0.5)", "medium(0.5-0.7)", "high(0.7-0.9)", "very_high(>0.9)"];
    let mut det_bucket_results = Vec::new();

    for bucket in &det_buckets {
        let mut bucket_positive = Vec::new();
        let mut bucket_negative = Vec::new();

        for (&pid, split) in splits {
            for &q_id in &split.query {
                let Some(q_img) = id_to_img.get(&q_id) else { continue; };
                let Some(q_emb) = &q_img.face_emb else { continue; };

                let img_bucket = if q_img.face_score < 0.5 { "low(<0.5)" }
                    else if q_img.face_score < 0.7 { "medium(0.5-0.7)" }
                    else if q_img.face_score < 0.9 { "high(0.7-0.9)" }
                    else { "very_high(>0.9)" };

                if img_bucket != *bucket { continue; }

                for &g_id in &split.gallery {
                    if g_id == q_id { continue; }
                    let Some(g_img) = id_to_img.get(&g_id) else { continue; };
                    if g_img.person_id != pid { continue; }
                    let Some(g_emb) = &g_img.face_emb else { continue; };
                    bucket_positive.push(cosine(q_emb, g_emb));
                }

                for (other_pid, other_split) in splits {
                    if *other_pid == pid { continue; }
                    for &g_id in &other_split.gallery {
                        let Some(g_img) = id_to_img.get(&g_id) else { continue; };
                        let Some(g_emb) = &g_img.face_emb else { continue; };
                        bucket_negative.push(cosine(q_emb, g_emb));
                    }
                }
            }
        }

        if !bucket_positive.is_empty() && !bucket_negative.is_empty() {
            let mut pos_sorted = bucket_positive.clone();
            pos_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pos_median = pos_sorted[pos_sorted.len()/2];

            let mut neg_sorted = bucket_negative.clone();
            neg_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let neg_median = neg_sorted[neg_sorted.len()/2];

            let margin_plus = bucket_positive.iter().filter(|&&s| s > neg_median).count() as f64 / bucket_positive.len() as f64;

            let mut margins: Vec<f32> = bucket_positive.iter().map(|&s| s - neg_median).collect();
            margins.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p10_idx = ((margins.len() as f32 * 0.1) as usize).min(margins.len() - 1);
            let p10_margin = margins[p10_idx];

            det_bucket_results.push(QualityBucket {
                bucket_name: bucket.to_string(),
                count: bucket_positive.len(),
                positive_median: pos_median as f64,
                negative_median: neg_median as f64,
                margin_plus_rate: margin_plus,
                margin_p10: p10_margin as f64,
            });
        }
    }

    QualityBucketReport {
        by_face_size: size_bucket_results,
        by_pose: pose_bucket_results,
        by_detection_confidence: det_bucket_results,
    }
}

// ============================================================================
// T27.7: Hard Negative Analysis
// ============================================================================

fn analyze_hard_negatives(
    images: &[ImageRecordV3],
    splits: &HashMap<usize, QueryGallerySplit>,
) -> HardNegativeReport {
    let face_images: Vec<&ImageRecordV3> = images.iter()
        .filter(|i| i.face_detected && i.face_emb.is_some())
        .collect();

    let id_to_img: HashMap<usize, &ImageRecordV3> = face_images.iter()
        .map(|i| (i.id, *i))
        .collect();

    // Collect all cross-person pairs with their similarity
    let mut all_pairs: Vec<(usize, usize, usize, usize, f32)> = Vec::new();

    for (&pid_a, split_a) in splits {
        for &img_id_a in &split_a.gallery {
            let Some(img_a) = id_to_img.get(&img_id_a) else { continue; };
            let Some(emb_a) = &img_a.face_emb else { continue; };

            for (&pid_b, split_b) in splits {
                if pid_b <= pid_a { continue; }

                for &img_id_b in &split_b.gallery {
                    let Some(img_b) = id_to_img.get(&img_id_b) else { continue; };
                    let Some(emb_b) = &img_b.face_emb else { continue; };

                    let sim = cosine(emb_a, emb_b);
                    all_pairs.push((pid_a, img_id_a, pid_b, img_id_b, sim));
                }
            }
        }
    }

    // Sort by similarity descending
    all_pairs.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());

    let top_100: Vec<HardNegativeCase> = all_pairs.iter().take(100).map(|p| {
        let img_a = id_to_img.get(&p.1).unwrap();
        let img_b = id_to_img.get(&p.3).unwrap();

        let category = if p.4 > 0.8 {
            if img_a.yaw.map(|y| y.abs() > 30.0).unwrap_or(false) ||
               img_b.yaw.map(|y| y.abs() > 30.0).unwrap_or(false) {
                "D: Alignment/Pose Error".to_string()
            } else if img_a.face_score < 0.5 || img_b.face_score < 0.5 {
                "C: Low Detection Confidence".to_string()
            } else {
                "A: Genuinely Similar Faces".to_string()
            }
        } else if p.4 > 0.6 {
            "E: Similar Appearance (hair/clothing)".to_string()
        } else {
            "G: Other".to_string()
        };

        HardNegativeCase {
            person_a: p.0,
            image_a: p.1,
            person_b: p.2,
            image_b: p.3,
            similarity: p.4 as f64,
            face_size_a: img_a.face_bbox.map(|(_, _, w, h)| w.min(h)).unwrap_or(0.0),
            face_size_b: img_b.face_bbox.map(|(_, _, w, h)| w.min(h)).unwrap_or(0.0),
            yaw_a: img_a.yaw,
            yaw_b: img_b.yaw,
            blur_a: img_a.blur,
            blur_b: img_b.blur,
            likely_category: category,
        }
    }).collect();

    let mut categories: HashMap<String, usize> = HashMap::new();
    for case in &top_100 {
        *categories.entry(case.likely_category.clone()).or_insert(0) += 1;
    }

    HardNegativeReport {
        top_hard_negatives: top_100,
        categories,
    }
}

// ============================================================================
// T27.9: Prototype Quality Experiment
// ============================================================================

fn experiment_prototype_quality_filtering(
    images: &[ImageRecordV3],
    splits: &HashMap<usize, QueryGallerySplit>,
) -> PrototypeQualityReport {
    let face_images: Vec<&ImageRecordV3> = images.iter()
        .filter(|i| i.face_detected && i.face_emb.is_some())
        .collect();

    let id_to_img: HashMap<usize, &ImageRecordV3> = face_images.iter()
        .map(|i| (i.id, *i))
        .collect();

    let mut person_images: HashMap<usize, Vec<&ImageRecordV3>> = HashMap::new();
    for img in &face_images {
        person_images.entry(img.person_id).or_default().push(img);
    }

    // Filter experiment: all images
    let mut all_positive = Vec::new();
    let mut all_negative = Vec::new();

    for (&pid, split) in splits {
        for &q_id in &split.query {
            let Some(q_img) = id_to_img.get(&q_id) else { continue; };
            let Some(q_emb) = &q_img.face_emb else { continue; };

            let person_imgs: Vec<_> = person_images.get(&pid).map(|imgs|
                imgs.iter().filter(|i| i.id != q_id).collect()
            ).unwrap_or_default();

            if person_imgs.is_empty() { continue; }

            let prototype: Vec<f32> = person_imgs.iter()
                .filter_map(|i| i.face_emb.clone())
                .fold(vec![0.0; FACE_DIM], |mut acc, emb| {
                    for (a, &e) in acc.iter_mut().zip(emb.iter()) { *a += e; }
                    acc
                });
            let mut norm_proto = prototype;
            l2_normalize(&mut norm_proto);

            let pos_score = cosine(&norm_proto, q_emb);
            all_positive.push(pos_score);

            let mut neg_max = f32::MIN;
            for (other_pid, other_imgs) in &person_images {
                if *other_pid == pid { continue; }
                let other_proto: Vec<f32> = other_imgs.iter()
                    .filter_map(|i| i.face_emb.clone())
                    .fold(vec![0.0; FACE_DIM], |mut acc, emb| {
                        for (a, &e) in acc.iter_mut().zip(emb.iter()) { *a += e; }
                        acc
                    });
                if other_proto.iter().all(|&x| x == 0.0) { continue; }
                let mut norm_other = other_proto;
                l2_normalize(&mut norm_other);
                let neg_score = cosine(&norm_other, q_emb);
                if neg_score > neg_max { neg_max = neg_score; }
            }
            all_negative.push(neg_max);
        }
    }

    // Compute stats
    let neg_median = {
        let mut s = all_negative.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len()/2]
    };
    let margin_plus = all_positive.iter().filter(|&&s| s > neg_median).count() as f64 / all_positive.len() as f64;
    let mut margins: Vec<f32> = all_positive.iter().map(|&s| s - neg_median).collect();
    margins.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_margin = margins[margins.len()/2];
    let p10_idx = ((margins.len() as f32 * 0.1) as usize).min(margins.len() - 1);
    let p10_margin = margins[p10_idx];
    let threshold_pass = all_positive.iter().filter(|&&s| s >= 0.55).count() as f64 / all_positive.len() as f64;

    let results = vec![
        PrototypeQualityResult {
            filter: "all_images".to_string(),
            margin_plus_rate: margin_plus,
            median_margin: median_margin as f64,
            p10_margin: p10_margin as f64,
            threshold_pass_rate: threshold_pass,
        },
    ];

    PrototypeQualityReport { results }
}

// ============================================================================
// T27.11: Model Ceiling Verdict
// ============================================================================

fn determine_model_ceiling(
    quality_report: &QualityBucketReport,
    hard_negatives: &HardNegativeReport,
    margin_plus: f64,
    pos_median: f64,
    neg_median: f64,
) -> ModelCeilingVerdict {
    let best_size_bucket = quality_report.by_face_size.iter()
        .max_by(|a, b| a.margin_plus_rate.partial_cmp(&b.margin_plus_rate).unwrap())
        .map(|b| (b.bucket_name.clone(), b.margin_plus_rate, b.count));

    let genuinely_similar = hard_negatives.categories.get("A: Genuinely Similar Faces")
        .copied().unwrap_or(0);

    let mut evidence = Vec::new();

    evidence.push(format!(
        "Overall LOO Margin+ = {:.2}% (positive median = {:.3}, negative median = {:.3})",
        margin_plus * 100.0, pos_median, neg_median
    ));

    if let Some((name, rate, count)) = &best_size_bucket {
        evidence.push(format!(
            "Best size bucket: {} with Margin+ = {:.2}% (n={})",
            name, rate * 100.0, count
        ));
    }

    evidence.push(format!(
        "Hard negatives >0.8 similarity: {} pairs",
        hard_negatives.top_hard_negatives.iter().filter(|c| c.similarity > 0.8).count()
    ));

    evidence.push(format!(
        "Genuinely similar faces (not pipeline error): {} of top 100 hard negatives",
        genuinely_similar
    ));

    let (verdict, confidence, recommendation) = if margin_plus >= 0.90 {
        if genuinely_similar >= 30 {
            ("MODEL_LIMITED", "HIGH",
             "Embeddings separate well overall (90%+ Margin+) but have genuine verification ceiling due to look-alike persons. Consider model upgrade (Phase 28).")
        } else {
            ("PIPELINE_ISSUE", "HIGH",
             "Embeddings work but pipeline introduces errors. Focus on quality filtering and alignment improvement.")
        }
    } else if margin_plus >= 0.70 {
        ("MODEL_LIMITED", "MEDIUM",
         "Significant embedding quality issues. Consider model upgrade or TTA enhancement.")
    } else {
        ("MODEL_INSUFFICIENT", "HIGH",
         "Embedding quality is insufficient for this dataset. Direct model upgrade is required.")
    };

    ModelCeilingVerdict {
        verdict: verdict.to_string(),
        confidence: confidence.to_string(),
        recall_at_far01: 0.0,
        margin_plus_rate: margin_plus,
        evidence,
        recommendation: recommendation.to_string(),
    }
}

// ============================================================================
// Main Test
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn phase27_model_capability() {
    println!("============================================================");
    println!("Phase 27 — Face Embedding Model Capability Benchmark");
    println!("============================================================\n");

    // ====== 1. Discover Dataset ======
    println!("Step 1: Discovering Dataset...");
    let mut images = match discover_dataset() {
        Some(imgs) => imgs,
        None => {
            println!("ERROR: Dataset not found at {}", BENCHMARK_DIR);
            return;
        }
    };
    println!("  Found {} images\n", images.len());

    // ====== 2. Load Models ======
    println!("Step 2: Loading Models...");
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

    // ====== 3. T27.1: Pipeline Audit ======
    println!("============================================================");
    println!("T27.1: Face Pipeline Audit");
    println!("============================================================\n");

    let pipeline_report = audit_face_pipeline_async(&mut images, &face_pipeline).await;

    println!("Face Size Distribution:");
    for (bucket, count) in &pipeline_report.face_size_distribution {
        println!("  {}: {}", bucket, count);
    }

    println!("\nPose Distribution:");
    for (bucket, count) in &pipeline_report.pose_distribution {
        println!("  {}: {}", bucket, count);
    }

    println!("\nQuality Distribution:");
    for (bucket, count) in &pipeline_report.quality_distribution {
        println!("  {}: {}", bucket, count);
    }

    println!("\nDetection Score Stats:");
    println!("  Min: {:.3}, P10: {:.3}, P50: {:.3}, P90: {:.3}, Max: {:.3}",
        pipeline_report.detection_score_stats.min,
        pipeline_report.detection_score_stats.p10,
        pipeline_report.detection_score_stats.p50,
        pipeline_report.detection_score_stats.p90,
        pipeline_report.detection_score_stats.max);

    // ====== 4. Create Splits ======
    println!("\n============================================================");
    println!("Step 4: Creating Query/Gallery Splits...");
    let splits = create_query_gallery_split(&images, 42);
    println!("  Created splits for {} persons\n", splits.len());

    // ====== 5. T27.2: Quality Bucket Analysis ======
    println!("============================================================");
    println!("T27.2: Quality Bucket Analysis");
    println!("============================================================\n");

    let quality_report = analyze_by_quality_bucket(&images, &splits);

    println!("By Face Size:");
    for bucket in &quality_report.by_face_size {
        println!("  {}: n={}, pos_median={:.3}, margin+={:.2}%, p10_margin={:.4}",
            bucket.bucket_name, bucket.count, bucket.positive_median,
            bucket.margin_plus_rate * 100.0, bucket.margin_p10);
    }

    println!("\nBy Pose:");
    for bucket in &quality_report.by_pose {
        println!("  {}: n={}, pos_median={:.3}, margin+={:.2}%, p10_margin={:.4}",
            bucket.bucket_name, bucket.count, bucket.positive_median,
            bucket.margin_plus_rate * 100.0, bucket.margin_p10);
    }

    println!("\nBy Detection Confidence:");
    for bucket in &quality_report.by_detection_confidence {
        println!("  {}: n={}, pos_median={:.3}, margin+={:.2}%, p10_margin={:.4}",
            bucket.bucket_name, bucket.count, bucket.positive_median,
            bucket.margin_plus_rate * 100.0, bucket.margin_p10);
    }

    // ====== 6. T27.7: Hard Negative Analysis ======
    println!("\n============================================================");
    println!("T27.7: Hard Negative Analysis");
    println!("============================================================\n");

    let hard_negative_report = analyze_hard_negatives(&images, &splits);

    println!("Hard Negative Categories (Top 100 pairs):");
    for (cat, count) in &hard_negative_report.categories {
        println!("  {}: {}", cat, count);
    }

    println!("\nTop 20 Hard Negatives:");
    println!("  Rank | PersonA | PersonB | Similarity | Category");
    println!("  -----|--------|--------|-----------|---------");
    for (i, case_) in hard_negative_report.top_hard_negatives.iter().take(20).enumerate() {
        println!("  {:4} | {:7} | {:7} | {:9.4} | {}",
            i + 1, case_.person_a, case_.person_b, case_.similarity, case_.likely_category);
    }

    // ====== 7. T27.9: Prototype Quality Filtering ======
    println!("\n============================================================");
    println!("T27.9: Prototype Quality Filtering Experiment");
    println!("============================================================\n");

    let prototype_report = experiment_prototype_quality_filtering(&images, &splits);

    println!("Prototype Quality Filtering Results:");
    println!("  Filter            | Margin+   | Median Margin | P10 Margin | ThreshPass");
    println!("  ------------------|-----------|----------------|------------|----------");
    for result in &prototype_report.results {
        println!("  {:17} | {:9.2}% | {:14.4} | {:10.4} | {:9.2}%",
            result.filter, result.margin_plus_rate * 100.0,
            result.median_margin, result.p10_margin, result.threshold_pass_rate * 100.0);
    }

    // ====== 8. Compute Overall Stats for Verdict ======
    let face_images: Vec<&ImageRecordV3> = images.iter()
        .filter(|i| i.face_detected && i.face_emb.is_some())
        .collect();

    let id_to_img: HashMap<usize, &ImageRecordV3> = face_images.iter()
        .map(|i| (i.id, *i))
        .collect();

    let mut all_positive = Vec::new();
    let mut all_negative = Vec::new();

    for (pid, split) in &splits {
        for &q_id in &split.query {
            let Some(q_img) = id_to_img.get(&q_id) else { continue; };
            let Some(q_emb) = &q_img.face_emb else { continue; };

            for &g_id in &split.gallery {
                if g_id == q_id { continue; }
                let Some(g_img) = id_to_img.get(&g_id) else { continue; };
                if g_img.person_id != *pid { continue; }
                let Some(g_emb) = &g_img.face_emb else { continue; };
                all_positive.push(cosine(q_emb, g_emb));
            }

            for (other_pid, other_split) in splits.iter() {
                if *other_pid == *pid { continue; }
                for &g_id in &other_split.gallery {
                    let Some(g_img) = id_to_img.get(&g_id) else { continue; };
                    let Some(g_emb) = &g_img.face_emb else { continue; };
                    all_negative.push(cosine(q_emb, g_emb));
                }
            }
        }
    }

    let pos_median_overall = {
        let mut s = all_positive.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len()/2]
    };
    let neg_median_overall = {
        let mut s = all_negative.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len()/2]
    };
    let margin_plus_overall = all_positive.iter().filter(|&&s| s > neg_median_overall).count() as f64 / all_positive.len() as f64;

    // ====== 9. T27.11: Model Ceiling Verdict ======
    println!("\n============================================================");
    println!("T27.11: Model Ceiling Verdict");
    println!("============================================================\n");

    let ceiling = determine_model_ceiling(
        &quality_report,
        &hard_negative_report,
        margin_plus_overall,
        pos_median_overall as f64,
        neg_median_overall as f64,
    );

    println!("VERDICT: {}", ceiling.verdict);
    println!("Confidence: {}\n", ceiling.confidence);
    println!("Evidence:");
    for e in &ceiling.evidence {
        println!("  - {}", e);
    }
    println!("\nRecommendation:");
    println!("  {}", ceiling.recommendation);

    // ====== 10. Final Summary ======
    println!("\n============================================================");
    println!("Phase 27 Final Summary");
    println!("============================================================");
    println!("Dataset: {} images, {} persons", images.len(), splits.len());
    println!("Face detected: {}/{}", face_images.len(), images.len());
    println!("");
    println!("OVERALL PAIRWISE METRICS:");
    println!("  Positive Median: {:.4}", pos_median_overall);
    println!("  Negative Median: {:.4}", neg_median_overall);
    println!("  LOO Margin+: {:.2}%", margin_plus_overall * 100.0);
    println!("");
    println!("MODEL CEILING: {}", ceiling.verdict);

    // ====== 11. Write JSON Report ======
    let report = Phase27Report {
        phase: "27".to_string(),
        dataset_summary: DatasetSummaryV3 {
            total_images: images.len(),
            total_persons: splits.len(),
            face_detected: face_images.len(),
        },
        pipeline_audit: pipeline_report,
        quality_bucket_analysis: quality_report,
        hard_negative_analysis: hard_negative_report,
        prototype_quality_experiment: prototype_report,
        model_ceiling: ceiling,
    };

    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    let output_path = PathBuf::from("/Users/mac/ai-project/photofinder-next-2/crates/pf_application/tests/phase27_model_capability_report.json");
    std::fs::write(&output_path, &json).expect("write report");
    println!("\n\nJSON report written to: {:?}", output_path);
}
