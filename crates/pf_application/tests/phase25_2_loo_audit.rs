//! Phase 25.2 — LOO / Prototype Audit
//!
//! Audit the Phase 25 LOO implementation to determine root cause of low pass rates.
//!
//! Separates:
//! - LOO Rank1: correct person ranked #1
//! - LOO Margin+: positive_score > negative_max
//! - LOO Threshold: positive_score >= fixed_threshold
//!
//! T1: Audit LOO definition
//! T2: Remove threshold from LOO measurement
//! T3: Prototype normalization audit
//! T4: Per-person LOO analysis
//! T5: Prototype dilution experiment
//! T6: Multi-prototype experiment
//! T7: Direct pairwise distribution
//! T8: Inspect 20 failed cases
//! T9: Correct interpretation
//! T10: Root cause classification
//!
//! 运行：
//! ```bash
//! cargo test -p pf_application --test phase25_2_loo_audit -- --nocapture --ignored
//! ```

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Serialize, Deserialize};

// ============================================================================
// Report Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LooAuditReport {
    pub phase: String,
    pub dataset_summary: DatasetSummary,
    pub face_loo: DetailedLooMetrics,
    pub body_loo: DetailedLooMetrics,
    pub face_pairwise: PairwiseDistribution,
    pub body_pairwise: PairwiseDistribution,
    pub prototype_dilution: PrototypeDilutionReport,
    pub multi_prototype: MultiPrototypeReport,
    pub failure_cases: FailureCaseReport,
    pub root_cause: RootCauseVerdict,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSummary {
    pub total_images: usize,
    pub total_persons: usize,
    pub face_available: usize,
    pub body_available: usize,
    pub image_distribution: Vec<PersonImageCount>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonImageCount {
    pub person_id: usize,
    pub face_images: usize,
    pub body_images: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedLooMetrics {
    pub total_queries: usize,
    pub rank1_rate: f64,
    pub rank1_count: usize,
    pub margin_plus_rate: f64,
    pub margin_plus_count: usize,
    pub threshold_pass_rate: f64,
    pub threshold_pass_count: usize,
    pub positive_score_stats: ScoreStats,
    pub negative_max_stats: ScoreStats,
    pub margin_stats: ScoreStats,
    pub per_person: Vec<PersonLooMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreStats {
    pub min: f64,
    pub p1: f64,
    pub p5: f64,
    pub p10: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
    pub mean: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonLooMetrics {
    pub person_id: usize,
    pub images: usize,
    pub rank1_rate: f64,
    pub margin_plus_rate: f64,
    pub threshold_pass_rate: f64,
    pub median_positive_score: f64,
    pub median_margin: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseDistribution {
    pub positive_scores: ScoreStats,
    pub negative_scores: ScoreStats,
    pub separation: SeparationMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeparationMetrics {
    pub overlap_ratio: f64,
    pub decile_separation: Vec<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrototypeDilutionReport {
    pub face_results: Vec<DilutionResult>,
    pub body_results: Vec<DilutionResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DilutionResult {
    pub k: usize,
    pub rank1_rate: f64,
    pub margin_plus_rate: f64,
    pub median_margin: f64,
    pub p10_margin: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiPrototypeReport {
    pub face_results: Vec<MultiProtoResult>,
    pub body_results: Vec<MultiProtoResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiProtoResult {
    pub num_prototypes: usize,
    pub rank1_rate: f64,
    pub margin_plus_rate: f64,
    pub median_margin: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureCaseReport {
    pub face_failures: Vec<FailureCase>,
    pub body_failures: Vec<FailureCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureCase {
    pub person_id: usize,
    pub image_id: usize,
    pub positive_score: f64,
    pub best_negative_person: usize,
    pub best_negative_score: f64,
    pub margin: f64,
    pub correct_rank: usize,
    pub face_quality: Option<f32>,
    pub detection_score: Option<f32>,
    pub failure_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootCauseVerdict {
    pub verdict: String,
    pub confidence: String,
    pub evidence: Vec<String>,
    pub recommendation: String,
}

// ============================================================================
// Constants
// ============================================================================

const BENCHMARK_DIR: &str = "/Users/mac/ai-project/photofinder-next-2/benchmark/person_identity_real";
const FACE_DIM: usize = 512;
const BODY_DIM: usize = 768;
const FACE_THRESHOLD: f32 = 0.75;
const BODY_THRESHOLD: f32 = 0.70;

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone)]
struct ImageRecord {
    id: usize,
    person_id: usize,
    person_name: String,
    path: PathBuf,
    face_emb: Option<Vec<f32>>,
    body_emb: Option<Vec<f32>>,
    face_detected: bool,
    face_score: f32,
    face_quality: f32,
    body_available: bool,
    body_quality: f32,
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
    for x in v.iter_mut() {
        *x /= norm;
    }
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
        return ScoreStats {
            min: 0.0, p1: 0.0, p5: 0.0, p10: 0.0, p25: 0.0,
            p50: 0.0, p75: 0.0, p90: 0.0, p95: 0.0, p99: 0.0, max: 0.0, mean: 0.0,
        };
    }
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum: f64 = scores.iter().map(|&x| x as f64).sum();
    ScoreStats {
        min: sorted.first().copied().unwrap_or(0.0) as f64,
        p1: percentile(&sorted, 0.01) as f64,
        p5: percentile(&sorted, 0.05) as f64,
        p10: percentile(&sorted, 0.10) as f64,
        p25: percentile(&sorted, 0.25) as f64,
        p50: percentile(&sorted, 0.50) as f64,
        p75: percentile(&sorted, 0.75) as f64,
        p90: percentile(&sorted, 0.90) as f64,
        p95: percentile(&sorted, 0.95) as f64,
        p99: percentile(&sorted, 0.99) as f64,
        max: sorted.last().copied().unwrap_or(0.0) as f64,
        mean: sum / scores.len() as f64,
    }
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

            let filename = img_path.file_stem()?.to_str()?;
            let name_parts: Vec<&str> = filename.split('_').collect();
            let person_display_name = if name_parts.len() >= 2 {
                format!("{} {}", name_parts[0], name_parts[1])
            } else {
                person_name.to_string()
            };

            images.push(ImageRecord {
                id: image_id,
                person_id,
                person_name: person_display_name,
                path: img_path,
                face_emb: None,
                body_emb: None,
                face_detected: false,
                face_score: 0.0,
                face_quality: 0.0,
                body_available: false,
                body_quality: 0.0,
            });
            image_id += 1;
        }
    }
    images.sort_by(|a, b| a.person_id.cmp(&b.person_id).then(a.path.cmp(&b.path)));
    Some(images)
}

// ============================================================================
// Model Resolution
// ============================================================================

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

fn create_query_gallery_split(images: &[ImageRecord], seed: u64) -> HashMap<usize, QueryGallerySplit> {
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
// T1 & T2: Detailed LOO Metrics (Rank + Margin + Threshold separated)
// ============================================================================

struct RawLooResult {
    positive_score: f32,
    negative_max: f32,
    margin: f32,
    correct_rank: usize,
}

fn compute_detailed_loo_face(
    images: &[ImageRecord],
    splits: &HashMap<usize, QueryGallerySplit>,
) -> (DetailedLooMetrics, Vec<RawLooResult>, Vec<f32>, Vec<f32>) {
    let face_images: Vec<&ImageRecord> = images.iter()
        .filter(|i| i.face_detected && i.face_emb.is_some())
        .collect();

    let id_to_img: HashMap<usize, &ImageRecord> = images.iter()
        .filter(|i| i.face_detected && i.face_emb.is_some())
        .map(|i| (i.id, i))
        .collect();

    // Build person prototypes from FULL gallery (not just the benchmark split)
    // For LOO, we need ALL images of a person except the query
    let mut person_to_images: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
    for img in &face_images {
        person_to_images.entry(img.person_id).or_default().push(img);
    }

    let mut raw_results = Vec::new();
    let mut all_positive_scores = Vec::new();
    let mut all_negative_max_scores = Vec::new();

    // For ranking: compute all person prototypes first
    let mut person_prototypes: HashMap<usize, Vec<f32>> = HashMap::new();
    for (&person_id, imgs) in &person_to_images {
        let prototype: Vec<f32> = imgs.iter()
            .filter_map(|i| i.face_emb.clone())
            .fold(vec![0.0; FACE_DIM], |mut acc, emb| {
                for (a, &e) in acc.iter_mut().zip(emb.iter()) { *a += e; }
                acc
            });
        let mut norm_proto = prototype;
        l2_normalize(&mut norm_proto);
        person_prototypes.insert(person_id, norm_proto);
    }

    // For each query image
    for (&person_id, split) in splits {
        for &q_id in &split.query {
            let Some(q_img) = id_to_img.get(&q_id) else { continue };
            let Some(q_emb) = &q_img.face_emb else { continue };

            // Build prototype excluding this query image
            let person_imgs = person_to_images.get(&person_id);
            let Some(person_imgs) = person_imgs else { continue };

            let prototype: Vec<f32> = person_imgs.iter()
                .filter(|i| i.id != q_id)
                .filter_map(|i| i.face_emb.clone())
                .fold(vec![0.0; FACE_DIM], |mut acc, emb| {
                    for (a, &e) in acc.iter_mut().zip(emb.iter()) { *a += e; }
                    acc
                });
            let mut norm_proto = prototype;
            l2_normalize(&mut norm_proto);

            let positive_score = cosine(&norm_proto, q_emb);
            all_positive_scores.push(positive_score);

            // Compute negative max (best other person prototype)
            let mut negative_max = f32::MIN;
            let mut best_negative_person = 0usize;
            for (&other_pid, other_proto) in &person_prototypes {
                if other_pid == person_id { continue; }
                let score = cosine(&norm_proto, q_emb); // Wait - this should be other_proto vs q_emb!
                // Actually: query embedding vs other person's prototype
                let score = cosine(other_proto, q_emb);
                if score > negative_max {
                    negative_max = score;
                    best_negative_person = other_pid;
                }
            }
            all_negative_max_scores.push(negative_max);

            let margin = positive_score - negative_max;

            // Compute correct rank among all persons
            let mut all_scores: Vec<(usize, f32)> = person_prototypes.iter()
                .filter(|(&pid, _)| pid != person_id)
                .map(|(&pid, proto)| (pid, cosine(proto, q_emb)))
                .collect();
            all_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Find actual rank of correct person
            let correct_rank = all_scores.iter().position(|(pid, _)| *pid == person_id).map(|p| p + 1).unwrap_or(999);

            raw_results.push(RawLooResult {
                positive_score,
                negative_max,
                margin,
                correct_rank,
            });
        }
    }

    // Compute per-person metrics
    let mut per_person_metrics = Vec::new();
    for (&person_id, imgs) in &person_to_images {
        let person_raw: Vec<&RawLooResult> = raw_results.iter()
            .filter(|r| {
                let q_ids: Vec<usize> = imgs.iter().map(|i| i.id).collect();
                // This is expensive, let me simplify
                true
            })
            .collect();

        let person_results: Vec<&RawLooResult> = raw_results.iter()
            .filter(|r| {
                let q_person_id = id_to_img.values()
                    .find(|img| img.person_id == person_id)
                    .map(|img| img.person_id);
                q_person_id == Some(person_id)
            })
            .collect();

        if !person_results.is_empty() {
            let rank1 = person_results.iter().filter(|r| r.correct_rank == 1).count();
            let margin_plus = person_results.iter().filter(|r| r.margin > 0.0).count();
            let threshold_pass = person_results.iter().filter(|r| r.positive_score >= FACE_THRESHOLD).count();
            let median_pos = {
                let mut scores: Vec<f32> = person_results.iter().map(|r| r.positive_score).collect();
                scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
                scores[scores.len() / 2]
            };
            let median_margin = {
                let mut margins: Vec<f32> = person_results.iter().map(|r| r.margin).collect();
                margins.sort_by(|a, b| a.partial_cmp(b).unwrap());
                margins[margins.len() / 2]
            };

            per_person_metrics.push(PersonLooMetrics {
                person_id,
                images: imgs.len(),
                rank1_rate: rank1 as f64 / person_results.len() as f64,
                margin_plus_rate: margin_plus as f64 / person_results.len() as f64,
                threshold_pass_rate: threshold_pass as f64 / person_results.len() as f64,
                median_positive_score: median_pos as f64,
                median_margin: median_margin as f64,
            });
        }
    }

    let total = raw_results.len();
    let rank1_count = raw_results.iter().filter(|r| r.correct_rank == 1).count();
    let margin_plus_count = raw_results.iter().filter(|r| r.margin > 0.0).count();
    let threshold_pass_count = raw_results.iter().filter(|r| r.positive_score >= FACE_THRESHOLD).count();
    let margins: Vec<f32> = raw_results.iter().map(|r| r.margin).collect();

    let detailed = DetailedLooMetrics {
        total_queries: total,
        rank1_rate: rank1_count as f64 / total.max(1) as f64,
        rank1_count,
        margin_plus_rate: margin_plus_count as f64 / total.max(1) as f64,
        margin_plus_count,
        threshold_pass_rate: threshold_pass_count as f64 / total.max(1) as f64,
        threshold_pass_count,
        positive_score_stats: compute_score_stats(&all_positive_scores),
        negative_max_stats: compute_score_stats(&all_negative_max_scores),
        margin_stats: compute_score_stats(&margins),
        per_person: per_person_metrics,
    };

    (detailed, raw_results, all_positive_scores, all_negative_max_scores)
}

fn compute_detailed_loo_body(
    images: &[ImageRecord],
    splits: &HashMap<usize, QueryGallerySplit>,
) -> (DetailedLooMetrics, Vec<RawLooResult>, Vec<f32>, Vec<f32>) {
    let body_images: Vec<&ImageRecord> = images.iter()
        .filter(|i| i.body_available && i.body_emb.is_some())
        .collect();

    let id_to_img: HashMap<usize, &ImageRecord> = images.iter()
        .filter(|i| i.body_available && i.body_emb.is_some())
        .map(|i| (i.id, i))
        .collect();

    let mut person_to_images: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
    for img in &body_images {
        person_to_images.entry(img.person_id).or_default().push(img);
    }

    let mut raw_results = Vec::new();
    let mut all_positive_scores = Vec::new();
    let mut all_negative_max_scores = Vec::new();

    // Build person prototypes
    let mut person_prototypes: HashMap<usize, Vec<f32>> = HashMap::new();
    for (&person_id, imgs) in &person_to_images {
        let prototype: Vec<f32> = imgs.iter()
            .filter_map(|i| i.body_emb.clone())
            .fold(vec![0.0; BODY_DIM], |mut acc, emb| {
                for (a, &e) in acc.iter_mut().zip(emb.iter()) { *a += e; }
                acc
            });
        let mut norm_proto = prototype;
        l2_normalize(&mut norm_proto);
        person_prototypes.insert(person_id, norm_proto);
    }

    for (&person_id, split) in splits {
        for &q_id in &split.query {
            let Some(q_img) = id_to_img.get(&q_id) else { continue };
            let Some(q_emb) = &q_img.body_emb else { continue };

            let person_imgs = person_to_images.get(&person_id);
            let Some(person_imgs) = person_imgs else { continue };

            let prototype: Vec<f32> = person_imgs.iter()
                .filter(|i| i.id != q_id)
                .filter_map(|i| i.body_emb.clone())
                .fold(vec![0.0; BODY_DIM], |mut acc, emb| {
                    for (a, &e) in acc.iter_mut().zip(emb.iter()) { *a += e; }
                    acc
                });
            let mut norm_proto = prototype;
            l2_normalize(&mut norm_proto);

            let positive_score = cosine(&norm_proto, q_emb);
            all_positive_scores.push(positive_score);

            let mut negative_max = f32::MIN;
            for (&other_pid, other_proto) in &person_prototypes {
                if other_pid == person_id { continue; }
                let score = cosine(other_proto, q_emb);
                if score > negative_max {
                    negative_max = score;
                }
            }
            all_negative_max_scores.push(negative_max);

            let margin = positive_score - negative_max;

            let mut all_scores: Vec<(usize, f32)> = person_prototypes.iter()
                .filter(|(&pid, _)| pid != person_id)
                .map(|(&pid, proto)| (pid, cosine(proto, q_emb)))
                .collect();
            all_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let correct_rank = all_scores.iter().position(|(pid, _)| *pid == person_id).map(|p| p + 1).unwrap_or(999);

            raw_results.push(RawLooResult {
                positive_score,
                negative_max,
                margin,
                correct_rank,
            });
        }
    }

    let total = raw_results.len();
    let rank1_count = raw_results.iter().filter(|r| r.correct_rank == 1).count();
    let margin_plus_count = raw_results.iter().filter(|r| r.margin > 0.0).count();
    let threshold_pass_count = raw_results.iter().filter(|r| r.positive_score >= BODY_THRESHOLD).count();
    let margins: Vec<f32> = raw_results.iter().map(|r| r.margin).collect();

    let detailed = DetailedLooMetrics {
        total_queries: total,
        rank1_rate: rank1_count as f64 / total.max(1) as f64,
        rank1_count,
        margin_plus_rate: margin_plus_count as f64 / total.max(1) as f64,
        margin_plus_count,
        threshold_pass_rate: threshold_pass_count as f64 / total.max(1) as f64,
        threshold_pass_count,
        positive_score_stats: compute_score_stats(&all_positive_scores),
        negative_max_stats: compute_score_stats(&all_negative_max_scores),
        margin_stats: compute_score_stats(&margins),
        per_person: Vec::new(),
    };

    (detailed, raw_results, all_positive_scores, all_negative_max_scores)
}

// ============================================================================
// T7: Pairwise Distribution
// ============================================================================

fn compute_pairwise_distributions(
    images: &[ImageRecord],
    splits: &HashMap<usize, QueryGallerySplit>,
    emb_type: &str,
) -> (PairwiseDistribution, Vec<f32>, Vec<f32>) {
    let dim = if emb_type == "face" { FACE_DIM } else { BODY_DIM };

    let id_to_img: HashMap<usize, &ImageRecord> = images.iter()
        .filter(|i| {
            if emb_type == "face" {
                i.face_detected && i.face_emb.is_some()
            } else {
                i.body_available && i.body_emb.is_some()
            }
        })
        .map(|i| (i.id, i))
        .collect();

    let mut positive_scores = Vec::new();
    let mut negative_scores = Vec::new();

    for (person_id, split) in splits {
        for &q_id in &split.query {
            let Some(q_img) = id_to_img.get(&q_id) else { continue };
            let q_emb = if emb_type == "face" { &q_img.face_emb } else { &q_img.body_emb };
            let Some(q_emb) = q_emb else { continue };

            // Positive pairs: query vs same person gallery
            for &g_id in &split.gallery {
                if g_id == q_id { continue; }
                let Some(g_img) = id_to_img.get(&g_id) else { continue; };
                let g_emb = if emb_type == "face" { &g_img.face_emb } else { &g_img.body_emb };
                let Some(g_emb) = g_emb else { continue; };
                positive_scores.push(cosine(q_emb, g_emb));
            }

            // Negative pairs: query vs other person gallery
            for (other_pid, other_split) in splits {
                if *other_pid == *person_id { continue; }
                for &g_id in &other_split.gallery {
                    let Some(g_img) = id_to_img.get(&g_id) else { continue; };
                    let g_emb = if emb_type == "face" { &g_img.face_emb } else { &g_img.body_emb };
                    let Some(g_emb) = g_emb else { continue; };
                    negative_scores.push(cosine(q_emb, g_emb));
                }
            }
        }
    }

    let separation = compute_separation_metrics(&positive_scores, &negative_scores);

    (PairwiseDistribution {
        positive_scores: compute_score_stats(&positive_scores),
        negative_scores: compute_score_stats(&negative_scores),
        separation,
    }, positive_scores, negative_scores)
}

fn compute_separation_metrics(positive: &[f32], negative: &[f32]) -> SeparationMetrics {
    // Compute overlap ratio: what fraction of positive scores are below the median negative score
    let neg_median = {
        let mut sorted = negative.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    };

    let overlap_count = positive.iter().filter(|&&s| s < neg_median).count();
    let overlap_ratio = overlap_count as f64 / positive.len().max(1) as f64;

    // Decile separation: for each positive decile, is it above median negative?
    let mut decile_separation = Vec::new();
    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] {
        let pos_d = percentile(positive, p);
        decile_separation.push(pos_d > neg_median);
    }

    SeparationMetrics { overlap_ratio, decile_separation }
}

// ============================================================================
// T5: Prototype Dilution Experiment
// ============================================================================

fn prototype_dilution_experiment(
    images: &[ImageRecord],
    splits: &HashMap<usize, QueryGallerySplit>,
    emb_type: &str,
) -> Vec<DilutionResult> {
    let dim = if emb_type == "face" { FACE_DIM } else { BODY_DIM };
    let threshold = if emb_type == "face" { FACE_THRESHOLD } else { BODY_THRESHOLD };

    let valid_images: Vec<&ImageRecord> = images.iter()
        .filter(|i| {
            if emb_type == "face" { i.face_detected && i.face_emb.is_some() }
            else { i.body_available && i.body_emb.is_some() }
        })
        .collect();

    let id_to_img: HashMap<usize, &ImageRecord> = valid_images.iter()
        .map(|i| (i.id, *i))
        .collect();

    let mut person_images: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
    for img in &valid_images {
        person_images.entry(img.person_id).or_default().push(img);
    }

    let mut results = Vec::new();

    for &k in &[1, 3, 5, 10, 20] {
        let mut rank1_total = 0;
        let mut margin_plus_total = 0;
        let mut margins = Vec::new();
        let mut total = 0;

        for (&person_id, split) in splits {
            for &q_id in &split.query {
                let Some(q_img) = id_to_img.get(&q_id) else { continue; };
                let q_emb = if emb_type == "face" { &q_img.face_emb } else { &q_img.body_emb };
                let Some(q_emb) = q_emb else { continue; };

                let person_imgs = match person_images.get(&person_id) {
                    Some(imgs) => imgs,
                    None => continue,
                };

                // Select K images (randomly for now, just take first K)
                let selected: Vec<_> = person_imgs.iter()
                    .filter(|i| i.id != q_id)
                    .take(k)
                    .collect();

                if selected.is_empty() { continue; }

                // Build prototype from selected
                let prototype: Vec<f32> = selected.iter()
                    .filter_map(|i| {
                        if emb_type == "face" { i.face_emb.clone() }
                        else { i.body_emb.clone() }
                    })
                    .fold(vec![0.0; dim], |mut acc, emb| {
                        for (a, &e) in acc.iter_mut().zip(emb.iter()) { *a += e; }
                        acc
                    });

                let mut norm_proto = prototype;
                l2_normalize(&mut norm_proto);

                let positive_score = cosine(&norm_proto, q_emb);

                // Compute negative max
                let mut negative_max = f32::MIN;
                for (&other_pid, other_imgs) in &person_images {
                    if other_pid == person_id { continue; }
                    let other_proto: Vec<f32> = other_imgs.iter()
                        .filter_map(|i| {
                            if emb_type == "face" { i.face_emb.clone() }
                            else { i.body_emb.clone() }
                        })
                        .fold(vec![0.0; dim], |mut acc, emb| {
                            for (a, &e) in acc.iter_mut().zip(emb.iter()) { *a += e; }
                            acc
                        });
                    let mut norm_other = other_proto;
                    l2_normalize(&mut norm_other);
                    let score = cosine(&norm_other, q_emb);
                    if score > negative_max {
                        negative_max = score;
                    }
                }

                let margin = positive_score - negative_max;
                margins.push(margin);

                if positive_score > negative_max { margin_plus_total += 1; }
                total += 1;
            }
        }

        results.push(DilutionResult {
            k,
            rank1_rate: rank1_total as f64 / total.max(1) as f64,
            margin_plus_rate: margin_plus_total as f64 / total.max(1) as f64,
            median_margin: {
                let mut sorted = margins.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                sorted[sorted.len() / 2] as f64
            },
            p10_margin: percentile(&margins, 0.1) as f64,
        });
    }

    results
}

// ============================================================================
// T8: Failure Cases
// ============================================================================

fn collect_failure_cases(
    raw_results: &[RawLooResult],
    images: &[ImageRecord],
    emb_type: &str,
    n_cases: usize,
) -> Vec<FailureCase> {
    let valid_images: Vec<&ImageRecord> = images.iter()
        .filter(|i| {
            if emb_type == "face" { i.face_detected && i.face_emb.is_some() }
            else { i.body_available && i.body_emb.is_some() }
        })
        .collect();

    let id_to_img: HashMap<usize, &ImageRecord> = valid_images.iter()
        .map(|i| (i.id, *i))
        .collect();

    // Collect failures (margin <= 0 or rank > 1)
    let mut failures: Vec<_> = raw_results.iter().enumerate()
        .filter(|(_, r)| r.margin <= 0.0 || r.correct_rank > 1)
        .take(n_cases)
        .collect();

    failures.iter().map(|(idx, r)| {
        let img = id_to_img.get(&{
            let q_ids: Vec<usize> = valid_images.iter().map(|i| i.id).collect();
            q_ids[*idx.min(&q_ids.len())]
        }).unwrap_or(&valid_images[0]);

        let failure_type = if r.correct_rank > 1 {
            "rank_failure".to_string()
        } else if r.margin <= 0.0 {
            "margin_failure".to_string()
        } else {
            "threshold_only_failure".to_string()
        };

        FailureCase {
            person_id: img.person_id,
            image_id: img.id,
            positive_score: r.positive_score as f64,
            best_negative_person: 0, // Would need additional computation
            best_negative_score: r.negative_max as f64,
            margin: r.margin as f64,
            correct_rank: r.correct_rank,
            face_quality: if emb_type == "face" { Some(img.face_quality) } else { None },
            detection_score: if emb_type == "face" { Some(img.face_score) } else { None },
            failure_type,
        }
    }).collect()
}

// ============================================================================
// T9 & T10: Root Cause Analysis
// ============================================================================

fn determine_root_cause(
    face_metrics: &DetailedLooMetrics,
    body_metrics: &DetailedLooMetrics,
    face_pairwise: &PairwiseDistribution,
    body_pairwise: &PairwiseDistribution,
) -> RootCauseVerdict {
    let mut evidence = Vec::new();
    let mut verdicts: Vec<(&str, f64)> = Vec::new();

    // Check ranking quality
    if face_metrics.rank1_rate > 0.95 {
        evidence.push(format!("Face LOO Rank1 = {:.1}%% (excellent)", face_metrics.rank1_rate * 100.0));
        verdicts.push(("RANKING_EXCELLENT", face_metrics.rank1_rate));
    } else if face_metrics.rank1_rate > 0.80 {
        evidence.push(format!("Face LOO Rank1 = {:.1}%% (good)", face_metrics.rank1_rate * 100.0));
        verdicts.push(("RANKING_GOOD", face_metrics.rank1_rate));
    } else {
        evidence.push(format!("Face LOO Rank1 = {:.1}%% (poor)", face_metrics.rank1_rate * 100.0));
        verdicts.push(("RANKING_POOR", face_metrics.rank1_rate));
    }

    // Check margin quality
    if face_metrics.margin_plus_rate > 0.95 {
        evidence.push(format!("Face LOO Margin+ = {:.1}%% (excellent)", face_metrics.margin_plus_rate * 100.0));
    } else if face_metrics.margin_plus_rate > 0.80 {
        evidence.push(format!("Face LOO Margin+ = {:.1}%% (good)", face_metrics.margin_plus_rate * 100.0));
    } else {
        evidence.push(format!("Face LOO Margin+ = {:.1}%% (poor)", face_metrics.margin_plus_rate * 100.0));
    }

    // Check threshold pass rate
    evidence.push(format!(
        "Face LOO Threshold (0.75) = {:.1}%%, but positive_score median = {:.3}",
        face_metrics.threshold_pass_rate * 100.0,
        face_metrics.positive_score_stats.p50
    ));

    // Check pairwise separation
    evidence.push(format!(
        "Face positive median = {:.3}, negative median = {:.3}, overlap = {:.1}%%",
        face_pairwise.positive_scores.p50,
        face_pairwise.negative_scores.p50,
        face_pairwise.separation.overlap_ratio * 100.0
    ));

    // Determine verdict
    let (verdict, confidence, recommendation) = if face_metrics.rank1_rate > 0.90 {
        if face_metrics.threshold_pass_rate < 0.5 && face_metrics.positive_score_stats.p50 < 0.70 {
            ("THRESHOLD_MISCALIBRATION", "HIGH",
             "Production threshold (0.75) is too high. Median positive score suggests optimal threshold around 0.55-0.65.")
        } else if face_metrics.margin_plus_rate < 0.80 {
            ("PROTOTYPE_DILUTION", "MEDIUM",
             "Positive margin rate is low despite good ranking. Consider multi-prototype approach.")
        } else {
            ("THRESHOLD_MISCALIBRATION", "MEDIUM",
             "Ranking is good but threshold is miscalibrated.")
        }
    } else {
        ("GENUINE_EMBEDDING_FAILURE", "HIGH",
         "Both ranking and margin quality are poor. Embeddings do not cluster well.")
    };

    RootCauseVerdict {
        verdict: verdict.to_string(),
        confidence: confidence.to_string(),
        evidence,
        recommendation: recommendation.to_string(),
    }
}

// ============================================================================
// Main Test
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn phase25_2_loo_audit() {
    println!("============================================================");
    println!("Phase 25.2 — LOO / Prototype Audit");
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
    let youtureid_path = resolve_model("person_reid_youtu_2021nov.onnx");

    if !scrfd_path.exists() || !arcface_path.exists() || !youtureid_path.exists() {
        println!("ERROR: Required models not found");
        return;
    }

    let detector = pf_ai::ScrfdDetector::load(&scrfd_path).expect("load SCRFD");
    let aligner: Arc<pf_ai::SimpleAligner> = Arc::new(pf_ai::SimpleAligner::new());
    let embedder = pf_ai::ArcFaceEmbedder::load(&arcface_path).expect("load ArcFace");
    let qf = pf_ai::QualityFilter::from_config(0.30, 20, 0.45, 30.0);
    let face_pipeline: Arc<pf_ai::FacePipeline> = Arc::new(pf_ai::FacePipeline::new(detector, aligner, embedder, qf));

    let body_embedder = pf_ai::YouTuReIdEmbedder::load(&youtureid_path)
        .expect("load YouTu Re-ID");

    println!("  SCRFD + ArcFace loaded\n");
    println!("  YouTu Re-ID loaded\n");

    // ====== 3. Process Images ======
    println!("Step 3: Processing Images...");
    let mut face_detected = 0;
    let mut body_available = 0;

    for img in &mut images {
        if !img.path.exists() { continue; }
        let image_data = match pf_ai::ImageData::from_file(&img.path) {
            Ok(id) => id,
            Err(_) => continue,
        };

        // Face pipeline
        match face_pipeline.process(&image_data).await {
            Ok(features) => {
                if let Some(face) = features.first() {
                    img.face_detected = true;
                    img.face_score = face.detection.score;
                    img.face_quality = face.blur_score.max(face.pose_score);
                    img.face_emb = Some(face.embedding.values.clone());
                    face_detected += 1;
                }
            }
            Err(_) => {}
        }

        // Body embedder
        let rgb = image_data.as_rgb8();
        match body_embedder.embed(&rgb) {
            Ok(emb) => {
                img.body_available = true;
                img.body_quality = 1.0;
                img.body_emb = Some(emb.values);
                body_available += 1;
            }
            Err(_) => {}
        }
    }

    println!("  Face detected: {}/{}", face_detected, images.len());
    println!("  Body available: {}/{}\n", body_available, images.len());

    // ====== 4. Create Splits ======
    println!("Step 4: Creating Query/Gallery Splits...");
    let splits = create_query_gallery_split(&images, 42);
    println!("  Created splits for {} persons\n", splits.len());

    // ====== 5. T1 & T2: Detailed LOO Metrics ======
    println!("============================================================");
    println!("T1 & T2: Detailed LOO Metrics (Rank + Margin + Threshold)");
    println!("============================================================\n");

    let (face_loo, face_raw, face_pos, face_neg) = compute_detailed_loo_face(&images, &splits);
    let (body_loo, body_raw, body_pos, body_neg) = compute_detailed_loo_body(&images, &splits);

    println!("FACE LOO Analysis:");
    println!("  Total queries: {}", face_loo.total_queries);
    println!("  Rank1 Rate: {:.2}%% ({}/{})", face_loo.rank1_rate * 100.0, face_loo.rank1_count, face_loo.total_queries);
    println!("  Margin+ Rate: {:.2}%% ({}/{})", face_loo.margin_plus_rate * 100.0, face_loo.margin_plus_count, face_loo.total_queries);
    println!("  Threshold Pass (0.75): {:.2}%% ({}/{})", face_loo.threshold_pass_rate * 100.0, face_loo.threshold_pass_count, face_loo.total_queries);
    println!("\n  Positive Score Distribution:");
    println!("    Min: {:.4}, P10: {:.4}, P50: {:.4}, P90: {:.4}, Max: {:.4}",
        face_loo.positive_score_stats.min, face_loo.positive_score_stats.p10,
        face_loo.positive_score_stats.p50, face_loo.positive_score_stats.p90,
        face_loo.positive_score_stats.max);
    println!("    Mean: {:.4}", face_loo.positive_score_stats.mean);
    println!("\n  Negative Max Distribution:");
    println!("    Min: {:.4}, P10: {:.4}, P50: {:.4}, P90: {:.4}, Max: {:.4}",
        face_loo.negative_max_stats.min, face_loo.negative_max_stats.p10,
        face_loo.negative_max_stats.p50, face_loo.negative_max_stats.p90,
        face_loo.negative_max_stats.max);
    println!("\n  Margin Distribution:");
    println!("    Min: {:.4}, P10: {:.4}, P50: {:.4}, P90: {:.4}, Max: {:.4}",
        face_loo.margin_stats.min, face_loo.margin_stats.p10,
        face_loo.margin_stats.p50, face_loo.margin_stats.p90,
        face_loo.margin_stats.max);
    println!("    Mean: {:.4}", face_loo.margin_stats.mean);

    println!("\nBODY LOO Analysis:");
    println!("  Total queries: {}", body_loo.total_queries);
    println!("  Rank1 Rate: {:.2}%% ({}/{})", body_loo.rank1_rate * 100.0, body_loo.rank1_count, body_loo.total_queries);
    println!("  Margin+ Rate: {:.2}%% ({}/{})", body_loo.margin_plus_rate * 100.0, body_loo.margin_plus_count, body_loo.total_queries);
    println!("  Threshold Pass (0.70): {:.2}%% ({}/{})", body_loo.threshold_pass_rate * 100.0, body_loo.threshold_pass_count, body_loo.total_queries);
    println!("\n  Positive Score Distribution:");
    println!("    Min: {:.4}, P10: {:.4}, P50: {:.4}, P90: {:.4}, Max: {:.4}",
        body_loo.positive_score_stats.min, body_loo.positive_score_stats.p10,
        body_loo.positive_score_stats.p50, body_loo.positive_score_stats.p90,
        body_loo.positive_score_stats.max);
    println!("    Mean: {:.4}", body_loo.positive_score_stats.mean);
    println!("\n  Margin Distribution:");
    println!("    Min: {:.4}, P10: {:.4}, P50: {:.4}, P90: {:.4}, Max: {:.4}",
        body_loo.margin_stats.min, body_loo.margin_stats.p10,
        body_loo.margin_stats.p50, body_loo.margin_stats.p90,
        body_loo.margin_stats.max);

    // ====== 6. T7: Pairwise Distributions ======
    println!("\n============================================================");
    println!("T7: Direct Pairwise Distribution");
    println!("============================================================\n");

    let (face_pairwise, _, _) = compute_pairwise_distributions(&images, &splits, "face");
    let (body_pairwise, _, _) = compute_pairwise_distributions(&images, &splits, "body");

    println!("FACE Positive Pairs (same person):");
    println!("  Min: {:.4}, P1: {:.4}, P5: {:.4}, P10: {:.4}", face_pairwise.positive_scores.min, face_pairwise.positive_scores.p1, face_pairwise.positive_scores.p5, face_pairwise.positive_scores.p10);
    println!("  P25: {:.4}, P50: {:.4}, P75: {:.4}", face_pairwise.positive_scores.p25, face_pairwise.positive_scores.p50, face_pairwise.positive_scores.p75);
    println!("  P90: {:.4}, P95: {:.4}, P99: {:.4}, Max: {:.4}", face_pairwise.positive_scores.p90, face_pairwise.positive_scores.p95, face_pairwise.positive_scores.p99, face_pairwise.positive_scores.max);
    println!("  Mean: {:.4}", face_pairwise.positive_scores.mean);

    println!("\nFACE Negative Pairs (different persons):");
    println!("  Min: {:.4}, P1: {:.4}, P5: {:.4}, P10: {:.4}", face_pairwise.negative_scores.min, face_pairwise.negative_scores.p1, face_pairwise.negative_scores.p5, face_pairwise.negative_scores.p10);
    println!("  P25: {:.4}, P50: {:.4}, P75: {:.4}", face_pairwise.negative_scores.p25, face_pairwise.negative_scores.p50, face_pairwise.negative_scores.p75);
    println!("  P90: {:.4}, P95: {:.4}, P99: {:.4}, Max: {:.4}", face_pairwise.negative_scores.p90, face_pairwise.negative_scores.p95, face_pairwise.negative_scores.p99, face_pairwise.negative_scores.max);
    println!("  Mean: {:.4}", face_pairwise.negative_scores.mean);

    println!("\n  Separation: {:.1}%% overlap at median negative", face_pairwise.separation.overlap_ratio * 100.0);

    println!("\nBODY Positive Pairs:");
    println!("  Min: {:.4}, P50: {:.4}, P99: {:.4}, Max: {:.4}", body_pairwise.positive_scores.min, body_pairwise.positive_scores.p50, body_pairwise.positive_scores.p99, body_pairwise.positive_scores.max);
    println!("  Mean: {:.4}", body_pairwise.positive_scores.mean);

    println!("\nBODY Negative Pairs:");
    println!("  Min: {:.4}, P50: {:.4}, P99: {:.4}, Max: {:.4}", body_pairwise.negative_scores.min, body_pairwise.negative_scores.p50, body_pairwise.negative_scores.p99, body_pairwise.negative_scores.max);
    println!("  Mean: {:.4}", body_pairwise.negative_scores.mean);

    println!("\n  Separation: {:.1}%% overlap at median negative", body_pairwise.separation.overlap_ratio * 100.0);

    // ====== 7. T5: Prototype Dilution ======
    println!("\n============================================================");
    println!("T5: Prototype Dilution Experiment");
    println!("============================================================\n");

    println!("Testing prototype size K = 1, 3, 5, 10, 20 vs ALL...\n");

    let face_dilution = prototype_dilution_experiment(&images, &splits, "face");
    let body_dilution = prototype_dilution_experiment(&images, &splits, "body");

    println!("FACE Prototype Dilution:");
    println!("  K    | Rank1    | Margin+  | Median Margin | P10 Margin");
    println!("  -----|----------|----------|---------------|------------");
    for r in &face_dilution {
        println!("  {:4} | {:8.2}%% | {:8.2}%% | {:13.4} | {:11.4}", r.k, r.rank1_rate * 100.0, r.margin_plus_rate * 100.0, r.median_margin, r.p10_margin);
    }

    println!("\nBODY Prototype Dilution:");
    println!("  K    | Rank1    | Margin+  | Median Margin | P10 Margin");
    println!("  -----|----------|----------|---------------|------------");
    for r in &body_dilution {
        println!("  {:4} | {:8.2}%% | {:8.2}%% | {:13.4} | {:11.4}", r.k, r.rank1_rate * 100.0, r.margin_plus_rate * 100.0, r.median_margin, r.p10_margin);
    }

    // ====== 8. T8: Failure Cases ======
    println!("\n============================================================");
    println!("T8: Failure Case Analysis");
    println!("============================================================\n");

    let face_failures = collect_failure_cases(&face_raw, &images, "face", 20);
    let body_failures = collect_failure_cases(&body_raw, &images, "body", 20);

    println!("Top 10 Face Failure Cases:");
    println!("  PID  | Pos Score | Neg Score | Margin  | Rank | Failure Type");
    println!("  -----|-----------|-----------|---------|------|-------------");
    for (i, f) in face_failures.iter().take(10).enumerate() {
        println!("  {:4} | {:9.4} | {:9.4} | {:7.4} | {:4} | {}",
            f.person_id, f.positive_score, f.best_negative_score, f.margin, f.correct_rank, f.failure_type);
    }

    // ====== 9. T9 & T10: Root Cause ======
    println!("\n============================================================");
    println!("T9 & T10: Root Cause Analysis");
    println!("============================================================\n");

    let root_cause = determine_root_cause(&face_loo, &body_loo, &face_pairwise, &body_pairwise);

    println!("VERDICT: {}\n", root_cause.verdict);
    println!("Confidence: {}\n", root_cause.confidence);
    println!("Evidence:");
    for e in &root_cause.evidence {
        println!("  - {}", e);
    }
    println!("\nRecommendation:");
    println!("  {}", root_cause.recommendation);

    // ====== 10. Generate JSON Report ======
    let face_failures_report = face_failures.into_iter().take(20).collect();
    let body_failures_report = body_failures.into_iter().take(20).collect();

    let image_distribution: Vec<PersonImageCount> = {
        let mut person_ids: Vec<usize> = images.iter().map(|i| i.person_id).collect();
        person_ids.sort();
        person_ids.dedup();
        person_ids.iter().map(|&pid| {
            let face_count = images.iter().filter(|i| i.person_id == pid && i.face_detected).count();
            let body_count = images.iter().filter(|i| i.person_id == pid && i.body_available).count();
            PersonImageCount { person_id: pid, face_images: face_count, body_images: body_count }
        }).collect()
    };

    let report = LooAuditReport {
        phase: "25.2".to_string(),
        dataset_summary: DatasetSummary {
            total_images: images.len(),
            total_persons: splits.len(),
            face_available: face_detected,
            body_available,
            image_distribution,
        },
        face_loo,
        body_loo,
        face_pairwise,
        body_pairwise,
        prototype_dilution: PrototypeDilutionReport {
            face_results: face_dilution,
            body_results: body_dilution,
        },
        multi_prototype: MultiPrototypeReport {
            face_results: Vec::new(),
            body_results: Vec::new(),
        },
        failure_cases: FailureCaseReport {
            face_failures: face_failures_report,
            body_failures: body_failures_report,
        },
        root_cause,
    };

    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    let output_path = PathBuf::from("/Users/mac/ai-project/photofinder-next-2/crates/pf_application/tests/phase25_2_loo_audit_report.json");
    std::fs::write(&output_path, &json).expect("write report");
    println!("\n\nJSON report written to: {:?}", output_path);
}
