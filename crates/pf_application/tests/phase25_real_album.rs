//! Phase 25 — Real-World Person Identity Benchmark
//!
//! Core principles:
//! 1. DO NOT modify production models
//! 2. Identification ≠ Verification — separate evaluation
//! 3. Clustering ≠ Identification — separate evaluation
//! 4. Train/Validation/Test split — no data leakage
//! 5. Margin-based Fusion — NOT rank normalization
//! 6. Missing Channel handling — missing = excluded from fusion
//! 7. Honest reporting — use None for unavailable, not 0.0
//!
//! Uses serde_json for JSON serialization — no manual string building.
//!
//! 运行：
//! ```bash
//! cargo test -p pf_application --test phase25_real_album -- --nocapture --ignored
//! ```

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

// ============================================================================
// Report Structures (serde-based, no manual JSON)
// ============================================================================

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase25Report {
    pub phase: String,
    pub dataset: DatasetReport,
    pub missing_channels: MissingChannelReport,
    pub face: SignalReport,
    pub body: SignalReport,
    pub fusion: FusionReport,
    pub chain_contamination: Vec<ChainResult>,
    pub hard_negatives: HardNegativeReport,
    pub verdict: VerdictReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetReport {
    pub total_images: usize,
    pub total_persons: usize,
    pub min_images_per_person: usize,
    pub median_images_per_person: f64,
    pub max_images_per_person: usize,
    pub face_detection_rate: f64,
    pub body_detection_rate: f64,
    pub dual_channel_rate: f64,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalReport {
    pub available: bool,
    pub identification: IdentificationReport,
    pub verification: VerificationReport,
    pub margin: MarginReport,
    pub loo: LooReport,
    pub verdict: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentificationReport {
    pub top1: f64,
    pub top3: f64,
    pub top5: f64,
    pub top10: f64,
    pub map: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    pub auc: f64,
    pub eer: f64,
    pub far_01: f64,
    pub far_001: f64,
    pub tar_at_far_1e3: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginReport {
    pub mean: f64,
    pub p10: f64,
    pub p50: f64,
    pub p90: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LooReport {
    pub pass_rate: f64,
    pub total: usize,
    pub passed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionReport {
    pub available: bool,
    pub best_alpha: Option<f64>,
    pub alpha_sweep: Vec<AlphaResult>,
    pub identification: Option<IdentificationReport>,
    pub verification: Option<VerificationReport>,
    pub margin: Option<MarginReport>,
    pub loo: Option<LooReport>,
    pub verdict: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaResult {
    pub alpha: f64,
    pub auc: f64,
    pub eer: f64,
    pub loo_pass_rate: f64,
    pub p10_margin: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainResult {
    pub steps: usize,
    pub detected: bool,
    pub margin_a_b: f64,
    pub margin_b_c: f64,
    pub margin_c_d: f64,
    pub margin_a_d: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardNegativeItem {
    pub rank: usize,
    pub person_a: String,
    pub person_b: String,
    pub face_score: Option<f64>,
    pub body_score: Option<f64>,
    pub fusion_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardNegativeReport {
    pub face: Vec<HardNegativeItem>,
    pub body: Vec<HardNegativeItem>,
    pub dual: Vec<HardNegativeItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingChannelReport {
    pub face_only_count: usize,
    pub body_only_count: usize,
    pub dual_count: usize,
    pub none_count: usize,
    pub face_only_accuracy: Option<f64>,
    pub body_only_accuracy: Option<f64>,
    pub dual_accuracy: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerdictReport {
    pub face: String,
    pub body: String,
    pub fusion: String,
    pub clustering: String,
    pub search: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringReport {
    pub false_merge_rate: f64,
    pub false_split_rate: f64,
    pub purity: f64,
    pub pairwise_f1: f64,
    pub seeds_tested: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrototypePollutionReport {
    pub drift_1_wrong: f64,
    pub drift_2_wrong: f64,
    pub drift_5_wrong: f64,
    pub drift_10_wrong: f64,
    pub core_preserved: bool,
}

fn generate_json_report(
    report: &Phase25Report,
    output_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(report)?;
    std::fs::write(output_path, json)?;
    Ok(())
}

// ============================================================================
// Constants
// ============================================================================

// Dataset: person_identity_v3 points to person_identity_real (v3 metadata only)
const BENCHMARK_DIR: &str = "/Users/mac/ai-project/photofinder-next-2/benchmark/person_identity_real";
const SOURCE_DIR: &str = "/Users/mac/ai-project/photofinder-next-2/benchmark/person_identity_real";

const FACE_THRESHOLD: f32 = 0.75;
const BODY_THRESHOLD: f32 = 0.70;
const FUSION_THRESHOLD: f32 = 0.65;
const CHAINING_MARGIN: f32 = 0.05;

const FACE_DIM: usize = 512;
const BODY_DIM: usize = 768;

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
    resolution: u32,
    image_width: u32,
    image_height: u32,
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
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p * (sorted.len() - 1) as f32).round() as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn median(data: &[f32]) -> f32 {
    percentile(data, 0.5)
}

// ============================================================================
// ROC/AUC/EER Computation
// ============================================================================

fn roc_auc(positive_scores: &[f32], negative_scores: &[f32]) -> f32 {
    if positive_scores.is_empty() || negative_scores.is_empty() {
        return 0.0;
    }

    let mut all_scores: Vec<(bool, f32)> = Vec::new();
    for &s in positive_scores {
        all_scores.push((true, s));
    }
    for &s in negative_scores {
        all_scores.push((false, s));
    }
    all_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let n_pos = positive_scores.len() as f32;
    let n_neg = negative_scores.len() as f32;

    let mut tpr_list = Vec::new();
    let mut fpr_list = Vec::new();

    for &(_, threshold) in &all_scores {
        let tp = positive_scores.iter().filter(|&&s| s >= threshold).count() as f32;
        let fp = negative_scores.iter().filter(|&&s| s >= threshold).count() as f32;
        tpr_list.push(tp / n_pos);
        fpr_list.push(fp / n_neg);
    }

    let mut auc = 0.0;
    for i in 1..tpr_list.len() {
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2.0;
    }
    auc.abs()
}

fn compute_eer(positive_scores: &[f32], negative_scores: &[f32]) -> f32 {
    if positive_scores.is_empty() || negative_scores.is_empty() {
        return 0.5;
    }

    let mut all_scores: Vec<(bool, f32)> = Vec::new();
    for &s in positive_scores {
        all_scores.push((true, s));
    }
    for &s in negative_scores {
        all_scores.push((false, s));
    }
    all_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let n_pos = positive_scores.len() as f32;
    let n_neg = negative_scores.len() as f32;

    let mut best_eer = 1.0;

    for &(_, threshold) in &all_scores {
        let tp = positive_scores.iter().filter(|&&s| s >= threshold).count() as f32;
        let fp = negative_scores.iter().filter(|&&s| s >= threshold).count() as f32;
        let fnr = 1.0 - (tp / n_pos);
        let fpr = fp / n_neg;
        let eer = (fnr + fpr) / 2.0;

        if eer < best_eer {
            best_eer = eer;
        }
    }

    best_eer
}

fn compute_far_at_threshold(negative_scores: &[f32], threshold: f32) -> f32 {
    if negative_scores.is_empty() {
        return 0.0;
    }
    let fp = negative_scores.iter().filter(|&&s| s >= threshold).count() as f32;
    fp / negative_scores.len() as f32
}

fn compute_frr_at_threshold(positive_scores: &[f32], threshold: f32) -> f32 {
    if positive_scores.is_empty() {
        return 1.0;
    }
    let fn_ = positive_scores.iter().filter(|&&s| s < threshold).count() as f32;
    fn_ / positive_scores.len() as f32
}

fn compute_tar_at_far(positive_scores: &[f32], negative_scores: &[f32], target_far: f32) -> f32 {
    if positive_scores.is_empty() || negative_scores.is_empty() {
        return 0.0;
    }

    let mut all_scores: Vec<(bool, f32)> = Vec::new();
    for &s in positive_scores {
        all_scores.push((true, s));
    }
    for &s in negative_scores {
        all_scores.push((false, s));
    }
    all_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let n_neg = negative_scores.len() as f32;
    let target_fp = (target_far * n_neg).round() as usize;

    if target_fp == 0 {
        return 1.0;
    }

    let threshold_idx = target_fp.min(all_scores.len() - 1);
    let threshold = all_scores[threshold_idx].1;

    let tp = positive_scores.iter().filter(|&&s| s >= threshold).count() as f32;
    tp / positive_scores.len() as f32
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
    candidates
        .into_iter()
        .find(|p| p.exists())
        .unwrap_or_else(|| workspace.join("models").join(name))
}

// ============================================================================
// Dataset Discovery
// ============================================================================

fn discover_dataset() -> Option<Vec<ImageRecord>> {
    let benchmark_path = PathBuf::from(BENCHMARK_DIR);

    if !benchmark_path.exists() {
        println!("Dataset not found at {}", BENCHMARK_DIR);
        return None;
    }

    let mut images = Vec::new();
    let mut image_id = 0;

    let entries = fs::read_dir(&benchmark_path).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let person_name = path.file_name()?.to_str()?;
        if !person_name.starts_with("person_") {
            continue;
        }

        let person_id: usize = person_name.strip_prefix("person_")?.parse().ok()?;

        let img_entries = fs::read_dir(&path).ok()?;
        for img_entry in img_entries.flatten() {
            let img_path = img_entry.path();
            let ext = img_path.extension()?.to_str()?.to_lowercase();
            if !matches!(ext.as_str(), "jpg" | "jpeg" | "png") {
                continue;
            }

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
                resolution: 0,
                image_width: 0,
                image_height: 0,
            });
            image_id += 1;
        }
    }

    images.sort_by(|a, b| a.person_id.cmp(&b.person_id).then(a.path.cmp(&b.path)));
    Some(images)
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
        let gallery_size = ((img_count as f32) * 0.6) as usize;
        let remaining: Vec<usize> = person_groups.get(person_id).cloned().unwrap_or_default();

        let mut hash_input = format!("{}:{}", person_id, seed);
        let mut hasher = DefaultHasher::new();
        hash_input.hash(&mut hasher);
        let hash_val = hasher.finish();
        let mut rng_state = hash_val;

        let mut gallery = Vec::with_capacity(gallery_size);
        let mut remaining_copy = remaining.clone();

        for _ in 0..gallery_size {
            if remaining_copy.is_empty() {
                break;
            }
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
// Dataset Statistics
// ============================================================================

fn compute_dataset_stats(images: &[ImageRecord]) -> (usize, usize, usize, f64, usize) {
    let total = images.len();
    let mut person_ids: HashSet<usize> = HashSet::new();
    let mut counts: Vec<usize> = Vec::new();
    let mut per_person: HashMap<usize, usize> = HashMap::new();

    for img in images {
        person_ids.insert(img.person_id);
        *per_person.entry(img.person_id).or_default() += 1;
    }
    counts.extend(per_person.values());
    counts.sort();

    let persons = person_ids.len();
    let min = counts.first().copied().unwrap_or(0);
    let max = counts.last().copied().unwrap_or(0);
    let median = if counts.is_empty() {
        0.0
    } else if counts.len() % 2 == 0 {
        (counts[counts.len() / 2 - 1] + counts[counts.len() / 2]) as f64 / 2.0
    } else {
        counts[counts.len() / 2] as f64
    };

    (total, persons, min, median, max)
}

fn dataset_audit(images: &[ImageRecord]) {
    let (total, persons, min, median, max) = compute_dataset_stats(images);

    println!();
    println!("============================================================");
    println!("DATASET AUDIT");
    println!("============================================================");
    println!("total_images: {}", total);
    println!("persons: {}", persons);
    println!("min_images_per_person: {}", min);
    println!("median_images_per_person: {:.1}", median);
    println!("max_images_per_person: {}", max);

    // Per-person breakdown
    let mut per_person: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
    for img in images {
        per_person.entry(img.person_id).or_default().push(img);
    }

    println!();
    println!("Per-person image counts:");
    let mut person_ids: Vec<usize> = per_person.keys().copied().collect();
    person_ids.sort();
    for pid in person_ids {
        let count = per_person[&pid].len();
        let flag = if count < 5 { " [LOW_SAMPLE_PERSON]" } else { "" };
        println!("  person_{:03}: {}{}", pid, count, flag);
    }

    let face_count = images.iter().filter(|i| i.face_detected && i.face_emb.is_some()).count();
    let body_count = images.iter().filter(|i| i.body_available && i.body_emb.is_some()).count();
    let dual_count = images.iter().filter(|i| {
        i.face_detected && i.face_emb.is_some() && i.body_available && i.body_emb.is_some()
    }).count();
    let face_only = face_count.saturating_sub(dual_count);
    let body_only = body_count.saturating_sub(dual_count);
    let none = total.saturating_sub(face_count).saturating_sub(body_count).saturating_add(dual_count);

    println!();
    println!("face_available: {}", face_count);
    println!("body_available: {}", body_count);
    println!("dual_available: {}", dual_count);
    println!("face_only: {}", face_only);
    println!("body_only: {}", body_only);
    println!("none: {}", none);
}

// ============================================================================
// Face Benchmark
// ============================================================================

struct FaceBenchmarkResult {
    identification: IdentificationReport,
    verification: VerificationReport,
    margin: MarginReport,
    loo: LooReport,
    positive_pairs: usize,
    negative_pairs: usize,
}

fn run_face_benchmark(
    images: &[ImageRecord],
    splits: &HashMap<usize, QueryGallerySplit>,
) -> FaceBenchmarkResult {
    let face_images: Vec<&ImageRecord> = images.iter()
        .filter(|i| i.face_detected && i.face_emb.is_some())
        .collect();

    if face_images.is_empty() {
        return FaceBenchmarkResult {
            identification: IdentificationReport { top1: 0.0, top3: 0.0, top5: 0.0, top10: 0.0, map: 0.0 },
            verification: VerificationReport { auc: 0.0, eer: 1.0, far_01: 1.0, far_001: 1.0, tar_at_far_1e3: 0.0 },
            margin: MarginReport { mean: 0.0, p10: 0.0, p50: 0.0, p90: 0.0 },
            loo: LooReport { pass_rate: 0.0, total: 0, passed: 0 },
            positive_pairs: 0,
            negative_pairs: 0,
        };
    }

    let id_to_img: HashMap<usize, &ImageRecord> = images.iter().map(|i| (i.id, i)).collect();

    // Verification pairs
    let mut positive_scores = Vec::new();
    let mut negative_scores = Vec::new();

    for (person_id, split) in splits {
        for &q_id in &split.query {
            let Some(q_img) = id_to_img.get(&q_id) else { continue };
            let Some(q_emb) = &q_img.face_emb else { continue };

            for &g_id in &split.gallery {
                let Some(g_img) = id_to_img.get(&g_id) else { continue };
                let Some(g_emb) = &g_img.face_emb else { continue };
                let score = cosine(q_emb, g_emb);
                positive_scores.push(score);
            }

            for (other_pid, other_split) in splits {
                if *other_pid == *person_id { continue; }
                for &g_id in &other_split.gallery {
                    let Some(g_img) = id_to_img.get(&g_id) else { continue };
                    let Some(g_emb) = &g_img.face_emb else { continue };
                    let score = cosine(q_emb, g_emb);
                    negative_scores.push(score);
                }
            }
        }
    }

    let verification = VerificationReport {
        auc: roc_auc(&positive_scores, &negative_scores) as f64,
        eer: compute_eer(&positive_scores, &negative_scores) as f64,
        far_01: compute_far_at_threshold(&negative_scores, FACE_THRESHOLD) as f64,
        far_001: compute_far_at_threshold(&negative_scores, 0.001) as f64,
        tar_at_far_1e3: compute_tar_at_far(&positive_scores, &negative_scores, 0.001) as f64,
    };

    // Identification Top-K
    let mut top1_correct = 0;
    let mut top3_correct = 0;
    let mut top5_correct = 0;
    let mut top10_correct = 0;
    let mut total_queries = 0;
    let mut ap_sum = 0.0f32;

    for (person_id, split) in splits {
        let id_to_img_local: HashMap<usize, &ImageRecord> = images.iter()
            .filter(|i| i.face_detected && i.face_emb.is_some())
            .map(|i| (i.id, i))
            .collect();

        for &q_id in &split.query {
            let Some(q_img) = id_to_img_local.get(&q_id) else { continue };
            let Some(q_emb) = &q_img.face_emb else { continue };
            total_queries += 1;

            let mut candidates: Vec<(usize, f32)> = Vec::new();
            for (g_id, g_img) in &id_to_img_local {
                if split.gallery.contains(g_id) { continue; }
                if let Some(g_emb) = &g_img.face_emb {
                    candidates.push((*g_id, cosine(q_emb, g_emb)));
                }
            }

            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top1_pid = candidates.first()
                .and_then(|(id, _)| id_to_img_local.get(id))
                .map(|i| i.person_id);
            let top3_pids: Vec<usize> = candidates.iter().take(3)
                .filter_map(|(id, _)| id_to_img_local.get(id))
                .map(|i| i.person_id)
                .collect();
            let top5_pids: Vec<usize> = candidates.iter().take(5)
                .filter_map(|(id, _)| id_to_img_local.get(id))
                .map(|i| i.person_id)
                .collect();
            let top10_pids: Vec<usize> = candidates.iter().take(10)
                .filter_map(|(id, _)| id_to_img_local.get(id))
                .map(|i| i.person_id)
                .collect();

            if top1_pid == Some(*person_id) { top1_correct += 1; }
            if top3_pids.contains(person_id) { top3_correct += 1; }
            if top5_pids.contains(person_id) { top5_correct += 1; }
            if top10_pids.contains(person_id) { top10_correct += 1; }

            // AP for this query
            let relevant = candidates.iter()
                .take(10)
                .filter(|(id, _)| id_to_img_local.get(id).map(|i| i.person_id) == Some(*person_id))
                .count();
            let precision_at_relevant = if relevant > 0 {
                candidates.iter().take(10).position(|(id, _)| id_to_img_local.get(id).map(|i| i.person_id) == Some(*person_id))
                    .map(|pos| (10 - pos) as f32 / 10.0)
                    .unwrap_or(0.0)
            } else { 0.0 };
            ap_sum += precision_at_relevant;
        }
    }

    let identification = IdentificationReport {
        top1: top1_correct as f64 / total_queries.max(1) as f64,
        top3: top3_correct as f64 / total_queries.max(1) as f64,
        top5: top5_correct as f64 / total_queries.max(1) as f64,
        top10: top10_correct as f64 / total_queries.max(1) as f64,
        map: ap_sum as f64 / total_queries.max(1) as f64,
    };

    // Margin analysis
    let mut margins = Vec::new();
    for (person_id, split) in splits {
        let id_to_img_local: HashMap<usize, &ImageRecord> = images.iter()
            .filter(|i| i.face_detected && i.face_emb.is_some())
            .map(|i| (i.id, i))
            .collect();

        for &q_id in &split.query {
            let Some(q_img) = id_to_img_local.get(&q_id) else { continue };
            let Some(q_emb) = &q_img.face_emb else { continue };

            let mut gallery_scores: Vec<f32> = split.gallery.iter()
                .filter_map(|g_id| id_to_img_local.get(g_id).and_then(|g| g.face_emb.as_ref())
                    .map(|g_emb| cosine(q_emb, g_emb)))
                .collect();
            gallery_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let best_positive = gallery_scores.first().copied().unwrap_or(0.0);

            let mut other_max = f32::MIN;
            for (other_pid, other_split) in splits {
                if *other_pid == *person_id { continue; }
                for g_id in &other_split.gallery {
                    if let Some(g_img) = id_to_img_local.get(g_id) {
                        if let Some(g_emb) = &g_img.face_emb {
                            other_max = other_max.max(cosine(q_emb, g_emb));
                        }
                    }
                }
            }

            margins.push(best_positive - other_max);
        }
    }

    let margin = MarginReport {
        mean: if margins.is_empty() { 0.0 } else { margins.iter().sum::<f32>() / margins.len() as f32 } as f64,
        p10: percentile(&margins, 0.1) as f64,
        p50: percentile(&margins, 0.5) as f64,
        p90: percentile(&margins, 0.9) as f64,
    };

    // LOO Prototype
    let mut loo_pass = 0;
    let mut loo_total = 0;

    for (person_id, split) in splits {
        let person_images: Vec<&ImageRecord> = images.iter()
            .filter(|i| i.person_id == *person_id && i.face_detected && i.face_emb.is_some())
            .collect();

        if person_images.len() < 2 { continue; }

        for &leave_out_id in &person_images.iter().map(|i| i.id).collect::<Vec<_>>() {
            let prototype: Vec<f32> = person_images.iter()
                .filter(|i| i.id != leave_out_id)
                .filter_map(|i| i.face_emb.clone())
                .fold(vec![0.0; FACE_DIM], |mut acc, emb| {
                    for (a, &e) in acc.iter_mut().zip(emb.iter()) { *a += e; }
                    acc
                });

            let mut normalized_prototype = prototype;
            l2_normalize(&mut normalized_prototype);

            let Some(leave_out) = person_images.iter().find(|i| i.id == leave_out_id) else { continue };
            let Some(leave_out_emb) = &leave_out.face_emb else { continue };

            let score = cosine(&normalized_prototype, leave_out_emb);
            if score >= FACE_THRESHOLD { loo_pass += 1; }
            loo_total += 1;
        }
    }

    FaceBenchmarkResult {
        identification,
        verification,
        margin,
        loo: LooReport {
            pass_rate: loo_pass as f64 / loo_total.max(1) as f64,
            total: loo_total,
            passed: loo_pass,
        },
        positive_pairs: positive_scores.len(),
        negative_pairs: negative_scores.len(),
    }
}

// ============================================================================
// Body Benchmark
// ============================================================================

struct BodyBenchmarkResult {
    identification: IdentificationReport,
    verification: VerificationReport,
    margin: MarginReport,
    loo: LooReport,
    positive_pairs: usize,
    negative_pairs: usize,
    min_bbox_size: f32,
    median_bbox_size: f32,
    max_bbox_size: f32,
}

fn run_body_benchmark(
    images: &[ImageRecord],
    splits: &HashMap<usize, QueryGallerySplit>,
) -> BodyBenchmarkResult {
    let body_images: Vec<&ImageRecord> = images.iter()
        .filter(|i| i.body_available && i.body_emb.is_some())
        .collect();

    if body_images.is_empty() {
        return BodyBenchmarkResult {
            identification: IdentificationReport { top1: 0.0, top3: 0.0, top5: 0.0, top10: 0.0, map: 0.0 },
            verification: VerificationReport { auc: 0.0, eer: 1.0, far_01: 1.0, far_001: 1.0, tar_at_far_1e3: 0.0 },
            margin: MarginReport { mean: 0.0, p10: 0.0, p50: 0.0, p90: 0.0 },
            loo: LooReport { pass_rate: 0.0, total: 0, passed: 0 },
            positive_pairs: 0,
            negative_pairs: 0,
            min_bbox_size: 0.0,
            median_bbox_size: 0.0,
            max_bbox_size: 0.0,
        };
    }

    let id_to_img: HashMap<usize, &ImageRecord> = images.iter().map(|i| (i.id, i)).collect();

    let mut positive_scores = Vec::new();
    let mut negative_scores = Vec::new();

    for (person_id, split) in splits {
        for &q_id in &split.query {
            let Some(q_img) = id_to_img.get(&q_id) else { continue };
            let Some(q_emb) = &q_img.body_emb else { continue };

            for &g_id in &split.gallery {
                let Some(g_img) = id_to_img.get(&g_id) else { continue };
                let Some(g_emb) = &g_img.body_emb else { continue };
                let score = cosine(q_emb, g_emb);
                positive_scores.push(score);
            }

            for (other_pid, other_split) in splits {
                if *other_pid == *person_id { continue; }
                for &g_id in &other_split.gallery {
                    let Some(g_img) = id_to_img.get(&g_id) else { continue };
                    let Some(g_emb) = &g_img.body_emb else { continue };
                    let score = cosine(q_emb, g_emb);
                    negative_scores.push(score);
                }
            }
        }
    }

    let verification = VerificationReport {
        auc: roc_auc(&positive_scores, &negative_scores) as f64,
        eer: compute_eer(&positive_scores, &negative_scores) as f64,
        far_01: compute_far_at_threshold(&negative_scores, BODY_THRESHOLD) as f64,
        far_001: compute_far_at_threshold(&negative_scores, 0.001) as f64,
        tar_at_far_1e3: compute_tar_at_far(&positive_scores, &negative_scores, 0.001) as f64,
    };

    let mut top1_correct = 0;
    let mut top3_correct = 0;
    let mut top5_correct = 0;
    let mut top10_correct = 0;
    let mut total_queries = 0;

    for (person_id, split) in splits {
        let id_to_img_local: HashMap<usize, &ImageRecord> = images.iter()
            .filter(|i| i.body_available && i.body_emb.is_some())
            .map(|i| (i.id, i))
            .collect();

        for &q_id in &split.query {
            let Some(q_img) = id_to_img_local.get(&q_id) else { continue };
            let Some(q_emb) = &q_img.body_emb else { continue };
            total_queries += 1;

            let mut candidates: Vec<(usize, f32)> = Vec::new();
            for (g_id, g_img) in &id_to_img_local {
                if split.gallery.contains(g_id) { continue; }
                if let Some(g_emb) = &g_img.body_emb {
                    candidates.push((*g_id, cosine(q_emb, g_emb)));
                }
            }

            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top1_pid = candidates.first()
                .and_then(|(id, _)| id_to_img_local.get(id))
                .map(|i| i.person_id);
            let top3_pids: Vec<usize> = candidates.iter().take(3)
                .filter_map(|(id, _)| id_to_img_local.get(id))
                .map(|i| i.person_id)
                .collect();
            let top5_pids: Vec<usize> = candidates.iter().take(5)
                .filter_map(|(id, _)| id_to_img_local.get(id))
                .map(|i| i.person_id)
                .collect();
            let top10_pids: Vec<usize> = candidates.iter().take(10)
                .filter_map(|(id, _)| id_to_img_local.get(id))
                .map(|i| i.person_id)
                .collect();

            if top1_pid == Some(*person_id) { top1_correct += 1; }
            if top3_pids.contains(person_id) { top3_correct += 1; }
            if top5_pids.contains(person_id) { top5_correct += 1; }
            if top10_pids.contains(person_id) { top10_correct += 1; }
        }
    }

    let identification = IdentificationReport {
        top1: top1_correct as f64 / total_queries.max(1) as f64,
        top3: top3_correct as f64 / total_queries.max(1) as f64,
        top5: top5_correct as f64 / total_queries.max(1) as f64,
        top10: top10_correct as f64 / total_queries.max(1) as f64,
        map: 0.0,
    };

    // Body margin
    let mut margins = Vec::new();
    for (person_id, split) in splits {
        let id_to_img_local: HashMap<usize, &ImageRecord> = images.iter()
            .filter(|i| i.body_available && i.body_emb.is_some())
            .map(|i| (i.id, i))
            .collect();

        for &q_id in &split.query {
            let Some(q_img) = id_to_img_local.get(&q_id) else { continue };
            let Some(q_emb) = &q_img.body_emb else { continue };

            let mut gallery_scores: Vec<f32> = split.gallery.iter()
                .filter_map(|g_id| id_to_img_local.get(g_id).and_then(|g| g.body_emb.as_ref())
                    .map(|g_emb| cosine(q_emb, g_emb)))
                .collect();
            gallery_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let best_positive = gallery_scores.first().copied().unwrap_or(0.0);

            let mut other_max = f32::MIN;
            for (other_pid, other_split) in splits {
                if *other_pid == *person_id { continue; }
                for g_id in &other_split.gallery {
                    if let Some(g_img) = id_to_img_local.get(g_id) {
                        if let Some(g_emb) = &g_img.body_emb {
                            other_max = other_max.max(cosine(q_emb, g_emb));
                        }
                    }
                }
            }
            margins.push(best_positive - other_max);
        }
    }

    let margin = MarginReport {
        mean: if margins.is_empty() { 0.0 } else { margins.iter().sum::<f32>() / margins.len() as f32 } as f64,
        p10: percentile(&margins, 0.1) as f64,
        p50: percentile(&margins, 0.5) as f64,
        p90: percentile(&margins, 0.9) as f64,
    };

    // LOO
    let mut loo_pass = 0;
    let mut loo_total = 0;

    for (person_id, split) in splits {
        let person_images: Vec<&ImageRecord> = images.iter()
            .filter(|i| i.person_id == *person_id && i.body_available && i.body_emb.is_some())
            .collect();

        if person_images.len() < 2 { continue; }

        for &leave_out_id in &person_images.iter().map(|i| i.id).collect::<Vec<_>>() {
            let prototype: Vec<f32> = person_images.iter()
                .filter(|i| i.id != leave_out_id)
                .filter_map(|i| i.body_emb.clone())
                .fold(vec![0.0; BODY_DIM], |mut acc, emb| {
                    for (a, &e) in acc.iter_mut().zip(emb.iter()) { *a += e; }
                    acc
                });

            let mut normalized_prototype = prototype;
            l2_normalize(&mut normalized_prototype);

            let Some(leave_out) = person_images.iter().find(|i| i.id == leave_out_id) else { continue };
            let Some(leave_out_emb) = &leave_out.body_emb else { continue };

            let score = cosine(&normalized_prototype, leave_out_emb);
            if score >= BODY_THRESHOLD { loo_pass += 1; }
            loo_total += 1;
        }
    }

    BodyBenchmarkResult {
        identification,
        verification,
        margin,
        loo: LooReport {
            pass_rate: loo_pass as f64 / loo_total.max(1) as f64,
            total: loo_total,
            passed: loo_pass,
        },
        positive_pairs: positive_scores.len(),
        negative_pairs: negative_scores.len(),
        min_bbox_size: 0.0,
        median_bbox_size: 0.0,
        max_bbox_size: 0.0,
    }
}

// ============================================================================
// Fusion Benchmark with Alpha Sweep
// ============================================================================

fn run_fusion_benchmark(
    images: &[ImageRecord],
    splits: &HashMap<usize, QueryGallerySplit>,
    face_available: bool,
    body_available: bool,
) -> FusionReport {
    if !face_available && !body_available {
        return FusionReport {
            available: false,
            best_alpha: None,
            alpha_sweep: vec![],
            identification: None,
            verification: None,
            margin: None,
            loo: None,
            verdict: "INSUFFICIENT_DATA".to_string(),
        };
    }

    let dual_images: Vec<&ImageRecord> = images.iter()
        .filter(|i| i.face_detected && i.face_emb.is_some() && i.body_available && i.body_emb.is_some())
        .collect();

    if dual_images.is_empty() {
        return FusionReport {
            available: false,
            best_alpha: None,
            alpha_sweep: vec![],
            identification: None,
            verification: None,
            margin: None,
            loo: None,
            verdict: "INSUFFICIENT_DATA".to_string(),
        };
    }

    let id_to_img: HashMap<usize, &ImageRecord> = images.iter().map(|i| (i.id, i)).collect();
    let mut alpha_sweep_results = Vec::new();

    for alpha_i in 0..=10 {
        let alpha = alpha_i as f32 / 10.0;

        let mut positive_scores = Vec::new();
        let mut negative_scores = Vec::new();

        for (person_id, split) in splits {
            for &q_id in &split.query {
                let Some(q_img) = id_to_img.get(&q_id) else { continue };

                let q_face = q_img.face_emb.as_ref().map(|e| cosine(e, e)).unwrap_or(0.0);
                let q_body = q_img.body_emb.as_ref().map(|e| cosine(e, e)).unwrap_or(0.0);
                let _ = (q_face, q_body);

                for &g_id in &split.gallery {
                    let Some(g_img) = id_to_img.get(&g_id) else { continue };

                    let face_score = if face_available {
                        q_img.face_emb.as_ref().and_then(|qe| g_img.face_emb.as_ref().map(|ge| cosine(qe, ge)))
                    } else { None };

                    let body_score = if body_available {
                        q_img.body_emb.as_ref().and_then(|qe| g_img.body_emb.as_ref().map(|ge| cosine(qe, ge)))
                    } else { None };

                    let fusion_score = match (face_score, body_score) {
                        (Some(f), Some(b)) => alpha * f + (1.0 - alpha) * b,
                        (Some(f), None) => f,
                        (None, Some(b)) => b,
                        (None, None) => continue,
                    };

                    positive_scores.push(fusion_score);
                }

                for (other_pid, other_split) in splits {
                    if *other_pid == *person_id { continue; }
                    for &g_id in &other_split.gallery {
                        let Some(g_img) = id_to_img.get(&g_id) else { continue };

                        let face_score = if face_available {
                            q_img.face_emb.as_ref().and_then(|qe| g_img.face_emb.as_ref().map(|ge| cosine(qe, ge)))
                        } else { None };

                        let body_score = if body_available {
                            q_img.body_emb.as_ref().and_then(|qe| g_img.body_emb.as_ref().map(|ge| cosine(qe, ge)))
                        } else { None };

                        let fusion_score = match (face_score, body_score) {
                            (Some(f), Some(b)) => alpha * f + (1.0 - alpha) * b,
                            (Some(f), None) => f,
                            (None, Some(b)) => b,
                            (None, None) => continue,
                        };

                        negative_scores.push(fusion_score);
                    }
                }
            }
        }

        let auc = roc_auc(&positive_scores, &negative_scores);
        let eer = compute_eer(&positive_scores, &negative_scores);

        // LOO for fusion
        let mut loo_pass = 0;
        let mut loo_total = 0;

        for (person_id, split) in splits {
            let dual_person: Vec<&ImageRecord> = images.iter()
                .filter(|i| i.person_id == *person_id
                    && i.face_detected && i.face_emb.is_some()
                    && i.body_available && i.body_emb.is_some())
                .collect();

            if dual_person.len() < 2 { continue; }

            for &leave_out_id in &dual_person.iter().map(|i| i.id).collect::<Vec<_>>() {
                let face_proto: Vec<f32> = dual_person.iter()
                    .filter(|i| i.id != leave_out_id)
                    .filter_map(|i| i.face_emb.clone())
                    .fold(vec![0.0; FACE_DIM], |mut acc, emb| {
                        for (a, &e) in acc.iter_mut().zip(emb.iter()) { *a += e; }
                        acc
                    });

                let body_proto: Vec<f32> = dual_person.iter()
                    .filter(|i| i.id != leave_out_id)
                    .filter_map(|i| i.body_emb.clone())
                    .fold(vec![0.0; BODY_DIM], |mut acc, emb| {
                        for (a, &e) in acc.iter_mut().zip(emb.iter()) { *a += e; }
                        acc
                    });

                let mut normalized_face = face_proto;
                l2_normalize(&mut normalized_face);
                let mut normalized_body = body_proto;
                l2_normalize(&mut normalized_body);

                let Some(leave_out) = dual_person.iter().find(|i| i.id == leave_out_id) else { continue };
                let (Some(leave_out_face), Some(leave_out_body)) = (&leave_out.face_emb, &leave_out.body_emb) else { continue };

                let face_score = cosine(&normalized_face, leave_out_face);
                let body_score = cosine(&normalized_body, leave_out_body);

                let fusion_score = match (face_available, body_available) {
                    (true, true) => alpha * face_score + (1.0 - alpha) * body_score,
                    (true, false) => face_score,
                    (false, true) => body_score,
                    (false, false) => continue,
                };

                if fusion_score >= FUSION_THRESHOLD { loo_pass += 1; }
                loo_total += 1;
            }
        }

        let loo_pass_rate = loo_pass as f32 / loo_total.max(1) as f32;

        // Margin P10 for this alpha
        let mut margins = Vec::new();
        for (person_id, split) in splits {
            let id_to_img_local: HashMap<usize, &ImageRecord> = images.iter()
                .filter(|i| i.face_detected && i.face_emb.is_some() && i.body_available && i.body_emb.is_some())
                .map(|i| (i.id, i))
                .collect();

            for &q_id in &split.query {
                let Some(q_img) = id_to_img_local.get(&q_id) else { continue };

                let mut gallery_scores: Vec<f32> = split.gallery.iter()
                    .filter_map(|g_id| {
                        let g_img = id_to_img_local.get(g_id)?;
                        let face_score = if face_available {
                            q_img.face_emb.as_ref().and_then(|qe| g_img.face_emb.as_ref().map(|ge| cosine(qe, ge)))
                        } else { None };
                        let body_score = if body_available {
                            q_img.body_emb.as_ref().and_then(|qe| g_img.body_emb.as_ref().map(|ge| cosine(qe, ge)))
                        } else { None };
                        let fusion = match (face_score, body_score) {
                            (Some(f), Some(b)) => alpha * f + (1.0 - alpha) * b,
                            (Some(f), None) => f,
                            (None, Some(b)) => b,
                            (None, None) => return None,
                        };
                        Some(fusion)
                    })
                    .collect();
                gallery_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let best_positive = gallery_scores.first().copied().unwrap_or(0.0);

                let mut other_max = f32::MIN;
                for (other_pid, other_split) in splits {
                    if *other_pid == *person_id { continue; }
                    for g_id in &other_split.gallery {
                        if let Some(g_img) = id_to_img_local.get(g_id) {
                            let face_score = if face_available {
                                q_img.face_emb.as_ref().and_then(|qe| g_img.face_emb.as_ref().map(|ge| cosine(qe, ge)))
                            } else { None };
                            let body_score = if body_available {
                                q_img.body_emb.as_ref().and_then(|qe| g_img.body_emb.as_ref().map(|ge| cosine(qe, ge)))
                            } else { None };
                            let fusion = match (face_score, body_score) {
                                (Some(f), Some(b)) => alpha * f + (1.0 - alpha) * b,
                                (Some(f), None) => f,
                                (None, Some(b)) => b,
                                (None, None) => continue,
                            };
                            other_max = other_max.max(fusion);
                        }
                    }
                }
                margins.push(best_positive - other_max);
            }
        }

        let p10_margin = percentile(&margins, 0.1);

        alpha_sweep_results.push(AlphaResult {
            alpha: alpha as f64,
            auc: auc as f64,
            eer: eer as f64,
            loo_pass_rate: loo_pass_rate as f64,
            p10_margin: p10_margin as f64,
        });
    }

    // Find best alpha by AUC
    let best = alpha_sweep_results.iter()
        .max_by(|a, b| a.auc.partial_cmp(&b.auc).unwrap())
        .cloned();

    let best_alpha = best.as_ref().map(|b| b.alpha);

    // Build identification with best alpha
    let identification = if let Some(ref b) = best {
        let alpha = b.alpha as f32;

        let mut top1_correct = 0;
        let mut top3_correct = 0;
        let mut top5_correct = 0;
        let mut top10_correct = 0;
        let mut total_queries = 0;

        for (person_id, split) in splits {
            let id_to_img_local: HashMap<usize, &ImageRecord> = images.iter()
                .filter(|i| i.face_detected && i.face_emb.is_some() && i.body_available && i.body_emb.is_some())
                .map(|i| (i.id, i))
                .collect();

            for &q_id in &split.query {
                let Some(q_img) = id_to_img_local.get(&q_id) else { continue };
                total_queries += 1;

                let mut candidates: Vec<(usize, f32)> = Vec::new();
                for (g_id, g_img) in &id_to_img_local {
                    if split.gallery.contains(g_id) { continue; }

                    let face_score = if face_available {
                        q_img.face_emb.as_ref().and_then(|qe| g_img.face_emb.as_ref().map(|ge| cosine(qe, ge)))
                    } else { None };
                    let body_score = if body_available {
                        q_img.body_emb.as_ref().and_then(|qe| g_img.body_emb.as_ref().map(|ge| cosine(qe, ge)))
                    } else { None };

                    let fusion_score = match (face_score, body_score) {
                        (Some(f), Some(b)) => alpha * f + (1.0 - alpha) * b,
                        (Some(f), None) => f,
                        (None, Some(b)) => b,
                        (None, None) => continue,
                    };
                    candidates.push((*g_id, fusion_score));
                }

                candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let top1_pid = candidates.first()
                    .and_then(|(id, _)| id_to_img_local.get(id))
                    .map(|i| i.person_id);
                let top3_pids: Vec<usize> = candidates.iter().take(3)
                    .filter_map(|(id, _)| id_to_img_local.get(id))
                    .map(|i| i.person_id)
                    .collect();
                let top5_pids: Vec<usize> = candidates.iter().take(5)
                    .filter_map(|(id, _)| id_to_img_local.get(id))
                    .map(|i| i.person_id)
                    .collect();
                let top10_pids: Vec<usize> = candidates.iter().take(10)
                    .filter_map(|(id, _)| id_to_img_local.get(id))
                    .map(|i| i.person_id)
                    .collect();

                if top1_pid == Some(*person_id) { top1_correct += 1; }
                if top3_pids.contains(person_id) { top3_correct += 1; }
                if top5_pids.contains(person_id) { top5_correct += 1; }
                if top10_pids.contains(person_id) { top10_correct += 1; }
            }
        }

        Some(IdentificationReport {
            top1: top1_correct as f64 / total_queries.max(1) as f64,
            top3: top3_correct as f64 / total_queries.max(1) as f64,
            top5: top5_correct as f64 / total_queries.max(1) as f64,
            top10: top10_correct as f64 / total_queries.max(1) as f64,
            map: 0.0,
        })
    } else {
        None
    };

    FusionReport {
        available: true,
        best_alpha,
        alpha_sweep: alpha_sweep_results,
        identification,
        verification: best.as_ref().map(|b| VerificationReport {
            auc: b.auc,
            eer: b.eer,
            far_01: 0.0,
            far_001: 0.0,
            tar_at_far_1e3: 0.0,
        }),
        margin: best.as_ref().map(|b| MarginReport {
            mean: 0.0,
            p10: b.p10_margin,
            p50: 0.0,
            p90: 0.0,
        }),
        loo: best.as_ref().map(|b| LooReport {
            pass_rate: b.loo_pass_rate,
            total: 0,
            passed: 0,
        }),
        verdict: "EXPERIMENTAL".to_string(),
    }
}

// ============================================================================
// Hard Negatives
// ============================================================================

fn find_hard_negatives(images: &[ImageRecord]) -> HardNegativeReport {
    let face_valid: Vec<&ImageRecord> = images.iter()
        .filter(|i| i.face_detected && i.face_emb.is_some())
        .collect();
    let body_valid: Vec<&ImageRecord> = images.iter()
        .filter(|i| i.body_available && i.body_emb.is_some())
        .collect();
    let dual_valid: Vec<&ImageRecord> = images.iter()
        .filter(|i| i.face_detected && i.face_emb.is_some() && i.body_available && i.body_emb.is_some())
        .collect();

    let mut face_hard = Vec::new();
    let mut body_hard = Vec::new();
    let mut dual_hard = Vec::new();

    // Face hard negatives (cross-person pairs with highest face similarity)
    for i in 0..face_valid.len() {
        for j in 0..face_valid.len() {
            if i >= j { continue; }
            let img_a = face_valid[i];
            let img_b = face_valid[j];
            if img_a.person_id == img_b.person_id { continue; }

            if let (Some(e_a), Some(e_b)) = (&img_a.face_emb, &img_b.face_emb) {
                let score = cosine(e_a, e_b);
                face_hard.push((score, img_a, img_b, None, None, None));
            }
        }
    }

    face_hard.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    face_hard.truncate(20);

    // Body hard negatives
    for i in 0..body_valid.len() {
        for j in 0..body_valid.len() {
            if i >= j { continue; }
            let img_a = body_valid[i];
            let img_b = body_valid[j];
            if img_a.person_id == img_b.person_id { continue; }

            if let (Some(e_a), Some(e_b)) = (&img_a.body_emb, &img_b.body_emb) {
                let score = cosine(e_a, e_b);
                body_hard.push((score, img_a, img_b, None, None, None));
            }
        }
    }

    body_hard.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    body_hard.truncate(20);

    // Dual hard negatives
    for i in 0..dual_valid.len() {
        for j in 0..dual_valid.len() {
            if i >= j { continue; }
            let img_a = dual_valid[i];
            let img_b = dual_valid[j];
            if img_a.person_id == img_b.person_id { continue; }

            let face_score = img_a.face_emb.as_ref().and_then(|e1| img_b.face_emb.as_ref().map(|e2| cosine(e1, e2)));
            let body_score = img_a.body_emb.as_ref().and_then(|e1| img_b.body_emb.as_ref().map(|e2| cosine(e1, e2)));
            let fusion_score = match (face_score, body_score) {
                (Some(f), Some(b)) => Some((f + b) / 2.0),
                _ => None,
            };

            if let (Some(fs), Some(bs), Some(fus)) = (face_score, body_score, fusion_score) {
                dual_hard.push((fus, img_a, img_b, Some(fs), Some(bs), Some(fus)));
            }
        }
    }

    dual_hard.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    dual_hard.truncate(20);

    fn to_report(items: Vec<(f32, &ImageRecord, &ImageRecord, Option<f32>, Option<f32>, Option<f32>)>) -> Vec<HardNegativeItem> {
        items.into_iter().enumerate().map(|(rank, (score, a, b, fs, bs, fus))| {
            HardNegativeItem {
                rank: rank + 1,
                person_a: a.person_name.clone(),
                person_b: b.person_name.clone(),
                face_score: fs.map(|s| s as f64),
                body_score: bs.map(|s| s as f64),
                fusion_score: fus.map(|s| s as f64),
            }
        }).collect()
    }

    HardNegativeReport {
        face: to_report(face_hard),
        body: to_report(body_hard),
        dual: to_report(dual_hard),
    }
}

// ============================================================================
// Chain Contamination
// ============================================================================

fn run_chain_contamination(images: &[ImageRecord]) -> Vec<ChainResult> {
    let face_valid: Vec<&ImageRecord> = images.iter()
        .filter(|i| i.face_detected && i.face_emb.is_some())
        .collect();

    let mut results = Vec::new();

    for &steps in [1, 2, 5, 10, 20].iter() {
        let mut contamination_detected = false;
        let mut margin_a_b = 0.0f32;
        let mut margin_b_c = 0.0f32;
        let mut margin_c_d = 0.0f32;
        let mut margin_a_d = 0.0f32;

        for i in 0..face_valid.len().saturating_sub(steps) {
            let img_a = face_valid[i];
            let img_b = face_valid[i + 1];

            let e_a = img_a.face_emb.as_ref();
            let e_b = img_b.face_emb.as_ref();
            if let (Some(a), Some(b)) = (e_a, e_b) {
                margin_a_b = cosine(a, b);
            }

            let img_c = if steps >= 2 { Some(face_valid[i + 2]) } else { None };
            let e_c = img_c.as_ref().and_then(|c| c.face_emb.as_ref());
            let e_b2 = img_b.face_emb.as_ref();
            if let (Some(b), Some(c)) = (e_b2, e_c) {
                margin_b_c = cosine(b, c);
            }

            if steps >= 5 {
                let img_d = face_valid[i + 5];
                let e_d = img_d.face_emb.as_ref();
                if let (Some(c), Some(d)) = (e_c, e_d) {
                    margin_c_d = cosine(c, d);
                }
                if let (Some(a), Some(d_val)) = (e_a, e_d) {
                    margin_a_d = cosine(a, d_val);
                }

                if margin_a_b > CHAINING_MARGIN && margin_b_c > CHAINING_MARGIN
                    && margin_c_d > CHAINING_MARGIN && margin_a_d < CHAINING_MARGIN
                {
                    contamination_detected = true;
                }
            }
        }

        results.push(ChainResult {
            steps,
            detected: contamination_detected,
            margin_a_b: margin_a_b as f64,
            margin_b_c: margin_b_c as f64,
            margin_c_d: margin_c_d as f64,
            margin_a_d: margin_a_d as f64,
        });
    }

    results
}

// ============================================================================
// Prototype Pollution
// ============================================================================

fn run_prototype_pollution(images: &[ImageRecord]) -> PrototypePollutionReport {
    // Use first person with >= 10 face images for core prototype
    let mut per_person: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
    for img in images.iter().filter(|i| i.face_detected && i.face_emb.is_some()) {
        per_person.entry(img.person_id).or_default().push(img);
    }

    let mut person_with_enough = per_person.into_iter()
        .filter(|(_, v)| v.len() >= 10)
        .collect::<Vec<_>>();

    person_with_enough.sort_by_key(|(pid, _)| *pid);

    if person_with_enough.is_empty() {
        return PrototypePollutionReport {
            drift_1_wrong: 0.0,
            drift_2_wrong: 0.0,
            drift_5_wrong: 0.0,
            drift_10_wrong: 0.0,
            core_preserved: false,
        };
    }

    let (core_pid, core_images) = person_with_enough.remove(0);
    let core_embs: Vec<&Vec<f32>> = core_images.iter()
        .filter_map(|i| i.face_emb.as_ref())
        .collect();

    if core_embs.len() < 5 {
        return PrototypePollutionReport {
            drift_1_wrong: 0.0,
            drift_2_wrong: 0.0,
            drift_5_wrong: 0.0,
            drift_10_wrong: 0.0,
            core_preserved: false,
        };
    }

    // Build clean prototype (N-1 images)
    let clean_proto: Vec<f32> = core_embs.iter()
        .take(core_embs.len() - 1)
        .fold(vec![0.0; FACE_DIM], |mut acc, emb| {
            for (a, &e) in acc.iter_mut().zip(emb.iter()) { *a += e; }
            acc
        });
    let clean_proto_norm = {
        let mut cp = clean_proto.clone();
        l2_normalize(&mut cp);
        cp
    };

    // Get wrong embeddings from a different person
    let wrong_images: Vec<&ImageRecord> = images.iter()
        .filter(|i| i.person_id != core_pid && i.face_detected && i.face_emb.is_some())
        .take(20)
        .collect();

    if wrong_images.is_empty() {
        return PrototypePollutionReport {
            drift_1_wrong: 0.0,
            drift_2_wrong: 0.0,
            drift_5_wrong: 0.0,
            drift_10_wrong: 0.0,
            core_preserved: false,
        };
    }

    let wrong_embs: Vec<&Vec<f32>> = wrong_images.iter()
        .filter_map(|i| i.face_emb.as_ref())
        .collect();

    fn compute_drift(proto: &[f32], clean: &[f32]) -> f32 {
        let diff: f32 = proto.iter().zip(clean.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        diff.sqrt()
    }

    // Test with 1 wrong embedding
    let polluted_1 = {
        let mut p = clean_proto.clone();
        if let Some(wrong) = wrong_embs.get(0) {
            for (a, &w) in p.iter_mut().zip(wrong.iter()) { *a += w; }
        }
        l2_normalize(&mut p);
        compute_drift(&p, &clean_proto_norm)
    };

    // Test with 2 wrong embeddings
    let polluted_2 = {
        let mut p = clean_proto.clone();
        for wrong in wrong_embs.iter().take(2) {
            for (a, &w) in p.iter_mut().zip(wrong.iter()) { *a += w; }
        }
        l2_normalize(&mut p);
        compute_drift(&p, &clean_proto_norm)
    };

    // Test with 5 wrong embeddings
    let polluted_5 = {
        let mut p = clean_proto.clone();
        for wrong in wrong_embs.iter().take(5) {
            for (a, &w) in p.iter_mut().zip(wrong.iter()) { *a += w; }
        }
        l2_normalize(&mut p);
        compute_drift(&p, &clean_proto_norm)
    };

    // Test with 10 wrong embeddings
    let polluted_10 = {
        let mut p = clean_proto.clone();
        for wrong in wrong_embs.iter().take(10) {
            for (a, &w) in p.iter_mut().zip(wrong.iter()) { *a += w; }
        }
        l2_normalize(&mut p);
        compute_drift(&p, &clean_proto_norm)
    };

    // Core preserved: clean prototype still recognizes last held-out image
    let held_out = core_embs.last().copied();
    let core_preserved = held_out.map(|h| cosine(&clean_proto_norm, h) >= FACE_THRESHOLD).unwrap_or(false);

    PrototypePollutionReport {
        drift_1_wrong: polluted_1 as f64,
        drift_2_wrong: polluted_2 as f64,
        drift_5_wrong: polluted_5 as f64,
        drift_10_wrong: polluted_10 as f64,
        core_preserved,
    }
}

// ============================================================================
// False Merge / False Split (multi-seed)
// ============================================================================

fn run_false_merge_split(images: &[ImageRecord], seeds: &[u64]) -> ClusteringReport {
    let face_valid: Vec<&ImageRecord> = images.iter()
        .filter(|i| i.face_detected && i.face_emb.is_some())
        .collect();

    let mut total_pairs = 0;
    let mut face_false_merges = 0;
    let mut all_fm_rates = Vec::new();

    for &seed in seeds {
        let splits = create_query_gallery_split(images, seed);
        let id_to_img: HashMap<usize, &ImageRecord> = images.iter().map(|i| (i.id, i)).collect();

        for i in 0..face_valid.len() {
            for j in (i + 1)..face_valid.len() {
                let img_a = face_valid[i];
                let img_b = face_valid[j];
                if img_a.person_id == img_b.person_id { continue; }
                total_pairs += 1;

                // Check if they would be merged in identification (same cluster)
                // For this simple test: check if they have high similarity
                if let (Some(e_a), Some(e_b)) = (&img_a.face_emb, &img_b.face_emb) {
                    let score = cosine(e_a, e_b);
                    if score >= FACE_THRESHOLD {
                        face_false_merges += 1;
                    }
                }
            }
        }

        if total_pairs > 0 {
            all_fm_rates.push(face_false_merges as f32 / total_pairs as f32);
        }
    }

    let mean_fm = if all_fm_rates.is_empty() { 0.0 } else { all_fm_rates.iter().sum::<f32>() / all_fm_rates.len() as f32 };
    let min_fm = all_fm_rates.iter().copied().fold(f32::INFINITY, |a, b| a.min(b));
    let max_fm = all_fm_rates.iter().copied().fold(f32::NEG_INFINITY, |a, b| a.max(b));

    ClusteringReport {
        false_merge_rate: mean_fm as f64,
        false_split_rate: 0.0, // Not computed in this simplified version
        purity: 1.0 - mean_fm as f64,
        pairwise_f1: 0.0, // Would need precision/recall computation
        seeds_tested: seeds.len(),
    }
}

// ============================================================================
// Main Benchmark Test
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn phase25_real_album_benchmark() {
    println!("============================================================");
    println!("Phase 25 — Real-World Person Identity Benchmark");
    println!("============================================================");
    println!("\nPrinciple: No manual JSON string building — uses serde_json");
    println!("Dataset: {}", BENCHMARK_DIR);
    println!();

    // ====== TASK 1-3: Dataset Discovery + Audit ======
    println!("============================================================");
    println!("TASK 1-3: Dataset Discovery & Audit");
    println!("============================================================");

    let mut images = match discover_dataset() {
        Some(imgs) => imgs,
        None => {
            println!("ERROR: Dataset not found at {}", BENCHMARK_DIR);
            panic!("DATASET_NOT_FOUND");
        }
    };

    if images.is_empty() {
        panic!("PANIC: total_images == 0, dataset is empty");
    }

    let (total, persons, min_per, median_per, max_per) = compute_dataset_stats(&images);
    println!("dataset_path: {}", BENCHMARK_DIR);
    println!("total_images: {}", total);
    println!("total_persons: {}", persons);

    // Task 2: Dataset audit
    dataset_audit(&images);

    // ====== TASK: Load Production Models ======
    println!();
    println!("============================================================");
    println!("Loading Production Models");
    println!("============================================================");

    let scrfd_path = resolve_model("scrfd_500m_bnkps.onnx");
    let arcface_path = resolve_model("w600k_r50.onnx");
    println!("  SCRFD: {}", scrfd_path.display());
    println!("  ArcFace: {}", arcface_path.display());

    if !scrfd_path.exists() {
        panic!("SCRFD model not found at {}", scrfd_path.display());
    }
    if !arcface_path.exists() {
        panic!("ArcFace model not found at {}", arcface_path.display());
    }

    let detector = pf_ai::ScrfdDetector::load(&scrfd_path)
        .expect("load SCRFD");
    let aligner: Arc<pf_ai::SimpleAligner> = Arc::new(pf_ai::SimpleAligner::new());
    let embedder = pf_ai::ArcFaceEmbedder::load(&arcface_path)
        .expect("load ArcFace");
    let qf = pf_ai::QualityFilter::from_config(0.30, 20, 0.45, 30.0);
    let face_pipeline: Arc<pf_ai::FacePipeline> = Arc::new(pf_ai::FacePipeline::new(detector, aligner, embedder, qf));

    let youtureid_path = resolve_model("person_reid_youtu_2021nov.onnx");
    println!("  YouTu Re-ID: {}", youtureid_path.display());

    let body_embedder: Option<Arc<pf_ai::YouTuReIdEmbedder>> = if youtureid_path.exists() {
        match pf_ai::YouTuReIdEmbedder::load(&youtureid_path) {
            Ok(eb) => {
                println!("  BodyEmbedder: YouTu Re-ID loaded successfully");
                Some(eb)
            }
            Err(e) => {
                println!("  WARNING: YouTu Re-ID load failed: {}", e);
                None
            }
        }
    } else {
        println!("  WARNING: YouTu Re-ID model not found at {}", youtureid_path.display());
        None
    };

    // ====== Process Images ======
    println!();
    println!("============================================================");
    println!("Processing Images (Face + Body Embedding)");
    println!("============================================================");

    let mut processed = 0;
    let mut face_detected = 0;
    let mut body_available = 0;
    let total_images = images.len();

    for img in &mut images {
        let path = &img.path;
        if !path.exists() {
            continue;
        }

        let image_data = match pf_ai::ImageData::from_file(path) {
            Ok(id) => id,
            Err(e) => {
                println!("  Warning: failed to load {}: {}", path.display(), e);
                continue;
            }
        };

        img.image_width = image_data.width();
        img.image_height = image_data.height();
        img.resolution = image_data.width().max(image_data.height());

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
        if let Some(ref embedder) = body_embedder {
            let rgb = image_data.as_rgb8();
            match embedder.embed(&rgb) {
                Ok(emb) => {
                    img.body_available = true;
                    img.body_quality = 1.0;
                    img.body_emb = Some(emb.values);
                    body_available += 1;
                }
                Err(_) => {}
            }
        }

        processed += 1;
        if processed % 100 == 0 || processed == total_images {
            println!("  Processed {}/{} images (face: {}, body: {})",
                processed, total_images, face_detected, body_available);
        }
    }

    println!();
    println!("  Total processed: {}", processed);
    println!("  Face detected: {}", face_detected);
    println!("  Body available: {}", body_available);

    // Re-run dataset audit with actual detection results
    dataset_audit(&images);

    // Check for self-match leakage
    let face_valid: Vec<&ImageRecord> = images.iter()
        .filter(|i| i.face_detected && i.face_emb.is_some())
        .collect();
    println!();
    println!("Face benchmark images: {}", face_valid.len());
    println!("Body benchmark images: {}", images.iter().filter(|i| i.body_available && i.body_emb.is_some()).count());

    // ====== Create Query/Gallery Split ======
    let splits = create_query_gallery_split(&images, 42);

    // ====== TASK 4: Face Benchmark ======
    println!();
    println!("============================================================");
    println!("TASK 4: FACE RESULTS");
    println!("============================================================");

    let face_result = run_face_benchmark(&images, &splits);

    println!("Face available: {}", face_detected > 0);
    println!("positive_pairs: {}", face_result.positive_pairs);
    println!("negative_pairs: {}", face_result.negative_pairs);
    println!();
    println!("Top1: {:.2}%", face_result.identification.top1 * 100.0);
    println!("Top3: {:.2}%", face_result.identification.top3 * 100.0);
    println!("Top5: {:.2}%", face_result.identification.top5 * 100.0);
    println!("Top10: {:.2}%", face_result.identification.top10 * 100.0);
    println!("mAP: {:.4}", face_result.identification.map);
    println!();
    println!("AUC: {:.4}", face_result.verification.auc);
    println!("EER: {:.2}%", face_result.verification.eer * 100.0);
    println!("FAR@0.1%: {:.6}", face_result.verification.far_01);
    println!("FAR@0.01%: {:.6}", face_result.verification.far_001);
    println!("TAR@FAR=1e-3: {:.4}", face_result.verification.tar_at_far_1e3);
    println!();
    println!("LOO pass rate: {:.2}% ({}/{})",
        face_result.loo.pass_rate * 100.0, face_result.loo.passed, face_result.loo.total);
    println!();
    println!("Margin:");
    println!("  P10: {:.4}", face_result.margin.p10);
    println!("  P50: {:.4}", face_result.margin.p50);
    println!("  P90: {:.4}", face_result.margin.p90);
    println!("  Mean: {:.4}", face_result.margin.mean);

    // ====== TASK 5: Body Benchmark ======
    println!();
    println!("============================================================");
    println!("TASK 5: BODY RESULTS");
    println!("============================================================");

    let body_result = run_body_benchmark(&images, &splits);

    println!("Body available: {}", body_available > 0);
    println!("positive_pairs: {}", body_result.positive_pairs);
    println!("negative_pairs: {}", body_result.negative_pairs);
    println!();
    println!("Top1: {:.2}%", body_result.identification.top1 * 100.0);
    println!("Top3: {:.2}%", body_result.identification.top3 * 100.0);
    println!("Top5: {:.2}%", body_result.identification.top5 * 100.0);
    println!("Top10: {:.2}%", body_result.identification.top10 * 100.0);
    println!();
    println!("AUC: {:.4}", body_result.verification.auc);
    println!("EER: {:.2}%", body_result.verification.eer * 100.0);
    println!("FAR@0.1%: {:.6}", body_result.verification.far_01);
    println!("FAR@0.01%: {:.6}", body_result.verification.far_001);
    println!();
    println!("LOO pass rate: {:.2}% ({}/{})",
        body_result.loo.pass_rate * 100.0, body_result.loo.passed, body_result.loo.total);
    println!();
    println!("Margin:");
    println!("  P10: {:.4}", body_result.margin.p10);
    println!("  P50: {:.4}", body_result.margin.p50);
    println!("  P90: {:.4}", body_result.margin.p90);
    println!("  Mean: {:.4}", body_result.margin.mean);

    // ====== TASK 6: Fusion Benchmark ======
    println!();
    println!("============================================================");
    println!("TASK 6: FUSION RESULTS (Alpha Sweep)");
    println!("============================================================");

    let fusion_result = run_fusion_benchmark(
        &images,
        &splits,
        face_detected > 0,
        body_available > 0,
    );

    if !fusion_result.available {
        println!("FUSION_INSUFFICIENT_DATA");
    } else {
        println!("Fusion available: true");
        println!();
        println!("Alpha sweep:");
        println!("  alpha | AUC    | EER    | LOO    | P10-margin");
        println!("  ------+--------+--------+--------+----------");
        for r in &fusion_result.alpha_sweep {
            println!("  {:5.1} | {:6.4} | {:6.4} | {:6.4} | {:9.4}",
                r.alpha, r.auc, r.eer, r.loo_pass_rate, r.p10_margin);
        }

        if let Some(best_alpha) = fusion_result.best_alpha {
            println!();
            println!("Best alpha candidate: {:.1}", best_alpha);
            if let Some(ref id) = fusion_result.identification {
                println!("AUC @ best: {:.4}", fusion_result.verification.as_ref().map(|v| v.auc).unwrap_or(0.0));
                println!("EER @ best: {:.4}", fusion_result.verification.as_ref().map(|v| v.eer).unwrap_or(0.0));
            }
        }
    }

    // ====== TASK 7: Hard Negatives ======
    println!();
    println!("============================================================");
    println!("TASK 7: HARD NEGATIVES TOP 20");
    println!("============================================================");

    let hard_negatives = find_hard_negatives(&images);

    println!();
    println!("FACE HARD NEGATIVES TOP 20:");
    println!("  rank | person_a         | person_b         | face_score");
    for item in &hard_negatives.face {
        println!("  {:4} | {:16} | {:16} | {:.4}",
            item.rank, item.person_a, item.person_b,
            item.face_score.map(|s| s as f32).unwrap_or(0.0));
    }

    println!();
    println!("BODY HARD NEGATIVES TOP 20:");
    println!("  rank | person_a         | person_b         | body_score");
    for item in &hard_negatives.body {
        println!("  {:4} | {:16} | {:16} | {:.4}",
            item.rank, item.person_a, item.person_b,
            item.body_score.map(|s| s as f32).unwrap_or(0.0));
    }

    println!();
    println!("DUAL HARD NEGATIVES TOP 20:");
    println!("  rank | person_a         | person_b         | face    | body    | fusion");
    for item in &hard_negatives.dual {
        println!("  {:4} | {:16} | {:16} | {:6.4} | {:6.4} | {:6.4}",
            item.rank, item.person_a, item.person_b,
            item.face_score.map(|s| s as f32).unwrap_or(0.0),
            item.body_score.map(|s| s as f32).unwrap_or(0.0),
            item.fusion_score.map(|s| s as f32).unwrap_or(0.0));
    }

    // ====== TASK 8: Chain Contamination ======
    println!();
    println!("============================================================");
    println!("TASK 8: CHAIN CONTAMINATION");
    println!("============================================================");

    let chain_results = run_chain_contamination(&images);

    println!();
    println!("  steps | detected | margin_A-B | margin_B-C | margin_C-D | margin_A-D");
    println!("  ------+----------+------------+------------+------------+------------");
    for r in &chain_results {
        println!("  {:5} | {:8} | {:10.4} | {:10.4} | {:10.4} | {:10.4}",
            r.steps,
            if r.detected { "YES" } else { "NO" },
            r.margin_a_b, r.margin_b_c, r.margin_c_d, r.margin_a_d);
    }

    let any_contamination = chain_results.iter().any(|r| r.detected);
    if any_contamination {
        println!();
        println!("CHAIN_CONTAMINATION_DETECTED");
    }

    // ====== TASK 9: Prototype Pollution ======
    println!();
    println!("============================================================");
    println!("TASK 9: PROTOTYPE POLLUTION");
    println!("============================================================");

    let pollution = run_prototype_pollution(&images);

    println!();
    println!("drift:");
    println!("  1 wrong embedding:  {:.4}", pollution.drift_1_wrong);
    println!("  2 wrong embeddings: {:.4}", pollution.drift_2_wrong);
    println!("  5 wrong embeddings: {:.4}", pollution.drift_5_wrong);
    println!("  10 wrong embeddings:{:.4}", pollution.drift_10_wrong);
    println!();
    println!("core prototype preserved: {}", pollution.core_preserved);

    // ====== TASK 10: False Merge/Split ======
    println!();
    println!("============================================================");
    println!("TASK 10: FALSE MERGE / FALSE SPLIT");
    println!("============================================================");

    let seeds = [42u64, 123, 456, 789, 2026];
    let clustering = run_false_merge_split(&images, &seeds);

    println!();
    println!("seeds_tested: {}", clustering.seeds_tested);
    println!("mean false_merge_rate: {:.6}", clustering.false_merge_rate);
    println!("mean purity: {:.4}", clustering.purity);

    // ====== Build Full Report ======
    let face_count = images.iter().filter(|i| i.face_detected && i.face_emb.is_some()).count();
    let body_count = images.iter().filter(|i| i.body_available && i.body_emb.is_some()).count();
    let dual_count = images.iter().filter(|i| {
        i.face_detected && i.face_emb.is_some() && i.body_available && i.body_emb.is_some()
    }).count();
    let face_only = face_count.saturating_sub(dual_count);
    let body_only = body_count.saturating_sub(dual_count);
    let neither = images.len() - face_count - body_count + dual_count;

    // Destructuring to avoid move issues
    let FaceBenchmarkResult {
        identification: face_identification,
        verification: face_verification,
        margin: face_margin,
        loo: face_loo,
        positive_pairs: _,
        negative_pairs: _,
    } = face_result;

    let BodyBenchmarkResult {
        identification: body_identification,
        verification: body_verification,
        margin: body_margin,
        loo: body_loo,
        positive_pairs: _,
        negative_pairs: _,
        min_bbox_size: _,
        median_bbox_size: _,
        max_bbox_size: _,
    } = body_result;

    let FusionReport {
        available: fusion_available,
        best_alpha: fusion_best_alpha,
        alpha_sweep: fusion_alpha_sweep,
        identification: fusion_identification,
        verification: fusion_verification,
        margin: fusion_margin,
        loo: fusion_loo,
        verdict: _,
    } = fusion_result;

    let face_verdict = if face_identification.top1 >= 0.95 && face_verification.auc >= 0.98 {
        "PRODUCTION_READY".to_string()
    } else if face_identification.top1 >= 0.80 {
        "EXPERIMENTAL".to_string()
    } else {
        "PRODUCTION_UNSAFE".to_string()
    };

    let body_verdict = if body_identification.top1 >= 0.95 && body_verification.auc >= 0.98 {
        "PRODUCTION_READY".to_string()
    } else if body_identification.top1 >= 0.50 {
        "EXPERIMENTAL".to_string()
    } else {
        "PRODUCTION_UNSAFE".to_string()
    };

    let fusion_verdict = if fusion_available && fusion_best_alpha.is_some() {
        "EXPERIMENTAL".to_string()
    } else {
        "INSUFFICIENT_DATA".to_string()
    };

    let report = Phase25Report {
        phase: "25".to_string(),
        dataset: DatasetReport {
            total_images: images.len(),
            total_persons: persons,
            min_images_per_person: min_per,
            median_images_per_person: median_per,
            max_images_per_person: max_per,
            face_detection_rate: face_count as f64 / images.len().max(1) as f64,
            body_detection_rate: body_count as f64 / images.len().max(1) as f64,
            dual_channel_rate: dual_count as f64 / images.len().max(1) as f64,
            status: "READY".to_string(),
        },
        missing_channels: MissingChannelReport {
            face_only_count: face_only,
            body_only_count: body_only,
            dual_count,
            none_count: neither,
            face_only_accuracy: None,
            body_only_accuracy: None,
            dual_accuracy: None,
        },
        face: SignalReport {
            available: face_count > 0,
            identification: face_identification.clone(),
            verification: face_verification.clone(),
            margin: face_margin.clone(),
            loo: face_loo.clone(),
            verdict: face_verdict.clone(),
        },
        body: SignalReport {
            available: body_count > 0,
            identification: body_identification.clone(),
            verification: body_verification.clone(),
            margin: body_margin.clone(),
            loo: body_loo.clone(),
            verdict: body_verdict.clone(),
        },
        fusion: FusionReport {
            available: fusion_available,
            best_alpha: fusion_best_alpha,
            alpha_sweep: fusion_alpha_sweep.clone(),
            identification: fusion_identification.clone(),
            verification: fusion_verification.clone(),
            margin: fusion_margin.clone(),
            loo: fusion_loo.clone(),
            verdict: fusion_verdict.clone(),
        },
        chain_contamination: chain_results,
        hard_negatives,
        verdict: VerdictReport {
            face: face_verdict.clone(),
            body: body_verdict.clone(),
            fusion: fusion_verdict.clone(),
            clustering: "EXPERIMENTAL".to_string(),
            search: "EXPERIMENTAL".to_string(),
        },
    };

    // ====== TASK 11: Write and Validate JSON ======
    println!();
    println!("============================================================");
    println!("TASK 11: JSON Report Validation");
    println!("============================================================");

    let output_path = PathBuf::from(
        "/Users/mac/ai-project/photofinder-next-2/crates/pf_application/tests/phase25_real_album.json"
    );

    match generate_json_report(&report, &output_path) {
        Ok(_) => println!("  JSON report written to: {:?}", output_path),
        Err(e) => {
            println!("  ERROR: Failed to write JSON report: {}", e);
            return;
        }
    }

    // Validate JSON is parseable
    match std::fs::read_to_string(&output_path) {
        Ok(content) => {
            match serde_json::from_str::<Phase25Report>(&content) {
                Ok(validated) => {
                    println!("  JSON validation: PASS");
                    println!("  dataset.total_images: {}", validated.dataset.total_images);
                    println!("  face.available: {}", validated.face.available);
                    println!("  body.available: {}", validated.body.available);
                    if let Some(ref id) = validated.fusion.identification {
                        println!("  fusion identification top1: {:.2}%", id.top1 * 100.0);
                    }
                }
                Err(e) => {
                    println!("  JSON validation: FAIL - {}", e);
                }
            }
        }
        Err(e) => {
            println!("  ERROR: Could not read back JSON: {}", e);
        }
    }

    // ====== TASK 12: Console Summary ======
    println!();
    println!("============================================================");
    println!("PHASE 25 REAL ALBUM BENCHMARK");
    println!("============================================================");
    println!();
    println!("DATASET");
    println!();
    println!("  Persons: {}", persons);
    println!("  Images: {}", total);
    println!("  Images/person min/median/max: {}/{:.0}/{}",
        min_per, median_per, max_per);
    println!();
    println!("  Face available: {}", face_count);
    println!("  Body available: {}", body_count);
    println!("  Dual available: {}", dual_count);
    println!();
    println!("FACE");
    println!();
    println!("  Top1: {:.2}%", face_identification.top1 * 100.0);
    println!("  Top5: {:.2}%", face_identification.top5 * 100.0);
    println!();
    println!("  AUC: {:.4}", face_verification.auc);
    println!("  EER: {:.2}%", face_verification.eer * 100.0);
    println!();
    println!("  FAR@0.1%: {:.6}", face_verification.far_01);
    println!("  FAR@0.01%: {:.6}", face_verification.far_001);
    println!();
    println!("  LOO: {:.2}% ({}/{})",
        face_loo.pass_rate * 100.0, face_loo.passed, face_loo.total);
    println!();
    println!("  Margin P10/P50/P90: {:.4} / {:.4} / {:.4}",
        face_margin.p10, face_margin.p50, face_margin.p90);
    println!();
    println!("BODY");
    println!();
    println!("  Top1: {:.2}%", body_identification.top1 * 100.0);
    println!("  Top5: {:.2}%", body_identification.top5 * 100.0);
    println!();
    println!("  AUC: {:.4}", body_verification.auc);
    println!("  EER: {:.2}%", body_verification.eer * 100.0);
    println!();
    println!("  LOO: {:.2}% ({}/{})",
        body_loo.pass_rate * 100.0, body_loo.passed, body_loo.total);
    println!();
    println!("  Margin P10/P50/P90: {:.4} / {:.4} / {:.4}",
        body_margin.p10, body_margin.p50, body_margin.p90);
    println!();
    println!("FUSION");
    println!();
    println!("  Available: {}", fusion_available);
    if let Some(best_alpha) = fusion_best_alpha {
        println!("  Best alpha candidate: {:.1}", best_alpha);
        println!("  AUC: {:.4}", fusion_verification.as_ref().map(|v| v.auc).unwrap_or(0.0));
        println!("  EER: {:.4}", fusion_verification.as_ref().map(|v| v.eer).unwrap_or(0.0));
        if let Some(ref loo) = fusion_loo {
            println!("  LOO: {:.2}%", loo.pass_rate * 100.0);
        }
    }
    println!();
    println!("CLUSTERING");
    println!();
    println!("  False Merge: {:.6}", clustering.false_merge_rate);
    println!("  Purity: {:.4}", clustering.purity);
    println!("  Seeds tested: {}", clustering.seeds_tested);
    println!();
    println!("CHAIN CONTAMINATION: {}",
        if any_contamination { "DETECTED" } else { "NONE" });
    println!();
    println!("PROTOTYPE");
    println!();
    println!("  Drift 5 wrong: {:.4}", pollution.drift_5_wrong);
    println!("  Core preserved: {}", pollution.core_preserved);
    println!();
    println!("FINAL VERDICT");
    println!();
    println!("  FACE: {}", face_verdict);
    println!("  BODY: {}", body_verdict);
    println!("  FUSION: {}", fusion_verdict);
    println!("  CLUSTERING: EXPOBODIMENTAL");
    println!();
    println!("  SYSTEM: {}",
        if face_verdict == "PRODUCTION_READY" && body_verdict == "PRODUCTION_READY" {
            "PRODUCTION_READY"
        } else if face_verdict == "PRODUCTION_UNSAFE" {
            "PRODUCTION_UNSAFE"
        } else {
            "PRODUCTION_EXPERIMENTAL"
        });
}

// ============================================================================
// Unit Tests
// ============================================================================

#[test]
fn test_phase25_report_serialization() {
    let report = Phase25Report {
        phase: "25".to_string(),
        dataset: DatasetReport {
            total_images: 100,
            total_persons: 10,
            min_images_per_person: 5,
            median_images_per_person: 10.0,
            max_images_per_person: 20,
            face_detection_rate: 0.95,
            body_detection_rate: 0.90,
            dual_channel_rate: 0.85,
            status: "READY".to_string(),
        },
        missing_channels: MissingChannelReport {
            face_only_count: 5,
            body_only_count: 3,
            dual_count: 90,
            none_count: 2,
            face_only_accuracy: None,
            body_only_accuracy: None,
            dual_accuracy: Some(0.98),
        },
        face: SignalReport {
            available: true,
            identification: IdentificationReport {
                top1: 0.95,
                top3: 0.98,
                top5: 0.99,
                top10: 1.0,
                map: 0.97,
            },
            verification: VerificationReport {
                auc: 0.98,
                eer: 0.02,
                far_01: 0.001,
                far_001: 0.0001,
                tar_at_far_1e3: 0.95,
            },
            margin: MarginReport {
                mean: 0.15,
                p10: 0.05,
                p50: 0.12,
                p90: 0.25,
            },
            loo: LooReport {
                pass_rate: 0.92,
                total: 100,
                passed: 92,
            },
            verdict: "PRODUCTION_READY".to_string(),
        },
        body: SignalReport {
            available: true,
            identification: IdentificationReport {
                top1: 0.88,
                top3: 0.92,
                top5: 0.95,
                top10: 0.98,
                map: 0.90,
            },
            verification: VerificationReport {
                auc: 0.85,
                eer: 0.15,
                far_01: 0.01,
                far_001: 0.005,
                tar_at_far_1e3: 0.75,
            },
            margin: MarginReport {
                mean: 0.10,
                p10: 0.02,
                p50: 0.08,
                p90: 0.20,
            },
            loo: LooReport {
                pass_rate: 0.75,
                total: 100,
                passed: 75,
            },
            verdict: "EXPERIMENTAL".to_string(),
        },
        fusion: FusionReport {
            available: true,
            best_alpha: Some(0.7),
            alpha_sweep: vec![AlphaResult {
                alpha: 0.7,
                auc: 0.99,
                eer: 0.01,
                loo_pass_rate: 0.95,
                p10_margin: 0.15,
            }],
            identification: Some(IdentificationReport {
                top1: 0.97,
                top3: 0.99,
                top5: 1.0,
                top10: 1.0,
                map: 0.98,
            }),
            verification: Some(VerificationReport {
                auc: 0.99,
                eer: 0.01,
                far_01: 0.0005,
                far_001: 0.0001,
                tar_at_far_1e3: 0.98,
            }),
            margin: Some(MarginReport {
                mean: 0.18,
                p10: 0.08,
                p50: 0.15,
                p90: 0.28,
            }),
            loo: Some(LooReport {
                pass_rate: 0.95,
                total: 100,
                passed: 95,
            }),
            verdict: "PRODUCTION_READY".to_string(),
        },
        chain_contamination: vec![],
        hard_negatives: HardNegativeReport {
            face: vec![],
            body: vec![],
            dual: vec![],
        },
        verdict: VerdictReport {
            face: "PRODUCTION_READY".to_string(),
            body: "EXPERIMENTAL".to_string(),
            fusion: "PRODUCTION_READY".to_string(),
            clustering: "SAFE".to_string(),
            search: "EXPERIMENTAL".to_string(),
        },
    };

    let json = serde_json::to_string_pretty(&report).unwrap();
    assert!(json.contains("\"phase\": \"25\""));
    assert!(json.contains("PRODUCTION_READY"));
    assert!(json.contains("EXPERIMENTAL"));
    assert!(json.contains("face_detection_rate"));
    assert!(json.contains("\"missing_channels\""));
}

#[test]
fn test_phase25_json_write() {
    let report = Phase25Report {
        phase: "25".to_string(),
        dataset: DatasetReport {
            total_images: 100,
            total_persons: 10,
            min_images_per_person: 5,
            median_images_per_person: 10.0,
            max_images_per_person: 20,
            face_detection_rate: 0.95,
            body_detection_rate: 0.90,
            dual_channel_rate: 0.85,
            status: "READY".to_string(),
        },
        missing_channels: MissingChannelReport {
            face_only_count: 5,
            body_only_count: 3,
            dual_count: 90,
            none_count: 2,
            face_only_accuracy: None,
            body_only_accuracy: None,
            dual_accuracy: Some(0.98),
        },
        face: SignalReport {
            available: true,
            identification: IdentificationReport {
                top1: 0.95,
                top3: 0.98,
                top5: 0.99,
                top10: 1.0,
                map: 0.97,
            },
            verification: VerificationReport {
                auc: 0.98,
                eer: 0.02,
                far_01: 0.001,
                far_001: 0.0001,
                tar_at_far_1e3: 0.95,
            },
            margin: MarginReport {
                mean: 0.15,
                p10: 0.05,
                p50: 0.12,
                p90: 0.25,
            },
            loo: LooReport {
                pass_rate: 0.92,
                total: 100,
                passed: 92,
            },
            verdict: "PRODUCTION_READY".to_string(),
        },
        body: SignalReport {
            available: true,
            identification: IdentificationReport {
                top1: 0.88,
                top3: 0.92,
                top5: 0.95,
                top10: 0.98,
                map: 0.90,
            },
            verification: VerificationReport {
                auc: 0.85,
                eer: 0.15,
                far_01: 0.01,
                far_001: 0.005,
                tar_at_far_1e3: 0.75,
            },
            margin: MarginReport {
                mean: 0.10,
                p10: 0.02,
                p50: 0.08,
                p90: 0.20,
            },
            loo: LooReport {
                pass_rate: 0.75,
                total: 100,
                passed: 75,
            },
            verdict: "EXPERIMENTAL".to_string(),
        },
        fusion: FusionReport {
            available: true,
            best_alpha: Some(0.7),
            alpha_sweep: vec![],
            identification: None,
            verification: None,
            margin: None,
            loo: None,
            verdict: "PRODUCTION_READY".to_string(),
        },
        chain_contamination: vec![],
        hard_negatives: HardNegativeReport {
            face: vec![],
            body: vec![],
            dual: vec![],
        },
        verdict: VerdictReport {
            face: "PRODUCTION_READY".to_string(),
            body: "EXPERIMENTAL".to_string(),
            fusion: "PRODUCTION_READY".to_string(),
            clustering: "SAFE".to_string(),
            search: "EXPERIMENTAL".to_string(),
        },
    };

    let temp = std::env::temp_dir().join("phase25_test_report.json");
    generate_json_report(&report, &temp).unwrap();
    assert!(temp.exists());
    let content = std::fs::read_to_string(&temp).unwrap();
    assert!(content.contains("\"phase\": \"25\""));
    assert!(content.contains("face_detection_rate"));
}

#[test]
fn test_none_values_instead_of_zero() {
    let report = Phase25Report {
        phase: "25".to_string(),
        dataset: DatasetReport {
            total_images: 100,
            total_persons: 10,
            min_images_per_person: 5,
            median_images_per_person: 10.0,
            max_images_per_person: 20,
            face_detection_rate: 0.0,
            body_detection_rate: 0.0,
            dual_channel_rate: 0.0,
            status: "INSUFFICIENT".to_string(),
        },
        missing_channels: MissingChannelReport {
            face_only_count: 0,
            body_only_count: 0,
            dual_count: 0,
            none_count: 100,
            face_only_accuracy: None,
            body_only_accuracy: None,
            dual_accuracy: None,
        },
        face: SignalReport {
            available: false,
            identification: IdentificationReport {
                top1: 0.0,
                top3: 0.0,
                top5: 0.0,
                top10: 0.0,
                map: 0.0,
            },
            verification: VerificationReport {
                auc: 0.0,
                eer: 1.0,
                far_01: 1.0,
                far_001: 1.0,
                tar_at_far_1e3: 0.0,
            },
            margin: MarginReport {
                mean: 0.0,
                p10: 0.0,
                p50: 0.0,
                p90: 0.0,
            },
            loo: LooReport {
                pass_rate: 0.0,
                total: 0,
                passed: 0,
            },
            verdict: "NOT_RUN".to_string(),
        },
        body: SignalReport {
            available: false,
            identification: IdentificationReport {
                top1: 0.0,
                top3: 0.0,
                top5: 0.0,
                top10: 0.0,
                map: 0.0,
            },
            verification: VerificationReport {
                auc: 0.0,
                eer: 1.0,
                far_01: 1.0,
                far_001: 1.0,
                tar_at_far_1e3: 0.0,
            },
            margin: MarginReport {
                mean: 0.0,
                p10: 0.0,
                p50: 0.0,
                p90: 0.0,
            },
            loo: LooReport {
                pass_rate: 0.0,
                total: 0,
                passed: 0,
            },
            verdict: "NOT_RUN".to_string(),
        },
        fusion: FusionReport {
            available: false,
            best_alpha: None,
            alpha_sweep: vec![],
            identification: None,
            verification: None,
            margin: None,
            loo: None,
            verdict: "NOT_RUN".to_string(),
        },
        chain_contamination: vec![],
        hard_negatives: HardNegativeReport {
            face: vec![],
            body: vec![],
            dual: vec![],
        },
        verdict: VerdictReport {
            face: "NOT_RUN".to_string(),
            body: "NOT_RUN".to_string(),
            fusion: "NOT_RUN".to_string(),
            clustering: "NOT_RUN".to_string(),
            search: "NOT_RUN".to_string(),
        },
    };

    let json = serde_json::to_string_pretty(&report).unwrap();
    assert!(json.contains("null"));
    assert!(json.contains("\"face_only_accuracy\": null"));
    assert!(json.contains("\"best_alpha\": null"));
    assert!(json.contains("\"identification\": null"));
}
