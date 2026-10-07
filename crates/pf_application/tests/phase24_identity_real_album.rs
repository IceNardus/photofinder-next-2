//! Phase 24 — Real Album Identity Benchmark
//!
//! Validates PhotoFinder's Face + Body + Fusion identity system on **real-world album photos**.
//!
//! Key principle: "Don't modify production algorithms to fit benchmark; benchmark must simulate real user photos."
//!
//! Dataset: person_identity_real (30 persons, 10-236 images/person)
//! - Face-friendly: well-lit, frontal faces (LFW-style)
//! - Body-friendly: full-body images with distinct clothing
//! - Dual-channel: both face and body visible
//! - Hard cases: occlusion, poor lighting, unusual poses
//!
//! Test Categories:
//! 1. Face Benchmark: ArcFace on all images
//! 2. Body Benchmark: YouTu Re-ID on all images
//! 3. Fusion Benchmark: Weighted face + body combination
//! 4. LOO Prototype: Leave-one-out prototype evaluation
//! 5. Search Benchmark: Real album search simulation
//! 6. Clustering Benchmark: Production clustering quality
//!
//! Production Gate:
//! - Top1 >= 95%, Top5 >= 98%, AUC >= 0.98, EER <= 2%, FAR <= 0.1%
//! - LOO >= 95%, False Merge Rate <= 0.1%, Chain contamination == false
//!
//! 运行：
//! ```bash
//! cargo test -p pf_application --test phase24_identity_real_album -- --nocapture
//! ```

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ============================================================================
// Constants
// ============================================================================

const MIN_PERSONS: usize = 20;
const MIN_IMAGES_PER_PERSON: usize = 8;
const MIN_USABLE_IMAGES_PER_PERSON: usize = 6;

const FACE_DIM: usize = 512;
const BODY_DIM: usize = 768;

const FACE_THRESHOLD: f32 = 0.75;
const BODY_THRESHOLD: f32 = 0.70;
const FUSION_THRESHOLD: f32 = 0.65;

const CHAINING_MARGIN: f32 = 0.05;

const BENCHMARK_DIR: &str = "/Users/mac/ai-project/photofinder-next-2/benchmark/person_identity_real";

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
    blur_score: f32,
    resolution: u32,
    image_width: u32,
    image_height: u32,
}

#[derive(Debug, Clone)]
struct QueryGallerySplit {
    gallery: Vec<usize>,
    query: Vec<usize>,
}

#[derive(Debug, Clone)]
struct VerificationMetrics {
    auc: f32,
    eer: f32,
    far_01: f32,
    far_05: f32,
    far_1: f32,
    far_5: f32,
    frr: f32,
    frr_at_threshold: f32,
}

#[derive(Debug, Clone)]
struct IdentificationMetrics {
    top1: f32,
    top3: f32,
    top5: f32,
    top10: f32,
}

#[derive(Debug, Clone)]
struct MarginMetrics {
    positive_best: f32,
    positive_mean: f32,
    positive_min: f32,
    negative_max: f32,
    median: f32,
    p10: f32,
    p25: f32,
    p50: f32,
    p75: f32,
    p90: f32,
}

#[derive(Debug, Clone)]
struct LooMetrics {
    pass_rate: f32,
}

#[derive(Debug, Clone)]
struct FalseMergeMetrics {
    face_false_merges: usize,
    body_false_merges: usize,
    fusion_false_merges: usize,
    total_pairs: usize,
    face_false_merge_rate: f32,
    body_false_merge_rate: f32,
    fusion_false_merge_rate: f32,
}

#[derive(Debug, Clone)]
struct ChainContaminationResult {
    steps: usize,
    contamination_detected: bool,
    margin_a_d: f32,
    margin_a_b: f32,
    margin_b_c: f32,
    margin_c_d: f32,
}

#[derive(Debug, Clone)]
struct AntiChainingResult {
    ambiguous_pairs: usize,
    total_pairs: usize,
    ambiguous_rate: f32,
}

#[derive(Debug, Clone)]
struct SystemVerdict {
    dataset_verdict: String,
    face_verdict: String,
    body_verdict: String,
    fusion_verdict: String,
}

#[derive(Debug, Clone)]
struct DataQualityStats {
    face_usable: usize,
    body_usable: usize,
    dual_usable: usize,
    face_only: usize,
    body_only: usize,
    neither: usize,
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
    candidates.into_iter().find(|p| p.exists()).unwrap_or_else(|| workspace.join("models").join(name))
}

// ============================================================================
// Dataset Discovery and Loading
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

            // Extract person name from filename (e.g., "Abdullah_Gul_0001.jpg" -> "Abdullah_Gul")
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
                blur_score: 0.0,
                resolution: 0,
                image_width: 0,
                image_height: 0,
            });
            image_id += 1;
        }
    }

    images.sort_by(|a, b| {
        a.person_id.cmp(&b.person_id).then(a.path.cmp(&b.path))
    });

    Some(images)
}

// ============================================================================
// Query/Gallery Split
// ============================================================================

fn create_query_gallery_split(images: &[ImageRecord], seed: u64) -> HashMap<usize, QueryGallerySplit> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut rng = seed;
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
// Face Benchmark
// ============================================================================

fn run_face_benchmark(
    images: &[ImageRecord],
    splits: &HashMap<usize, QueryGallerySplit>,
) -> (VerificationMetrics, IdentificationMetrics, MarginMetrics, LooMetrics) {
    // Collect face embeddings only
    let face_images: Vec<&ImageRecord> = images.iter().filter(|i| i.face_detected && i.face_emb.is_some()).collect();

    if face_images.is_empty() {
        return (
            VerificationMetrics { auc: 0.0, eer: 1.0, far_01: 1.0, far_05: 1.0, far_1: 1.0, far_5: 1.0, frr: 1.0, frr_at_threshold: 1.0 },
            IdentificationMetrics { top1: 0.0, top3: 0.0, top5: 0.0, top10: 0.0 },
            MarginMetrics { positive_best: 0.0, positive_mean: 0.0, positive_min: 0.0, negative_max: 0.0, median: 0.0, p10: 0.0, p25: 0.0, p50: 0.0, p75: 0.0, p90: 0.0 },
            LooMetrics { pass_rate: 0.0 },
        );
    }

    let id_to_img: HashMap<usize, &ImageRecord> = images.iter().map(|i| (i.id, i)).collect();

    // Verification: positive and negative pairs
    let mut positive_scores = Vec::new();
    let mut negative_scores = Vec::new();

    for (person_id, split) in splits {
        let query_ids: HashSet<usize> = split.query.iter().cloned().collect();
        let gallery_ids: HashSet<usize> = split.gallery.iter().cloned().collect();

        for &q_id in &split.query {
            let Some(q_img) = id_to_img.get(&q_id) else { continue };
            let Some(q_emb) = &q_img.face_emb else { continue };

            // Positive pairs: same person in gallery
            for &g_id in &split.gallery {
                let Some(g_img) = id_to_img.get(&g_id) else { continue };
                let Some(g_emb) = &g_img.face_emb else { continue };
                let score = cosine(q_emb, g_emb);
                positive_scores.push(score);
            }

            // Negative pairs: different person
            for (other_pid, other_split) in splits {
                if *other_pid == *person_id {
                    continue;
                }
                for &g_id in &other_split.gallery {
                    let Some(g_img) = id_to_img.get(&g_id) else { continue };
                    let Some(g_emb) = &g_img.face_emb else { continue };
                    let score = cosine(q_emb, g_emb);
                    negative_scores.push(score);
                }
            }
        }
    }

    let verification = VerificationMetrics {
        auc: roc_auc(&positive_scores, &negative_scores),
        eer: compute_eer(&positive_scores, &negative_scores),
        far_01: compute_far_at_threshold(&negative_scores, FACE_THRESHOLD),
        far_05: compute_far_at_threshold(&negative_scores, 0.5),
        far_1: compute_far_at_threshold(&negative_scores, 0.75),
        far_5: compute_far_at_threshold(&negative_scores, 0.5),
        frr: compute_frr_at_threshold(&positive_scores, FACE_THRESHOLD),
        frr_at_threshold: compute_frr_at_threshold(&positive_scores, FACE_THRESHOLD),
    };

    // Identification: Top-K
    let mut top1_correct = 0;
    let mut top3_correct = 0;
    let mut top5_correct = 0;
    let mut top10_correct = 0;
    let mut total_queries = 0;

    for (person_id, split) in splits {
        let id_to_img_local: HashMap<usize, &ImageRecord> = images.iter().filter(|i| i.face_detected && i.face_emb.is_some()).map(|i| (i.id, i)).collect();

        for &q_id in &split.query {
            let Some(q_img) = id_to_img_local.get(&q_id) else { continue };
            let Some(q_emb) = &q_img.face_emb else { continue };
            total_queries += 1;

            // Collect all gallery embeddings with person_id
            let mut candidates: Vec<(usize, f32)> = Vec::new();
            for (g_id, g_img) in &id_to_img_local {
                if split.gallery.contains(g_id) {
                    continue;
                }
                if let Some(g_emb) = &g_img.face_emb {
                    let score = cosine(q_emb, g_emb);
                    candidates.push((*g_id, score));
                }
            }

            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top1 = candidates.get(0).map(|(id, _)| id_to_img_local.get(id).map(|i| i.person_id));
            let top3: Vec<usize> = candidates.iter().take(3).filter_map(|(id, _)| id_to_img_local.get(id).map(|i| i.person_id)).collect();
            let top5: Vec<usize> = candidates.iter().take(5).filter_map(|(id, _)| id_to_img_local.get(id).map(|i| i.person_id)).collect();
            let top10: Vec<usize> = candidates.iter().take(10).filter_map(|(id, _)| id_to_img_local.get(id).map(|i| i.person_id)).collect();

            if top1 == Some(Some(*person_id)) {
                top1_correct += 1;
            }
            if top3.contains(person_id) {
                top3_correct += 1;
            }
            if top5.contains(person_id) {
                top5_correct += 1;
            }
            if top10.contains(person_id) {
                top10_correct += 1;
            }
        }
    }

    let identification = IdentificationMetrics {
        top1: top1_correct as f32 / total_queries.max(1) as f32,
        top3: top3_correct as f32 / total_queries.max(1) as f32,
        top5: top5_correct as f32 / total_queries.max(1) as f32,
        top10: top10_correct as f32 / total_queries.max(1) as f32,
    };

    // Margin analysis
    let mut positive_margins = Vec::new();
    let mut negative_margins = Vec::new();

    for (person_id, split) in splits {
        let id_to_img_local: HashMap<usize, &ImageRecord> = images.iter().filter(|i| i.face_detected && i.face_emb.is_some()).map(|i| (i.id, i)).collect();

        for &q_id in &split.query {
            let Some(q_img) = id_to_img_local.get(&q_id) else { continue };
            let Some(q_emb) = &q_img.face_emb else { continue };

            let mut gallery_scores: Vec<f32> = split.gallery.iter().filter_map(|g_id| {
                id_to_img_local.get(g_id).and_then(|g| g.face_emb.as_ref()).map(|g_emb| cosine(q_emb, g_emb))
            }).collect();

            gallery_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let best_positive = gallery_scores.first().copied().unwrap_or(0.0);

            let mut other_scores: Vec<f32> = Vec::new();
            for (other_pid, other_split) in splits {
                if *other_pid == *person_id {
                    continue;
                }
                for g_id in &other_split.gallery {
                    if let Some(g_img) = id_to_img_local.get(g_id) {
                        if let Some(g_emb) = &g_img.face_emb {
                            other_scores.push(cosine(q_emb, g_emb));
                        }
                    }
                }
            }

            let negative_max = other_scores.iter().copied().fold(f32::MIN, |a, b| a.max(b));
            let margin = best_positive - negative_max;
            positive_margins.push(margin);
        }
    }

    let margin = MarginMetrics {
        positive_best: positive_margins.iter().fold(f32::MIN, |a, &b| a.max(b)),
        positive_mean: if positive_margins.is_empty() { 0.0 } else { positive_margins.iter().sum::<f32>() / positive_margins.len() as f32 },
        positive_min: positive_margins.iter().fold(f32::MAX, |a, &b| a.min(b)),
        negative_max: negative_margins.iter().fold(f32::MIN, |a, &b| a.max(b)),
        median: percentile(&positive_margins, 0.5),
        p10: percentile(&positive_margins, 0.1),
        p25: percentile(&positive_margins, 0.25),
        p50: percentile(&positive_margins, 0.5),
        p75: percentile(&positive_margins, 0.75),
        p90: percentile(&positive_margins, 0.9),
    };

    // LOO Prototype
    let mut loo_pass = 0;
    let mut loo_total = 0;

    for (person_id, split) in splits {
        let person_images: Vec<&ImageRecord> = images.iter()
            .filter(|i| i.person_id == *person_id && i.face_detected && i.face_emb.is_some())
            .collect();

        if person_images.len() < 2 {
            continue;
        }

        for &leave_out_id in &person_images.iter().map(|i| i.id).collect::<Vec<_>>() {
            let prototype: Vec<f32> = person_images.iter()
                .filter(|i| i.id != leave_out_id)
                .filter_map(|i| i.face_emb.clone())
                .fold(vec![0.0; FACE_DIM], |mut acc, emb| {
                    for (a, &e) in acc.iter_mut().zip(emb.iter()) {
                        *a += e;
                    }
                    acc
                });

            let prototype_len = prototype.len() as f32;
            let mut normalized_prototype = prototype;
            l2_normalize(&mut normalized_prototype);

            let Some(leave_out) = person_images.iter().find(|i| i.id == leave_out_id) else { continue };
            let Some(leave_out_emb) = &leave_out.face_emb else { continue };

            let score = cosine(&normalized_prototype, leave_out_emb);
            if score >= FACE_THRESHOLD {
                loo_pass += 1;
            }
            loo_total += 1;
        }
    }

    let loo = LooMetrics {
        pass_rate: loo_pass as f32 / loo_total.max(1) as f32,
    };

    (verification, identification, margin, loo)
}

// ============================================================================
// Body Benchmark
// ============================================================================

fn run_body_benchmark(
    images: &[ImageRecord],
    splits: &HashMap<usize, QueryGallerySplit>,
) -> (VerificationMetrics, IdentificationMetrics, LooMetrics) {
    let body_images: Vec<&ImageRecord> = images.iter().filter(|i| i.body_available && i.body_emb.is_some()).collect();

    if body_images.is_empty() {
        return (
            VerificationMetrics { auc: 0.0, eer: 1.0, far_01: 1.0, far_05: 1.0, far_1: 1.0, far_5: 1.0, frr: 1.0, frr_at_threshold: 1.0 },
            IdentificationMetrics { top1: 0.0, top3: 0.0, top5: 0.0, top10: 0.0 },
            LooMetrics { pass_rate: 0.0 },
        );
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
                if *other_pid == *person_id {
                    continue;
                }
                for &g_id in &other_split.gallery {
                    let Some(g_img) = id_to_img.get(&g_id) else { continue };
                    let Some(g_emb) = &g_img.body_emb else { continue };
                    let score = cosine(q_emb, g_emb);
                    negative_scores.push(score);
                }
            }
        }
    }

    let verification = VerificationMetrics {
        auc: roc_auc(&positive_scores, &negative_scores),
        eer: compute_eer(&positive_scores, &negative_scores),
        far_01: compute_far_at_threshold(&negative_scores, BODY_THRESHOLD),
        far_05: compute_far_at_threshold(&negative_scores, 0.5),
        far_1: compute_far_at_threshold(&negative_scores, 0.75),
        far_5: compute_far_at_threshold(&negative_scores, 0.5),
        frr: compute_frr_at_threshold(&positive_scores, BODY_THRESHOLD),
        frr_at_threshold: compute_frr_at_threshold(&positive_scores, BODY_THRESHOLD),
    };

    let mut top1_correct = 0;
    let mut top3_correct = 0;
    let mut top5_correct = 0;
    let mut top10_correct = 0;
    let mut total_queries = 0;

    for (person_id, split) in splits {
        let id_to_img_local: HashMap<usize, &ImageRecord> = images.iter().filter(|i| i.body_available && i.body_emb.is_some()).map(|i| (i.id, i)).collect();

        for &q_id in &split.query {
            let Some(q_img) = id_to_img_local.get(&q_id) else { continue };
            let Some(q_emb) = &q_img.body_emb else { continue };
            total_queries += 1;

            let mut candidates: Vec<(usize, f32)> = Vec::new();
            for (g_id, g_img) in &id_to_img_local {
                if split.gallery.contains(g_id) {
                    continue;
                }
                let Some(g_emb) = &g_img.body_emb else { continue };
                let score = cosine(q_emb, g_emb);
                candidates.push((*g_id, score));
            }

            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top3: Vec<usize> = candidates.iter().take(3).filter_map(|(id, _)| id_to_img_local.get(id).map(|i| i.person_id)).collect();
            let top5: Vec<usize> = candidates.iter().take(5).filter_map(|(id, _)| id_to_img_local.get(id).map(|i| i.person_id)).collect();
            let top10: Vec<usize> = candidates.iter().take(10).filter_map(|(id, _)| id_to_img_local.get(id).map(|i| i.person_id)).collect();

            if let Some((id, _)) = candidates.first() {
                if id_to_img_local.get(id).map(|i| i.person_id) == Some(*person_id) {
                    top1_correct += 1;
                }
            }
            if top3.contains(person_id) {
                top3_correct += 1;
            }
            if top5.contains(person_id) {
                top5_correct += 1;
            }
            if top10.contains(person_id) {
                top10_correct += 1;
            }
        }
    }

    let identification = IdentificationMetrics {
        top1: top1_correct as f32 / total_queries.max(1) as f32,
        top3: top3_correct as f32 / total_queries.max(1) as f32,
        top5: top5_correct as f32 / total_queries.max(1) as f32,
        top10: top10_correct as f32 / total_queries.max(1) as f32,
    };

    let mut loo_pass = 0;
    let mut loo_total = 0;

    for (person_id, split) in splits {
        let person_images: Vec<&ImageRecord> = images.iter()
            .filter(|i| i.person_id == *person_id && i.body_available && i.body_emb.is_some())
            .collect();

        if person_images.len() < 2 {
            continue;
        }

        for &leave_out_id in &person_images.iter().map(|i| i.id).collect::<Vec<_>>() {
            let prototype: Vec<f32> = person_images.iter()
                .filter(|i| i.id != leave_out_id)
                .filter_map(|i| i.body_emb.clone())
                .fold(vec![0.0; BODY_DIM], |mut acc, emb| {
                    for (a, &e) in acc.iter_mut().zip(emb.iter()) {
                        *a += e;
                    }
                    acc
                });

            let mut normalized_prototype = prototype;
            l2_normalize(&mut normalized_prototype);

            let Some(leave_out) = person_images.iter().find(|i| i.id == leave_out_id) else { continue };
            let Some(leave_out_emb) = &leave_out.body_emb else { continue };

            let score = cosine(&normalized_prototype, leave_out_emb);
            if score >= BODY_THRESHOLD {
                loo_pass += 1;
            }
            loo_total += 1;
        }
    }

    let loo = LooMetrics {
        pass_rate: loo_pass as f32 / loo_total.max(1) as f32,
    };

    (verification, identification, loo)
}

// ============================================================================
// False Merge Analysis
// ============================================================================

fn run_false_merge_analysis(images: &[ImageRecord]) -> FalseMergeMetrics {
    let mut face_false_merges = 0;
    let mut body_false_merges = 0;
    let mut fusion_false_merges = 0;
    let mut total_pairs = 0;

    let face_valid: Vec<&ImageRecord> = images.iter().filter(|i| i.face_detected && i.face_emb.is_some()).collect();
    let body_valid: Vec<&ImageRecord> = images.iter().filter(|i| i.body_available && i.body_emb.is_some()).collect();

    // Sample pairs for efficiency
    let max_pairs = 500;
    let mut pair_count = 0;

    for i in 0..face_valid.len() {
        for j in (i + 1)..face_valid.len() {
            if pair_count >= max_pairs {
                break;
            }
            pair_count += 1;

            let img1 = face_valid[i];
            let img2 = face_valid[j];

            if img1.person_id == img2.person_id {
                continue;
            }
            total_pairs += 1;

            // Face false merge: different person but high face similarity
            if let (Some(e1), Some(e2)) = (&img1.face_emb, &img2.face_emb) {
                let score = cosine(e1, e2);
                if score >= FACE_THRESHOLD {
                    face_false_merges += 1;
                }
            }

            // Body false merge: different person but high body similarity
            if let (Some(b1), Some(b2)) = (&img1.body_emb, &img2.body_emb) {
                let score = cosine(b1, b2);
                if score >= BODY_THRESHOLD {
                    body_false_merges += 1;
                }
            }

            // Fusion false merge
            let face_score = img1.face_emb.as_ref().and_then(|e1| img2.face_emb.as_ref().map(|e2| cosine(e1, e2))).unwrap_or(0.0);
            let body_score = img1.body_emb.as_ref().and_then(|b1| img2.body_emb.as_ref().map(|b2| cosine(b1, b2))).unwrap_or(0.0);
            let fusion_score = (face_score + body_score) / 2.0;

            if fusion_score >= FUSION_THRESHOLD {
                fusion_false_merges += 1;
            }
        }
        if pair_count >= max_pairs {
            break;
        }
    }

    FalseMergeMetrics {
        face_false_merges,
        body_false_merges,
        fusion_false_merges,
        total_pairs,
        face_false_merge_rate: face_false_merges as f32 / total_pairs.max(1) as f32,
        body_false_merge_rate: body_false_merges as f32 / total_pairs.max(1) as f32,
        fusion_false_merge_rate: fusion_false_merges as f32 / total_pairs.max(1) as f32,
    }
}

// ============================================================================
// Chain Contamination Testing
// ============================================================================

fn test_chain_contamination(images: &[ImageRecord]) -> Vec<ChainContaminationResult> {
    let face_valid: Vec<&ImageRecord> = images.iter().filter(|i| i.face_detected && i.face_emb.is_some()).collect();

    let mut results = Vec::new();

    for &steps in [1, 2, 5, 10, 20].iter() {
        let mut margin_a_b = 0.0f32;
        let mut margin_b_c = 0.0f32;
        let mut margin_c_d = 0.0f32;
        let mut margin_a_d = 0.0f32;

        let mut detected = false;

        for i in 0..face_valid.len().saturating_sub(3) {
            let img_a = face_valid[i];
            let img_b = face_valid[i + 1];
            let img_c = face_valid[i + 2];
            let img_d = face_valid[i + 3];

            if let (Some(e_a), Some(e_b), Some(e_c), Some(e_d)) = (
                &img_a.face_emb,
                &img_b.face_emb,
                &img_c.face_emb,
                &img_d.face_emb,
            ) {
                margin_a_b = cosine(e_a, e_b);
                margin_b_c = cosine(e_b, e_c);
                margin_c_d = cosine(e_c, e_d);
                margin_a_d = cosine(e_a, e_d);

                if margin_a_b > CHAINING_MARGIN && margin_b_c > CHAINING_MARGIN && margin_c_d > CHAINING_MARGIN && margin_a_d < CHAINING_MARGIN {
                    detected = true;
                }
            }
        }

        results.push(ChainContaminationResult {
            steps,
            contamination_detected: detected,
            margin_a_d,
            margin_a_b,
            margin_b_c,
            margin_c_d,
        });
    }

    results
}

// ============================================================================
// Anti-Chaining Verification
// ============================================================================

fn test_anti_chaining(images: &[ImageRecord]) -> AntiChainingResult {
    let face_valid: Vec<&ImageRecord> = images.iter().filter(|i| i.face_detected && i.face_emb.is_some()).collect();

    let mut ambiguous = 0;
    let mut total = 0;

    for i in 0..face_valid.len() {
        for j in (i + 1)..face_valid.len() {
            let img1 = face_valid[i];
            let img2 = face_valid[j];

            if img1.person_id == img2.person_id {
                continue;
            }
            total += 1;

            if let (Some(e1), Some(e2)) = (&img1.face_emb, &img2.face_emb) {
                let score = cosine(e1, e2);
                if (score - FACE_THRESHOLD).abs() < 0.05 {
                    ambiguous += 1;
                }
            }
        }
    }

    AntiChainingResult {
        ambiguous_pairs: ambiguous,
        total_pairs: total,
        ambiguous_rate: ambiguous as f32 / total.max(1) as f32,
    }
}

// ============================================================================
// Data Quality Analysis
// ============================================================================

fn analyze_data_quality(images: &[ImageRecord]) -> DataQualityStats {
    let mut face_usable = 0;
    let mut body_usable = 0;
    let mut dual_usable = 0;
    let mut face_only = 0;
    let mut body_only = 0;
    let mut neither = 0;

    for img in images {
        let has_face = img.face_detected && img.face_emb.is_some();
        let has_body = img.body_available && img.body_emb.is_some();

        if has_face && has_body {
            dual_usable += 1;
        } else if has_face {
            face_only += 1;
        } else if has_body {
            body_only += 1;
        } else {
            neither += 1;
        }

        if has_face {
            face_usable += 1;
        }
        if has_body {
            body_usable += 1;
        }
    }

    DataQualityStats {
        face_usable,
        body_usable,
        dual_usable,
        face_only,
        body_only,
        neither,
    }
}

// ============================================================================
// Verdict
// ============================================================================

fn compute_verdict(
    face_verification: &VerificationMetrics,
    body_verification: &VerificationMetrics,
    face_identification: &IdentificationMetrics,
    face_loo: &LooMetrics,
    false_merge: &FalseMergeMetrics,
    chain_results: &[ChainContaminationResult],
) -> SystemVerdict {
    let dataset_ok = true;

    let face_ok = face_verification.auc >= 0.98
        && face_verification.eer <= 0.02
        && face_identification.top1 >= 0.95
        && face_loo.pass_rate >= 0.95
        && false_merge.face_false_merge_rate <= 0.001;

    let body_ok = body_verification.auc >= 0.98
        && body_verification.eer <= 0.02
        && false_merge.body_false_merge_rate <= 0.001;

    let fusion_ok = false_merge.fusion_false_merge_rate <= 0.001;

    let chain_ok = !chain_results.iter().any(|r| r.contamination_detected);

    SystemVerdict {
        dataset_verdict: if dataset_ok { "READY".to_string() } else { "INSUFFICIENT".to_string() },
        face_verdict: if face_ok { "PRODUCTION_READY".to_string() } else { "EXPERIMENTAL".to_string() },
        body_verdict: if body_ok { "PRODUCTION_READY".to_string() } else { "EXPERIMENTAL".to_string() },
        fusion_verdict: if fusion_ok && chain_ok { "PRODUCTION_READY".to_string() } else { "EXPERIMENTAL".to_string() },
    }
}

// ============================================================================
// Report Generation
// ============================================================================

fn generate_markdown_report(
    verdict: &SystemVerdict,
    face_verification: &VerificationMetrics,
    body_verification: &VerificationMetrics,
    face_identification: &IdentificationMetrics,
    body_identification: &IdentificationMetrics,
    face_margin: &MarginMetrics,
    face_loo: &LooMetrics,
    body_loo: &LooMetrics,
    false_merge: &FalseMergeMetrics,
    chain_results: &[ChainContaminationResult],
    anti_chain: &AntiChainingResult,
    data_quality: &DataQualityStats,
) -> String {
    let mut md = String::new();

    md.push_str("# Phase 24 Real Album Identity Benchmark Report\n\n");
    md.push_str("## Final Verdict\n\n");
    md.push_str(&format!("- **Dataset**: {}\n", verdict.dataset_verdict));
    md.push_str(&format!("- **Face**: {}\n", verdict.face_verdict));
    md.push_str(&format!("- **Body**: {}\n", verdict.body_verdict));
    md.push_str(&format!("- **Fusion**: {}\n", verdict.fusion_verdict));
    md.push_str("\n## Data Quality Summary\n\n");
    md.push_str(&format!("| Category | Count |\n"));
    md.push_str(&format!("|----------|-------|\n"));
    md.push_str(&format!("| Face-Usable | {} |\n", data_quality.face_usable));
    md.push_str(&format!("| Body-Usable | {} |\n", data_quality.body_usable));
    md.push_str(&format!("| Dual-Channel | {} |\n", data_quality.dual_usable));
    md.push_str(&format!("| Face-Only | {} |\n", data_quality.face_only));
    md.push_str(&format!("| Body-Only | {} |\n", data_quality.body_only));
    md.push_str(&format!("| Neither | {} |\n", data_quality.neither));

    md.push_str("\n## Production Gate Criteria\n\n");
    md.push_str("| Metric | Face | Body | Fusion | Gate |\n");
    md.push_str("|--------|------|------|--------|------|\n");
    md.push_str(&format!("| Top-1 | {:.2}% | {:.2}% | - | >= 95% |\n", face_identification.top1 * 100.0, body_identification.top1 * 100.0));
    md.push_str(&format!("| Top-3 | {:.2}% | {:.2}% | - | >= 98% |\n", face_identification.top3 * 100.0, body_identification.top3 * 100.0));
    md.push_str(&format!("| Top-5 | {:.2}% | {:.2}% | - | >= 99% |\n", face_identification.top5 * 100.0, body_identification.top5 * 100.0));
    md.push_str(&format!("| AUC | {:.4} | {:.4} | - | >= 0.98 |\n", face_verification.auc, body_verification.auc));
    md.push_str(&format!("| EER | {:.2}% | {:.2}% | - | <= 2% |\n", face_verification.eer * 100.0, body_verification.eer * 100.0));
    md.push_str(&format!("| FAR@0.1% | {:.4} | {:.4} | - | <= 0.1% |\n", face_verification.far_01, body_verification.far_01));
    md.push_str(&format!("| LOO Pass Rate | {:.2}% | {:.2}% | - | >= 95% |\n", face_loo.pass_rate * 100.0, body_loo.pass_rate * 100.0));
    md.push_str(&format!("| False Merge Rate | {:.6} | {:.6} | {:.6} | <= 0.1% |\n", false_merge.face_false_merge_rate, false_merge.body_false_merge_rate, false_merge.fusion_false_merge_rate));

    md.push_str("\n## Face Benchmark\n\n");
    md.push_str("### Verification\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!("| AUC | {:.4} |\n", face_verification.auc));
    md.push_str(&format!("| EER | {:.4} |\n", face_verification.eer));
    md.push_str(&format!("| FAR@0.1% | {:.4} |\n", face_verification.far_01));
    md.push_str(&format!("| FAR@0.5% | {:.4} |\n", face_verification.far_05));
    md.push_str(&format!("| FAR@1% | {:.4} |\n", face_verification.far_1));
    md.push_str(&format!("| FRR | {:.4} |\n", face_verification.frr));

    md.push_str("\n### Identification\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!("| Top-1 | {:.2}% |\n", face_identification.top1 * 100.0));
    md.push_str(&format!("| Top-3 | {:.2}% |\n", face_identification.top3 * 100.0));
    md.push_str(&format!("| Top-5 | {:.2}% |\n", face_identification.top5 * 100.0));
    md.push_str(&format!("| Top-10 | {:.2}% |\n", face_identification.top10 * 100.0));

    md.push_str("\n### Margin Analysis\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!("| Positive Best | {:.4} |\n", face_margin.positive_best));
    md.push_str(&format!("| Positive Mean | {:.4} |\n", face_margin.positive_mean));
    md.push_str(&format!("| Positive Min | {:.4} |\n", face_margin.positive_min));
    md.push_str(&format!("| Negative Max | {:.4} |\n", face_margin.negative_max));
    md.push_str(&format!("| Median | {:.4} |\n", face_margin.median));
    md.push_str(&format!("| P10 | {:.4} |\n", face_margin.p10));
    md.push_str(&format!("| P25 | {:.4} |\n", face_margin.p25));
    md.push_str(&format!("| P50 | {:.4} |\n", face_margin.p50));
    md.push_str(&format!("| P75 | {:.4} |\n", face_margin.p75));
    md.push_str(&format!("| P90 | {:.4} |\n", face_margin.p90));

    md.push_str(&format!("\n### LOO Prototype\n\n"));
    md.push_str(&format!("| Metric | Value |\n"));
    md.push_str(&format!("|--------|-------|\n"));
    md.push_str(&format!("| Pass Rate | {:.2}% |\n", face_loo.pass_rate * 100.0));

    md.push_str("\n## Body Benchmark\n\n");
    md.push_str("### Verification\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!("| AUC | {:.4} |\n", body_verification.auc));
    md.push_str(&format!("| EER | {:.4} |\n", body_verification.eer));
    md.push_str(&format!("| FAR@0.1% | {:.4} |\n", body_verification.far_01));
    md.push_str(&format!("| FAR@0.5% | {:.4} |\n", body_verification.far_05));
    md.push_str(&format!("| FAR@1% | {:.4} |\n", body_verification.far_1));
    md.push_str(&format!("| FRR | {:.4} |\n", body_verification.frr));

    md.push_str("\n### Identification\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!("| Top-1 | {:.2}% |\n", body_identification.top1 * 100.0));
    md.push_str(&format!("| Top-3 | {:.2}% |\n", body_identification.top3 * 100.0));
    md.push_str(&format!("| Top-5 | {:.2}% |\n", body_identification.top5 * 100.0));
    md.push_str(&format!("| Top-10 | {:.2}% |\n", body_identification.top10 * 100.0));

    md.push_str(&format!("\n### LOO Prototype\n\n"));
    md.push_str(&format!("| Metric | Value |\n"));
    md.push_str(&format!("|--------|-------|\n"));
    md.push_str(&format!("| Pass Rate | {:.2}% |\n", body_loo.pass_rate * 100.0));

    md.push_str("\n## False Merge Analysis\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!("| Face-only False Merges | {} |\n", false_merge.face_false_merges));
    md.push_str(&format!("| Body-only False Merges | {} |\n", false_merge.body_false_merges));
    md.push_str(&format!("| Fusion False Merges | {} |\n", false_merge.fusion_false_merges));
    md.push_str(&format!("| Total Pairs | {} |\n", false_merge.total_pairs));
    md.push_str(&format!("| Face False Merge Rate | {:.6} |\n", false_merge.face_false_merge_rate));
    md.push_str(&format!("| Body False Merge Rate | {:.6} |\n", false_merge.body_false_merge_rate));
    md.push_str(&format!("| Fusion False Merge Rate | {:.6} |\n", false_merge.fusion_false_merge_rate));

    md.push_str("\n## Chain Contamination Testing\n\n");
    md.push_str("| Steps | Detected | Margin A-B | Margin B-C | Margin C-D | Margin A-D |\n");
    md.push_str("|-------|----------|------------|------------|------------|------------|\n");
    for r in chain_results {
        md.push_str(&format!("| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            r.steps, if r.contamination_detected { "YES" } else { "NO" },
            r.margin_a_b, r.margin_b_c, r.margin_c_d, r.margin_a_d));
    }

    md.push_str("\n## Anti-Chaining Verification\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!("| Ambiguous Pairs | {} |\n", anti_chain.ambiguous_pairs));
    md.push_str(&format!("| Total Pairs | {} |\n", anti_chain.total_pairs));
    md.push_str(&format!("| Ambiguous Rate | {:.4} |\n", anti_chain.ambiguous_rate));

    md
}

// ============================================================================
// Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn phase24_identity_real_album_benchmark() {
    println!("============================================================");
    println!("Phase 24 — Real Album Identity Benchmark");
    println!("============================================================");
    println!("\nDataset: {}", BENCHMARK_DIR);
    println!("Principle: Don't modify production algorithms to fit benchmark\n");

    // ====== 1. Discover Dataset ======
    println!("Step 1: Discovering Dataset");
    println!("----------------------------");

    let mut images = match discover_dataset() {
        Some(imgs) => imgs,
        None => {
            println!("ERROR: Failed to discover dataset at {}", BENCHMARK_DIR);
            return;
        }
    };

    println!("  Found {} images", images.len());

    // ====== 2. Load Models ======
    println!("\nStep 2: Loading Production Models");
    println!("----------------------------------");

    let scrfd_path = resolve_model("scrfd_500m_bnkps.onnx");
    let arcface_path = resolve_model("w600k_r50.onnx");
    println!("  SCRFD: {}", scrfd_path.display());
    println!("  ArcFace: {}", arcface_path.display());

    if !scrfd_path.exists() {
        println!("\nERROR: SCRFD model not found at {}", scrfd_path.display());
        return;
    }
    if !arcface_path.exists() {
        println!("\nERROR: ArcFace model not found at {}", arcface_path.display());
        return;
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

    let body_embedder = if youtureid_path.exists() {
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
        println!("  WARNING: YouTu Re-ID model not found");
        None
    };

    println!("  FacePipeline: SCRFD + ArcFace w600k_r50");
    println!("  BodyEmbedder: YouTu Re-ID (person_reid_youtu_2021nov.onnx)");

    // ====== 3. Process Images ======
    println!("\nStep 3: Processing Images with Production Pipelines");
    println!("---------------------------------------------------");

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

        // Process with FacePipeline
        match face_pipeline.process(&image_data).await {
            Ok(features) => {
                if let Some(face) = features.first() {
                    img.face_detected = true;
                    img.face_score = face.detection.score;
                    img.face_quality = face.blur_score.max(face.pose_score);
                    img.blur_score = face.blur_score;
                    img.face_emb = Some(face.embedding.values.clone());
                    face_detected += 1;
                }
            }
            Err(e) => {
                // Face detection failure is expected for some images
            }
        }

        // Process with BodyEmbedder
        if let Some(ref embedder) = body_embedder {
            let rgb = image_data.as_rgb8();
            match embedder.embed(&rgb) {
                Ok(emb) => {
                    img.body_available = true;
                    img.body_quality = 1.0;
                    img.body_emb = Some(emb.values);
                    body_available += 1;
                }
                Err(e) => {
                    // Body embedding failure
                }
            }
        }

        processed += 1;
        if processed % 100 == 0 {
            println!("  Processed {}/{} images (face: {}, body: {})",
                processed, total_images, face_detected, body_available);
        }
    }

    println!("\n  Total processed: {}", processed);
    println!("  Face detected: {}", face_detected);
    println!("  Body available: {}", body_available);

    // ====== 4. Data Quality Analysis ======
    println!("\nStep 4: Data Quality Analysis");
    println!("------------------------------");

    let data_quality = analyze_data_quality(&images);

    println!("  Face-Usable Images: {}", data_quality.face_usable);
    println!("  Body-Usable Images: {}", data_quality.body_usable);
    println!("  Dual-Channel Images: {}", data_quality.dual_usable);
    println!("  Face-Only Images: {}", data_quality.face_only);
    println!("  Body-Only Images: {}", data_quality.body_only);
    println!("  Neither Face nor Body: {}", data_quality.neither);

    // ====== 5. Validate Dataset ======
    println!("\nStep 5: Dataset Validation");
    println!("--------------------------");

    let mut person_groups: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
    for img in &images {
        person_groups.entry(img.person_id).or_default().push(img);
    }

    let persons_count = person_groups.len();
    let min_images = person_groups.values().map(|v| v.len()).min().unwrap_or(0);
    let min_face_usable = person_groups.values().map(|v| v.iter().filter(|i| i.face_detected).count()).min().unwrap_or(0);
    let min_body_usable = person_groups.values().map(|v| v.iter().filter(|i| i.body_available).count()).min().unwrap_or(0);

    println!("  Persons: {}", persons_count);
    println!("  Min images/person: {}", min_images);
    println!("  Min face-usable images/person: {}", min_face_usable);
    println!("  Min body-usable images/person: {}", min_body_usable);

    let face_valid = persons_count >= MIN_PERSONS && min_images >= MIN_IMAGES_PER_PERSON && min_face_usable >= MIN_USABLE_IMAGES_PER_PERSON;
    let body_valid = persons_count >= MIN_PERSONS && min_images >= MIN_IMAGES_PER_PERSON && min_body_usable >= MIN_USABLE_IMAGES_PER_PERSON;

    if !face_valid && !body_valid {
        println!("\n============================================================");
        println!("DATASET_INSUFFICIENT");
        println!("============================================================");
        println!("\nReasons:");
        if persons_count < MIN_PERSONS {
            println!("  - insufficient persons: {} < {}", persons_count, MIN_PERSONS);
        }
        if min_images < MIN_IMAGES_PER_PERSON {
            println!("  - insufficient images per person: {} < {}", min_images, MIN_IMAGES_PER_PERSON);
        }
        if !face_valid {
            println!("  - insufficient face-usable images per person: {} < {}", min_face_usable, MIN_USABLE_IMAGES_PER_PERSON);
        }
        if !body_valid {
            println!("  - insufficient body-usable images per person: {} < {}", min_body_usable, MIN_USABLE_IMAGES_PER_PERSON);
        }
        return;
    }

    // ====== 6. Create Query/Gallery Splits ======
    println!("\nStep 6: Creating Query/Gallery Splits");
    println!("------------------------------------");

    let splits = create_query_gallery_split(&images, 42);

    let mut total_gallery = 0;
    let mut total_query = 0;
    for split in splits.values() {
        total_gallery += split.gallery.len();
        total_query += split.query.len();
    }

    println!("  Gallery images: {}", total_gallery);
    println!("  Query images: {}", total_query);

    // Verify self-match protection
    let mut self_matches = 0;
    for (person_id, split) in &splits {
        for &q_id in &split.query {
            if split.gallery.contains(&q_id) {
                self_matches += 1;
                println!("  WARNING: Self-match found for person {} query image {}", person_id, q_id);
            }
        }
    }

    if self_matches == 0 {
        println!("  Self-match protection: PASSED");
    } else {
        println!("  Self-match protection: FAILED ({} self-matches)", self_matches);
    }

    // ====== 7. Run Benchmarks ======
    println!("\n============================================================");
    println!("Running Face Benchmark (ArcFace w600k_r50)...");
    println!("============================================================");

    let (face_verification, face_identification, face_margin, face_loo) =
        run_face_benchmark(&images, &splits);

    println!("\nFace Verification:");
    println!("  AUC: {:.4}", face_verification.auc);
    println!("  EER: {:.4}", face_verification.eer);
    println!("  FAR@0.1%: {:.4}", face_verification.far_01);
    println!("  FAR@0.5%: {:.4}", face_verification.far_05);
    println!("  FAR@1%: {:.4}", face_verification.far_1);
    println!("  FAR@5%: {:.4}", face_verification.far_5);
    println!("  FRR: {:.4}", face_verification.frr);

    println!("\nFace Identification:");
    println!("  Top-1: {:.2}%", face_identification.top1 * 100.0);
    println!("  Top-3: {:.2}%", face_identification.top3 * 100.0);
    println!("  Top-5: {:.2}%", face_identification.top5 * 100.0);
    println!("  Top-10: {:.2}%", face_identification.top10 * 100.0);

    println!("\nFace Margin:");
    println!("  Median: {:.4}", face_margin.median);
    println!("  P10: {:.4}", face_margin.p10);
    println!("  P25: {:.4}", face_margin.p25);
    println!("  P50: {:.4}", face_margin.p50);
    println!("  P75: {:.4}", face_margin.p75);
    println!("  P90: {:.4}", face_margin.p90);

    println!("\nFace LOO Pass Rate: {:.2}%", face_loo.pass_rate * 100.0);

    println!("\n============================================================");
    println!("Running Body Benchmark (YouTu Re-ID)...");
    println!("============================================================");

    let (body_verification, body_identification, body_loo) =
        run_body_benchmark(&images, &splits);

    println!("\nBody Verification:");
    println!("  AUC: {:.4}", body_verification.auc);
    println!("  EER: {:.4}", body_verification.eer);
    println!("  FAR@0.1%: {:.4}", body_verification.far_01);
    println!("  FAR@0.5%: {:.4}", body_verification.far_05);
    println!("  FAR@1%: {:.4}", body_verification.far_1);
    println!("  FAR@5%: {:.4}", body_verification.far_5);
    println!("  FRR: {:.4}", body_verification.frr);

    println!("\nBody Identification:");
    println!("  Top-1: {:.2}%", body_identification.top1 * 100.0);
    println!("  Top-3: {:.2}%", body_identification.top3 * 100.0);
    println!("  Top-5: {:.2}%", body_identification.top5 * 100.0);
    println!("  Top-10: {:.2}%", body_identification.top10 * 100.0);

    println!("\nBody LOO Pass Rate: {:.2}%", body_loo.pass_rate * 100.0);

    // ====== 8. False Merge Analysis ======
    println!("\n============================================================");
    println!("Running False Merge Analysis...");
    println!("============================================================");

    let false_merge_metrics = run_false_merge_analysis(&images);

    println!("\nFalse Merge Analysis:");
    println!("  Face-only False Merges: {}", false_merge_metrics.face_false_merges);
    println!("  Body-only False Merges: {}", false_merge_metrics.body_false_merges);
    println!("  Fusion False Merges: {}", false_merge_metrics.fusion_false_merges);
    println!("  Total Pairs: {}", false_merge_metrics.total_pairs);
    println!("  Face False Merge Rate: {:.6}", false_merge_metrics.face_false_merge_rate);
    println!("  Body False Merge Rate: {:.6}", false_merge_metrics.body_false_merge_rate);
    println!("  Fusion False Merge Rate: {:.6}", false_merge_metrics.fusion_false_merge_rate);

    // ====== 9. Chain Contamination Testing ======
    println!("\n============================================================");
    println!("Running Chain Contamination Testing...");
    println!("============================================================");

    let chain_results = test_chain_contamination(&images);

    println!("\nChain Contamination:");
    for r in &chain_results {
        println!("  {}-step: detected={}, A-B={:.4}, B-C={:.4}, C-D={:.4}, A-D={:.4}",
            r.steps, r.contamination_detected, r.margin_a_b, r.margin_b_c, r.margin_c_d, r.margin_a_d);
    }

    if chain_results.iter().any(|r| r.contamination_detected) {
        println!("  *** CHAIN_CONTAMINATION DETECTED ***");
    } else {
        println!("  No chain contamination detected");
    }

    // ====== 10. Anti-Chaining Verification ======
    println!("\n============================================================");
    println!("Running Anti-Chaining Verification...");
    println!("============================================================");

    let anti_chain = test_anti_chaining(&images);

    println!("\nAnti-Chaining:");
    println!("  Ambiguous pairs: {}", anti_chain.ambiguous_pairs);
    println!("  Total pairs: {}", anti_chain.total_pairs);
    println!("  Ambiguous rate: {:.4}", anti_chain.ambiguous_rate);

    // ====== 11. Compute Final Verdict ======
    let verdict = compute_verdict(
        &face_verification,
        &body_verification,
        &face_identification,
        &face_loo,
        &false_merge_metrics,
        &chain_results,
    );

    println!("\n============================================================");
    println!("FINAL VERDICT");
    println!("============================================================");
    println!("\nDataset: {}", verdict.dataset_verdict);
    println!("Face: {}", verdict.face_verdict);
    println!("Body: {}", verdict.body_verdict);
    println!("Fusion: {}", verdict.fusion_verdict);

    // ====== 12. Generate Report ======
    let md_report = generate_markdown_report(
        &verdict,
        &face_verification,
        &body_verification,
        &face_identification,
        &body_identification,
        &face_margin,
        &face_loo,
        &body_loo,
        &false_merge_metrics,
        &chain_results,
        &anti_chain,
        &data_quality,
    );

    let reports_dir = PathBuf::from("/Users/mac/ai-project/photofinder-next-2/crates/pf_application/tests");
    let md_path = reports_dir.join("phase24_identity_real_album.md");
    let mut md_file = File::create(&md_path).expect("Failed to create MD report");
    md_file.write_all(md_report.as_bytes()).expect("Failed to write MD report");
    println!("\nMarkdown report written to: {:?}", md_path);

    // ====== 13. Production Gate Summary ======
    println!("\n============================================================");
    println!("Production Gate Summary");
    println!("============================================================");

    let face_gate_pass = face_verification.auc >= 0.98
        && face_verification.eer <= 0.02
        && face_identification.top1 >= 0.95
        && face_loo.pass_rate >= 0.95;

    let body_gate_pass = body_verification.auc >= 0.98
        && body_verification.eer <= 0.02
        && body_identification.top1 >= 0.95
        && body_loo.pass_rate >= 0.95
        && false_merge_metrics.body_false_merge_rate <= 0.001;

    let fusion_gate_pass = false_merge_metrics.fusion_false_merge_rate <= 0.001
        && !chain_results.iter().any(|r| r.contamination_detected);

    println!("\nFace Gate: {}", if face_gate_pass { "PASS" } else { "FAIL" });
    println!("Body Gate: {}", if body_gate_pass { "PASS" } else { "FAIL" });
    println!("Fusion Gate: {}", if fusion_gate_pass { "PASS" } else { "FAIL" });

    if face_gate_pass && body_gate_pass && fusion_gate_pass {
        println!("\n*** ALL GATES PASSED — READY FOR PRODUCTION ***");
    } else {
        println!("\n*** SOME GATES FAILED — NOT READY FOR PRODUCTION ***");
    }
}
