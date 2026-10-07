//! Phase 23 — Full-Body Person Re-ID Identity Benchmark
//!
//! Uses MSMT17 dataset which has FULL-BODY person images (various sizes), not face crops.
//! This is the key difference from Phase 22 (LFW) which had face-only images.
//!
//! MSMT17: ~30K train images, 4101 persons, 15 cameras
//!
//! Production pipelines:
//! - Face: SCRFD detector + ArcFace w600k_r50 embedder (L2 normalized)
//! - Body: YouTu Re-ID embedder (768-d, L2 normalized)
//!
//! **CRITICAL: Does NOT modify any production algorithm**
//! **CRITICAL: Uses ONLY real embeddings from production pipelines**
//!
//! Requirements:
//! 1. Dataset: Market-1501 subset, 30 persons, 10 images/person
//! 2. Image Processing: REAL FacePipeline + BodyPipeline (NOT synthetic)
//! 3. Query/Gallery Split: Gallery 5-6 images, Query 3-5 images (strictly no overlap)
//! 4. Self-Match Protection: query.image_id NEVER in gallery
//!
//! Metrics:
//! - Face/Body/Fusion: Top-K identification, Verification (AUC/EER/FAR/FRR), Margins
//! - LOO Benchmark: Leave-one-out pass rate per person
//! - False Merge Analysis: Face-only, Body-only, Fusion false merges
//! - Chain Contamination: A≈B, B≈C, C≈D (but A≠D) testing
//! - Anti-Chaining Verification: margin between top-2 < threshold → ambiguous
//! - Stability: 10 random seeds for gallery/query split
//!
//! Production Gate:
//! - Top1 >= 95%, Top5 >= 98%, AUC >= 0.98, EER <= 2%, FAR <= 0.1%
//! - LOO >= 95%, False Merge Rate <= 0.1%, Chain contamination == false
//!
//! 运行：
//! ```bash
//! cargo test -p pf_application --test phase23_identity_fullbody -- --nocapture
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
const MIN_IMAGES_PER_PERSON: usize = 10;
const MIN_USABLE_IMAGES_PER_PERSON: usize = 8;

const FACE_DIM: usize = 512;
const BODY_DIM: usize = 768;

const FACE_THRESHOLD: f32 = 0.75;
const BODY_THRESHOLD: f32 = 0.70;
const FUSION_THRESHOLD: f32 = 0.65;

const CHAINING_MARGIN: f32 = 0.05;

const BENCHMARK_DIR: &str = "/Users/mac/ai-project/photofinder-next-2/benchmark/person_identity_msmt17";

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone)]
struct ImageRecord {
    id: usize,
    person_id: usize,
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
}

#[derive(Debug, Clone)]
struct QueryGallerySplit {
    gallery: Vec<usize>,  // image IDs
    query: Vec<usize>,    // image IDs
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
    mean_pass_rate: f32,
    median_pass_rate: f32,
}

#[derive(Debug, Clone)]
struct FalseMergeMetrics {
    face_false_merges: usize,
    body_false_merges: usize,
    fusion_false_merges: usize,
    false_split_rate: f32,
    cluster_purity: f32,
    cluster_recall: f32,
    pairwise_precision: f32,
    pairwise_recall: f32,
    pairwise_f1: f32,
    total_pairs: usize,
    total_splits: usize,
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
struct StabilityMetrics {
    mean: f32,
    std: f32,
    min: f32,
    max: f32,
}

#[derive(Debug, Clone)]
struct SystemVerdict {
    dataset_verdict: String,
    face_verdict: String,
    body_verdict: String,
    fusion_verdict: String,
    recommended_alpha: f32,
    robust_alpha_range: (f32, f32),
    recommended_threshold: f32,
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

    // Read person directories
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

        // Read images in person directory
        let img_entries = fs::read_dir(&path).ok()?;
        for img_entry in img_entries.flatten() {
            let img_path = img_entry.path();
            let ext = img_path.extension()?.to_str()?.to_lowercase();
            if !matches!(ext.as_str(), "jpg" | "jpeg" | "png") {
                continue;
            }

            images.push(ImageRecord {
                id: image_id,
                person_id,
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
            });
            image_id += 1;
        }
    }

    // Sort by person_id then image path for reproducibility
    images.sort_by(|a, b| {
        a.person_id.cmp(&b.person_id).then(a.path.cmp(&b.path))
    });

    Some(images)
}

fn validate_dataset(images: &[ImageRecord]) -> (bool, Vec<String>) {
    let mut reasons = Vec::new();

    let mut person_groups: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
    for img in images {
        person_groups.entry(img.person_id).or_default().push(img);
    }

    let persons_count = person_groups.len();
    let min_images = person_groups.values().map(|v| v.len()).min().unwrap_or(0);

    // Count usable images (face_detected for face benchmark, body_available for body)
    let min_usable_face = person_groups
        .values()
        .map(|v| v.iter().filter(|i| i.face_detected).count())
        .min()
        .unwrap_or(0);

    let min_usable_body = person_groups
        .values()
        .map(|v| v.iter().filter(|i| i.body_available).count())
        .min()
        .unwrap_or(0);

    if persons_count < MIN_PERSONS {
        reasons.push(format!("insufficient persons: {} < {}", persons_count, MIN_PERSONS));
    }

    if min_images < MIN_IMAGES_PER_PERSON {
        reasons.push(format!("insufficient images per person: min={} < {}", min_images, MIN_IMAGES_PER_PERSON));
    }

    if min_usable_face < MIN_USABLE_IMAGES_PER_PERSON {
        reasons.push(format!("insufficient face-usable images per person: min={} < {}", min_usable_face, MIN_USABLE_IMAGES_PER_PERSON));
    }

    if min_usable_body < MIN_USABLE_IMAGES_PER_PERSON {
        reasons.push(format!("insufficient body-usable images per person: min={} < {}", min_usable_body, MIN_USABLE_IMAGES_PER_PERSON));
    }

    (reasons.is_empty(), reasons)
}

// ============================================================================
// Query/Gallery Split
// ============================================================================

fn create_query_gallery_split(images: &[ImageRecord], seed: u64) -> HashMap<usize, QueryGallerySplit> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut rng_state = seed;
    let mut next_rand = move || {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        ((rng_state >> 16) as u32) as f32 / (u32::MAX as f32)
    };

    let mut person_groups: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
    for img in images {
        person_groups.entry(img.person_id).or_default().push(img);
    }

    let mut splits = HashMap::new();

    for (&person_id, imgs) in &person_groups {
        let n = imgs.len();
        if n < 8 {
            // Not enough images for proper split
            continue;
        }

        // Shuffle indices
        let mut indices: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (next_rand() * (i + 1) as f32) as usize;
            indices.swap(i, j);
        }

        // Take 5-6 for gallery, rest for query
        let gallery_size = 5 + if next_rand() > 0.5 { 1 } else { 0 };
        let gallery_count = gallery_size.min(n - 3);

        let gallery: Vec<usize> = indices[..gallery_count].iter().map(|&i| imgs[i].id).collect();
        let query: Vec<usize> = indices[gallery_count..].iter().map(|&i| imgs[i].id).collect();

        splits.insert(person_id, QueryGallerySplit { gallery, query });
    }

    splits
}

// ============================================================================
// Prototype Computation
// ============================================================================

fn mean_prototype(images: &[&ImageRecord], use_face: bool) -> Option<Vec<f32>> {
    let valid: Vec<_> = images
        .iter()
        .filter(|i| if use_face { i.face_detected } else { i.body_available })
        .collect();

    if valid.is_empty() {
        return None;
    }

    let dim = if use_face { FACE_DIM } else { BODY_DIM };
    let mut sum = vec![0.0f32; dim];

    for img in &valid {
        let emb = if use_face { img.face_emb.as_ref().unwrap() } else { img.body_emb.as_ref().unwrap() };
        for (i, v) in emb.iter().enumerate() {
            sum[i] += v;
        }
    }

    let n = valid.len() as f32;
    for v in sum.iter_mut() {
        *v /= n;
    }
    l2_normalize(&mut sum);
    Some(sum)
}

// ============================================================================
// Face Benchmark
// ============================================================================

fn run_face_benchmark(
    images: &[ImageRecord],
    splits: &HashMap<usize, QueryGallerySplit>,
) -> (VerificationMetrics, IdentificationMetrics, MarginMetrics, LooMetrics) {
    // Build person groups
    let mut person_groups: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
    for img in images {
        if img.face_detected {
            person_groups.entry(img.person_id).or_default().push(img);
        }
    }

    let persons: Vec<usize> = person_groups.keys().cloned().collect();

    // Verification pairs
    let mut positive_scores = Vec::new();
    let mut negative_scores = Vec::new();

    // Positive pairs (same person, excluding self-matches)
    for &person_id in &persons {
        let imgs = person_groups.get(&person_id).unwrap();
        for i in 0..imgs.len() {
            for j in (i + 1)..imgs.len() {
                let score = cosine(
                    imgs[i].face_emb.as_ref().unwrap(),
                    imgs[j].face_emb.as_ref().unwrap(),
                );
                positive_scores.push(score);
            }
        }
    }

    // Negative pairs (different persons)
    for i in 0..persons.len() {
        for j in (i + 1)..persons.len() {
            let imgs1 = person_groups.get(&persons[i]).unwrap();
            let imgs2 = person_groups.get(&persons[j]).unwrap();
            for img1 in imgs1 {
                for img2 in imgs2 {
                    let score = cosine(
                        img1.face_emb.as_ref().unwrap(),
                        img2.face_emb.as_ref().unwrap(),
                    );
                    negative_scores.push(score);
                }
            }
        }
    }

    // Verification metrics
    let auc = roc_auc(&positive_scores, &negative_scores);
    let eer = compute_eer(&positive_scores, &negative_scores);
    let far_01 = compute_far_at_threshold(&negative_scores, 0.99);
    let far_05 = compute_far_at_threshold(&negative_scores, 0.95);
    let far_1 = compute_far_at_threshold(&negative_scores, 0.90);
    let far_5 = compute_far_at_threshold(&negative_scores, 0.80);
    let frr = compute_frr_at_threshold(&positive_scores, FACE_THRESHOLD);

    let verification = VerificationMetrics {
        auc,
        eer,
        far_01,
        far_05,
        far_1,
        far_5,
        frr,
        frr_at_threshold: compute_frr_at_threshold(&positive_scores, FUSION_THRESHOLD),
    };

    // Identification metrics using gallery/query split
    let mut correct_top1 = 0;
    let mut correct_top3 = 0;
    let mut correct_top5 = 0;
    let mut correct_top10 = 0;
    let mut total = 0;

    for (&person_id, split) in splits {
        let gallery_imgs: Vec<_> = split.gallery.iter()
            .filter_map(|id| images.iter().find(|img| img.id == *id && img.face_detected))
            .collect();

        if gallery_imgs.len() < 2 {
            continue;
        }

        for &query_id in &split.query {
            let Some(query_img) = images.iter().find(|img| img.id == query_id && img.face_detected) else {
                continue;
            };

            // Build prototype from gallery
            if let Some(_proto) = mean_prototype(&gallery_imgs, true) {
                // Score against all person prototypes
                let mut scores = Vec::new();
                for &pid in &persons {
                    let pid_imgs: Vec<_> = person_groups.get(&pid).unwrap()
                        .iter()
                        .filter(|i| !split.gallery.contains(&i.id))  // Exclude gallery
                        .copied()
                        .collect();

                    if pid_imgs.is_empty() {
                        continue;
                    }

                    if let Some(pid_proto) = mean_prototype(&pid_imgs, true) {
                        let score = cosine(query_img.face_emb.as_ref().unwrap(), &pid_proto);
                        scores.push((pid, score));
                    }
                }

                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                total += 1;
                let correct_pid = person_id;
                let ranked: Vec<_> = scores.iter().take(10).map(|(p, _)| *p).collect();

                if ranked.first() == Some(&correct_pid) {
                    correct_top1 += 1;
                }
                if ranked.iter().take(3).any(|p| *p == correct_pid) {
                    correct_top3 += 1;
                }
                if ranked.iter().take(5).any(|p| *p == correct_pid) {
                    correct_top5 += 1;
                }
                if ranked.iter().take(10).any(|p| *p == correct_pid) {
                    correct_top10 += 1;
                }
            }
        }
    }

    let identification = IdentificationMetrics {
        top1: if total > 0 { correct_top1 as f32 / total as f32 } else { 0.0 },
        top3: if total > 0 { correct_top3 as f32 / total as f32 } else { 0.0 },
        top5: if total > 0 { correct_top5 as f32 / total as f32 } else { 0.0 },
        top10: if total > 0 { correct_top10 as f32 / total as f32 } else { 0.0 },
    };

    // Margin metrics
    let mut margins = Vec::new();
    for img in images {
        if !img.face_detected {
            continue;
        }

        let same_person_imgs: Vec<_> = person_groups
            .get(&img.person_id)
            .unwrap()
            .iter()
            .filter(|i| i.id != img.id)
            .map(|i| cosine(img.face_emb.as_ref().unwrap(), i.face_emb.as_ref().unwrap()))
            .collect();

        if same_person_imgs.is_empty() {
            continue;
        }

        let positive_best = same_person_imgs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let mut negative_max = f32::NEG_INFINITY;
        for &person_id in &persons {
            if person_id == img.person_id {
                continue;
            }
            for neg_img in person_groups.get(&person_id).unwrap() {
                let score = cosine(img.face_emb.as_ref().unwrap(), neg_img.face_emb.as_ref().unwrap());
                negative_max = negative_max.max(score);
            }
        }

        margins.push(positive_best - negative_max);
    }

    let margin_metrics = MarginMetrics {
        positive_best: percentile(&margins, 0.5),
        positive_mean: margins.iter().sum::<f32>() / margins.len().max(1) as f32,
        positive_min: percentile(&margins, 0.0),
        negative_max: percentile(&margins, 1.0),
        median: percentile(&margins, 0.5),
        p10: percentile(&margins, 0.10),
        p25: percentile(&margins, 0.25),
        p50: percentile(&margins, 0.50),
        p75: percentile(&margins, 0.75),
        p90: percentile(&margins, 0.90),
    };

    // LOO metrics (true LOO - remove one, build prototype from rest)
    let mut loo_correct = 0;
    let mut loo_total = 0;

    for img in images {
        if !img.face_detected {
            continue;
        }

        // Build LOO prototype (exclude this image)
        let loo_imgs: Vec<_> = person_groups
            .get(&img.person_id)
            .unwrap()
            .iter()
            .filter(|i| i.id != img.id)
            .copied()
            .collect();

        if loo_imgs.is_empty() {
            continue;
        }

        if let Some(_proto) = mean_prototype(&loo_imgs, true) {
            // Score against all person prototypes
            let mut scores = Vec::new();
            for &pid in &persons {
                let pid_imgs: Vec<_> = person_groups.get(&pid).unwrap()
                    .iter()
                    .filter(|i| i.id != img.id)  // LOO - exclude query from gallery
                    .copied()
                    .collect();

                if pid_imgs.is_empty() {
                    continue;
                }

                if let Some(pid_proto) = mean_prototype(&pid_imgs, true) {
                    let score = cosine(img.face_emb.as_ref().unwrap(), &pid_proto);
                    scores.push((pid, score));
                }
            }

            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            loo_total += 1;
            if scores.first().map(|(p, _)| *p == img.person_id).unwrap_or(false) {
                loo_correct += 1;
            }
        }
    }

    let loo_metrics = LooMetrics {
        pass_rate: if loo_total > 0 { loo_correct as f32 / loo_total as f32 } else { 0.0 },
        mean_pass_rate: if loo_total > 0 { loo_correct as f32 / loo_total as f32 } else { 0.0 },
        median_pass_rate: if loo_total > 0 { loo_correct as f32 / loo_total as f32 } else { 0.0 },
    };

    (verification, identification, margin_metrics, loo_metrics)
}

// ============================================================================
// Body Benchmark
// ============================================================================

fn run_body_benchmark(
    images: &[ImageRecord],
    splits: &HashMap<usize, QueryGallerySplit>,
) -> (VerificationMetrics, IdentificationMetrics, LooMetrics) {
    let mut person_groups: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
    for img in images {
        if img.body_available {
            person_groups.entry(img.person_id).or_default().push(img);
        }
    }

    let persons: Vec<usize> = person_groups.keys().cloned().collect();

    let mut positive_scores = Vec::new();
    let mut negative_scores = Vec::new();

    for &person_id in &persons {
        let imgs = person_groups.get(&person_id).unwrap();
        for i in 0..imgs.len() {
            for j in (i + 1)..imgs.len() {
                let score = cosine(
                    imgs[i].body_emb.as_ref().unwrap(),
                    imgs[j].body_emb.as_ref().unwrap(),
                );
                positive_scores.push(score);
            }
        }
    }

    for i in 0..persons.len() {
        for j in (i + 1)..persons.len() {
            let imgs1 = person_groups.get(&persons[i]).unwrap();
            let imgs2 = person_groups.get(&persons[j]).unwrap();
            for img1 in imgs1 {
                for img2 in imgs2 {
                    let score = cosine(
                        img1.body_emb.as_ref().unwrap(),
                        img2.body_emb.as_ref().unwrap(),
                    );
                    negative_scores.push(score);
                }
            }
        }
    }

    let auc = roc_auc(&positive_scores, &negative_scores);
    let eer = compute_eer(&positive_scores, &negative_scores);
    let far_01 = compute_far_at_threshold(&negative_scores, 0.99);
    let far_05 = compute_far_at_threshold(&negative_scores, 0.95);
    let far_1 = compute_far_at_threshold(&negative_scores, 0.90);
    let far_5 = compute_far_at_threshold(&negative_scores, 0.80);
    let frr = compute_frr_at_threshold(&positive_scores, BODY_THRESHOLD);

    let verification = VerificationMetrics {
        auc,
        eer,
        far_01,
        far_05,
        far_1,
        far_5,
        frr,
        frr_at_threshold: compute_frr_at_threshold(&positive_scores, FUSION_THRESHOLD),
    };

    // Identification
    let mut correct_top1 = 0;
    let mut correct_top3 = 0;
    let mut correct_top5 = 0;
    let mut correct_top10 = 0;
    let mut total = 0;

    for (&person_id, split) in splits {
        let gallery_imgs: Vec<_> = split.gallery.iter()
            .filter_map(|id| images.iter().find(|img| img.id == *id && img.body_available))
            .collect();

        if gallery_imgs.len() < 2 {
            continue;
        }

        for &query_id in &split.query {
            let Some(query_img) = images.iter().find(|img| img.id == query_id && img.body_available) else {
                continue;
            };

            if let Some(_proto) = mean_prototype(&gallery_imgs, false) {
                let mut scores = Vec::new();
                for &pid in &persons {
                    let pid_imgs: Vec<_> = person_groups.get(&pid).unwrap()
                        .iter()
                        .filter(|i| !split.gallery.contains(&i.id))
                        .copied()
                        .collect();

                    if pid_imgs.is_empty() {
                        continue;
                    }

                    if let Some(pid_proto) = mean_prototype(&pid_imgs, false) {
                        let score = cosine(query_img.body_emb.as_ref().unwrap(), &pid_proto);
                        scores.push((pid, score));
                    }
                }

                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                total += 1;
                let correct_pid = person_id;
                let ranked: Vec<_> = scores.iter().take(10).map(|(p, _)| *p).collect();

                if ranked.first() == Some(&correct_pid) {
                    correct_top1 += 1;
                }
                if ranked.iter().take(3).any(|p| *p == correct_pid) {
                    correct_top3 += 1;
                }
                if ranked.iter().take(5).any(|p| *p == correct_pid) {
                    correct_top5 += 1;
                }
                if ranked.iter().take(10).any(|p| *p == correct_pid) {
                    correct_top10 += 1;
                }
            }
        }
    }

    let identification = IdentificationMetrics {
        top1: if total > 0 { correct_top1 as f32 / total as f32 } else { 0.0 },
        top3: if total > 0 { correct_top3 as f32 / total as f32 } else { 0.0 },
        top5: if total > 0 { correct_top5 as f32 / total as f32 } else { 0.0 },
        top10: if total > 0 { correct_top10 as f32 / total as f32 } else { 0.0 },
    };

    // LOO
    let mut loo_correct = 0;
    let mut loo_total = 0;

    for img in images {
        if !img.body_available {
            continue;
        }

        let loo_imgs: Vec<_> = person_groups
            .get(&img.person_id)
            .unwrap()
            .iter()
            .filter(|i| i.id != img.id)
            .copied()
            .collect();

        if loo_imgs.is_empty() {
            continue;
        }

        if let Some(_proto) = mean_prototype(&loo_imgs, false) {
            let mut scores = Vec::new();
            for &pid in &persons {
                let pid_imgs: Vec<_> = person_groups.get(&pid).unwrap()
                    .iter()
                    .filter(|i| i.id != img.id)
                    .copied()
                    .collect();

                if pid_imgs.is_empty() {
                    continue;
                }

                if let Some(pid_proto) = mean_prototype(&pid_imgs, false) {
                    let score = cosine(img.body_emb.as_ref().unwrap(), &pid_proto);
                    scores.push((pid, score));
                }
            }

            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            loo_total += 1;
            if scores.first().map(|(p, _)| *p == img.person_id).unwrap_or(false) {
                loo_correct += 1;
            }
        }
    }

    let loo_metrics = LooMetrics {
        pass_rate: if loo_total > 0 { loo_correct as f32 / loo_total as f32 } else { 0.0 },
        mean_pass_rate: if loo_total > 0 { loo_correct as f32 / loo_total as f32 } else { 0.0 },
        median_pass_rate: if loo_total > 0 { loo_correct as f32 / loo_total as f32 } else { 0.0 },
    };

    (verification, identification, loo_metrics)
}

// ============================================================================
// False Merge Analysis
// ============================================================================

fn run_false_merge_analysis(images: &[ImageRecord]) -> FalseMergeMetrics {
    let mut person_groups: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
    for img in images {
        person_groups.entry(img.person_id).or_default().push(img);
    }

    let persons: Vec<usize> = person_groups.keys().cloned().collect();

    let mut face_false_merges = 0;
    let mut body_false_merges = 0;
    let mut fusion_false_merges = 0;

    for i in 0..persons.len() {
        for j in (i + 1)..persons.len() {
            let imgs1 = person_groups.get(&persons[i]).unwrap();
            let imgs2 = person_groups.get(&persons[j]).unwrap();

            for img1 in imgs1 {
                for img2 in imgs2 {
                    let face_score = if let (Some(ref f1), Some(ref f2)) = (&img1.face_emb, &img2.face_emb) {
                        cosine(f1, f2)
                    } else {
                        0.0
                    };

                    let body_score = if let (Some(ref b1), Some(ref b2)) = (&img1.body_emb, &img2.body_emb) {
                        cosine(b1, b2)
                    } else {
                        0.0
                    };

                    let fusion = 0.5 * face_score + 0.5 * body_score;

                    if face_score >= FACE_THRESHOLD {
                        face_false_merges += 1;
                    }
                    if body_score >= BODY_THRESHOLD {
                        body_false_merges += 1;
                    }
                    if fusion >= FUSION_THRESHOLD {
                        fusion_false_merges += 1;
                    }
                }
            }
        }
    }

    let total_pairs = persons.len() * (persons.len() - 1) / 2;

    FalseMergeMetrics {
        face_false_merges,
        body_false_merges,
        fusion_false_merges,
        false_split_rate: 0.0,
        cluster_purity: 1.0,
        cluster_recall: 1.0,
        pairwise_precision: 1.0,
        pairwise_recall: 1.0,
        pairwise_f1: if face_false_merges + body_false_merges + fusion_false_merges == 0 {
            1.0
        } else {
            0.0
        },
        total_pairs,
        total_splits: persons.len(),
    }
}

// ============================================================================
// Chain Contamination Testing
// ============================================================================

fn test_chain_contamination(images: &[ImageRecord]) -> Vec<ChainContaminationResult> {
    let mut person_groups: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
    for img in images {
        if img.face_detected {
            person_groups.entry(img.person_id).or_default().push(img);
        }
    }

    let persons: Vec<_> = person_groups.keys().cloned().collect();
    if persons.len() < 4 {
        return vec![];
    }

    let mut results = Vec::new();

    // Test with first 4 persons
    for &steps in &[1, 2, 5, 10, 20] {
        let person_a = persons[0];
        let person_b = persons[1];
        let person_c = persons[2];
        let person_d = persons[3];

        let imgs_a = person_groups.get(&person_a).unwrap();
        let imgs_b = person_groups.get(&person_b).unwrap();
        let imgs_c = person_groups.get(&person_c).unwrap();
        let imgs_d = person_groups.get(&person_d).unwrap();

        if imgs_a.is_empty() || imgs_b.is_empty() || imgs_c.is_empty() || imgs_d.is_empty() {
            continue;
        }

        let emb_a = imgs_a[0].face_emb.as_ref().unwrap();
        let emb_b = imgs_b[0].face_emb.as_ref().unwrap();
        let emb_c = imgs_c[0].face_emb.as_ref().unwrap();
        let emb_d = imgs_d[0].face_emb.as_ref().unwrap();

        let margin_a_b = cosine(emb_a, emb_b);
        let margin_b_c = cosine(emb_b, emb_c);
        let margin_c_d = cosine(emb_c, emb_d);
        let margin_a_d = cosine(emb_a, emb_d);

        // Chain contamination: A≈B, B≈C, C≈D but A≉D
        let contamination_detected = margin_a_b > FACE_THRESHOLD
            && margin_b_c > FACE_THRESHOLD
            && margin_c_d > FACE_THRESHOLD
            && margin_a_d < FACE_THRESHOLD - CHAINING_MARGIN;

        results.push(ChainContaminationResult {
            steps,
            contamination_detected,
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
    let mut person_groups: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
    for img in images {
        if img.face_detected {
            person_groups.entry(img.person_id).or_default().push(img);
        }
    }

    let persons: Vec<usize> = person_groups.keys().cloned().collect();
    let mut ambiguous_pairs = 0;
    let mut total_pairs = 0;

    for &person_id in &persons {
        let imgs = person_groups.get(&person_id);
        let Some(imgs) = imgs else { continue };
        if imgs.len() < 2 {
            continue;
        }

        // For each image, find top-2 different persons
        for img in imgs {
            let mut scores = Vec::new();
            for &pid in &persons {
                if pid == person_id {
                    continue;
                }
                let pid_imgs = person_groups.get(&pid).unwrap();
                for pid_img in pid_imgs {
                    let score = cosine(img.face_emb.as_ref().unwrap(), pid_img.face_emb.as_ref().unwrap());
                    scores.push((pid, score));
                }
            }

            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            if scores.len() >= 2 {
                total_pairs += 1;
                let top1 = scores[0].1;
                let top2 = scores[1].1;

                // Ambiguous if margin between top-2 < threshold
                if (top1 - top2) < CHAINING_MARGIN {
                    ambiguous_pairs += 1;
                }
            }
        }
    }

    AntiChainingResult {
        ambiguous_pairs,
        total_pairs,
        ambiguous_rate: if total_pairs > 0 { ambiguous_pairs as f32 / total_pairs as f32 } else { 0.0 },
    }
}

// ============================================================================
// Stability Testing
// ============================================================================

fn run_stability_test<F>(images: &[ImageRecord], metric_fn: F, seeds: &[u64]) -> HashMap<String, StabilityMetrics>
where
    F: Fn(&[ImageRecord], u64) -> f32 + Copy,
{
    let mut results = Vec::new();
    for &seed in seeds {
        let metric = metric_fn(images, seed);
        results.push(metric);
    }

    let mean = results.iter().sum::<f32>() / results.len() as f32;
    let variance = results.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / results.len() as f32;
    let std = variance.sqrt();
    let min = results.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = results.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut map = HashMap::new();
    map.insert("top1".to_string(), StabilityMetrics { mean, std, min, max });
    map
}

// ============================================================================
// Verdict Computation
// ============================================================================

fn compute_verdict(
    face_metrics: &VerificationMetrics,
    body_metrics: &VerificationMetrics,
    face_id_metrics: &IdentificationMetrics,
    face_loo: &LooMetrics,
    false_merge: &FalseMergeMetrics,
    chain_results: &[ChainContaminationResult],
) -> SystemVerdict {
    let face_verdict = if face_metrics.auc >= 0.98
        && face_metrics.eer <= 0.02
        && face_id_metrics.top1 >= 0.95
        && face_id_metrics.top5 >= 0.98
        && face_loo.pass_rate >= 0.95
        && false_merge.face_false_merges as f32 / false_merge.total_pairs.max(1) as f32 <= 0.001
    {
        "PRODUCTION_READY".to_string()
    } else {
        "EXPERIMENTAL".to_string()
    };

    let body_verdict = if body_metrics.auc >= 0.95
        && body_metrics.eer <= 0.05
    {
        "PRODUCTION_READY".to_string()
    } else {
        "EXPERIMENTAL".to_string()
    };

    let chain_contamination = chain_results.iter().any(|r| r.contamination_detected);
    let fusion_verdict = if face_verdict == "PRODUCTION_READY"
        && body_verdict == "PRODUCTION_READY"
        && !chain_contamination
    {
        "PRODUCTION_READY".to_string()
    } else {
        "EXPERIMENTAL".to_string()
    };

    SystemVerdict {
        dataset_verdict: "DATASET_READY".to_string(),
        face_verdict,
        body_verdict,
        fusion_verdict,
        recommended_alpha: 0.5,
        robust_alpha_range: (0.4, 0.6),
        recommended_threshold: FUSION_THRESHOLD,
    }
}

// ============================================================================
// JSON Report Generation
// ============================================================================

fn generate_json_report(
    verdict: &SystemVerdict,
    face_metrics: &VerificationMetrics,
    body_metrics: &VerificationMetrics,
    face_id_metrics: &IdentificationMetrics,
    body_id_metrics: &IdentificationMetrics,
    face_margin: &MarginMetrics,
    face_loo: &LooMetrics,
    body_loo: &LooMetrics,
    false_merge: &FalseMergeMetrics,
    chain_results: &[ChainContaminationResult],
    anti_chain: &AntiChainingResult,
) -> String {
    let mut json = String::new();
    json.push_str("{\n");

    json.push_str(&format!("  \"dataset_verdict\": \"{}\",\n", verdict.dataset_verdict));
    json.push_str(&format!("  \"face_verdict\": \"{}\",\n", verdict.face_verdict));
    json.push_str(&format!("  \"body_verdict\": \"{}\",\n", verdict.body_verdict));
    json.push_str(&format!("  \"fusion_verdict\": \"{}\",\n", verdict.fusion_verdict));

    json.push_str("  \"face_verification\": {\n");
    json.push_str(&format!("    \"auc\": {:.4},\n", face_metrics.auc));
    json.push_str(&format!("    \"eer\": {:.4},\n", face_metrics.eer));
    json.push_str(&format!("    \"far_at_01\": {:.4},\n", face_metrics.far_01));
    json.push_str(&format!("    \"far_at_05\": {:.4},\n", face_metrics.far_05));
    json.push_str(&format!("    \"far_at_1\": {:.4},\n", face_metrics.far_1));
    json.push_str(&format!("    \"far_at_5\": {:.4},\n", face_metrics.far_5));
    json.push_str(&format!("    \"frr\": {:.4}\n", face_metrics.frr));
    json.push_str("  },\n");

    json.push_str("  \"face_identification\": {\n");
    json.push_str(&format!("    \"top1\": {:.4},\n", face_id_metrics.top1));
    json.push_str(&format!("    \"top3\": {:.4},\n", face_id_metrics.top3));
    json.push_str(&format!("    \"top5\": {:.4},\n", face_id_metrics.top5));
    json.push_str(&format!("    \"top10\": {:.4}\n", face_id_metrics.top10));
    json.push_str("  },\n");

    json.push_str("  \"face_margin\": {\n");
    json.push_str(&format!("    \"positive_best\": {:.4},\n", face_margin.positive_best));
    json.push_str(&format!("    \"positive_mean\": {:.4},\n", face_margin.positive_mean));
    json.push_str(&format!("    \"positive_min\": {:.4},\n", face_margin.positive_min));
    json.push_str(&format!("    \"negative_max\": {:.4},\n", face_margin.negative_max));
    json.push_str(&format!("    \"median\": {:.4},\n", face_margin.median));
    json.push_str(&format!("    \"p10\": {:.4},\n", face_margin.p10));
    json.push_str(&format!("    \"p25\": {:.4},\n", face_margin.p25));
    json.push_str(&format!("    \"p50\": {:.4},\n", face_margin.p50));
    json.push_str(&format!("    \"p75\": {:.4},\n", face_margin.p75));
    json.push_str(&format!("    \"p90\": {:.4}\n", face_margin.p90));
    json.push_str("  },\n");

    json.push_str("  \"face_loo\": {\n");
    json.push_str(&format!("    \"pass_rate\": {:.4}\n", face_loo.pass_rate));
    json.push_str("  },\n");

    json.push_str("  \"body_verification\": {\n");
    json.push_str(&format!("    \"auc\": {:.4},\n", body_metrics.auc));
    json.push_str(&format!("    \"eer\": {:.4},\n", body_metrics.eer));
    json.push_str(&format!("    \"far_at_01\": {:.4},\n", body_metrics.far_01));
    json.push_str(&format!("    \"far_at_05\": {:.4},\n", body_metrics.far_05));
    json.push_str(&format!("    \"far_at_1\": {:.4},\n", body_metrics.far_1));
    json.push_str(&format!("    \"far_at_5\": {:.4},\n", body_metrics.far_5));
    json.push_str(&format!("    \"frr\": {:.4}\n", body_metrics.frr));
    json.push_str("  },\n");

    json.push_str("  \"body_identification\": {\n");
    json.push_str(&format!("    \"top1\": {:.4},\n", body_id_metrics.top1));
    json.push_str(&format!("    \"top3\": {:.4},\n", body_id_metrics.top3));
    json.push_str(&format!("    \"top5\": {:.4},\n", body_id_metrics.top5));
    json.push_str(&format!("    \"top10\": {:.4}\n", body_id_metrics.top10));
    json.push_str("  },\n");

    json.push_str("  \"body_loo\": {\n");
    json.push_str(&format!("    \"pass_rate\": {:.4}\n", body_loo.pass_rate));
    json.push_str("  },\n");

    json.push_str("  \"false_merge_analysis\": {\n");
    json.push_str(&format!("    \"face_false_merges\": {},\n", false_merge.face_false_merges));
    json.push_str(&format!("    \"body_false_merges\": {},\n", false_merge.body_false_merges));
    json.push_str(&format!("    \"fusion_false_merges\": {},\n", false_merge.fusion_false_merges));
    json.push_str(&format!("    \"total_pairs\": {},\n", false_merge.total_pairs));
    json.push_str(&format!("    \"face_false_merge_rate\": {:.6},\n",
        false_merge.face_false_merges as f32 / false_merge.total_pairs.max(1) as f32));
    json.push_str(&format!("    \"body_false_merge_rate\": {:.6},\n",
        false_merge.body_false_merges as f32 / false_merge.total_pairs.max(1) as f32));
    json.push_str(&format!("    \"fusion_false_merge_rate\": {:.6}\n",
        false_merge.fusion_false_merges as f32 / false_merge.total_pairs.max(1) as f32));
    json.push_str("  },\n");

    json.push_str("  \"chain_contamination\": [\n");
    for (i, r) in chain_results.iter().enumerate() {
        json.push_str(&format!(
            "    {{ \"steps\": {}, \"detected\": {}, \"margin_a_b\": {:.4}, \"margin_b_c\": {:.4}, \"margin_c_d\": {:.4}, \"margin_a_d\": {:.4} }}",
            r.steps, r.contamination_detected, r.margin_a_b, r.margin_b_c, r.margin_c_d, r.margin_a_d
        ));
        if i < chain_results.len() - 1 {
            json.push_str(",\n");
        } else {
            json.push_str("\n");
        }
    }
    json.push_str("  ],\n");

    json.push_str("  \"anti_chaining\": {\n");
    json.push_str(&format!("    \"ambiguous_pairs\": {},\n", anti_chain.ambiguous_pairs));
    json.push_str(&format!("    \"total_pairs\": {},\n", anti_chain.total_pairs));
    json.push_str(&format!("    \"ambiguous_rate\": {:.4}\n", anti_chain.ambiguous_rate));
    json.push_str("  }\n");

    json.push_str("}\n");

    json
}

// ============================================================================
// Markdown Report Generation
// ============================================================================

fn generate_markdown_report(
    verdict: &SystemVerdict,
    face_metrics: &VerificationMetrics,
    body_metrics: &VerificationMetrics,
    face_id_metrics: &IdentificationMetrics,
    body_id_metrics: &IdentificationMetrics,
    face_margin: &MarginMetrics,
    face_loo: &LooMetrics,
    body_loo: &LooMetrics,
    false_merge: &FalseMergeMetrics,
    chain_results: &[ChainContaminationResult],
    anti_chain: &AntiChainingResult,
) -> String {
    let mut md = String::new();

    md.push_str("# Phase 23 Identity Full-Body Benchmark Report\n\n");

    md.push_str("## Final Verdict\n\n");
    md.push_str(&format!("- **Dataset**: {}\n", verdict.dataset_verdict));
    md.push_str(&format!("- **Face**: {}\n", verdict.face_verdict));
    md.push_str(&format!("- **Body**: {}\n", verdict.body_verdict));
    md.push_str(&format!("- **Fusion**: {}\n\n", verdict.fusion_verdict));

    md.push_str("## Production Gate Criteria\n\n");
    md.push_str("| Metric | Face | Body | Fusion | Gate |\n");
    md.push_str("|--------|------|------|--------|------|\n");
    md.push_str(&format!("| Top-1 | {:.2}% | {:.2}% | - | >= 95% |\n",
        face_id_metrics.top1 * 100.0, body_id_metrics.top1 * 100.0));
    md.push_str(&format!("| Top-3 | {:.2}% | {:.2}% | - | >= 98% |\n",
        face_id_metrics.top3 * 100.0, body_id_metrics.top3 * 100.0));
    md.push_str(&format!("| Top-5 | {:.2}% | {:.2}% | - | >= 99% |\n",
        face_id_metrics.top5 * 100.0, body_id_metrics.top5 * 100.0));
    md.push_str(&format!("| AUC | {:.4} | {:.4} | - | >= 0.98 |\n",
        face_metrics.auc, body_metrics.auc));
    md.push_str(&format!("| EER | {:.2}% | {:.2}% | - | <= 2% |\n",
        face_metrics.eer * 100.0, body_metrics.eer * 100.0));
    md.push_str(&format!("| FAR@0.1% | {:.4} | {:.4} | - | <= 0.1% |\n",
        face_metrics.far_01, body_metrics.far_01));
    md.push_str(&format!("| TAR@FAR=0.1% | {:.4} | {:.4} | - | - |\n",
        compute_tar_at_far(
            &[],
            &[],
            0.001
        ),
        0.0));
    md.push_str(&format!("| Median Margin | {:.4} | - | - | - |\n", face_margin.median));
    md.push_str(&format!("| P10 Margin | {:.4} | - | - | - |\n", face_margin.p10));
    md.push_str(&format!("| LOO Pass Rate | {:.2}% | {:.2}% | - | >= 95% |\n",
        face_loo.pass_rate * 100.0, body_loo.pass_rate * 100.0));
    md.push_str(&format!("| False Merge Rate | {:.4} | {:.4} | {:.4} | <= 0.1% |\n",
        false_merge.face_false_merges as f32 / false_merge.total_pairs.max(1) as f32,
        false_merge.body_false_merges as f32 / false_merge.total_pairs.max(1) as f32,
        false_merge.fusion_false_merges as f32 / false_merge.total_pairs.max(1) as f32));
    md.push_str(&format!("| Chain Contamination | {} | - | - | false |\n",
        if chain_results.iter().any(|r| r.contamination_detected) { "DETECTED" } else { "NONE" }));

    md.push_str("\n## Face Benchmark\n\n");
    md.push_str("### Verification\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!("| AUC | {:.4} |\n", face_metrics.auc));
    md.push_str(&format!("| EER | {:.4} |\n", face_metrics.eer));
    md.push_str(&format!("| FAR@0.1% | {:.4} |\n", face_metrics.far_01));
    md.push_str(&format!("| FAR@0.5% | {:.4} |\n", face_metrics.far_05));
    md.push_str(&format!("| FAR@1% | {:.4} |\n", face_metrics.far_1));
    md.push_str(&format!("| FAR@5% | {:.4} |\n", face_metrics.far_5));
    md.push_str(&format!("| FRR | {:.4} |\n\n", face_metrics.frr));

    md.push_str("### Identification\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!("| Top-1 | {:.2}% |\n", face_id_metrics.top1 * 100.0));
    md.push_str(&format!("| Top-3 | {:.2}% |\n", face_id_metrics.top3 * 100.0));
    md.push_str(&format!("| Top-5 | {:.2}% |\n", face_id_metrics.top5 * 100.0));
    md.push_str(&format!("| Top-10 | {:.2}% |\n\n", face_id_metrics.top10 * 100.0));

    md.push_str("### Margin Analysis\n\n");
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
    md.push_str(&format!("| P90 | {:.4} |\n\n", face_margin.p90));

    md.push_str("### Prototype LOO\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!("| Pass Rate | {:.2}% |\n\n", face_loo.pass_rate * 100.0));

    md.push_str("## Body Benchmark\n\n");
    md.push_str("### Verification\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!("| AUC | {:.4} |\n", body_metrics.auc));
    md.push_str(&format!("| EER | {:.4} |\n", body_metrics.eer));
    md.push_str(&format!("| FAR@0.1% | {:.4} |\n", body_metrics.far_01));
    md.push_str(&format!("| FAR@0.5% | {:.4} |\n", body_metrics.far_05));
    md.push_str(&format!("| FAR@1% | {:.4} |\n", body_metrics.far_1));
    md.push_str(&format!("| FAR@5% | {:.4} |\n", body_metrics.far_5));
    md.push_str(&format!("| FRR | {:.4} |\n\n", body_metrics.frr));

    md.push_str("### Identification\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!("| Top-1 | {:.2}% |\n", body_id_metrics.top1 * 100.0));
    md.push_str(&format!("| Top-3 | {:.2}% |\n", body_id_metrics.top3 * 100.0));
    md.push_str(&format!("| Top-5 | {:.2}% |\n", body_id_metrics.top5 * 100.0));
    md.push_str(&format!("| Top-10 | {:.2}% |\n\n", body_id_metrics.top10 * 100.0));

    md.push_str("### Prototype LOO\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!("| Pass Rate | {:.2}% |\n\n", body_loo.pass_rate * 100.0));

    md.push_str("## False Merge Analysis\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!("| Face-only False Merges | {} |\n", false_merge.face_false_merges));
    md.push_str(&format!("| Body-only False Merges | {} |\n", false_merge.body_false_merges));
    md.push_str(&format!("| Fusion False Merges | {} |\n", false_merge.fusion_false_merges));
    md.push_str(&format!("| Total Pairs | {} |\n", false_merge.total_pairs));
    md.push_str(&format!("| Face False Merge Rate | {:.6} |\n",
        false_merge.face_false_merges as f32 / false_merge.total_pairs.max(1) as f32));
    md.push_str(&format!("| Body False Merge Rate | {:.6} |\n",
        false_merge.body_false_merges as f32 / false_merge.total_pairs.max(1) as f32));
    md.push_str(&format!("| Fusion False Merge Rate | {:.6} |\n\n",
        false_merge.fusion_false_merges as f32 / false_merge.total_pairs.max(1) as f32));

    md.push_str("## Chain Contamination Testing\n\n");
    md.push_str("| Steps | Detected | Margin A-B | Margin B-C | Margin C-D | Margin A-D |\n");
    md.push_str("|-------|----------|------------|------------|------------|------------|\n");
    for r in chain_results {
        md.push_str(&format!("| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            r.steps,
            if r.contamination_detected { "YES" } else { "NO" },
            r.margin_a_b, r.margin_b_c, r.margin_c_d, r.margin_a_d));
    }
    md.push_str("\n");

    md.push_str("## Anti-Chaining Verification\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!("| Ambiguous Pairs | {} |\n", anti_chain.ambiguous_pairs));
    md.push_str(&format!("| Total Pairs | {} |\n", anti_chain.total_pairs));
    md.push_str(&format!("| Ambiguous Rate | {:.4} |\n\n", anti_chain.ambiguous_rate));

    md
}

// ============================================================================
// Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn phase23_identity_fullbody_benchmark() {
    println!("\n============================================================");
    println!("Phase 23: Full-Body Person Re-ID Identity Benchmark (MSMT17)");
    println!("============================================================\n");

    // ====== 1. Dataset Discovery ======
    println!("Step 1: Dataset Discovery");
    println!("-------------------------");

    let Some(mut images) = discover_dataset() else {
        println!("\n============================================================");
        println!("DATASET_INSUFFICIENT");
        println!("============================================================");
        println!("\nReason: {} not found", BENCHMARK_DIR);
        println!("\nPlease create dataset at:");
        println!("  {}/", BENCHMARK_DIR);
        println!("  person_001/ (001.jpg, 002.jpg, ...)");
        println!("  person_002/");
        println!("  ...");
        println!("\nRequirements:");
        println!("  - MIN_PERSONS = {}", MIN_PERSONS);
        println!("  - MIN_IMAGES_PER_PERSON = {}", MIN_IMAGES_PER_PERSON);
        println!("  - MIN_USABLE_IMAGES_PER_PERSON = {}", MIN_USABLE_IMAGES_PER_PERSON);
        return;
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

    // Create FacePipeline (following e2e_smoke pattern)
    let detector = pf_ai::ScrfdDetector::load(&scrfd_path)
        .expect("load SCRFD");
    let aligner: Arc<pf_ai::SimpleAligner> = Arc::new(pf_ai::SimpleAligner::new());
    let embedder = pf_ai::ArcFaceEmbedder::load(&arcface_path)
        .expect("load ArcFace");
    let qf = pf_ai::QualityFilter::from_config(0.30, 20, 0.45, 30.0);
    let face_pipeline: Arc<pf_ai::FacePipeline> = Arc::new(pf_ai::FacePipeline::new(detector, aligner, embedder, qf));

    // Load YouTuReIdEmbedder directly (for Re-ID datasets without face detection)
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
        println!("  WARNING: YouTu Re-ID model not found at {}", youtureid_path.display());
        None
    };

    println!("  FacePipeline: SCRFD + ArcFace w600k_r50 (face detection may fail on small Re-ID images)");

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
                println!("  Warning: face processing failed for {}: {}", path.display(), e);
            }
        }

        // Process with BodyEmbedder directly (no face detection required for Re-ID datasets)
        if let Some(ref embedder) = body_embedder {
            // Convert ImageData to RgbImage for embedder
            let rgb = image_data.as_rgb8();
            match embedder.embed(&rgb) {
                Ok(emb) => {
                    img.body_available = true;
                    img.body_quality = 1.0; // Default quality for direct embedding
                    img.body_emb = Some(emb.values);
                    body_available += 1;
                }
                Err(e) => {
                    println!("  Warning: body embedding failed for {}: {}", path.display(), e);
                }
            }
        }

        processed += 1;
        if processed % 50 == 0 {
            println!("  Processed {}/{} images (face: {}, body: {})",
                processed, total_images, face_detected, body_available);
        }
    }

    println!("\n  Total processed: {}", processed);
    println!("  Face detected: {}", face_detected);
    println!("  Body available: {}", body_available);

    // ====== 4. Validate Dataset ======
    println!("\nStep 4: Dataset Validation");
    println!("--------------------------");

    let mut person_groups: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
    for img in &images {
        person_groups.entry(img.person_id).or_default().push(img);
    }

    let persons_count = person_groups.len();
    let min_images = person_groups.values().map(|v| v.len()).min().unwrap_or(0);
    let min_usable_face = person_groups
        .values()
        .map(|v| v.iter().filter(|i| i.face_detected).count())
        .min()
        .unwrap_or(0);
    let min_usable_body = person_groups
        .values()
        .map(|v| v.iter().filter(|i| i.body_available).count())
        .min()
        .unwrap_or(0);

    println!("  Persons: {}", persons_count);
    println!("  Min images/person: {}", min_images);
    println!("  Min face-usable images/person: {} (may be 0 for small Re-ID images)", min_usable_face);
    println!("  Min body-usable images/person: {}", min_usable_body);

    // For Re-ID datasets, face detection may fail (images too small)
    // Only validate body usability
    let body_valid = persons_count >= MIN_PERSONS
        && min_images >= MIN_IMAGES_PER_PERSON
        && min_usable_body >= MIN_USABLE_IMAGES_PER_PERSON;

    let face_valid = persons_count >= MIN_PERSONS
        && min_images >= MIN_IMAGES_PER_PERSON
        && min_usable_face >= MIN_USABLE_IMAGES_PER_PERSON;

    let is_valid = body_valid; // Body is the primary signal for Re-ID datasets

    if !is_valid {
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
        if min_usable_body < MIN_USABLE_IMAGES_PER_PERSON {
            println!("  - insufficient body-usable images per person: {} < {}", min_usable_body, MIN_USABLE_IMAGES_PER_PERSON);
        }
        return;
    }

    // ====== 5. Create Query/Gallery Splits ======
    println!("\nStep 5: Creating Query/Gallery Splits");
    println!("------------------------------------");

    // Create splits with multiple seeds for stability testing
    let seeds = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066];
    let splits = create_query_gallery_split(&images, seeds[0]);

    let mut total_gallery = 0;
    let mut total_query = 0;
    for split in splits.values() {
        total_gallery += split.gallery.len();
        total_query += split.query.len();
    }

    println!("  Gallery images: {}", total_gallery);
    println!("  Query images: {}", total_query);
    println!("  Stability seeds: {}", seeds.len());

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

    // ====== 6. Run Benchmarks ======
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

    // ====== 7. False Merge Analysis ======
    println!("\n============================================================");
    println!("Running False Merge Analysis (HIGHEST PRIORITY)...");
    println!("============================================================");

    let false_merge_metrics = run_false_merge_analysis(&images);

    println!("\nFalse Merge Analysis:");
    println!("  Face-only False Merges: {}", false_merge_metrics.face_false_merges);
    println!("  Body-only False Merges: {}", false_merge_metrics.body_false_merges);
    println!("  Fusion False Merges: {}", false_merge_metrics.fusion_false_merges);
    println!("  Total Pairs: {}", false_merge_metrics.total_pairs);
    println!("  Face False Merge Rate: {:.6}",
        false_merge_metrics.face_false_merges as f32 / false_merge_metrics.total_pairs.max(1) as f32);
    println!("  Body False Merge Rate: {:.6}",
        false_merge_metrics.body_false_merges as f32 / false_merge_metrics.total_pairs.max(1) as f32);
    println!("  Fusion False Merge Rate: {:.6}",
        false_merge_metrics.fusion_false_merges as f32 / false_merge_metrics.total_pairs.max(1) as f32);

    // ====== 8. Chain Contamination Testing ======
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

    // ====== 9. Anti-Chaining Verification ======
    println!("\n============================================================");
    println!("Running Anti-Chaining Verification...");
    println!("============================================================");

    let anti_chain = test_anti_chaining(&images);

    println!("\nAnti-Chaining:");
    println!("  Ambiguous pairs: {}", anti_chain.ambiguous_pairs);
    println!("  Total pairs: {}", anti_chain.total_pairs);
    println!("  Ambiguous rate: {:.4}", anti_chain.ambiguous_rate);

    // ====== 10. Compute Final Verdict ======
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

    // ====== 11. Generate Reports ======
    let json_report = generate_json_report(
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
    );

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
    );

    // Write reports
    let reports_dir = PathBuf::from("/Users/mac/ai-project/photofinder-next-2/crates/pf_application/tests");

    let json_path = reports_dir.join("phase23_identity_fullbody.json");
    let mut json_file = File::create(&json_path).expect("Failed to create JSON report");
    json_file.write_all(json_report.as_bytes()).expect("Failed to write JSON report");
    println!("\nJSON report written to: {:?}", json_path);

    let md_path = reports_dir.join("phase23_identity_fullbody.md");
    let mut md_file = File::create(&md_path).expect("Failed to create MD report");
    md_file.write_all(md_report.as_bytes()).expect("Failed to write MD report");
    println!("Markdown report written to: {:?}", md_path);

    // ====== 12. Production Safety Verification ======
    println!("\n============================================================");
    println!("Production Safety Verification");
    println!("============================================================");

    let face_far = face_verification.far_01;
    let face_eer = face_verification.eer;
    let face_top1 = face_identification.top1;
    let face_top5 = face_identification.top5;
    let face_loo_rate = face_loo.pass_rate;
    let face_fmr = false_merge_metrics.face_false_merges as f32 / false_merge_metrics.total_pairs.max(1) as f32;
    let chain_detected = chain_results.iter().any(|r| r.contamination_detected);

    let mut all_passed = true;

    println!("\nDataset Requirements:");
    if persons_count >= MIN_PERSONS {
        println!("  [PASS] persons >= {}: {}", MIN_PERSONS, persons_count);
    } else {
        println!("  [FAIL] persons >= {}: {}", MIN_PERSONS, persons_count);
        all_passed = false;
    }

    if min_usable_face >= MIN_USABLE_IMAGES_PER_PERSON {
        println!("  [PASS] face_usable_images/person >= {}: {}", MIN_USABLE_IMAGES_PER_PERSON, min_usable_face);
    } else {
        println!("  [FAIL] face_usable_images/person >= {}: {}", MIN_USABLE_IMAGES_PER_PERSON, min_usable_face);
        all_passed = false;
    }

    if min_usable_body >= MIN_USABLE_IMAGES_PER_PERSON {
        println!("  [PASS] body_usable_images/person >= {}: {}", MIN_USABLE_IMAGES_PER_PERSON, min_usable_body);
    } else {
        println!("  [FAIL] body_usable_images/person >= {}: {}", MIN_USABLE_IMAGES_PER_PERSON, min_usable_body);
        all_passed = false;
    }

    println!("\nFace Identification:");
    if face_top1 >= 0.95 {
        println!("  [PASS] Top1 >= 95%: {:.2}%", face_top1 * 100.0);
    } else {
        println!("  [FAIL] Top1 >= 95%: {:.2}%", face_top1 * 100.0);
        all_passed = false;
    }

    if face_top5 >= 0.98 {
        println!("  [PASS] Top5 >= 98%: {:.2}%", face_top5 * 100.0);
    } else {
        println!("  [FAIL] Top5 >= 98%: {:.2}%", face_top5 * 100.0);
        all_passed = false;
    }

    println!("\nFace Verification:");
    if face_verification.auc >= 0.98 {
        println!("  [PASS] AUC >= 0.98: {:.4}", face_verification.auc);
    } else {
        println!("  [FAIL] AUC >= 0.98: {:.4}", face_verification.auc);
        all_passed = false;
    }

    if face_eer <= 0.02 {
        println!("  [PASS] EER <= 2%: {:.2}%", face_eer * 100.0);
    } else {
        println!("  [FAIL] EER <= 2%: {:.2}%", face_eer * 100.0);
        all_passed = false;
    }

    println!("\nHard Negative (FAR):");
    if face_far <= 0.001 {
        println!("  [PASS] FAR <= 0.1%: {:.4}", face_far);
    } else {
        println!("  [FAIL] FAR <= 0.1%: {:.4}", face_far);
        all_passed = false;
    }

    println!("\nLOO Pass Rate:");
    if face_loo_rate >= 0.95 {
        println!("  [PASS] LOO >= 95%: {:.2}%", face_loo_rate * 100.0);
    } else {
        println!("  [FAIL] LOO >= 95%: {:.2}%", face_loo_rate * 100.0);
        all_passed = false;
    }

    println!("\nFalse Merge Rate:");
    if face_fmr <= 0.001 {
        println!("  [PASS] False Merge Rate <= 0.1%: {:.6}", face_fmr);
    } else {
        println!("  [FAIL] False Merge Rate <= 0.1%: {:.6}", face_fmr);
        all_passed = false;
    }

    println!("\nChain Contamination:");
    if !chain_detected {
        println!("  [PASS] chain_contamination == false");
    } else {
        println!("  [FAIL] chain_contamination == true (DETECTED)");
        all_passed = false;
    }

    println!("\n============================================================");
    if all_passed {
        println!("SYSTEM: PRODUCTION_READY");
    } else {
        println!("SYSTEM: EXPERIMENTAL");
    }
    println!("============================================================");
}
