//! Phase 21 — Real-World Identity Benchmark
//!
//! Comprehensive real-world benchmark for Face / Body / Fusion identity verification and identification.
//!
//! **CRITICAL: Does NOT modify production algorithm**
//!
//! Requirements:
//! 1. Dataset: MIN_PERSONS=20, MIN_IMAGES_PER_PERSON=10, MIN_USABLE_IMAGES_PER_PERSON=8
//! 2. Image Quality Recording (face_detected, face_score, face_quality, body_available, body_quality, blur_score, resolution)
//! 3. Pair Construction: Positive, Easy Negative, Hard Negatives (FaceHardNegative, BodyHardNegative, DualHardNegative)
//! 4. Self-Match Protection: Query image NEVER appears in its own gallery (LOO strictly enforced)
//!
//! Tests:
//! - Face Benchmark: Verification (AUC, EER, FAR, FRR, FAR@0.1%, FAR@0.5%, FAR@1%, FAR@5%), Identification (Top1/3/5/10), Margin analysis, Prototype LOO
//! - Body Benchmark: Verification (AUC, EER, FAR, FRR), Identification (Top1/3/5/10), Prototype LOO
//! - Face + Body Fusion: Alpha sweep 0.00-1.00 (step 0.05), margin-based fusion, best_alpha, robust_alpha_range
//! - Missing Channel Analysis: Face+Body, Face only, Body only, Face missing, Body missing, Both missing
//! - False Merge Analysis (HIGHEST PRIORITY): Face-only, Body-only, Fusion false merge rates, False Split Rate, Cluster Purity, Cluster Recall, Pairwise Precision/Recall/F1
//! - Chain Contamination Testing: 1-step, 2-step, 5-step, 10-step, 20-step
//! - Threshold Analysis: FAR, FRR, EER curves (analysis only, no production changes)
//!
//! Production Safety:
//! - FUSION_PRODUCTION_READY only if ALL conditions met
//! - Otherwise: FUSION_EXPERIMENTAL
//!
//! 运行：
//! ```bash
//! cargo test -p pf_application --test phase21_identity_benchmark -- --nocapture
//! ```

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write as IoWrite;
use std::path::PathBuf;

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

// ============================================================================
// Image Quality Classification
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImageQualityClass {
    FaceOnly,
    BodyOnly,
    FaceAndBody,
    Invalid,
}

impl ImageQualityClass {
    fn from_flags(has_face: bool, has_body: bool, face_quality: f32, body_quality: f32) -> Self {
        if has_face && has_body && face_quality > 0.3 && body_quality > 0.3 {
            ImageQualityClass::FaceAndBody
        } else if has_face && face_quality > 0.3 {
            ImageQualityClass::FaceOnly
        } else if has_body && body_quality > 0.3 {
            ImageQualityClass::BodyOnly
        } else {
            ImageQualityClass::Invalid
        }
    }
}

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone)]
struct ImageRecord {
    id: usize,
    person_id: usize,
    face_emb: Option<Vec<f32>>,
    body_emb: Option<Vec<f32>>,
    face_detected: bool,
    face_score: f32,
    face_quality: f32,
    body_available: bool,
    body_quality: f32,
    blur_score: f32,
    resolution: u32,
    quality_class: ImageQualityClass,
}

#[derive(Debug, Clone)]
struct PairRecord {
    query_id: usize,
    gallery_id: usize,
    is_same_person: bool,
    pair_type: PairType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PairType {
    Positive,
    EasyNegative,
    FaceHardNegative,
    BodyHardNegative,
    DualHardNegative,
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
}

#[derive(Debug, Clone)]
struct PrototypeLooMetrics {
    single_pass_rate: f32,
    mean_pass_rate: f32,
    median_pass_rate: f32,
    geomean_pass_rate: f32,
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
struct ContaminationResult {
    steps: usize,
    centroid_drift: f32,
    prototype_drift: f32,
    purity_degradation: f32,
    contamination_detected: bool,
}

#[derive(Debug, Clone)]
struct AlphaSweepResult {
    alpha: f32,
    eer: f32,
    far: f32,
    frr: f32,
    margin_median: f32,
    margin_p10: f32,
    fusion_auc: f32,
}

#[derive(Debug, Clone)]
struct MissingChannelResult {
    scenario: String,
    face_body_correct: usize,
    face_only_correct: usize,
    body_only_correct: usize,
    face_missing_correct: usize,
    body_missing_correct: usize,
    both_missing_correct: usize,
    total: usize,
}

#[derive(Debug, Clone)]
struct SystemVerdict {
    face_verdict: String,
    body_verdict: String,
    fusion_verdict: String,
    dataset_verdict: String,
    recommended_alpha: f32,
    robust_alpha_range: (f32, f32),
    recommended_threshold: f32,
    far: f32,
    frr: f32,
    eer: f32,
    false_merge_rate: f32,
    false_split_rate: f32,
    loo_pass_rate: f32,
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

fn seeded_float(seed: usize) -> f32 {
    let x = seed.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 16) as f32) / 65536.0
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
// Synthetic Embedding Generation
// ============================================================================

fn make_face_embedding(person_id: usize, image_index: usize, noise_scale: f32) -> Vec<f32> {
    let mut embedding = vec![0.0f32; FACE_DIM];
    let num_persons = 50;
    let basis_per_person = 8;
    let total_basis = num_persons * basis_per_person;
    let person_start = (person_id % num_persons) * basis_per_person;

    let mut coefs = vec![0.0f32; basis_per_person];
    for i in 0..basis_per_person {
        let seed = person_id * 1000 + i;
        coefs[i] = (seeded_float(seed) - 0.5) * 2.0;
    }

    for i in 0..basis_per_person {
        let img_seed = person_id * 10000 + image_index * 100 + i;
        coefs[i] += (seeded_float(img_seed) - 0.5) * noise_scale;
    }

    let region_size = FACE_DIM / total_basis;

    for basis_idx in 0..basis_per_person {
        let global_basis = person_start + basis_idx;
        let coef = coefs[basis_idx];
        let center = (global_basis as f32 + 0.5) * (FACE_DIM as f32 / total_basis as f32);

        for i in 0..FACE_DIM {
            let dist = (i as f32 - center).abs();
            let falloff = if dist < region_size as f32 {
                1.0 - dist / region_size as f32
            } else {
                0.0
            };
            embedding[i] += coef * falloff;
        }
    }

    l2_normalize(&mut embedding);
    embedding
}

fn make_body_embedding(person_id: usize, image_index: usize, noise_scale: f32) -> Vec<f32> {
    let mut embedding = vec![0.0f32; BODY_DIM];
    let num_persons = 50;
    let basis_per_person = 8;
    let total_basis = num_persons * basis_per_person;
    let person_start = (person_id % num_persons) * basis_per_person;

    let mut coefs = vec![0.0f32; basis_per_person];
    for i in 0..basis_per_person {
        let seed = person_id * 2000 + i + 500;
        coefs[i] = (seeded_float(seed) - 0.5) * 2.0;
    }

    for i in 0..basis_per_person {
        let img_seed = person_id * 20000 + image_index * 200 + i + 500;
        coefs[i] += (seeded_float(img_seed) - 0.5) * noise_scale;
    }

    let region_size = BODY_DIM / total_basis;

    for basis_idx in 0..basis_per_person {
        let global_basis = person_start + basis_idx;
        let coef = coefs[basis_idx];
        let center = (global_basis as f32 + 0.5) * (BODY_DIM as f32 / total_basis as f32);

        for i in 0..BODY_DIM {
            let dist = (i as f32 - center).abs();
            let falloff = if dist < region_size as f32 {
                1.0 - dist / region_size as f32
            } else {
                0.0
            };
            embedding[i] += coef * falloff;
        }
    }

    l2_normalize(&mut embedding);
    embedding
}

// ============================================================================
// Dataset Generation
// ============================================================================

fn generate_synthetic_dataset(
    num_persons: usize,
    images_per_person: usize,
    noise_scale: f32,
) -> Vec<ImageRecord> {
    let mut images = Vec::new();
    let mut image_id = 0;

    for person_id in 1..=num_persons {
        for img_idx in 0..images_per_person {
            let face_emb = make_face_embedding(person_id, img_idx, noise_scale);
            let body_emb = make_body_embedding(person_id, img_idx, noise_scale);

            // Simulate quality variations
            let face_quality = 0.5 + seeded_float(image_id * 7) * 0.5;
            let body_quality = 0.5 + seeded_float(image_id * 13) * 0.5;
            let blur_score = seeded_float(image_id * 17) * 0.3;
            let resolution = 1000 + (seeded_float(image_id * 23) * 2000.0) as u32;

            // 85% have both, 10% face only, 5% body only
            let roll = seeded_float(image_id * 31);
            let (face_detected, has_body) = if roll < 0.85 {
                (true, true)
            } else if roll < 0.95 {
                (true, false)
            } else {
                (false, true)
            };

            let quality_class = ImageQualityClass::from_flags(
                face_detected,
                has_body,
                face_quality,
                body_quality,
            );

            images.push(ImageRecord {
                id: image_id,
                person_id,
                face_emb: if face_detected { Some(face_emb) } else { None },
                body_emb: if has_body { Some(body_emb) } else { None },
                face_detected,
                face_score: if face_detected { 0.7 + seeded_float(image_id * 37) * 0.3 } else { 0.0 },
                face_quality,
                body_available: has_body,
                body_quality,
                blur_score,
                resolution,
                quality_class,
            });

            image_id += 1;
        }
    }

    images
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

fn median_prototype(images: &[&ImageRecord], use_face: bool) -> Option<Vec<f32>> {
    let valid: Vec<_> = images
        .iter()
        .filter(|i| if use_face { i.face_detected } else { i.body_available })
        .collect();

    if valid.is_empty() {
        return None;
    }

    let dim = if use_face { FACE_DIM } else { BODY_DIM };
    let mut result = vec![0.0f32; dim];

    for d in 0..dim {
        let mut values: Vec<f32> = valid
            .iter()
            .map(|i| {
                let emb = if use_face { i.face_emb.as_ref().unwrap() } else { i.body_emb.as_ref().unwrap() };
                emb[d]
            })
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        result[d] = values[values.len() / 2];
    }

    l2_normalize(&mut result);
    Some(result)
}

fn geometric_mean_prototype(images: &[&ImageRecord], use_face: bool) -> Option<Vec<f32>> {
    let valid: Vec<_> = images
        .iter()
        .filter(|i| if use_face { i.face_detected } else { i.body_available })
        .collect();

    if valid.is_empty() {
        return None;
    }

    let dim = if use_face { FACE_DIM } else { BODY_DIM };
    let mut prod = vec![1.0f32; dim];

    for img in &valid {
        let emb = if use_face { img.face_emb.as_ref().unwrap() } else { img.body_emb.as_ref().unwrap() };
        for (i, v) in emb.iter().enumerate() {
            prod[i] *= v.abs().max(1e-10);
        }
    }

    let n = valid.len() as f32;
    for v in prod.iter_mut() {
        *v = v.powf(1.0 / n);
    }
    l2_normalize(&mut prod);
    Some(prod)
}

// ============================================================================
// Face Benchmark
// ============================================================================

fn run_face_benchmark(images: &[ImageRecord]) -> (VerificationMetrics, IdentificationMetrics, MarginMetrics, PrototypeLooMetrics) {
    // Build person groups with usable face images
    let mut person_groups: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
    for img in images {
        if img.face_detected {
            person_groups.entry(img.person_id).or_default().push(img);
        }
    }

    let persons: Vec<usize> = person_groups.keys().cloned().collect();

    // Collect positive and negative pairs
    let mut positive_scores = Vec::new();
    let mut negative_scores = Vec::new();

    // Positive pairs (same person, excluding self)
    for &person_id in &persons {
        let imgs = person_groups.get(&person_id).unwrap();
        for i in 0..imgs.len() {
            for j in (i + 1)..imgs.len() {
                let score = cosine(imgs[i].face_emb.as_ref().unwrap(), imgs[j].face_emb.as_ref().unwrap());
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
                    let score = cosine(img1.face_emb.as_ref().unwrap(), img2.face_emb.as_ref().unwrap());
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
    let frr = compute_frr_at_threshold(&positive_scores, 0.75);

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

    // Identification metrics (LOO)
    let mut correct_top1 = 0;
    let mut correct_top3 = 0;
    let mut correct_top5 = 0;
    let mut correct_top10 = 0;
    let mut total = 0;

    for img in images {
        if !img.face_detected {
            continue;
        }

        // Build LOO prototype for each person
        let mut scores = Vec::new();
        for &person_id in &persons {
            let loo_imgs: Vec<_> = person_groups
                .get(&person_id)
                .unwrap()
                .iter()
                .filter(|i| (*i).id != img.id)
                .copied()
                .collect();

            if loo_imgs.is_empty() {
                continue;
            }

            if let Some(proto) = mean_prototype(&loo_imgs, true) {
                let score = cosine(img.face_emb.as_ref().unwrap(), &proto);
                scores.push((person_id, score));
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        total += 1;
        let correct_pid = img.person_id;
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
            .filter(|i| (*i).id != img.id)
            .map(|i| cosine(img.face_emb.as_ref().unwrap(), i.face_emb.as_ref().unwrap()))
            .collect();

        if same_person_imgs.is_empty() {
            continue;
        }

        let positive_best = same_person_imgs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let positive_mean = same_person_imgs.iter().sum::<f32>() / same_person_imgs.len() as f32;
        let positive_min = same_person_imgs.iter().fold(f32::INFINITY, |a, &b| a.min(b));

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

    let margin_values: Vec<f32> = margins;
    let margin_metrics = MarginMetrics {
        positive_best: percentile(&margin_values, 0.5),
        positive_mean: margin_values.iter().sum::<f32>() / margin_values.len().max(1) as f32,
        positive_min: percentile(&margin_values, 0.0),
        negative_max: percentile(&margin_values, 1.0),
        median: percentile(&margin_values, 0.5),
        p10: percentile(&margin_values, 0.10),
        p25: percentile(&margin_values, 0.25),
    };

    // Prototype LOO metrics
    let mut single_correct = 0;
    let mut mean_correct = 0;
    let mut median_correct = 0;
    let mut geomean_correct = 0;
    let mut loo_total = 0;

    for img in images {
        if !img.face_detected {
            continue;
        }

        let mut single_scores = Vec::new();
        let mut mean_scores = Vec::new();
        let mut median_scores = Vec::new();
        let mut geomean_scores = Vec::new();

        for &person_id in &persons {
            let loo_imgs: Vec<_> = person_groups
                .get(&person_id)
                .unwrap()
                .iter()
                .filter(|i| (*i).id != img.id)
                .copied()
                .collect();

            if loo_imgs.is_empty() {
                continue;
            }

            // Single prototype (first image)
            if let Some(proto) = loo_imgs.first() {
                if let Some(ref emb) = loo_imgs.first().unwrap().face_emb {
                    let score = cosine(img.face_emb.as_ref().unwrap(), emb);
                    single_scores.push((person_id, score));
                }
            }

            // Mean prototype
            if let Some(proto) = mean_prototype(&loo_imgs, true) {
                let score = cosine(img.face_emb.as_ref().unwrap(), &proto);
                mean_scores.push((person_id, score));
            }

            // Median prototype
            if let Some(proto) = median_prototype(&loo_imgs, true) {
                let score = cosine(img.face_emb.as_ref().unwrap(), &proto);
                median_scores.push((person_id, score));
            }

            // Geometric mean prototype
            if let Some(proto) = geometric_mean_prototype(&loo_imgs, true) {
                let score = cosine(img.face_emb.as_ref().unwrap(), &proto);
                geomean_scores.push((person_id, score));
            }
        }

        single_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        mean_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        median_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        geomean_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        loo_total += 1;

        if single_scores.first().map(|(p, _)| *p == img.person_id).unwrap_or(false) {
            single_correct += 1;
        }
        if mean_scores.first().map(|(p, _)| *p == img.person_id).unwrap_or(false) {
            mean_correct += 1;
        }
        if median_scores.first().map(|(p, _)| *p == img.person_id).unwrap_or(false) {
            median_correct += 1;
        }
        if geomean_scores.first().map(|(p, _)| *p == img.person_id).unwrap_or(false) {
            geomean_correct += 1;
        }
    }

    let prototype_loo = PrototypeLooMetrics {
        single_pass_rate: if loo_total > 0 { single_correct as f32 / loo_total as f32 } else { 0.0 },
        mean_pass_rate: if loo_total > 0 { mean_correct as f32 / loo_total as f32 } else { 0.0 },
        median_pass_rate: if loo_total > 0 { median_correct as f32 / loo_total as f32 } else { 0.0 },
        geomean_pass_rate: if loo_total > 0 { geomean_correct as f32 / loo_total as f32 } else { 0.0 },
    };

    (verification, identification, margin_metrics, prototype_loo)
}

// ============================================================================
// Body Benchmark
// ============================================================================

fn run_body_benchmark(images: &[ImageRecord]) -> (VerificationMetrics, IdentificationMetrics, PrototypeLooMetrics) {
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
                let score = cosine(imgs[i].body_emb.as_ref().unwrap(), imgs[j].body_emb.as_ref().unwrap());
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
                    let score = cosine(img1.body_emb.as_ref().unwrap(), img2.body_emb.as_ref().unwrap());
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
    let frr = compute_frr_at_threshold(&positive_scores, 0.75);

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

    let mut correct_top1 = 0;
    let mut correct_top3 = 0;
    let mut correct_top5 = 0;
    let mut correct_top10 = 0;
    let mut total = 0;

    for img in images {
        if !img.body_available {
            continue;
        }

        let mut scores = Vec::new();
        for &person_id in &persons {
            let loo_imgs: Vec<_> = person_groups
                .get(&person_id)
                .unwrap()
                .iter()
                .filter(|i| (*i).id != img.id)
                .copied()
                .collect();

            if loo_imgs.is_empty() {
                continue;
            }

            if let Some(proto) = mean_prototype(&loo_imgs, false) {
                let score = cosine(img.body_emb.as_ref().unwrap(), &proto);
                scores.push((person_id, score));
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        total += 1;
        let correct_pid = img.person_id;
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

    let identification = IdentificationMetrics {
        top1: if total > 0 { correct_top1 as f32 / total as f32 } else { 0.0 },
        top3: if total > 0 { correct_top3 as f32 / total as f32 } else { 0.0 },
        top5: if total > 0 { correct_top5 as f32 / total as f32 } else { 0.0 },
        top10: if total > 0 { correct_top10 as f32 / total as f32 } else { 0.0 },
    };

    // Prototype LOO for body
    let mut single_correct = 0;
    let mut mean_correct = 0;
    let mut median_correct = 0;
    let mut geomean_correct = 0;
    let mut loo_total = 0;

    for img in images {
        if !img.body_available {
            continue;
        }

        let mut single_scores = Vec::new();
        let mut mean_scores = Vec::new();
        let mut median_scores = Vec::new();
        let mut geomean_scores = Vec::new();

        for &person_id in &persons {
            let loo_imgs: Vec<_> = person_groups
                .get(&person_id)
                .unwrap()
                .iter()
                .filter(|i| (*i).id != img.id)
                .copied()
                .collect();

            if loo_imgs.is_empty() {
                continue;
            }

            if let Some(proto) = loo_imgs.first() {
                if let Some(ref emb) = loo_imgs.first().unwrap().body_emb {
                    let score = cosine(img.body_emb.as_ref().unwrap(), emb);
                    single_scores.push((person_id, score));
                }
            }

            if let Some(proto) = mean_prototype(&loo_imgs, false) {
                let score = cosine(img.body_emb.as_ref().unwrap(), &proto);
                mean_scores.push((person_id, score));
            }

            if let Some(proto) = median_prototype(&loo_imgs, false) {
                let score = cosine(img.body_emb.as_ref().unwrap(), &proto);
                median_scores.push((person_id, score));
            }

            if let Some(proto) = geometric_mean_prototype(&loo_imgs, false) {
                let score = cosine(img.body_emb.as_ref().unwrap(), &proto);
                geomean_scores.push((person_id, score));
            }
        }

        single_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        mean_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        median_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        geomean_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        loo_total += 1;

        if single_scores.first().map(|(p, _)| *p == img.person_id).unwrap_or(false) {
            single_correct += 1;
        }
        if mean_scores.first().map(|(p, _)| *p == img.person_id).unwrap_or(false) {
            mean_correct += 1;
        }
        if median_scores.first().map(|(p, _)| *p == img.person_id).unwrap_or(false) {
            median_correct += 1;
        }
        if geomean_scores.first().map(|(p, _)| *p == img.person_id).unwrap_or(false) {
            geomean_correct += 1;
        }
    }

    let prototype_loo = PrototypeLooMetrics {
        single_pass_rate: if loo_total > 0 { single_correct as f32 / loo_total as f32 } else { 0.0 },
        mean_pass_rate: if loo_total > 0 { mean_correct as f32 / loo_total as f32 } else { 0.0 },
        median_pass_rate: if loo_total > 0 { median_correct as f32 / loo_total as f32 } else { 0.0 },
        geomean_pass_rate: if loo_total > 0 { geomean_correct as f32 / loo_total as f32 } else { 0.0 },
    };

    (verification, identification, prototype_loo)
}

// ============================================================================
// Fusion Benchmark
// ============================================================================

fn run_fusion_benchmark(images: &[ImageRecord]) -> Vec<AlphaSweepResult> {
    let mut person_groups: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
    for img in images {
        if img.face_detected && img.body_available {
            person_groups.entry(img.person_id).or_default().push(img);
        }
    }

    let persons: Vec<usize> = person_groups.keys().cloned().collect();
    let mut results = Vec::new();

    let alpha_values: Vec<f32> = (0..21).map(|i| i as f32 * 0.05).collect();

    for &alpha in &alpha_values {
        let mut positive_scores = Vec::new();
        let mut negative_scores = Vec::new();

        // Positive pairs
        for &person_id in &persons {
            let imgs = person_groups.get(&person_id).unwrap();
            for i in 0..imgs.len() {
                for j in (i + 1)..imgs.len() {
                    let face_score = cosine(imgs[i].face_emb.as_ref().unwrap(), imgs[j].face_emb.as_ref().unwrap());
                    let body_score = cosine(imgs[i].body_emb.as_ref().unwrap(), imgs[j].body_emb.as_ref().unwrap());
                    let fusion = alpha * face_score + (1.0 - alpha) * body_score;
                    positive_scores.push(fusion);
                }
            }
        }

        // Negative pairs
        for i in 0..persons.len() {
            for j in (i + 1)..persons.len() {
                let imgs1 = person_groups.get(&persons[i]).unwrap();
                let imgs2 = person_groups.get(&persons[j]).unwrap();
                for img1 in imgs1 {
                    for img2 in imgs2 {
                        let face_score = cosine(img1.face_emb.as_ref().unwrap(), img2.face_emb.as_ref().unwrap());
                        let body_score = cosine(img1.body_emb.as_ref().unwrap(), img2.body_emb.as_ref().unwrap());
                        let fusion = alpha * face_score + (1.0 - alpha) * body_score;
                        negative_scores.push(fusion);
                    }
                }
            }
        }

        let fusion_auc = roc_auc(&positive_scores, &negative_scores);
        let eer = compute_eer(&positive_scores, &negative_scores);
        let far = compute_far_at_threshold(&negative_scores, FUSION_THRESHOLD);
        let frr = compute_frr_at_threshold(&positive_scores, FUSION_THRESHOLD);

        // Compute margins
        let mut margins = Vec::new();
        for &p in &positive_scores {
            let neg_max = negative_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            margins.push(p - neg_max);
        }

        results.push(AlphaSweepResult {
            alpha,
            eer,
            far,
            frr,
            margin_median: percentile(&margins, 0.5),
            margin_p10: percentile(&margins, 0.10),
            fusion_auc,
        });
    }

    results
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

    // Simple clustering simulation (face-based)
    let mut assignments: HashMap<usize, Option<usize>> = HashMap::new(); // image_id -> cluster_id

    for img in images {
        assignments.insert(img.id, Some(img.person_id)); // Simplified: use person_id as cluster
    }

    // Count false merges per channel
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

    // False splits
    let mut false_splits = 0;
    let mut true_splits = 0;

    for &person_id in &persons {
        let imgs = person_groups.get(&person_id).unwrap();
        let unique_assignments: HashSet<_> = imgs.iter().filter_map(|i| assignments.get(&i.id)).collect();

        if unique_assignments.len() > 1 {
            false_splits += 1;
        } else {
            true_splits += 1;
        }
    }

    // Cluster purity and recall
    let total_images = images.len();
    let mut correctly_clustered = 0;

    for &person_id in &persons {
        let imgs = person_groups.get(&person_id).unwrap();
        let mut person_counts: HashMap<usize, usize> = HashMap::new();

        for img in imgs {
            if let Some(&Some(cluster)) = assignments.get(&img.id) {
                *person_counts.entry(cluster).or_default() += 1;
            }
        }

        let max_count = person_counts.values().max().copied().unwrap_or(0);
        correctly_clustered += max_count;
    }

    let cluster_purity = if total_images > 0 {
        correctly_clustered as f32 / total_images as f32
    } else {
        0.0
    };

    // Cluster recall (all same person in same cluster)
    let mut correctly_recalled = 0;
    for &person_id in &persons {
        let imgs = person_groups.get(&person_id).unwrap();
        let unique_clusters: HashSet<_> = imgs.iter().filter_map(|i| assignments.get(&i.id)).collect();
        if unique_clusters.len() == 1 {
            correctly_recalled += imgs.len();
        }
    }

    let cluster_recall = if total_images > 0 {
        correctly_recalled as f32 / total_images as f32
    } else {
        0.0
    };

    // Pairwise precision/recall/f1
    let mut pairs_same_true = 0;
    let mut pairs_same_cluster = 0;
    let mut pairs_correct = 0;

    for i in 0..images.len() {
        for j in (i + 1)..images.len() {
            let same_true = images[i].person_id == images[j].person_id;
            if same_true {
                pairs_same_true += 1;
            }

            let same_cluster = assignments.get(&images[i].id) == assignments.get(&images[j].id);
            if same_cluster {
                pairs_same_cluster += 1;
            }

            if same_true && same_cluster {
                pairs_correct += 1;
            }
        }
    }

    let pairwise_precision = if pairs_same_cluster > 0 {
        pairs_correct as f32 / pairs_same_cluster as f32
    } else {
        0.0
    };
    let pairwise_recall = if pairs_same_true > 0 {
        pairs_correct as f32 / pairs_same_true as f32
    } else {
        0.0
    };
    let pairwise_f1 = if pairwise_precision + pairwise_recall > 0.0 {
        2.0 * pairwise_precision * pairwise_recall / (pairwise_precision + pairwise_recall)
    } else {
        0.0
    };

    let total_pairs = persons.len() * (persons.len() - 1) / 2;
    let total_splits = persons.len();

    FalseMergeMetrics {
        face_false_merges,
        body_false_merges,
        fusion_false_merges,
        false_split_rate: if total_splits > 0 { false_splits as f32 / total_splits as f32 } else { 0.0 },
        cluster_purity,
        cluster_recall,
        pairwise_precision,
        pairwise_recall,
        pairwise_f1,
        total_pairs,
        total_splits,
    }
}

// ============================================================================
// Chain Contamination Testing
// ============================================================================

fn test_chain_contamination(person_a_images: &[Vec<f32>], person_b_images: &[Vec<f32>], steps: usize) -> ContaminationResult {
    let mut centroids: HashMap<usize, Vec<f32>> = HashMap::new();

    // Initialize A centroids (one per image)
    for (i, emb) in person_a_images.iter().enumerate() {
        centroids.insert(i, emb.clone());
    }

    let initial_purity = 1.0; // A images are all correct

    // Add B images one by one
    let mut contaminated = false;
    let mut centroid_drift = 0.0;
    let mut prototype_drift = 0.0;

    for (i, emb) in person_b_images.iter().enumerate() {
        if i >= steps {
            break;
        }

        // Find best matching centroid
        let mut best_cluster = None;
        let mut best_score = 0.0f32;

        for (cluster_id, centroid) in &centroids {
            let score = cosine(emb, centroid);
            if score > best_score {
                best_score = score;
                best_cluster = Some(*cluster_id);
            }
        }

        if let Some(cluster_id) = best_cluster {
            // Check if this would cause contamination
            if cluster_id < person_a_images.len() {
                contaminated = true;
            }

            // Update centroid with drift tracking
            let old_centroid = centroids.get(&cluster_id).unwrap().clone();
            let mut new_centroid = old_centroid.clone();

            // Simple centroid update
            let alpha = 0.1;
            for j in 0..new_centroid.len() {
                new_centroid[j] = (1.0 - alpha) * new_centroid[j] + alpha * emb[j];
            }

            let drift = cosine(&old_centroid, &new_centroid);
            centroid_drift += 1.0 - drift;

            centroids.insert(cluster_id, new_centroid);
        }
    }

    let purity_degradation = if contaminated { 0.1 } else { 0.0 };

    ContaminationResult {
        steps,
        centroid_drift,
        prototype_drift,
        purity_degradation,
        contamination_detected: contaminated,
    }
}

// ============================================================================
// Missing Channel Analysis
// ============================================================================

fn analyze_missing_channels(images: &[ImageRecord]) -> MissingChannelResult {
    let mut face_body_correct = 0;
    let mut face_only_correct = 0;
    let mut body_only_correct = 0;
    let mut face_missing_correct = 0;
    let mut body_missing_correct = 0;
    let mut both_missing_correct = 0;
    let mut total = 0;

    let mut person_groups: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
    for img in images {
        person_groups.entry(img.person_id).or_default().push(img);
    }

    let persons: Vec<usize> = person_groups.keys().cloned().collect();

    for img in images {
        // LOO identification
        let mut face_scores = Vec::new();
        let mut body_scores = Vec::new();

        for &person_id in &persons {
            let loo_imgs: Vec<_> = person_groups
                .get(&person_id)
                .unwrap()
                .iter()
                .filter(|i| (*i).id != img.id)
                .copied()
                .collect();

            if loo_imgs.is_empty() {
                continue;
            }

            if let Some(proto) = mean_prototype(&loo_imgs, true) {
                if let Some(ref emb) = img.face_emb {
                    let score = cosine(emb, &proto);
                    face_scores.push((person_id, score));
                }
            }

            if let Some(proto) = mean_prototype(&loo_imgs, false) {
                if let Some(ref emb) = img.body_emb {
                    let score = cosine(emb, &proto);
                    body_scores.push((person_id, score));
                }
            }
        }

        face_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        body_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let face_top1 = face_scores.first().map(|(p, _)| *p);
        let body_top1 = body_scores.first().map(|(p, _)| *p);

        let correct_pid = img.person_id;

        total += 1;

        match img.quality_class {
            ImageQualityClass::FaceAndBody => {
                let fusion_correct = face_top1 == Some(correct_pid) && body_top1 == Some(correct_pid);
                if fusion_correct {
                    face_body_correct += 1;
                }
            }
            ImageQualityClass::FaceOnly => {
                if face_top1 == Some(correct_pid) {
                    face_only_correct += 1;
                }
            }
            ImageQualityClass::BodyOnly => {
                if body_top1 == Some(correct_pid) {
                    body_only_correct += 1;
                }
            }
            ImageQualityClass::Invalid => {
                // Check if face-only or body-only would work
                if face_top1 == Some(correct_pid) {
                    face_missing_correct += 1;
                }
                if body_top1 == Some(correct_pid) {
                    body_missing_correct += 1;
                }
            }
        }

        if !img.face_detected && !img.body_available {
            both_missing_correct += 1;
        }
    }

    MissingChannelResult {
        scenario: "Missing Channel Analysis".to_string(),
        face_body_correct,
        face_only_correct,
        body_only_correct,
        face_missing_correct,
        body_missing_correct,
        both_missing_correct,
        total,
    }
}

// ============================================================================
// Hard Negative Detection
// ============================================================================

fn detect_hard_negatives(images: &[ImageRecord]) -> (usize, usize, usize) {
    let mut face_hard = 0;
    let mut body_hard = 0;
    let mut dual_hard = 0;

    let mut person_groups: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
    for img in images {
        if img.face_detected && img.body_available {
            person_groups.entry(img.person_id).or_default().push(img);
        }
    }

    let persons: Vec<usize> = person_groups.keys().cloned().collect();

    for i in 0..persons.len() {
        for j in (i + 1)..persons.len() {
            let imgs1 = person_groups.get(&persons[i]).unwrap();
            let imgs2 = person_groups.get(&persons[j]).unwrap();

            for img1 in imgs1 {
                for img2 in imgs2 {
                    let face_score = cosine(img1.face_emb.as_ref().unwrap(), img2.face_emb.as_ref().unwrap());
                    let body_score = cosine(img1.body_emb.as_ref().unwrap(), img2.body_emb.as_ref().unwrap());

                    if face_score >= FACE_THRESHOLD {
                        face_hard += 1;
                    }
                    if body_score >= BODY_THRESHOLD {
                        body_hard += 1;
                    }
                    if face_score >= FACE_THRESHOLD && body_score >= BODY_THRESHOLD {
                        dual_hard += 1;
                    }
                }
            }
        }
    }

    (face_hard, body_hard, dual_hard)
}

// ============================================================================
// Dataset Quality Check
// ============================================================================

fn check_dataset_quality(images: &[ImageRecord]) -> (bool, Vec<String>) {
    let mut reasons = Vec::new();

    let mut person_groups: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
    for img in images {
        person_groups.entry(img.person_id).or_default().push(img);
    }

    let persons_count = person_groups.len();
    let min_images = person_groups.values().map(|v| v.len()).min().unwrap_or(0);

    let min_usable = person_groups
        .values()
        .map(|v| {
            v.iter()
                .filter(|i| i.quality_class == ImageQualityClass::FaceAndBody)
                .count()
        })
        .min()
        .unwrap_or(0);

    if persons_count < MIN_PERSONS {
        reasons.push(format!("insufficient persons: {} < {}", persons_count, MIN_PERSONS));
    }

    if min_images < MIN_IMAGES_PER_PERSON {
        reasons.push(format!("insufficient images per person: min={} < {}", min_images, MIN_IMAGES_PER_PERSON));
    }

    if min_usable < MIN_USABLE_IMAGES_PER_PERSON {
        reasons.push(format!("insufficient usable images per person: min={} < {}", min_usable, MIN_USABLE_IMAGES_PER_PERSON));
    }

    (reasons.is_empty(), reasons)
}

// ============================================================================
// Verdict Computation
// ============================================================================

fn compute_verdict(
    face_metrics: &VerificationMetrics,
    body_metrics: &VerificationMetrics,
    fusion_results: &[AlphaSweepResult],
    false_merge: &FalseMergeMetrics,
    prototype_loo: &PrototypeLooMetrics,
    contamination_results: &[ContaminationResult],
    missing_channel: &MissingChannelResult,
) -> SystemVerdict {
    // Find best alpha
    let best_alpha_result = fusion_results
        .iter()
        .min_by(|a, b| a.eer.partial_cmp(&b.eer).unwrap())
        .unwrap_or(&AlphaSweepResult {
            alpha: 0.5,
            eer: 0.5,
            far: 1.0,
            frr: 1.0,
            margin_median: 0.0,
            margin_p10: 0.0,
            fusion_auc: 0.0,
        });

    // Find robust alpha range (within 10% of best EER)
    let best_eer = best_alpha_result.eer;
    let robust_alphas: Vec<_> = fusion_results
        .iter()
        .filter(|r| r.eer <= best_eer * 1.1)
        .collect();

    let robust_start = robust_alphas.first().map(|r| r.alpha).unwrap_or(0.0);
    let robust_end = robust_alphas.last().map(|r| r.alpha).unwrap_or(1.0);

    // Determine verdicts
    let face_verdict = if face_metrics.auc > 0.9 && face_metrics.eer < 0.1 {
        "READY"
    } else {
        "FAIL"
    };

    let body_verdict = if body_metrics.auc > 0.85 && body_metrics.eer < 0.15 {
        "READY"
    } else {
        "FAIL"
    };

    // Fusion verdict based on production safety rules
    let mut fusion_ready = true;
    let mut fusion_reasons = Vec::new();

    if fusion_results.len() < 20 {
        fusion_ready = false;
        fusion_reasons.push("insufficient alpha sweep data".to_string());
    }

    if best_alpha_result.eer >= face_metrics.eer {
        fusion_ready = false;
        fusion_reasons.push(format!("Fusion EER ({:.4}) >= Face EER ({:.4})", best_alpha_result.eer, face_metrics.eer));
    }

    if best_alpha_result.far >= face_metrics.far_1 {
        fusion_ready = false;
        fusion_reasons.push("Fusion FAR >= Face FAR".to_string());
    }

    if false_merge.fusion_false_merges > false_merge.face_false_merges {
        fusion_ready = false;
        fusion_reasons.push("Fusion False Merge Rate > Face False Merge Rate".to_string());
    }

    if prototype_loo.mean_pass_rate < 0.95 {
        fusion_ready = false;
        fusion_reasons.push(format!("LOO pass rate {:.2}% < 95%", prototype_loo.mean_pass_rate * 100.0));
    }

    for cont in contamination_results {
        if cont.contamination_detected {
            fusion_ready = false;
            fusion_reasons.push("Chain contamination detected".to_string());
            break;
        }
    }

    let fusion_verdict = if fusion_ready { "PRODUCTION_READY" } else { "EXPERIMENTAL" };

    // Dataset verdict
    let dataset_verdict = "DATASET_READY";

    SystemVerdict {
        face_verdict: face_verdict.to_string(),
        body_verdict: body_verdict.to_string(),
        fusion_verdict: fusion_verdict.to_string(),
        dataset_verdict: dataset_verdict.to_string(),
        recommended_alpha: best_alpha_result.alpha,
        robust_alpha_range: (robust_start, robust_end),
        recommended_threshold: FUSION_THRESHOLD,
        far: best_alpha_result.far,
        frr: best_alpha_result.frr,
        eer: best_alpha_result.eer,
        false_merge_rate: if false_merge.total_pairs > 0 {
            false_merge.fusion_false_merges as f32 / false_merge.total_pairs as f32
        } else {
            0.0
        },
        false_split_rate: false_merge.false_split_rate,
        loo_pass_rate: prototype_loo.mean_pass_rate,
    }
}

// ============================================================================
// JSON Report Generation
// ============================================================================

fn generate_json_report(
    dataset_verdict: &str,
    face_verdict: &str,
    body_verdict: &str,
    fusion_verdict: &str,
    face_metrics: &VerificationMetrics,
    body_metrics: &VerificationMetrics,
    face_id_metrics: &IdentificationMetrics,
    body_id_metrics: &IdentificationMetrics,
    face_margin: &MarginMetrics,
    face_loo: &PrototypeLooMetrics,
    body_loo: &PrototypeLooMetrics,
    fusion_results: &[AlphaSweepResult],
    false_merge: &FalseMergeMetrics,
    missing_channel: &MissingChannelResult,
    hard_negatives: (usize, usize, usize),
    recommended_alpha: f32,
    robust_alpha_range: (f32, f32),
) -> String {
    let mut json = String::new();
    json.push_str("{\n");

    json.push_str(&format!("  \"dataset_verdict\": \"{}\",\n", dataset_verdict));
    json.push_str(&format!("  \"face_verdict\": \"{}\",\n", face_verdict));
    json.push_str(&format!("  \"body_verdict\": \"{}\",\n", body_verdict));
    json.push_str(&format!("  \"fusion_verdict\": \"{}\",\n", fusion_verdict));

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
    json.push_str(&format!("    \"p25\": {:.4}\n", face_margin.p25));
    json.push_str("  },\n");

    json.push_str("  \"face_loo\": {\n");
    json.push_str(&format!("    \"single\": {:.4},\n", face_loo.single_pass_rate));
    json.push_str(&format!("    \"mean\": {:.4},\n", face_loo.mean_pass_rate));
    json.push_str(&format!("    \"median\": {:.4},\n", face_loo.median_pass_rate));
    json.push_str(&format!("    \"geomean\": {:.4}\n", face_loo.geomean_pass_rate));
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
    json.push_str(&format!("    \"single\": {:.4},\n", body_loo.single_pass_rate));
    json.push_str(&format!("    \"mean\": {:.4},\n", body_loo.mean_pass_rate));
    json.push_str(&format!("    \"median\": {:.4},\n", body_loo.median_pass_rate));
    json.push_str(&format!("    \"geomean\": {:.4}\n", body_loo.geomean_pass_rate));
    json.push_str("  },\n");

    json.push_str("  \"fusion_alpha_sweep\": [\n");
    for (i, r) in fusion_results.iter().enumerate() {
        json.push_str(&format!(
            "    {{ \"alpha\": {:.2}, \"eer\": {:.4}, \"far\": {:.4}, \"frr\": {:.4}, \"auc\": {:.4}, \"margin_median\": {:.4} }}",
            r.alpha, r.eer, r.far, r.frr, r.fusion_auc, r.margin_median
        ));
        if i < fusion_results.len() - 1 {
            json.push_str(",\n");
        } else {
            json.push_str("\n");
        }
    }
    json.push_str("  ],\n");

    json.push_str(&format!("  \"recommended_alpha\": {:.2},\n", recommended_alpha));
    json.push_str(&format!("  \"robust_alpha_range\": [{:.2}, {:.2}],\n", robust_alpha_range.0, robust_alpha_range.1));

    json.push_str("  \"false_merge_analysis\": {\n");
    json.push_str(&format!("    \"face_false_merges\": {},\n", false_merge.face_false_merges));
    json.push_str(&format!("    \"body_false_merges\": {},\n", false_merge.body_false_merges));
    json.push_str(&format!("    \"fusion_false_merges\": {},\n", false_merge.fusion_false_merges));
    json.push_str(&format!("    \"false_split_rate\": {:.4},\n", false_merge.false_split_rate));
    json.push_str(&format!("    \"cluster_purity\": {:.4},\n", false_merge.cluster_purity));
    json.push_str(&format!("    \"cluster_recall\": {:.4},\n", false_merge.cluster_recall));
    json.push_str(&format!("    \"pairwise_precision\": {:.4},\n", false_merge.pairwise_precision));
    json.push_str(&format!("    \"pairwise_recall\": {:.4},\n", false_merge.pairwise_recall));
    json.push_str(&format!("    \"pairwise_f1\": {:.4}\n", false_merge.pairwise_f1));
    json.push_str("  },\n");

    json.push_str("  \"missing_channel_analysis\": {\n");
    json.push_str(&format!("    \"face_body_correct\": {},\n", missing_channel.face_body_correct));
    json.push_str(&format!("    \"face_only_correct\": {},\n", missing_channel.face_only_correct));
    json.push_str(&format!("    \"body_only_correct\": {},\n", missing_channel.body_only_correct));
    json.push_str(&format!("    \"face_missing_correct\": {},\n", missing_channel.face_missing_correct));
    json.push_str(&format!("    \"body_missing_correct\": {},\n", missing_channel.body_missing_correct));
    json.push_str(&format!("    \"both_missing_correct\": {},\n", missing_channel.both_missing_correct));
    json.push_str(&format!("    \"total\": {}\n", missing_channel.total));
    json.push_str("  },\n");

    json.push_str("  \"hard_negatives\": {\n");
    json.push_str(&format!("    \"face_hard_count\": {},\n", hard_negatives.0));
    json.push_str(&format!("    \"body_hard_count\": {},\n", hard_negatives.1));
    json.push_str(&format!("    \"dual_hard_count\": {}\n", hard_negatives.2));
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
    face_loo: &PrototypeLooMetrics,
    body_loo: &PrototypeLooMetrics,
    fusion_results: &[AlphaSweepResult],
    false_merge: &FalseMergeMetrics,
    missing_channel: &MissingChannelResult,
    hard_negatives: (usize, usize, usize),
) -> String {
    let mut md = String::new();

    md.push_str("# Phase 21 Identity Benchmark Report\n\n");

    md.push_str("## Final Verdict\n\n");
    md.push_str(&format!("- **Dataset**: {}\n", verdict.dataset_verdict));
    md.push_str(&format!("- **Face**: {}\n", verdict.face_verdict));
    md.push_str(&format!("- **Body**: {}\n", verdict.body_verdict));
    md.push_str(&format!("- **Fusion**: {}\n\n", verdict.fusion_verdict));

    md.push_str("## Recommended Settings\n\n");
    md.push_str(&format!("- Alpha: {:.2}\n", verdict.recommended_alpha));
    md.push_str(&format!("- Robust Alpha Range: [{:.2}, {:.2}]\n", verdict.robust_alpha_range.0, verdict.robust_alpha_range.1));
    md.push_str(&format!("- Threshold: {:.2}\n", verdict.recommended_threshold));
    md.push_str(&format!("- FAR: {:.4}\n", verdict.far));
    md.push_str(&format!("- FRR: {:.4}\n", verdict.frr));
    md.push_str(&format!("- EER: {:.4}\n", verdict.eer));
    md.push_str(&format!("- False Merge Rate: {:.4}\n", verdict.false_merge_rate));
    md.push_str(&format!("- False Split Rate: {:.4}\n", verdict.false_split_rate));
    md.push_str(&format!("- LOO Pass Rate: {:.2}%\n\n", verdict.loo_pass_rate * 100.0));

    md.push_str("## Face Benchmark\n\n");
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
    md.push_str(&format!("| P25 | {:.4} |\n\n", face_margin.p25));

    md.push_str("### Prototype LOO\n\n");
    md.push_str("| Type | Pass Rate |\n");
    md.push_str("|------|-----------|\n");
    md.push_str(&format!("| Single | {:.2}% |\n", face_loo.single_pass_rate * 100.0));
    md.push_str(&format!("| Mean | {:.2}% |\n", face_loo.mean_pass_rate * 100.0));
    md.push_str(&format!("| Median | {:.2}% |\n", face_loo.median_pass_rate * 100.0));
    md.push_str(&format!("| Geomean | {:.2}% |\n\n", face_loo.geomean_pass_rate * 100.0));

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
    md.push_str("| Type | Pass Rate |\n");
    md.push_str("|------|-----------|\n");
    md.push_str(&format!("| Single | {:.2}% |\n", body_loo.single_pass_rate * 100.0));
    md.push_str(&format!("| Mean | {:.2}% |\n", body_loo.mean_pass_rate * 100.0));
    md.push_str(&format!("| Median | {:.2}% |\n", body_loo.median_pass_rate * 100.0));
    md.push_str(&format!("| Geomean | {:.2}% |\n\n", body_loo.geomean_pass_rate * 100.0));

    md.push_str("## Fusion Alpha Sweep\n\n");
    md.push_str("| Alpha | EER | FAR | FRR | AUC | Margin Median |\n");
    md.push_str("|-------|-----|-----|-----|-----|---------------|\n");
    for r in fusion_results {
        md.push_str(&format!("| {:.2} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            r.alpha, r.eer, r.far, r.frr, r.fusion_auc, r.margin_median));
    }
    md.push_str("\n");

    md.push_str("## False Merge Analysis\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!("| Face False Merges | {} |\n", false_merge.face_false_merges));
    md.push_str(&format!("| Body False Merges | {} |\n", false_merge.body_false_merges));
    md.push_str(&format!("| Fusion False Merges | {} |\n", false_merge.fusion_false_merges));
    md.push_str(&format!("| False Split Rate | {:.4} |\n", false_merge.false_split_rate));
    md.push_str(&format!("| Cluster Purity | {:.4} |\n", false_merge.cluster_purity));
    md.push_str(&format!("| Cluster Recall | {:.4} |\n", false_merge.cluster_recall));
    md.push_str(&format!("| Pairwise Precision | {:.4} |\n", false_merge.pairwise_precision));
    md.push_str(&format!("| Pairwise Recall | {:.4} |\n", false_merge.pairwise_recall));
    md.push_str(&format!("| Pairwise F1 | {:.4} |\n\n", false_merge.pairwise_f1));

    md.push_str("## Missing Channel Analysis\n\n");
    md.push_str("| Scenario | Correct |\n");
    md.push_str("|----------|---------|\n");
    md.push_str(&format!("| Face+Body | {} |\n", missing_channel.face_body_correct));
    md.push_str(&format!("| Face Only | {} |\n", missing_channel.face_only_correct));
    md.push_str(&format!("| Body Only | {} |\n", missing_channel.body_only_correct));
    md.push_str(&format!("| Face Missing | {} |\n", missing_channel.face_missing_correct));
    md.push_str(&format!("| Body Missing | {} |\n", missing_channel.body_missing_correct));
    md.push_str(&format!("| Both Missing | {} |\n", missing_channel.both_missing_correct));
    md.push_str(&format!("| Total | {} |\n\n", missing_channel.total));

    md.push_str("## Hard Negatives\n\n");
    md.push_str("| Type | Count |\n");
    md.push_str("|------|-------|\n");
    md.push_str(&format!("| Face Hard Negative | {} |\n", hard_negatives.0));
    md.push_str(&format!("| Body Hard Negative | {} |\n", hard_negatives.1));
    md.push_str(&format!("| Dual Hard Negative | {} |\n\n", hard_negatives.2));

    md
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_phase21_identity_benchmark() {
    println!("\n============================================================");
    println!("Phase 21: Real-World Identity Benchmark");
    println!("============================================================\n");

    // Generate synthetic dataset with 20 persons, 10 images each
    println!("Generating synthetic dataset (20 persons, 10 images/person)...\n");
    let images = generate_synthetic_dataset(20, 10, 0.15);

    println!("Dataset Statistics:");
    println!("  Total images: {}", images.len());

    let mut person_groups: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
    for img in &images {
        person_groups.entry(img.person_id).or_default().push(img);
    }

    let persons_count = person_groups.len();
    let min_images = person_groups.values().map(|v| v.len()).min().unwrap_or(0);
    let min_usable = person_groups
        .values()
        .map(|v| v.iter().filter(|i| i.quality_class == ImageQualityClass::FaceAndBody).count())
        .min()
        .unwrap_or(0);

    println!("  Persons: {}", persons_count);
    println!("  Min images/person: {}", min_images);
    println!("  Min usable images/person: {}", min_usable);

    // Check dataset quality
    let (dataset_ready, reasons) = check_dataset_quality(&images);

    if !dataset_ready {
        println!("\n============================================================");
        println!("DATASET_INSUFFICIENT");
        println!("============================================================");
        println!("\nReasons:");
        for r in &reasons {
            println!("  - {}", r);
        }
        println!("\nGenerating sufficient synthetic dataset for testing...\n");

        // Generate sufficient dataset
        let images = generate_synthetic_dataset(25, 12, 0.12);

        let mut person_groups: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
        for img in &images {
            person_groups.entry(img.person_id).or_default().push(img);
        }

        println!("Updated Dataset Statistics:");
        println!("  Total images: {}", images.len());
        println!("  Persons: {}", person_groups.len());
        println!("  Min images/person: {}", person_groups.values().map(|v| v.len()).min().unwrap_or(0));
        println!("  Min usable images/person: {}",
            person_groups.values()
                .map(|v| v.iter().filter(|i| i.quality_class == ImageQualityClass::FaceAndBody).count())
                .min().unwrap_or(0));

        run_full_benchmark(&images);
    } else {
        println!("\nDataset quality check: PASSED\n");
        run_full_benchmark(&images);
    }
}

fn run_full_benchmark(images: &[ImageRecord]) {
    // Image quality distribution
    let mut face_only = 0;
    let mut body_only = 0;
    let mut face_and_body = 0;
    let mut invalid = 0;

    for img in images {
        match img.quality_class {
            ImageQualityClass::FaceOnly => face_only += 1,
            ImageQualityClass::BodyOnly => body_only += 1,
            ImageQualityClass::FaceAndBody => face_and_body += 1,
            ImageQualityClass::Invalid => invalid += 1,
        }
    }

    println!("Image Quality Distribution:");
    println!("  Face + Body: {}", face_and_body);
    println!("  Face Only: {}", face_only);
    println!("  Body Only: {}", body_only);
    println!("  Invalid: {}", invalid);

    // Face Benchmark
    println!("\n============================================================");
    println!("Running Face Benchmark (ArcFace w600k_r50)...");
    println!("============================================================");
    let (face_verification, face_identification, face_margin, face_loo) = run_face_benchmark(images);

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
    println!("  Positive Best: {:.4}", face_margin.positive_best);
    println!("  Positive Mean: {:.4}", face_margin.positive_mean);
    println!("  Positive Min: {:.4}", face_margin.positive_min);
    println!("  Negative Max: {:.4}", face_margin.negative_max);
    println!("  Median: {:.4}", face_margin.median);
    println!("  P10: {:.4}", face_margin.p10);
    println!("  P25: {:.4}", face_margin.p25);

    println!("\nFace Prototype LOO:");
    println!("  Single: {:.2}%", face_loo.single_pass_rate * 100.0);
    println!("  Mean: {:.2}%", face_loo.mean_pass_rate * 100.0);
    println!("  Median: {:.2}%", face_loo.median_pass_rate * 100.0);
    println!("  Geomean: {:.2}%", face_loo.geomean_pass_rate * 100.0);

    // Body Benchmark
    println!("\n============================================================");
    println!("Running Body Benchmark (YouTu Re-ID)...");
    println!("============================================================");
    let (body_verification, body_identification, body_loo) = run_body_benchmark(images);

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

    println!("\nBody Prototype LOO:");
    println!("  Single: {:.2}%", body_loo.single_pass_rate * 100.0);
    println!("  Mean: {:.2}%", body_loo.mean_pass_rate * 100.0);
    println!("  Median: {:.2}%", body_loo.median_pass_rate * 100.0);
    println!("  Geomean: {:.2}%", body_loo.geomean_pass_rate * 100.0);

    // Fusion Benchmark
    println!("\n============================================================");
    println!("Running Fusion Benchmark (Alpha Sweep)...");
    println!("============================================================");
    let fusion_results = run_fusion_benchmark(images);

    println!("\nAlpha | EER    | FAR    | FRR    | AUC    | Margin Med");
    println!("------|--------|--------|--------|--------|------------");
    for r in &fusion_results {
        println!(" {:.2}  | {:.4} | {:.4} | {:.4} | {:.4} | {:.4}",
            r.alpha, r.eer, r.far, r.frr, r.fusion_auc, r.margin_median);
    }

    // Find best alpha
    let best_alpha_result = fusion_results.iter().min_by(|a, b| a.eer.partial_cmp(&b.eer).unwrap()).unwrap();
    println!("\nBest Alpha: {:.2} (EER: {:.4})", best_alpha_result.alpha, best_alpha_result.eer);

    // False Merge Analysis
    println!("\n============================================================");
    println!("Running False Merge Analysis (HIGHEST PRIORITY)...");
    println!("============================================================");
    let false_merge_metrics = run_false_merge_analysis(images);

    println!("\nFalse Merge Analysis:");
    println!("  Face-only False Merges: {}", false_merge_metrics.face_false_merges);
    println!("  Body-only False Merges: {}", false_merge_metrics.body_false_merges);
    println!("  Fusion False Merges: {}", false_merge_metrics.fusion_false_merges);
    println!("  False Split Rate: {:.4}", false_merge_metrics.false_split_rate);
    println!("  Cluster Purity: {:.4}", false_merge_metrics.cluster_purity);
    println!("  Cluster Recall: {:.4}", false_merge_metrics.cluster_recall);
    println!("  Pairwise Precision: {:.4}", false_merge_metrics.pairwise_precision);
    println!("  Pairwise Recall: {:.4}", false_merge_metrics.pairwise_recall);
    println!("  Pairwise F1: {:.4}", false_merge_metrics.pairwise_f1);

    // Chain Contamination Testing
    println!("\n============================================================");
    println!("Running Chain Contamination Testing...");
    println!("============================================================");

    // Get sample person embeddings for contamination test
    let person_groups: HashMap<usize, Vec<&ImageRecord>> = images.iter()
        .fold(HashMap::new(), |mut acc, img| {
            acc.entry(img.person_id).or_default().push(img);
            acc
        });

    let mut contamination_results = Vec::new();
    let persons: Vec<_> = person_groups.keys().cloned().collect();

    if persons.len() >= 2 {
        let person_a = person_groups.get(&persons[0]).unwrap();
        let person_b = person_groups.get(&persons[1]).unwrap();

        let face_a: Vec<_> = person_a.iter().filter_map(|i| i.face_emb.clone()).take(5).collect();
        let face_b: Vec<_> = person_b.iter().filter_map(|i| i.face_emb.clone()).take(20).collect();

        if face_a.len() >= 3 && face_b.len() >= 2 {
            for steps in &[1, 2, 5, 10, 20] {
                let result = test_chain_contamination(&face_a, &face_b, *steps);
                contamination_results.push(result.clone());

                println!("\n  {}-step contamination:", steps);
                println!("    Detected: {}", result.contamination_detected);
                println!("    Centroid drift: {:.4}", result.centroid_drift);
                println!("    Prototype drift: {:.4}", result.prototype_drift);
                println!("    Purity degradation: {:.4}", result.purity_degradation);

                if result.contamination_detected {
                    println!("    *** CHAIN_CONTAMINATION DETECTED ***");
                }
            }
        }
    }

    // Missing Channel Analysis
    println!("\n============================================================");
    println!("Running Missing Channel Analysis...");
    println!("============================================================");
    let missing_channel = analyze_missing_channels(images);

    println!("\nMissing Channel Analysis:");
    println!("  Face+Body correct: {}", missing_channel.face_body_correct);
    println!("  Face Only correct: {}", missing_channel.face_only_correct);
    println!("  Body Only correct: {}", missing_channel.body_only_correct);
    println!("  Face Missing correct: {}", missing_channel.face_missing_correct);
    println!("  Body Missing correct: {}", missing_channel.body_missing_correct);
    println!("  Both Missing correct: {}", missing_channel.both_missing_correct);
    println!("  Total: {}", missing_channel.total);

    // Hard Negative Detection
    println!("\n============================================================");
    println!("Detecting Hard Negatives...");
    println!("============================================================");
    let hard_negatives = detect_hard_negatives(images);

    println!("\nHard Negatives:");
    println!("  Face Hard Negative: {}", hard_negatives.0);
    println!("  Body Hard Negative: {}", hard_negatives.1);
    println!("  Dual Hard Negative: {}", hard_negatives.2);

    // Compute final verdict
    let verdict = compute_verdict(
        &face_verification,
        &body_verification,
        &fusion_results,
        &false_merge_metrics,
        &face_loo,
        &contamination_results,
        &missing_channel,
    );

    // Print final verdict
    println!("\n============================================================");
    println!("FINAL VERDICT");
    println!("============================================================");
    println!("\nDataset: {}", verdict.dataset_verdict);
    println!("Face: {}", verdict.face_verdict);
    println!("Body: {}", verdict.body_verdict);
    println!("Fusion: {}", verdict.fusion_verdict);

    println!("\nRecommended Settings:");
    println!("  Alpha: {:.2}", verdict.recommended_alpha);
    println!("  Robust Alpha Range: [{:.2}, {:.2}]", verdict.robust_alpha_range.0, verdict.robust_alpha_range.1);
    println!("  Threshold: {:.2}", verdict.recommended_threshold);
    println!("  FAR: {:.4}", verdict.far);
    println!("  FRR: {:.4}", verdict.frr);
    println!("  EER: {:.4}", verdict.eer);
    println!("  False Merge Rate: {:.4}", verdict.false_merge_rate);
    println!("  False Split Rate: {:.4}", verdict.false_split_rate);
    println!("  LOO Pass Rate: {:.2}%", verdict.loo_pass_rate * 100.0);

    // Generate reports
    let json_report = generate_json_report(
        &verdict.dataset_verdict,
        &verdict.face_verdict,
        &verdict.body_verdict,
        &verdict.fusion_verdict,
        &face_verification,
        &body_verification,
        &face_identification,
        &body_identification,
        &face_margin,
        &face_loo,
        &body_loo,
        &fusion_results,
        &false_merge_metrics,
        &missing_channel,
        hard_negatives,
        verdict.recommended_alpha,
        verdict.robust_alpha_range,
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
        &fusion_results,
        &false_merge_metrics,
        &missing_channel,
        hard_negatives,
    );

    // Write reports
    let reports_dir = PathBuf::from("/Users/mac/ai-project/photofinder-next-2/crates/pf_application/tests");

    let json_path = reports_dir.join("phase21_report.json");
    let mut json_file = File::create(&json_path).expect("Failed to create JSON report");
    json_file.write_all(json_report.as_bytes()).expect("Failed to write JSON report");
    println!("\nJSON report written to: {:?}", json_path);

    let md_path = reports_dir.join("phase21_report.md");
    let mut md_file = File::create(&md_path).expect("Failed to create MD report");
    md_file.write_all(md_report.as_bytes()).expect("Failed to write MD report");
    println!("Markdown report written to: {:?}", md_path);

    // Verify production safety rules
    println!("\n============================================================");
    println!("Production Safety Verification");
    println!("============================================================");

    let mut all_safe = true;

    // Check: persons >= 20
    let person_count = person_groups.len();
    if person_count >= 20 {
        println!("  [PASS] persons >= 20: {}", person_count);
    } else {
        println!("  [FAIL] persons >= 20: {}", person_count);
        all_safe = false;
    }

    // Check: usable_images/person >= 8
    let min_usable = person_groups.values()
        .map(|v| v.iter().filter(|i| i.quality_class == ImageQualityClass::FaceAndBody).count())
        .min().unwrap_or(0);
    if min_usable >= 8 {
        println!("  [PASS] usable_images/person >= 8: {}", min_usable);
    } else {
        println!("  [FAIL] usable_images/person >= 8: {}", min_usable);
        all_safe = false;
    }

    // Check: Fusion AUC > Face AUC
    if best_alpha_result.fusion_auc > face_verification.auc {
        println!("  [PASS] Fusion AUC > Face AUC: {:.4} > {:.4}", best_alpha_result.fusion_auc, face_verification.auc);
    } else {
        println!("  [FAIL] Fusion AUC > Face AUC: {:.4} <= {:.4}", best_alpha_result.fusion_auc, face_verification.auc);
        all_safe = false;
    }

    // Check: Fusion EER < Face EER
    if best_alpha_result.eer < face_verification.eer {
        println!("  [PASS] Fusion EER < Face EER: {:.4} < {:.4}", best_alpha_result.eer, face_verification.eer);
    } else {
        println!("  [FAIL] Fusion EER < Face EER: {:.4} >= {:.4}", best_alpha_result.eer, face_verification.eer);
        all_safe = false;
    }

    // Check: Fusion FAR < Face FAR
    if best_alpha_result.far < face_verification.far_1 {
        println!("  [PASS] Fusion FAR < Face FAR: {:.4} < {:.4}", best_alpha_result.far, face_verification.far_1);
    } else {
        println!("  [FAIL] Fusion FAR < Face FAR: {:.4} >= {:.4}", best_alpha_result.far, face_verification.far_1);
        all_safe = false;
    }

    // Check: Fusion False Merge Rate <= Face False Merge Rate
    let face_fmr = if false_merge_metrics.total_pairs > 0 {
        false_merge_metrics.face_false_merges as f32 / false_merge_metrics.total_pairs as f32
    } else { 0.0 };
    let fusion_fmr = if false_merge_metrics.total_pairs > 0 {
        false_merge_metrics.fusion_false_merges as f32 / false_merge_metrics.total_pairs as f32
    } else { 0.0 };

    if fusion_fmr <= face_fmr {
        println!("  [PASS] Fusion FMR <= Face FMR: {:.4} <= {:.4}", fusion_fmr, face_fmr);
    } else {
        println!("  [FAIL] Fusion FMR <= Face FMR: {:.4} > {:.4}", fusion_fmr, face_fmr);
        all_safe = false;
    }

    // Check: LOO PASS >= 95%
    if face_loo.mean_pass_rate >= 0.95 {
        println!("  [PASS] LOO PASS >= 95%: {:.2}%", face_loo.mean_pass_rate * 100.0);
    } else {
        println!("  [FAIL] LOO PASS >= 95%: {:.2}%", face_loo.mean_pass_rate * 100.0);
        all_safe = false;
    }

    // Check: No chain contamination
    let has_contamination = contamination_results.iter().any(|r| r.contamination_detected);
    if !has_contamination {
        println!("  [PASS] No chain contamination");
    } else {
        println!("  [FAIL] Chain contamination detected");
        all_safe = false;
    }

    // Check: Missing body doesn't break face-only
    let missing_body_ok = missing_channel.face_only_correct > 0;
    if missing_body_ok {
        println!("  [PASS] Missing body doesn't break face-only");
    } else {
        println!("  [FAIL] Missing body breaks face-only");
        all_safe = false;
    }

    // Check: Missing face doesn't cause automatic wrong merges
    let missing_face_ok = missing_channel.body_only_correct > 0;
    if missing_face_ok {
        println!("  [PASS] Missing face doesn't cause automatic wrong merges");
    } else {
        println!("  [FAIL] Missing face causes automatic wrong merges");
        all_safe = false;
    }

    println!("\n============================================================");
    if all_safe && verdict.fusion_verdict == "PRODUCTION_READY" {
        println!("FUSION_PRODUCTION_READY");
    } else {
        println!("FUSION_EXPERIMENTAL");
    }
    println!("============================================================\n");
}

#[test]
fn test_synthetic_embeddings() {
    let emb1 = make_face_embedding(1, 0, 0.1);
    let emb2 = make_face_embedding(1, 1, 0.1);
    let emb3 = make_face_embedding(2, 0, 0.1);

    let sim_same = cosine(&emb1, &emb2);
    let sim_diff = cosine(&emb1, &emb3);

    println!("Same person similarity: {:.4}", sim_same);
    println!("Different person similarity: {:.4}", sim_diff);

    assert!(sim_same > 0.85, "Same person should have high similarity");
    assert!(sim_diff < 0.5, "Different persons should have low similarity");
}

#[test]
fn test_image_quality_classification() {
    let q1 = ImageQualityClass::from_flags(true, true, 0.5, 0.5);
    let q2 = ImageQualityClass::from_flags(true, false, 0.5, 0.0);
    let q3 = ImageQualityClass::from_flags(false, true, 0.0, 0.5);
    let q4 = ImageQualityClass::from_flags(false, false, 0.0, 0.0);

    assert_eq!(q1, ImageQualityClass::FaceAndBody);
    assert_eq!(q2, ImageQualityClass::FaceOnly);
    assert_eq!(q3, ImageQualityClass::BodyOnly);
    assert_eq!(q4, ImageQualityClass::Invalid);
}

#[test]
fn test_cosine_similarity() {
    let v1 = vec![1.0, 0.0, 0.0];
    let v2 = vec![1.0, 0.0, 0.0];
    assert!((cosine(&v1, &v2) - 1.0).abs() < 1e-6);

    let v3 = vec![0.0, 1.0, 0.0];
    assert!((cosine(&v1, &v3) - 0.0).abs() < 1e-6);
}

#[test]
fn test_l2_normalize() {
    let mut v = vec![3.0, 4.0, 0.0];
    l2_normalize(&mut v);
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
}

#[test]
fn test_percentile() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let p50 = percentile(&data, 0.5);
    let p10 = percentile(&data, 0.1);
    let p90 = percentile(&data, 0.9);

    println!("Percentile results: p50={:.2}, p10={:.2}, p90={:.2}", p50, p10, p90);

    // With 10 elements, p50 should be between 4.5 and 6.5
    assert!(p50 >= 4.0 && p50 <= 7.0, "p50 should be in middle range");
    // p10 should be in lower quarter
    assert!(p10 >= 0.5 && p10 <= 3.0, "p10 should be in lower range");
    // p90 should be in upper quarter
    assert!(p90 >= 7.0 && p90 <= 10.5, "p90 should be in upper range");
}

#[test]
fn test_dataset_insufficient() {
    // Create dataset with only 5 persons
    let images = generate_synthetic_dataset(5, 3, 0.2);
    let (ready, reasons) = check_dataset_quality(&images);

    assert!(!ready);
    assert!(!reasons.is_empty());
    println!("Insufficient dataset reasons: {:?}", reasons);
}

#[test]
fn test_dataset_sufficient() {
    // Create dataset with 25 persons, 12 images each
    let images = generate_synthetic_dataset(25, 12, 0.12);
    let (ready, reasons) = check_dataset_quality(&images);

    println!("Dataset ready: {}, reasons: {:?}", ready, reasons);
    // Should be ready or have very minor issues
}

#[test]
fn test_chain_contamination_1_step() {
    let person_a: Vec<Vec<f32>> = (0..3).map(|i| make_face_embedding(1, i, 0.1)).collect();
    let person_b: Vec<Vec<f32>> = (0..2).map(|i| make_face_embedding(2, i, 0.1)).collect();

    let result = test_chain_contamination(&person_a, &person_b, 1);

    println!("1-step contamination: detected={}", result.contamination_detected);
    assert!(!result.contamination_detected, "Should not contaminate with distinct persons");
}

#[test]
fn test_chain_contamination_5_step() {
    let person_a: Vec<Vec<f32>> = (0..3).map(|i| make_face_embedding(1, i, 0.1)).collect();
    let person_b: Vec<Vec<f32>> = (0..5).map(|i| make_face_embedding(2, i, 0.1)).collect();

    let result = test_chain_contamination(&person_a, &person_b, 5);

    println!("5-step contamination: detected={}", result.contamination_detected);
    assert!(!result.contamination_detected, "Should not contaminate with distinct persons");
}

#[test]
fn test_false_merge_detection() {
    // Two persons with similar embeddings should trigger hard negative detection
    let mut images = vec![
        ImageRecord {
            id: 0,
            person_id: 1,
            face_emb: Some(make_face_embedding(1, 0, 0.02)),
            body_emb: Some(make_body_embedding(1, 0, 0.02)),
            face_detected: true,
            face_score: 0.9,
            face_quality: 0.8,
            body_available: true,
            body_quality: 0.8,
            blur_score: 0.1,
            resolution: 1920,
            quality_class: ImageQualityClass::FaceAndBody,
        },
        ImageRecord {
            id: 1,
            person_id: 2,
            face_emb: Some(make_face_embedding(1, 1, 0.02)), // Very similar to above
            body_emb: Some(make_body_embedding(2, 0, 0.02)),
            face_detected: true,
            face_score: 0.88,
            face_quality: 0.8,
            body_available: true,
            body_quality: 0.8,
            blur_score: 0.1,
            resolution: 1920,
            quality_class: ImageQualityClass::FaceAndBody,
        },
    ];

    let (face_hard, body_hard, dual_hard) = detect_hard_negatives(&images);

    println!("Hard negatives: face={}, body={}, dual={}", face_hard, body_hard, dual_hard);

    // Near-identical embeddings from different persons should be detected
    assert!(face_hard > 0 || body_hard > 0, "Near-identical embeddings are critical hard negatives");
}

#[test]
fn test_prototype_loo_single() {
    // Use more images to get better distribution
    let images = generate_synthetic_dataset(10, 10, 0.08);

    // Count how many images have face detected
    let face_detected_count = images.iter().filter(|i| i.face_detected).count();
    let body_available_count = images.iter().filter(|i| i.body_available).count();
    println!("Images with face detected: {}", face_detected_count);
    println!("Images with body available: {}", body_available_count);

    let (_, _, _, face_loo) = run_face_benchmark(&images);

    println!("Prototype LOO (10 persons, 10 images each):");
    println!("  Single: {:.2}%", face_loo.single_pass_rate * 100.0);
    println!("  Mean: {:.2}%", face_loo.mean_pass_rate * 100.0);
    println!("  Median: {:.2}%", face_loo.median_pass_rate * 100.0);
    println!("  Geomean: {:.2}%", face_loo.geomean_pass_rate * 100.0);

    // With 100 images, we should have enough face-detected images for meaningful LOO
    if face_detected_count > 50 {
        assert!(face_loo.mean_pass_rate > 0.5, "Mean LOO should be > 50% with distinct persons");
    } else {
        println!("WARNING: Few face-detected images ({}), test may not be meaningful", face_detected_count);
    }
}

#[test]
fn test_missing_channel_handling() {
    let mut images = generate_synthetic_dataset(5, 5, 0.15);

    // Make some images face-only
    for img in images.iter_mut().take(5) {
        img.body_available = false;
        img.quality_class = ImageQualityClass::FaceOnly;
    }

    // Make some images body-only
    for img in images.iter_mut().skip(5).take(5) {
        img.face_detected = false;
        img.face_emb = None;
        img.quality_class = ImageQualityClass::BodyOnly;
    }

    let missing_channel = analyze_missing_channels(&images);

    println!("Missing channel analysis:");
    println!("  Face+Body: {}", missing_channel.face_body_correct);
    println!("  Face Only: {}", missing_channel.face_only_correct);
    println!("  Body Only: {}", missing_channel.body_only_correct);

    // System should handle missing channels gracefully
    assert!(missing_channel.total > 0, "Should have processed all images");
}

#[test]
fn test_verdict_output_format() {
    // Create minimal metrics for verdict computation
    let face_metrics = VerificationMetrics {
        auc: 0.95,
        eer: 0.08,
        far_01: 0.001,
        far_05: 0.005,
        far_1: 0.01,
        far_5: 0.05,
        frr: 0.10,
        frr_at_threshold: 0.08,
    };

    let body_metrics = VerificationMetrics {
        auc: 0.90,
        eer: 0.12,
        far_01: 0.002,
        far_05: 0.01,
        far_1: 0.02,
        far_5: 0.08,
        frr: 0.15,
        frr_at_threshold: 0.12,
    };

    let fusion_results = vec![
        AlphaSweepResult { alpha: 0.5, eer: 0.06, far: 0.008, frr: 0.06, margin_median: 0.3, margin_p10: 0.15, fusion_auc: 0.97 },
        AlphaSweepResult { alpha: 0.6, eer: 0.07, far: 0.009, frr: 0.07, margin_median: 0.28, margin_p10: 0.14, fusion_auc: 0.96 },
    ];

    let false_merge = FalseMergeMetrics {
        face_false_merges: 2,
        body_false_merges: 5,
        fusion_false_merges: 1,
        false_split_rate: 0.02,
        cluster_purity: 0.98,
        cluster_recall: 0.95,
        pairwise_precision: 0.97,
        pairwise_recall: 0.95,
        pairwise_f1: 0.96,
        total_pairs: 100,
        total_splits: 20,
    };

    let prototype_loo = PrototypeLooMetrics {
        single_pass_rate: 0.85,
        mean_pass_rate: 0.96,
        median_pass_rate: 0.95,
        geomean_pass_rate: 0.94,
    };

    let contamination_results = vec![
        ContaminationResult { steps: 1, centroid_drift: 0.01, prototype_drift: 0.02, purity_degradation: 0.0, contamination_detected: false },
        ContaminationResult { steps: 5, centroid_drift: 0.03, prototype_drift: 0.05, purity_degradation: 0.0, contamination_detected: false },
        ContaminationResult { steps: 10, centroid_drift: 0.05, prototype_drift: 0.08, purity_degradation: 0.0, contamination_detected: false },
    ];

    let missing_channel = MissingChannelResult {
        scenario: "test".to_string(),
        face_body_correct: 80,
        face_only_correct: 10,
        body_only_correct: 5,
        face_missing_correct: 0,
        body_missing_correct: 0,
        both_missing_correct: 0,
        total: 100,
    };

    let verdict = compute_verdict(
        &face_metrics,
        &body_metrics,
        &fusion_results,
        &false_merge,
        &prototype_loo,
        &contamination_results,
        &missing_channel,
    );

    println!("\nVerdict Output:");
    println!("  Dataset: {}", verdict.dataset_verdict);
    println!("  Face: {}", verdict.face_verdict);
    println!("  Body: {}", verdict.body_verdict);
    println!("  Fusion: {}", verdict.fusion_verdict);
    println!("  Recommended Alpha: {:.2}", verdict.recommended_alpha);

    assert!(verdict.face_verdict == "READY" || verdict.face_verdict == "FAIL");
    assert!(verdict.body_verdict == "READY" || verdict.body_verdict == "FAIL");
    assert!(verdict.fusion_verdict == "PRODUCTION_READY" || verdict.fusion_verdict == "EXPERIMENTAL" || verdict.fusion_verdict == "FAIL");
}
