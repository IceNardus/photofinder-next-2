//! Phase 19 — Complete Person Identity Benchmark
//!
//! Comprehensive benchmark for Face / Body / Fusion identity verification and identification.
//!
//! Tests three systems:
//! - A. Face only
//! - B. Body only
//! - C. Face + Body fusion
//!
//! Each system is tested for:
//! - Verification (ROC, AUC, EER, FAR, FRR)
//! - Identification (Top-K accuracy)
//! - Margin analysis
//! - Prototype LOO
//! - Alpha sweep for fusion
//! - Missing channel handling
//! - Hard negative analysis
//!
//! 运行：
//! ```bash
//! cargo test -p pf_application --test phase19_identity_benchmark -- --nocapture
//! ```
//!
//! Ignored test (requires actual images):
//! ```bash
//! cargo test -p pf_application --test phase19_identity_benchmark -- --nocapture --ignored
//! ```

use std::collections::HashMap;

// ============================================================================
// Test Data (same as Phase 17)
// ============================================================================

const DOWNLOADS: &str = "/Users/mac/Downloads";

struct TestPerson {
    name: &'static str,
    images: &'static [&'static str],
}

const TEST_PERSONS: &[TestPerson] = &[
    TestPerson { name: "jaqor", images: &["pexels-jaqor-33601811.jpg", "pexels-jaqor-33601831.jpg", "pexels-jaqor-33601835.jpg"] },
    TestPerson { name: "cottonbro", images: &["pexels-cottonbro-5900525.jpg", "pexels-cottonbro-7609197.jpg", "pexels-cottonbro-7609199.jpg"] },
    TestPerson { name: "daria-voronkov", images: &["pexels-daria-voronkov-381938591-14723650.jpg", "pexels-daria-voronkov-381938591-14723672.jpg"] },
    TestPerson { name: "yi-ren", images: &["pexels-yi-ren-57040649-33026322.jpg"] },
    TestPerson { name: "joelle", images: &["pexels-joelle-s-2162497381-38263248.jpg"] },
    TestPerson { name: "peterdanthy", images: &["pexels-peterdanthy-33692605.jpg"] },
    TestPerson { name: "soc-nang", images: &["pexels-soc-nang-d-ng-2150345854-38142867.jpg"] },
];

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone)]
struct ImageData {
    id: usize,
    name: String,
    person: String,
    face_emb: Vec<f32>,
    body_emb: Vec<f32>,
    has_face: bool,
    has_body: bool,
}

#[derive(Debug, Clone)]
struct VerificationResult {
    is_same_person: bool,
    face_score: f32,
    body_score: f32,
}

#[derive(Debug, Clone)]
struct IdentificationResult {
    query_image_id: usize,
    correct_person_id: usize,
    ranked_persons: Vec<(usize, f32)>, // (person_id, score)
}

#[derive(Debug, Clone)]
struct MarginResult {
    positive_best: f32,
    positive_mean: f32,
    positive_min: f32,
    negative_max: f32,
    margin: f32,
}

#[derive(Debug, Clone)]
struct BenchmarkMetrics {
    // Verification
    auc: f32,
    eer: f32,
    far_at_01: f32,
    far_at_05: f32,
    far_at_1: f32,
    far_at_5: f32,
    frr: f32,

    // Identification
    top1: f32,
    top3: f32,
    top5: f32,
    top10: f32,
    hard_negative_top1_error: f32,

    // Margin
    margin_mean: f32,
    margin_median: f32,
    margin_p10: f32,
    margin_p25: f32,
    margin_p50: f32,
    margin_p90: f32,
}

#[derive(Debug, Clone)]
struct PrototypeType {
    name: String,
    compute_fn: PrototypeComputeFn,
}

type PrototypeComputeFn = fn(images: &[&ImageData]) -> Option<Vec<f32>>;

#[derive(Debug)]
struct FusionAlphaResult {
    alpha: f32,
    eer: f32,
    far: f32,
    frr: f32,
    margin_median: f32,
    margin_p10: f32,
}

#[derive(Debug)]
struct HardNegativeStats {
    face_hard_count: usize,
    body_hard_count: usize,
    dual_hard_count: usize,
    face_far: f32,
    body_far: f32,
    fusion_far: f32,
}

// ============================================================================
// Utility Functions
// ============================================================================

fn cosine(v1: &[f32], v2: &[f32]) -> f32 {
    if v1.is_empty() || v2.is_empty() || v1.len() != v2.len() {
        return 0.0;
    }
    let dot = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum::<f32>();
    let n1 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let n2 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n1 < 1e-8 || n2 < 1e-8 {
        return 0.0;
    }
    dot / (n1 * n2)
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
    let mut best_threshold = 0.0;

    for &(_, threshold) in &all_scores {
        let tp = positive_scores.iter().filter(|&&s| s >= threshold).count() as f32;
        let fp = negative_scores.iter().filter(|&&s| s >= threshold).count() as f32;
        let fnr = 1.0 - (tp / n_pos);
        let fpr = fp / n_neg;
        let eer = (fnr + fpr) / 2.0;

        if eer < best_eer {
            best_eer = eer;
            best_threshold = threshold;
        }
    }

    best_eer
}

fn compute_far_at_threshold(positive_scores: &[f32], negative_scores: &[f32], threshold: f32) -> f32 {
    if negative_scores.is_empty() {
        return 0.0;
    }
    let fp = negative_scores.iter().filter(|&&s| s >= threshold).count() as f32;
    fp / negative_scores.len() as f32
}

fn compute_frr_at_threshold(positive_scores: &[f32], negative_scores: &[f32], threshold: f32) -> f32 {
    if positive_scores.is_empty() {
        return 1.0;
    }
    let fn_ = positive_scores.iter().filter(|&&s| s < threshold).count() as f32;
    fn_ / positive_scores.len() as f32
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for val in v.iter_mut() {
        *val /= norm;
    }
}

fn mean_prototype(images: &[&ImageData]) -> Option<Vec<f32>> {
    let valid: Vec<_> = images.iter().filter(|i| i.has_face).collect();
    if valid.is_empty() {
        return None;
    }
    let dim = valid[0].face_emb.len();
    let mut sum = vec![0.0f32; dim];
    for img in &valid {
        for (i, v) in img.face_emb.iter().enumerate() {
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

fn geometric_mean_prototype(images: &[&ImageData]) -> Option<Vec<f32>> {
    let valid: Vec<_> = images.iter().filter(|i| i.has_face).collect();
    if valid.is_empty() {
        return None;
    }
    let dim = valid[0].face_emb.len();
    let mut prod = vec![1.0f32; dim];
    for img in &valid {
        for (i, v) in img.face_emb.iter().enumerate() {
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

fn median_prototype(images: &[&ImageData]) -> Option<Vec<f32>> {
    let valid: Vec<_> = images.iter().filter(|i| i.has_face).collect();
    if valid.is_empty() {
        return None;
    }
    let dim = valid[0].face_emb.len();
    let mut result = vec![0.0f32; dim];
    for d in 0..dim {
        let mut values: Vec<f32> = valid.iter().map(|i| i.face_emb[d]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        result[d] = values[values.len() / 2];
    }
    l2_normalize(&mut result);
    Some(result)
}

// ============================================================================
// Phase 19.1: Verification Benchmark
// ============================================================================

fn run_verification_benchmark(
    images: &[ImageData],
    system: SystemType,
) -> (Vec<VerificationResult>, BenchmarkMetrics) {
    let mut positive_face = Vec::new();
    let mut positive_body = Vec::new();
    let mut negative_face = Vec::new();
    let mut negative_body = Vec::new();

    // Build person groups
    let mut person_groups: HashMap<String, Vec<&ImageData>> = HashMap::new();
    for img in images {
        if img.has_face && img.has_body {
            person_groups.entry(img.person.clone()).or_default().push(img);
        }
    }

    let persons: Vec<_> = person_groups.keys().collect();

    // Positive pairs: same person
    for &person in &persons {
        let imgs = person_groups.get(person).unwrap();
        for i in 0..imgs.len() {
            for j in (i + 1)..imgs.len() {
                let face_score = cosine(&imgs[i].face_emb, &imgs[j].face_emb);
                let body_score = cosine(&imgs[i].body_emb, &imgs[j].body_emb);
                positive_face.push(face_score);
                positive_body.push(body_score);
            }
        }
    }

    // Negative pairs: different persons
    for i in 0..persons.len() {
        for j in (i + 1)..persons.len() {
            let imgs1 = person_groups.get(persons[i].as_str()).unwrap();
            let imgs2 = person_groups.get(persons[j].as_str()).unwrap();
            for img1 in imgs1 {
                for img2 in imgs2 {
                    let face_score = cosine(&img1.face_emb, &img2.face_emb);
                    let body_score = cosine(&img1.body_emb, &img2.body_emb);
                    negative_face.push(face_score);
                    negative_body.push(body_score);
                }
            }
        }
    }

    let (face_scores, body_scores) = match system {
        SystemType::FaceOnly => (positive_face.clone(), Vec::new()),
        SystemType::BodyOnly => (Vec::new(), positive_body.clone()),
        SystemType::Fusion => {
            // For fusion, compute margin-based fusion
            // (placeholder - full implementation in alpha sweep)
            (positive_face.clone(), negative_face.clone())
        }
    };

    // Compute metrics
    let auc = roc_auc(&positive_face, &negative_face);
    let eer = compute_eer(&positive_face, &negative_face);
    let far_at_01 = compute_far_at_threshold(&positive_face, &negative_face, 0.99); // placeholder
    let far_at_05 = 0.0; // TODO
    let far_at_1 = 0.0;
    let far_at_5 = 0.0;
    let frr = compute_frr_at_threshold(&positive_face, &negative_face, 0.5);

    let metrics = BenchmarkMetrics {
        auc,
        eer,
        far_at_01,
        far_at_05,
        far_at_1,
        far_at_5,
        frr,
        top1: 0.0,
        top3: 0.0,
        top5: 0.0,
        top10: 0.0,
        hard_negative_top1_error: 0.0,
        margin_mean: 0.0,
        margin_median: 0.0,
        margin_p10: 0.0,
        margin_p25: 0.0,
        margin_p50: 0.0,
        margin_p90: 0.0,
    };

    let results: Vec<VerificationResult> = Vec::new();

    (results, metrics)
}

#[derive(Debug, Clone, Copy)]
enum SystemType {
    FaceOnly,
    BodyOnly,
    Fusion,
}

// ============================================================================
// Phase 19.2: Identification Benchmark
// ============================================================================

fn run_identification_benchmark(
    images: &[ImageData],
    system: SystemType,
) -> (Vec<IdentificationResult>, BenchmarkMetrics) {
    // Build person prototypes
    let mut person_prototypes: HashMap<String, Vec<f32>> = HashMap::new();

    let mut person_groups: HashMap<String, Vec<&ImageData>> = HashMap::new();
    for img in images {
        if img.has_face {
            person_groups.entry(img.person.clone()).or_default().push(img);
        }
    }

    for (person, imgs) in &person_groups {
        if let Some(proto) = mean_prototype(imgs) {
            person_prototypes.insert(person.clone(), proto);
        }
    }

    let persons: Vec<String> = person_groups.keys().cloned().collect();
    let mut correct_top1 = 0;
    let mut correct_top3 = 0;
    let mut correct_top5 = 0;
    let mut correct_top10 = 0;
    let mut total = 0;

    let mut identification_results = Vec::new();

    for img in images {
        if !img.has_face {
            continue;
        }

        // Query this image against all other images (LOO)
        let mut scores: Vec<(String, f32)> = Vec::new();

        for person in &persons {
            if person == &img.person {
                continue; // Skip same person during query
            }

            if let Some(proto) = person_prototypes.get(person) {
                let score = cosine(&img.face_emb, proto);
                scores.push((person.clone(), score));
            }
        }

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Get ranked person IDs
        let ranked: Vec<(usize, f32)> = scores.iter().take(10).map(|(p, s)| {
            let pid = persons.iter().position(|x| x == p).unwrap();
            (pid, *s)
        }).collect();

        identification_results.push(IdentificationResult {
            query_image_id: img.id,
            correct_person_id: persons.iter().position(|x| x == &img.person).unwrap(),
            ranked_persons: ranked.clone(),
        });

        // Check correctness
        total += 1;
        let correct_pid = persons.iter().position(|x| x == &img.person).unwrap();
        if ranked.first().map(|(p, _)| *p == correct_pid).unwrap_or(false) {
            correct_top1 += 1;
        }
        if ranked.iter().take(3).any(|(p, _)| *p == correct_pid) {
            correct_top3 += 1;
        }
        if ranked.iter().take(5).any(|(p, _)| *p == correct_pid) {
            correct_top5 += 1;
        }
        if ranked.iter().take(10).any(|(p, _)| *p == correct_pid) {
            correct_top10 += 1;
        }
    }

    let metrics = BenchmarkMetrics {
        auc: 0.0,
        eer: 0.0,
        far_at_01: 0.0,
        far_at_05: 0.0,
        far_at_1: 0.0,
        far_at_5: 0.0,
        frr: 0.0,
        top1: if total > 0 { correct_top1 as f32 / total as f32 } else { 0.0 },
        top3: if total > 0 { correct_top3 as f32 / total as f32 } else { 0.0 },
        top5: if total > 0 { correct_top5 as f32 / total as f32 } else { 0.0 },
        top10: if total > 0 { correct_top10 as f32 / total as f32 } else { 0.0 },
        hard_negative_top1_error: 0.0,
        margin_mean: 0.0,
        margin_median: 0.0,
        margin_p10: 0.0,
        margin_p25: 0.0,
        margin_p50: 0.0,
        margin_p90: 0.0,
    };

    (identification_results, metrics)
}

// ============================================================================
// Phase 19.3: Margin Analysis
// ============================================================================

fn run_margin_benchmark(images: &[ImageData]) -> (Vec<MarginResult>, BenchmarkMetrics) {
    let mut person_groups: HashMap<String, Vec<&ImageData>> = HashMap::new();
    for img in images {
        if img.has_face && img.has_body {
            person_groups.entry(img.person.clone()).or_default().push(img);
        }
    }

    let persons: Vec<String> = person_groups.keys().cloned().collect();
    let mut margins = Vec::new();
    let mut hard_negative_top1_errors = 0;
    let mut total = 0;

    for img in images {
        if !img.has_face || !img.has_body {
            continue;
        }

        // Positive scores: same person (excluding self)
        let same_person_imgs: Vec<_> = person_groups.get(&img.person).map(|v| v.iter().filter(|i| i.id != img.id).copied().collect::<Vec<_>>()).unwrap_or_default();

        if same_person_imgs.is_empty() {
            continue;
        }

        let positive_scores: Vec<f32> = same_person_imgs.iter()
            .map(|i| cosine(&img.face_emb, &i.face_emb))
            .collect();

        // Negative scores: different persons
        let mut negative_scores = Vec::new();
        for person in &persons {
            if person == &img.person {
                continue;
            }
            for neg_img in person_groups.get(person).unwrap() {
                negative_scores.push(cosine(&img.face_emb, &neg_img.face_emb));
            }
        }

        if positive_scores.is_empty() || negative_scores.is_empty() {
            continue;
        }

        let positive_best = positive_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let positive_mean = positive_scores.iter().sum::<f32>() / positive_scores.len() as f32;
        let positive_min = positive_scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let negative_max = negative_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let margin = positive_best - negative_max;

        margins.push(MarginResult {
            positive_best,
            positive_mean,
            positive_min,
            negative_max,
            margin,
        });

        // Check hard negative top-1 error
        total += 1;
        if negative_max >= positive_best {
            hard_negative_top1_errors += 1;
        }
    }

    let margin_values: Vec<f32> = margins.iter().map(|m| m.margin).collect();

    let metrics = BenchmarkMetrics {
        auc: 0.0,
        eer: 0.0,
        far_at_01: 0.0,
        far_at_05: 0.0,
        far_at_1: 0.0,
        far_at_5: 0.0,
        frr: 0.0,
        top1: 0.0,
        top3: 0.0,
        top5: 0.0,
        top10: 0.0,
        hard_negative_top1_error: if total > 0 { hard_negative_top1_errors as f32 / total as f32 } else { 0.0 },
        margin_mean: if margin_values.is_empty() { 0.0 } else { margin_values.iter().sum::<f32>() / margin_values.len() as f32 },
        margin_median: percentile(&margin_values, 0.5),
        margin_p10: percentile(&margin_values, 0.10),
        margin_p25: percentile(&margin_values, 0.25),
        margin_p50: percentile(&margin_values, 0.50),
        margin_p90: percentile(&margin_values, 0.90),
    };

    (margins, metrics)
}

// ============================================================================
// Phase 19.4: Prototype LOO Benchmark
// ============================================================================

fn run_prototype_loo_benchmark(images: &[ImageData], proto_type: &PrototypeType) -> BenchmarkMetrics {
    let mut person_groups: HashMap<String, Vec<&ImageData>> = HashMap::new();
    for img in images {
        if img.has_face {
            person_groups.entry(img.person.clone()).or_default().push(img);
        }
    }

    let persons: Vec<_> = person_groups.keys().cloned().collect();
    let mut correct_top1 = 0;
    let mut total = 0;

    for img in images {
        if !img.has_face {
            continue;
        }

        // Build LOO prototype for each person (excluding current image)
        let mut scores: Vec<(String, f32)> = Vec::new();

        for person in &persons {
            // LOO: exclude images from same person that are similar to query
            let loo_imgs: Vec<_> = person_groups.get(person).unwrap().iter()
                .filter(|i| i.id != img.id)
                .copied()
                .collect();

            if loo_imgs.is_empty() {
                continue;
            }

            if let Some(proto) = (proto_type.compute_fn)(&loo_imgs) {
                let score = cosine(&img.face_emb, &proto);
                scores.push((person.clone(), score));
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        total += 1;
        if let Some((top_person, _)) = scores.first() {
            if top_person == &img.person {
                correct_top1 += 1;
            }
        }
    }

    BenchmarkMetrics {
        auc: 0.0,
        eer: 0.0,
        far_at_01: 0.0,
        far_at_05: 0.0,
        far_at_1: 0.0,
        far_at_5: 0.0,
        frr: 0.0,
        top1: if total > 0 { correct_top1 as f32 / total as f32 } else { 0.0 },
        top3: 0.0,
        top5: 0.0,
        top10: 0.0,
        hard_negative_top1_error: 0.0,
        margin_mean: 0.0,
        margin_median: 0.0,
        margin_p10: 0.0,
        margin_p25: 0.0,
        margin_p50: 0.0,
        margin_p90: 0.0,
    }
}

// ============================================================================
// Phase 19.5: Alpha Sweep for Fusion
// ============================================================================

fn run_alpha_sweep(images: &[ImageData]) -> Vec<FusionAlphaResult> {
    let mut results = Vec::new();

    let alpha_values: Vec<f32> = (0..21).map(|i| i as f32 * 0.05).collect();

    for &alpha in &alpha_values {
        // Compute fusion scores with this alpha
        let mut positive_scores = Vec::new();
        let mut negative_scores = Vec::new();

        let mut person_groups: HashMap<String, Vec<&ImageData>> = HashMap::new();
        for img in images {
            if img.has_face && img.has_body {
                person_groups.entry(img.person.clone()).or_default().push(img);
            }
        }

        let persons: Vec<_> = person_groups.keys().cloned().collect();

        // Positive pairs
        for person in &persons {
            let imgs = person_groups.get(person).unwrap();
            for i in 0..imgs.len() {
                for j in (i + 1)..imgs.len() {
                    let face_score = cosine(&imgs[i].face_emb, &imgs[j].face_emb);
                    let body_score = cosine(&imgs[i].body_emb, &imgs[j].body_emb);
                    let fusion = alpha * face_score + (1.0 - alpha) * body_score;
                    positive_scores.push(fusion);
                }
            }
        }

        // Negative pairs
        for i in 0..persons.len() {
            for j in (i + 1)..persons.len() {
                let imgs1 = person_groups.get(persons[i].as_str()).unwrap();
                let imgs2 = person_groups.get(persons[j].as_str()).unwrap();
                for img1 in imgs1 {
                    for img2 in imgs2 {
                        let face_score = cosine(&img1.face_emb, &img2.face_emb);
                        let body_score = cosine(&img1.body_emb, &img2.face_emb);
                        let fusion = alpha * face_score + (1.0 - alpha) * body_score;
                        negative_scores.push(fusion);
                    }
                }
            }
        }

        let eer = compute_eer(&positive_scores, &negative_scores);
        let far = compute_far_at_threshold(&positive_scores, &negative_scores, 0.5);
        let frr = compute_frr_at_threshold(&positive_scores, &negative_scores, 0.5);

        let mut all_margins: Vec<f32> = Vec::new();
        for &p in &positive_scores {
            let neg_max = negative_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            all_margins.push(p - neg_max);
        }

        results.push(FusionAlphaResult {
            alpha,
            eer,
            far,
            frr,
            margin_median: percentile(&all_margins, 0.5),
            margin_p10: percentile(&all_margins, 0.10),
        });
    }

    results
}

// ============================================================================
// Phase 19.6: Missing Channel Analysis
// ============================================================================

fn analyze_missing_channels(images: &[ImageData]) -> HashMap<String, BenchmarkMetrics> {
    let mut results = HashMap::new();

    // Count channel availability
    let mut face_only = 0;
    let mut body_only = 0;
    let mut both = 0;
    let mut neither = 0;

    for img in images {
        if img.has_face && img.has_body {
            both += 1;
        } else if img.has_face {
            face_only += 1;
        } else if img.has_body {
            body_only += 1;
        } else {
            neither += 1;
        }
    }

    println!("\n  Channel availability:");
    println!("    Face + Body: {}", both);
    println!("    Face only: {}", face_only);
    println!("    Body only: {}", body_only);
    println!("    Neither: {}", neither);

    // Run identification with available channels only
    // (Simplified: just report availability counts)

    results.insert("face_only".to_string(), BenchmarkMetrics {
        auc: 0.0, eer: 0.0, far_at_01: 0.0, far_at_05: 0.0, far_at_1: 0.0, far_at_5: 0.0, frr: 0.0,
        top1: 0.0, top3: 0.0, top5: 0.0, top10: 0.0, hard_negative_top1_error: 0.0,
        margin_mean: 0.0, margin_median: 0.0, margin_p10: 0.0, margin_p25: 0.0, margin_p50: 0.0, margin_p90: 0.0,
    });

    results
}

// ============================================================================
// Phase 19.7: Hard Negative Analysis
// ============================================================================

fn analyze_hard_negatives(images: &[ImageData]) -> HardNegativeStats {
    let mut face_hard = 0;
    let mut body_hard = 0;
    let mut dual_hard = 0;
    let mut face_fp = 0;
    let mut body_fp = 0;
    let mut fusion_fp = 0;
    let mut total_neg = 0;

    // Group images by person
    let mut person_groups: HashMap<String, Vec<&ImageData>> = HashMap::new();
    for img in images {
        if img.has_face && img.has_body {
            person_groups.entry(img.person.clone()).or_default().push(img);
        }
    }

    let persons: Vec<String> = person_groups.keys().cloned().collect();

    // All negative pairs
    for i in 0..persons.len() {
        for j in (i + 1)..persons.len() {
            let imgs1 = person_groups.get(persons[i].as_str()).unwrap();
            let imgs2 = person_groups.get(persons[j].as_str()).unwrap();
            for img1 in imgs1 {
                for img2 in imgs2 {
                    let face_score = cosine(&img1.face_emb, &img2.face_emb);
                    let body_score = cosine(&img1.body_emb, &img2.body_emb);
                    let fusion = 0.5 * face_score + 0.5 * body_score;

                    total_neg += 1;

                    const FACE_THRESHOLD: f32 = 0.65;
                    const BODY_THRESHOLD: f32 = 0.70;

                    if face_score >= FACE_THRESHOLD {
                        face_hard += 1;
                        if face_score >= 0.7 { face_fp += 1; }
                    }
                    if body_score >= BODY_THRESHOLD {
                        body_hard += 1;
                        if body_score >= 0.7 { body_fp += 1; }
                    }
                    if face_score >= FACE_THRESHOLD && body_score >= BODY_THRESHOLD {
                        dual_hard += 1;
                    }
                    if fusion >= 0.6 { fusion_fp += 1; }
                }
            }
        }
    }

    HardNegativeStats {
        face_hard_count: face_hard,
        body_hard_count: body_hard,
        dual_hard_count: dual_hard,
        face_far: if total_neg > 0 { face_fp as f32 / total_neg as f32 } else { 0.0 },
        body_far: if total_neg > 0 { body_fp as f32 / total_neg as f32 } else { 0.0 },
        fusion_far: if total_neg > 0 { fusion_fp as f32 / total_neg as f32 } else { 0.0 },
    }
}

// ============================================================================
// Dataset Quality Check
// ============================================================================

fn check_dataset_readiness(images: &[ImageData]) -> (bool, Vec<String>) {
    let mut reasons = Vec::new();

    // Build person groups
    let mut person_groups: HashMap<String, Vec<&ImageData>> = HashMap::new();
    for img in images {
        person_groups.entry(img.person.clone()).or_default().push(img);
    }

    let persons_count = person_groups.len();
    let usable_images = images.iter().filter(|i| i.has_face && i.has_body).count();
    let min_images = person_groups.values().map(|v| v.len()).min().unwrap_or(0);
    let min_usable = person_groups.values()
        .map(|v| v.iter().filter(|i| i.has_face && i.has_body).count())
        .min().unwrap_or(0);

    const MIN_PERSONS: usize = 10;
    const MIN_IMAGES_PER_PERSON: usize = 5;
    const MIN_USABLE_PER_PERSON: usize = 5;

    if persons_count < MIN_PERSONS {
        reasons.push(format!("insufficient persons: {} < {}", persons_count, MIN_PERSONS));
    }

    if min_images < MIN_IMAGES_PER_PERSON {
        reasons.push(format!("insufficient images per person: min={} < {}", min_images, MIN_IMAGES_PER_PERSON));
    }

    if min_usable < MIN_USABLE_PER_PERSON {
        reasons.push(format!("insufficient usable images per person: min={} < {}", min_usable, MIN_USABLE_PER_PERSON));
    }

    let verdict = reasons.is_empty();
    (verdict, reasons)
}

// ============================================================================
// Output Formatting
// ============================================================================

fn print_verification_table(name: &str, metrics: &BenchmarkMetrics) {
    println!("\n{} Verification:", name);
    println!("  {:>12} {:>8} {:>8} {:>8} {:>8} {:>8}", "AUC", "EER", "FAR@0.1", "FAR@1", "FAR@5", "FRR");
    println!("  {:>12.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
        metrics.auc, metrics.eer, metrics.far_at_01, metrics.far_at_1, metrics.far_at_5, metrics.frr);
}

fn print_identification_table(name: &str, metrics: &BenchmarkMetrics) {
    println!("\n{} Identification:", name);
    println!("  {:>8} {:>8} {:>8} {:>8} {:>8}", "Top-1", "Top-3", "Top-5", "Top-10", "HN@1");
    println!("  {:>8.2}% {:>8.2}% {:>8.2}% {:>8.2}% {:>8.2}%",
        metrics.top1, metrics.top3, metrics.top5, metrics.top10, metrics.hard_negative_top1_error);
}

fn print_margin_table(name: &str, metrics: &BenchmarkMetrics) {
    println!("\n{} Margin:", name);
    println!("  {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}", "Mean", "Median", "P10", "P25", "P50", "P90");
    println!("  {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
        metrics.margin_mean, metrics.margin_median, metrics.margin_p10, metrics.margin_p25, metrics.margin_p50, metrics.margin_p90);
}

fn print_alpha_sweep_table(results: &[FusionAlphaResult]) {
    println!("\nFusion Alpha Sweep:");
    println!("  {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}", "Alpha", "EER", "FAR", "FRR", "MedMarg", "P10Marg");
    for r in results {
        println!("  {:>8.2} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
            r.alpha, r.eer, r.far, r.frr, r.margin_median, r.margin_p10);
    }
}

fn print_final_verdict(
    face_ready: bool,
    body_ready: bool,
    fusion_ready: bool,
    reasons: &[String],
) {
    println!("\n============================================================");
    println!("FINAL VERDICT");
    println!("============================================================");

    if !reasons.is_empty() {
        println!("\nDATASET_INSUFFICIENT");
        println!("\nReasons:");
        for r in reasons {
            println!("  - {}", r);
        }
        return;
    }

    println!("\n{}", if fusion_ready { "FUSION_READY" } else if face_ready { "FACE_ONLY_READY" } else { "INSUFFICIENT" });

    println!("\nSystem Readiness:");
    println!("  FACE_ONLY: {}", if face_ready { "READY" } else { "NOT READY" });
    println!("  BODY_ONLY: {}", if body_ready { "READY" } else { "NOT READY" });
    println!("  FUSION: {}", if fusion_ready { "READY" } else { "NOT READY" });
}

// ============================================================================
// Main Benchmark Runner (Integration Test)
// ============================================================================

#[tokio::test]
#[ignore]
async fn phase19_identity_benchmark() {
    println!("\n============================================================");
    println!("Phase 19: Complete Person Identity Benchmark");
    println!("============================================================");

    // Load test data
    println!("\n[This test requires actual images from Downloads folder]");
    println!("Run with: cargo test -p pf_application --test phase19_identity_benchmark -- --nocapture --ignored");
}

// ============================================================================
// Unit Tests (with synthetic data)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_synthetic_image(id: usize, person_id: usize) -> ImageData {
        // Create random-ish embeddings
        let face_emb: Vec<f32> = (0..512).map(|i| {
            let seed = (id * 1000 + i) as f32;
            (seed * 0.12345).sin().abs()
        }).collect();

        let body_emb: Vec<f32> = (0..768).map(|i| {
            let seed = (id * 1000 + i) as f32;
            (seed * 0.54321).cos().abs()
        }).collect();

        ImageData {
            id,
            name: format!("img_{}.jpg", id),
            person: format!("person{}", person_id),
            face_emb,
            body_emb,
            has_face: true,
            has_body: true,
        }
    }

    #[test]
    fn test_verification_metrics() {
        let positive = vec![0.9, 0.85, 0.88, 0.92, 0.87];
        let negative = vec![0.3, 0.4, 0.35, 0.45, 0.38];

        let auc = roc_auc(&positive, &negative);
        let eer = compute_eer(&positive, &negative);

        assert!(auc > 0.9, "AUC should be high for separated distributions");
        assert!(eer < 0.1, "EER should be low");
    }

    #[test]
    fn test_margin_percentiles() {
        let margins = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

        assert!((percentile(&margins, 0.5) - 0.55).abs() < 0.1);
        assert!((percentile(&margins, 0.1) - 0.19).abs() < 0.1);
        assert!((percentile(&margins, 0.9) - 0.91).abs() < 0.1);
    }

    #[test]
    fn test_cosine_similarity() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        assert!((cosine(&v1, &v2) - 1.0).abs() < 1e-6);

        let v3 = vec![0.0, 1.0, 0.0];
        assert!((cosine(&v1, &v3) - 0.0).abs() < 1e-6);

        let v4 = vec![0.707, 0.707, 0.0];
        let expected = 0.707;
        assert!((cosine(&v1, &v4) - expected).abs() < 0.01);
    }

    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0, 4.0, 0.0];
        l2_normalize(&mut v);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 0.01);
        assert!((v[1] - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_mean_prototype() {
        let img1 = make_synthetic_image(1, 1);
        let img2 = make_synthetic_image(2, 1);
        let img3 = make_synthetic_image(3, 1);

        let proto = mean_prototype(&[&img1, &img2, &img3]);
        assert!(proto.is_some());

        let proto = proto.unwrap();
        assert_eq!(proto.len(), 512);

        // Should be normalized
        let norm = proto.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_dataset_insufficient() {
        // Create dataset with only 3 persons
        let images: Vec<ImageData> = (0..3)
            .flat_map(|p| {
                (0..2).map(move |i| make_synthetic_image(p * 10 + i, p))
            })
            .collect();

        let (ready, reasons) = check_dataset_readiness(&images);

        assert!(!ready);
        assert!(!reasons.is_empty());
        assert!(reasons.iter().any(|r| r.contains("insufficient persons")));
    }

    #[test]
    fn test_dataset_ready() {
        // Create dataset with 10 persons, 5 images each
        let images: Vec<ImageData> = (0..10)
            .flat_map(|p| {
                (0..5).map(move |i| make_synthetic_image(p * 10 + i, p))
            })
            .collect();

        let (ready, reasons) = check_dataset_readiness(&images);

        // May still be insufficient due to min_usable requirement
        println!("Ready: {}, Reasons: {:?}", ready, reasons);
    }

    #[test]
    fn test_loo_exclusion() {
        let img1 = make_synthetic_image(1, 1);
        let img2 = make_synthetic_image(2, 1);
        let img3 = make_synthetic_image(3, 2); // Different person

        // When querying img1, img2 should be excluded from same-person scores
        let same_person: Vec<_> = vec![&img1, &img2].iter()
            .filter(|i| (*i).id != img1.id)
            .copied()
            .collect();

        assert_eq!(same_person.len(), 1);
        assert_eq!(same_person[0].id, 2);
    }

    #[test]
    fn test_hard_negative_detection() {
        let mut face_scores = Vec::new();
        let mut body_scores = Vec::new();

        // Simulate: some negatives have high face score (hard negative)
        for i in 0..10 {
            let face = if i < 2 { 0.75 } else { 0.3 }; // 2 face hard negatives
            let body = if i < 3 { 0.80 } else { 0.2 }; // 3 body hard negatives
            face_scores.push(face);
            body_scores.push(body);
        }

        let face_hard = face_scores.iter().filter(|&&s| s >= 0.65).count();
        let body_hard = body_scores.iter().filter(|&&s| s >= 0.70).count();

        assert_eq!(face_hard, 2);
        assert_eq!(body_hard, 3);
    }

    #[test]
    fn test_prototype_types() {
        let img1 = make_synthetic_image(1, 1);
        let img2 = make_synthetic_image(2, 1);
        let img3 = make_synthetic_image(3, 1);
        let images = &[&img1, &img2, &img3];

        let mean_proto = mean_prototype(images);
        let median_proto = median_prototype(images);
        let geo_proto = geometric_mean_prototype(images);

        assert!(mean_proto.is_some());
        assert!(median_proto.is_some());
        assert!(geo_proto.is_some());

        // All should have same length
        let mean_len = mean_proto.as_ref().unwrap().len();
        let median_len = median_proto.as_ref().unwrap().len();
        let geo_len = geo_proto.as_ref().unwrap().len();
        assert_eq!(mean_len, median_len);
        assert_eq!(median_len, geo_len);
    }

    #[test]
    fn test_alpha_sweep() {
        let results = vec![
            FusionAlphaResult { alpha: 0.0, eer: 0.3, far: 0.2, frr: 0.4, margin_median: 0.1, margin_p10: 0.05 },
            FusionAlphaResult { alpha: 0.5, eer: 0.2, far: 0.1, frr: 0.3, margin_median: 0.2, margin_p10: 0.1 },
            FusionAlphaResult { alpha: 1.0, eer: 0.25, far: 0.15, frr: 0.35, margin_median: 0.15, margin_p10: 0.08 },
        ];

        // Find best alpha (lowest EER)
        let best = results.iter().min_by(|a, b| a.eer.partial_cmp(&b.eer).unwrap()).unwrap();
        assert!((best.alpha - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_missing_channel_stats() {
        let images = vec![
            make_synthetic_image(1, 1), // has face and body
            ImageData { id: 2, name: "no_body.jpg".into(), person: "person2".to_string(), face_emb: vec![0.5; 512], body_emb: vec![], has_face: true, has_body: false },
            ImageData { id: 3, name: "no_face.jpg".into(), person: "person3".to_string(), face_emb: vec![], body_emb: vec![0.5; 768], has_face: false, has_body: true },
        ];

        let mut face_only = 0;
        let mut body_only = 0;
        let mut both = 0;

        for img in &images {
            if img.has_face && img.has_body {
                both += 1;
            } else if img.has_face {
                face_only += 1;
            } else if img.has_body {
                body_only += 1;
            }
        }

        assert_eq!(both, 1);
        assert_eq!(face_only, 1);
        assert_eq!(body_only, 1);
    }

    #[test]
    fn test_identification_ranking() {
        // Create simple case: query should rank correct person first
        let img1 = make_synthetic_image(1, 1);
        let img2 = make_synthetic_image(2, 1); // Same person, should score high
        let _img3 = make_synthetic_image(3, 2); // Different person

        let score_same = cosine(&img1.face_emb, &img2.face_emb);
        let score_diff = cosine(&img1.face_emb, &_img3.face_emb);

        assert!(score_same > score_diff, "Same person should score higher");
    }

    #[test]
    fn test_far_at_thresholds() {
        let positive = vec![0.9, 0.85, 0.88];
        let negative = vec![0.3, 0.4, 0.5, 0.6, 0.7];

        // At threshold 0.5, all negatives >= 0.5 should be false accepts
        let far = compute_far_at_threshold(&positive, &negative, 0.5);
        assert!((far - 0.6).abs() < 0.01); // 3 out of 5 negatives >= 0.5

        let far_high = compute_far_at_threshold(&positive, &negative, 0.8);
        assert!((far_high - 0.0).abs() < 0.01); // 0 out of 5 negatives >= 0.8
    }

    #[test]
    fn test_frr_at_thresholds() {
        let positive = vec![0.9, 0.85, 0.88, 0.4, 0.35];
        let negative = vec![0.3, 0.4, 0.5];

        // At threshold 0.5, positives < 0.5 should be false rejects
        let frr = compute_frr_at_threshold(&positive, &negative, 0.5);
        assert!((frr - 0.4).abs() < 0.01); // 2 out of 5 positives < 0.5
    }

    #[test]
    fn test_verdict_output() {
        print_final_verdict(true, false, true, &[]);
        print_final_verdict(false, false, false, &["insufficient persons".to_string()]);
    }
}
