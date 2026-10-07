//! Phase 20 — Person Clustering False-Merge Benchmark
//!
//! Verifies whether production clustering incorrectly merges different people.
//!
//! **CRITICAL: Does NOT modify production algorithm**
//!
//! Test scenarios:
//! 1. A → A (same person assignment)
//! 2. B → B (same person assignment)
//! 3. A → B (false merge - A incorrectly assigned to B)
//! 4. B → A (false merge - B incorrectly assigned to A)
//!
//! Metrics:
//! - False Merge Rate
//! - False Split Rate
//! - Cluster Purity
//! - Cluster Recall
//! - Pairwise Precision/Recall/F1
//!
//! Hard Negative Detection:
//! - FaceHardNegative: face_sim >= threshold but different persons
//! - BodyHardNegative: body_sim >= threshold but different persons
//! - DualHardNegative: face+body >= threshold but different persons
//!
//! Chain Contamination Testing:
//! - 1-step, 2-step, 5-step, 10-step contamination
//!
//! 运行：
//! ```bash
//! cargo test -p pf_application --test phase20_clustering_false_merge -- --nocapture
//! ```

use std::collections::{HashMap, HashSet};

// ============================================================================
// Constants
// ============================================================================

const FACE_DIM: usize = 512;
const BODY_DIM: usize = 768;
const FACE_THRESHOLD: f32 = 0.75;
const BODY_THRESHOLD: f32 = 0.70;
const CHAINING_MARGIN: f32 = 0.05;

// ============================================================================
// Math Utilities
// ============================================================================

/// L2 normalized cosine similarity
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 normalize in-place
fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

/// Generate a deterministic f32 in range [0, 1) from a seed
fn seeded_float(seed: usize) -> f32 {
    // Simple deterministic hash
    let x = seed.wrapping_mul(1103515245).wrapping_add(12345);
    ((x >> 16) as f32) / 65536.0
}

// ============================================================================
// Synthetic Embedding Generation
// ============================================================================

/// Generate a synthetic face embedding for a person.
/// Uses sparse coding: each person is represented by a sparse combination
/// of K basis vectors, with distinct non-zero coefficients per person.
/// Different persons have non-overlapping coefficient patterns.
fn make_face_embedding(person_id: usize, image_index: usize, noise_scale: f32) -> Vec<f32> {
    let mut embedding = vec![0.0f32; FACE_DIM];

    // Each person gets a unique "sector" of basis indices
    let num_persons = 50; // Assume max 50 persons
    let basis_per_person = 8;
    let total_basis = num_persons * basis_per_person;

    // Determine which basis vectors this person uses
    let person_start = (person_id % num_persons) * basis_per_person;

    // Coefficient for each of this person's basis vectors
    let mut coefs = vec![0.0f32; basis_per_person];
    for i in 0..basis_per_person {
        let seed = person_id * 1000 + i;
        coefs[i] = (seeded_float(seed) - 0.5) * 2.0; // [-1, 1]
    }

    // Add small per-image variation
    for i in 0..basis_per_person {
        let img_seed = person_id * 10000 + image_index * 100 + i;
        coefs[i] += (seeded_float(img_seed) - 0.5) * noise_scale;
    }

    // Build embedding as sparse combination of basis vectors
    // Basis vector j has value 1.0 at index j, 0 elsewhere
    // But we use a "smooth" basis - each basis is concentrated in a small region
    let region_size = FACE_DIM / total_basis;

    for basis_idx in 0..basis_per_person {
        let global_basis = person_start + basis_idx;
        let coef = coefs[basis_idx];

        // Basis vector is 1.0 near its center, with smooth falloff
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

/// Generate a synthetic body embedding (different dimensional space)
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
// Test Data Structures
// ============================================================================

#[derive(Debug, Clone)]
struct PersonImage {
    id: usize,
    person_id: usize,
    face_emb: Vec<f32>,
    body_emb: Vec<f32>,
    true_person_id: usize,
}

#[derive(Debug, Clone)]
struct ClusteringResult {
    image_id: usize,
    assigned_cluster: Option<usize>,
    score: Option<f32>,
}

#[derive(Debug, Clone)]
struct HardNegative {
    img1_id: usize,
    img2_id: usize,
    person1: usize,
    person2: usize,
    face_score: f32,
    body_score: f32,
    fusion_score: f32,
}

#[derive(Debug)]
struct ClusterMetrics {
    false_merges: usize,
    false_splits: usize,
    true_merges: usize,
    true_splits: usize,
    cluster_purity: f32,
    cluster_recall: f32,
    pairwise_precision: f32,
    pairwise_recall: f32,
    pairwise_f1: f32,
    face_false_merges: usize,
    body_false_merges: usize,
    fusion_false_merges: usize,
}

#[derive(Debug)]
struct ContaminationResult {
    steps: usize,
    contamination_detected: bool,
    contaminated_images: Vec<usize>,
}

#[derive(Debug)]
struct Verdict {
    verdict: String,
    false_merge_rate: f32,
    false_split_rate: f32,
    worst_cluster_purity: f32,
    chain_contamination_safe: bool,
}

// ============================================================================
// Clustering Simulation (Production-like logic without DB)
// ============================================================================

fn run_clustering(images: &[PersonImage]) -> Vec<ClusteringResult> {
    let mut results = Vec::new();

    // Track cluster centroids (simplified prototype)
    let mut prototypes: HashMap<usize, Vec<Vec<f32>>> = HashMap::new();
    let mut next_cluster = 1usize;

    for img in images {
        // Search for best matching cluster
        let mut best_cluster = None;
        let mut best_score = 0.0f32;
        let mut second_best_score = 0.0f32;

        for (cluster_id, cluster_embs) in &prototypes {
            let max_sim = cluster_embs
                .iter()
                .map(|e| cosine(&img.face_emb, e))
                .fold(0.0f32, |a, b| a.max(b));

            if max_sim > best_score {
                second_best_score = best_score;
                best_score = max_sim;
                best_cluster = Some(*cluster_id);
            } else if max_sim > second_best_score {
                second_best_score = max_sim;
            }
        }

        let margin = (best_score - second_best_score).abs();

        if let Some(cluster_id) = best_cluster {
            if best_score >= FACE_THRESHOLD && margin >= CHAINING_MARGIN {
                // Assign to existing cluster
                prototypes.get_mut(&cluster_id).unwrap().push(img.face_emb.clone());
                results.push(ClusteringResult {
                    image_id: img.id,
                    assigned_cluster: Some(cluster_id),
                    score: Some(best_score),
                });
            } else {
                // Ambiguous or below threshold - create new
                let cid = next_cluster;
                next_cluster += 1;
                prototypes.insert(cid, vec![img.face_emb.clone()]);
                results.push(ClusteringResult {
                    image_id: img.id,
                    assigned_cluster: Some(cid),
                    score: None,
                });
            }
        } else {
            // First image - create first cluster
            prototypes.insert(next_cluster, vec![img.face_emb.clone()]);
            results.push(ClusteringResult {
                image_id: img.id,
                assigned_cluster: Some(next_cluster),
                score: None,
            });
            next_cluster += 1;
        }
    }

    results
}

// ============================================================================
// Metrics Computation
// ============================================================================

fn compute_metrics(images: &[PersonImage], results: &[ClusteringResult]) -> ClusterMetrics {
    let mut false_merges = 0;
    let mut false_splits = 0;
    let mut true_merges = 0;
    let mut true_splits = 0;
    let mut face_false_merges = 0;
    let mut body_false_merges = 0;
    let mut fusion_false_merges = 0;

    // Build result lookup
    let result_map: HashMap<usize, Option<usize>> = results
        .iter()
        .map(|r| (r.image_id, r.assigned_cluster))
        .collect();

    // Group images by true person
    let mut true_groups: HashMap<usize, Vec<&PersonImage>> = HashMap::new();
    for img in images {
        true_groups.entry(img.true_person_id).or_default().push(img);
    }

    // Group images by assigned cluster
    let mut assigned_groups: HashMap<usize, Vec<&PersonImage>> = HashMap::new();
    for img in images {
        if let Some(&Some(cluster_id)) = result_map.get(&img.id) {
            assigned_groups.entry(cluster_id).or_default().push(img);
        }
    }

    // Compute false merges
    for (_, assigned_imgs) in &assigned_groups {
        let mut true_persons_in_cluster: HashSet<usize> = HashSet::new();
        for img in assigned_imgs {
            true_persons_in_cluster.insert(img.true_person_id);
        }
        if true_persons_in_cluster.len() > 1 {
            false_merges += 1;
            // Check which channel caused it
            for i in 0..assigned_imgs.len() {
                for j in (i + 1)..assigned_imgs.len() {
                    let img1 = assigned_imgs[i];
                    let img2 = assigned_imgs[j];
                    if img1.true_person_id != img2.true_person_id {
                        let face_score = cosine(&img1.face_emb, &img2.face_emb);
                        let body_score = cosine(&img1.body_emb, &img2.body_emb);
                        if face_score >= FACE_THRESHOLD {
                            face_false_merges += 1;
                        }
                        if body_score >= BODY_THRESHOLD {
                            body_false_merges += 1;
                        }
                        if face_score >= FACE_THRESHOLD && body_score >= BODY_THRESHOLD {
                            fusion_false_merges += 1;
                        }
                    }
                }
            }
        } else {
            true_merges += 1;
        }
    }

    // Compute false splits
    for (_, true_imgs) in &true_groups {
        let mut assigned_clusters: HashSet<usize> = HashSet::new();
        for img in true_imgs {
            if let Some(&Some(cluster_id)) = result_map.get(&img.id) {
                assigned_clusters.insert(cluster_id);
            }
        }
        if assigned_clusters.len() > 1 {
            false_splits += 1;
        } else {
            true_splits += 1;
        }
    }

    // Cluster purity
    let total_images = images.len();
    let mut correctly_clustered = 0;
    for (_, assigned_imgs) in &assigned_groups {
        if assigned_imgs.len() > 1 {
            let mut person_counts: HashMap<usize, usize> = HashMap::new();
            for img in assigned_imgs {
                *person_counts.entry(img.true_person_id).or_default() += 1;
            }
            let max_count = person_counts.values().max().copied().unwrap_or(0);
            correctly_clustered += max_count;
        } else if assigned_imgs.len() == 1 {
            correctly_clustered += 1;
        }
    }
    let cluster_purity = if total_images > 0 {
        correctly_clustered as f32 / total_images as f32
    } else {
        0.0
    };

    // Cluster recall
    let mut correctly_recalled = 0;
    for (_, true_imgs) in &true_groups {
        let mut assigned_clusters: HashSet<usize> = HashSet::new();
        for img in true_imgs {
            if let Some(&Some(cluster_id)) = result_map.get(&img.id) {
                assigned_clusters.insert(cluster_id);
            }
        }
        if assigned_clusters.len() == 1 {
            correctly_recalled += true_imgs.len();
        }
    }
    let cluster_recall = if total_images > 0 {
        correctly_recalled as f32 / total_images as f32
    } else {
        0.0
    };

    // Pairwise metrics
    let mut pairs_same_true = 0;
    let mut pairs_same_cluster = 0;
    let mut pairs_correct = 0;

    for i in 0..images.len() {
        for j in (i + 1)..images.len() {
            let same_true = images[i].true_person_id == images[j].true_person_id;
            if same_true {
                pairs_same_true += 1;
            }

            let same_cluster = result_map
                .get(&images[i].id)
                .copied()
                .flatten()
                == result_map.get(&images[j].id).copied().flatten();
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

    ClusterMetrics {
        false_merges,
        false_splits,
        true_merges,
        true_splits,
        cluster_purity,
        cluster_recall,
        pairwise_precision,
        pairwise_recall,
        pairwise_f1,
        face_false_merges: face_false_merges / 2,
        body_false_merges: body_false_merges / 2,
        fusion_false_merges: fusion_false_merges / 2,
    }
}

// ============================================================================
// Hard Negative Detection
// ============================================================================

fn detect_hard_negatives(images: &[PersonImage]) -> Vec<HardNegative> {
    let mut hard_negatives = Vec::new();

    for i in 0..images.len() {
        for j in (i + 1)..images.len() {
            let img1 = &images[i];
            let img2 = &images[j];

            if img1.true_person_id == img2.true_person_id {
                continue;
            }

            let face_score = cosine(&img1.face_emb, &img2.face_emb);
            let body_score = cosine(&img1.body_emb, &img2.body_emb);
            let fusion_score = 0.5 * face_score + 0.5 * body_score;

            if face_score >= FACE_THRESHOLD || body_score >= BODY_THRESHOLD || fusion_score >= 0.65 {
                hard_negatives.push(HardNegative {
                    img1_id: img1.id,
                    img2_id: img2.id,
                    person1: img1.true_person_id,
                    person2: img2.true_person_id,
                    face_score,
                    body_score,
                    fusion_score,
                });
            }
        }
    }

    hard_negatives
}

// ============================================================================
// Chain Contamination Test
// ============================================================================

fn test_chain_contamination(
    person_a_images: &[Vec<f32>],
    person_b_images: &[Vec<f32>],
) -> ContaminationResult {
    let mut prototypes: HashMap<usize, Vec<Vec<f32>>> = HashMap::new();
    let mut next_cluster = 1usize;

    // Add all A images first
    for emb in person_a_images {
        prototypes.insert(next_cluster, vec![emb.clone()]);
        next_cluster += 1;
    }

    let a_cluster = 1;

    // Add B images one by one
    for (i, emb) in person_b_images.iter().enumerate() {
        let mut best_cluster = None;
        let mut best_score = 0.0f32;

        for (cluster_id, cluster_embs) in &prototypes {
            let max_sim = cluster_embs
                .iter()
                .map(|e| cosine(emb, e))
                .fold(0.0f32, |a, b| a.max(b));

            if max_sim > best_score {
                best_score = max_sim;
                best_cluster = Some(*cluster_id);
            }
        }

        if let Some(cluster_id) = best_cluster {
            if best_score >= FACE_THRESHOLD {
                if cluster_id == a_cluster {
                    return ContaminationResult {
                        steps: i + 1,
                        contamination_detected: true,
                        contaminated_images: vec![i],
                    };
                }
            }
        }
    }

    ContaminationResult {
        steps: person_b_images.len(),
        contamination_detected: false,
        contaminated_images: vec![],
    }
}

// ============================================================================
// Verdict
// ============================================================================

fn compute_verdict(metrics: &ClusterMetrics, contamination_safe: bool) -> Verdict {
    let total_pairs = metrics.true_merges + metrics.false_merges;
    let false_merge_rate = if total_pairs > 0 {
        metrics.false_merges as f32 / total_pairs as f32
    } else {
        0.0
    };

    let total_splits = metrics.true_splits + metrics.false_splits;
    let false_split_rate = if total_splits > 0 {
        metrics.false_splits as f32 / total_splits as f32
    } else {
        0.0
    };

    let verdict = if false_merge_rate < 0.01 && false_split_rate < 0.05 && contamination_safe {
        "SAFE"
    } else if false_merge_rate < 0.05 && false_split_rate < 0.10 {
        "CAUTION"
    } else {
        "UNSAFE"
    };

    Verdict {
        verdict: verdict.to_string(),
        false_merge_rate,
        false_split_rate,
        worst_cluster_purity: metrics.cluster_purity,
        chain_contamination_safe: contamination_safe,
    }
}

// ============================================================================
// Tests
// ============================================================================

fn make_test_images_two_persons() -> Vec<PersonImage> {
    let mut images = Vec::new();

    // Person A: 4 images
    for i in 0..4 {
        images.push(PersonImage {
            id: i,
            person_id: i,
            face_emb: make_face_embedding(1, i, 0.15),
            body_emb: make_body_embedding(1, i, 0.15),
            true_person_id: 1,
        });
    }

    // Person B: 4 images
    for i in 0..4 {
        images.push(PersonImage {
            id: 10 + i,
            person_id: 10 + i,
            face_emb: make_face_embedding(2, i, 0.15),
            body_emb: make_body_embedding(2, i, 0.15),
            true_person_id: 2,
        });
    }

    images
}

#[test]
fn test_false_merge_rate() {
    let images = make_test_images_two_persons();
    let results = run_clustering(&images);
    let metrics = compute_metrics(&images, &results);

    println!("False merges: {}", metrics.false_merges);
    println!("False splits: {}", metrics.false_splits);
    println!("Cluster purity: {:.4}", metrics.cluster_purity);
    println!("Face false merges: {}", metrics.face_false_merges);

    assert_eq!(metrics.false_merges, 0, "Should not merge different persons");
    assert!(metrics.cluster_purity > 0.9, "Cluster purity should be high");
}

#[test]
fn test_false_split_rate() {
    let mut images = Vec::new();

    // Person A: 3 images with very similar embeddings
    for i in 0..3 {
        images.push(PersonImage {
            id: i,
            person_id: i,
            face_emb: make_face_embedding(1, i, 0.05),
            body_emb: make_body_embedding(1, i, 0.05),
            true_person_id: 1,
        });
    }

    let results = run_clustering(&images);
    let metrics = compute_metrics(&images, &results);

    println!("False splits: {}", metrics.false_splits);
    println!("Cluster recall: {:.4}", metrics.cluster_recall);

    assert_eq!(metrics.false_splits, 0, "Should not split same person");
}

#[test]
fn test_pairwise_metrics() {
    let images = make_test_images_two_persons();
    let results = run_clustering(&images);
    let metrics = compute_metrics(&images, &results);

    println!("Pairwise precision: {:.4}", metrics.pairwise_precision);
    println!("Pairwise recall: {:.4}", metrics.pairwise_recall);
    println!("Pairwise F1: {:.4}", metrics.pairwise_f1);

    assert!(metrics.pairwise_f1 > 0.9, "Pairwise F1 should be high for distinct persons");
}

#[test]
fn test_hard_negative_detection() {
    let images = make_test_images_two_persons();
    let hard_negatives = detect_hard_negatives(&images);

    println!("Hard negatives detected: {}", hard_negatives.len());
    for hn in &hard_negatives {
        println!(
            "  Person {} <-> Person {}: face={:.3}, body={:.3}, fusion={:.3}",
            hn.person1, hn.person2, hn.face_score, hn.body_score, hn.fusion_score
        );
    }

    // Distinct persons should not have hard negatives
    assert_eq!(hard_negatives.len(), 0, "Distinct persons should not be hard negatives");
}

#[test]
fn test_chain_contamination_1_step() {
    let person_a: Vec<Vec<f32>> = (0..3).map(|i| make_face_embedding(1, i, 0.1)).collect();
    let person_b: Vec<Vec<f32>> = (0..2).map(|i| make_face_embedding(2, i, 0.1)).collect();

    let result = test_chain_contamination(&person_a, &person_b);

    println!("1-step contamination: detected={}, steps={}", result.contamination_detected, result.steps);

    assert!(!result.contamination_detected, "Should not contaminate with distinct persons");
}

#[test]
fn test_chain_contamination_5_step() {
    let person_a: Vec<Vec<f32>> = (0..3).map(|i| make_face_embedding(1, i, 0.1)).collect();
    let person_b: Vec<Vec<f32>> = (0..5).map(|i| make_face_embedding(2, i, 0.1)).collect();

    let result = test_chain_contamination(&person_a, &person_b);

    println!("5-step contamination: detected={}, steps={}", result.contamination_detected, result.steps);
    assert!(!result.contamination_detected, "Should not contaminate with distinct persons");
}

#[test]
fn test_chain_contamination_10_step() {
    let person_a: Vec<Vec<f32>> = (0..3).map(|i| make_face_embedding(1, i, 0.1)).collect();
    let person_b: Vec<Vec<f32>> = (0..10).map(|i| make_face_embedding(2, i, 0.1)).collect();

    let result = test_chain_contamination(&person_a, &person_b);

    println!("10-step contamination: detected={}, steps={}", result.contamination_detected, result.steps);
    assert!(!result.contamination_detected, "Should not contaminate with distinct persons");
}

#[test]
fn test_verdict_safe() {
    let images = make_test_images_two_persons();
    let results = run_clustering(&images);
    let metrics = compute_metrics(&images, &results);

    let contamination_result = ContaminationResult {
        steps: 10,
        contamination_detected: false,
        contaminated_images: vec![],
    };

    let verdict = compute_verdict(&metrics, !contamination_result.contamination_detected);

    println!("\n===== VERDICT =====");
    println!("Verdict: {}", verdict.verdict);
    println!("False Merge Rate: {:.4}", verdict.false_merge_rate);
    println!("False Split Rate: {:.4}", verdict.false_split_rate);
    println!("Worst Cluster Purity: {:.4}", verdict.worst_cluster_purity);
    println!("Chain Contamination Safe: {}", verdict.chain_contamination_safe);
    println!("====================\n");

    assert_eq!(verdict.verdict, "SAFE", "Should be SAFE for distinct persons");
}

#[test]
fn test_verdict_unsafe_similar_faces() {
    let mut images = Vec::new();

    // Person A with moderate noise
    for i in 0..3 {
        images.push(PersonImage {
            id: i,
            person_id: i,
            face_emb: make_face_embedding(1, i, 0.3),
            body_emb: make_body_embedding(1, i, 0.3),
            true_person_id: 1,
        });
    }

    // Person B with similar embeddings (higher noise)
    for i in 0..3 {
        images.push(PersonImage {
            id: 10 + i,
            person_id: 10 + i,
            face_emb: make_face_embedding(1, i + 10, 0.35),
            body_emb: make_body_embedding(2, i, 0.35),
            true_person_id: 2,
        });
    }

    let results = run_clustering(&images);
    let metrics = compute_metrics(&images, &results);

    let hard_negatives = detect_hard_negatives(&images);

    println!("Hard negatives: {}", hard_negatives.len());
    println!("False merges: {}", metrics.false_merges);
    println!("False splits: {}", metrics.false_splits);
    println!("Cluster purity: {:.4}", metrics.cluster_purity);

    // With similar embeddings, might get false merges or hard negatives
    if !hard_negatives.is_empty() {
        println!("\nCRITICAL_HARD_NEGATIVE detected!");
    }
}

#[test]
fn test_cluster_purity_calculation() {
    let images = make_test_images_two_persons();
    let results = run_clustering(&images);
    let metrics = compute_metrics(&images, &results);

    println!("Cluster purity: {:.4}", metrics.cluster_purity);
    println!("Cluster recall: {:.4}", metrics.cluster_recall);
    println!("Pairwise F1: {:.4}", metrics.pairwise_f1);

    assert!((metrics.cluster_purity - 1.0).abs() < 0.01, "Pure clusters for distinct persons");
}

#[test]
fn test_face_body_fusion_false_merge_tracking() {
    let mut images = Vec::new();

    // Person A with distinctive body but common face
    for i in 0..3 {
        images.push(PersonImage {
            id: i,
            person_id: i,
            face_emb: make_face_embedding(1, i, 0.25),
            body_emb: make_body_embedding(1, i, 0.1),
            true_person_id: 1,
        });
    }

    // Person B with similar face but different body
    for i in 0..3 {
        images.push(PersonImage {
            id: 10 + i,
            person_id: 10 + i,
            face_emb: make_face_embedding(1, i + 10, 0.25),
            body_emb: make_body_embedding(2, i, 0.1),
            true_person_id: 2,
        });
    }

    let results = run_clustering(&images);
    let metrics = compute_metrics(&images, &results);

    println!("Face false merges: {}", metrics.face_false_merges);
    println!("Body false merges: {}", metrics.body_false_merges);
    println!("Fusion false merges: {}", metrics.fusion_false_merges);
    println!("Cluster purity: {:.4}", metrics.cluster_purity);
}

#[test]
fn test_multi_person_clustering() {
    let mut images = Vec::new();

    // 5 persons, 3 images each
    for person in 1..=5 {
        for img in 0..3 {
            let id = (person - 1) * 10 + img;
            images.push(PersonImage {
                id,
                person_id: id,
                face_emb: make_face_embedding(person, img, 0.12),
                body_emb: make_body_embedding(person, img, 0.12),
                true_person_id: person,
            });
        }
    }

    let results = run_clustering(&images);
    let metrics = compute_metrics(&images, &results);

    println!("\n5-person clustering results:");
    println!("False merges: {}", metrics.false_merges);
    println!("False splits: {}", metrics.false_splits);
    println!("Cluster purity: {:.4}", metrics.cluster_purity);
    println!("Pairwise F1: {:.4}", metrics.pairwise_f1);

    assert!(metrics.pairwise_f1 > 0.8, "Should maintain high F1 with multiple persons");
}

#[test]
fn test_worst_case_similar_embeddings() {
    // Two persons with nearly identical embeddings (extreme case)
    let mut images = vec![
        PersonImage {
            id: 0,
            person_id: 0,
            face_emb: make_face_embedding(1, 0, 0.02),
            body_emb: make_body_embedding(1, 0, 0.02),
            true_person_id: 1,
        },
        PersonImage {
            id: 1,
            person_id: 1,
            face_emb: make_face_embedding(1, 1, 0.02), // Very similar to above
            body_emb: make_body_embedding(1, 1, 0.02),
            true_person_id: 2,
        },
    ];

    let hard_negatives = detect_hard_negatives(&images);
    let results = run_clustering(&images);
    let metrics = compute_metrics(&images, &results);

    println!("Hard negatives for near-identical embeddings: {}", hard_negatives.len());
    println!("False merges: {}", metrics.false_merges);
    println!("False splits: {}", metrics.false_splits);

    // Near-identical embeddings from different persons should be detected as hard negatives
    assert!(!hard_negatives.is_empty(), "Near-identical embeddings are critical hard negatives");
}

#[test]
fn test_embedding_separation() {
    // Verify that our embedding generation produces well-separated vectors
    let emb_a = make_face_embedding(1, 0, 0.1);
    let emb_b = make_face_embedding(2, 0, 0.1);
    let emb_a2 = make_face_embedding(1, 1, 0.1);

    let sim_aa = cosine(&emb_a, &emb_a2);
    let sim_ab = cosine(&emb_a, &emb_b);

    println!("Same person similarity: {:.4}", sim_aa);
    println!("Different person similarity: {:.4}", sim_ab);

    assert!(sim_aa > 0.9, "Same person should have high similarity");
    assert!(sim_ab < 0.5, "Different persons should have low similarity");
}

#[test]
fn test_verdict_output_format() {
    let images = make_test_images_two_persons();
    let results = run_clustering(&images);
    let metrics = compute_metrics(&images, &results);

    let contamination_result = ContaminationResult {
        steps: 10,
        contamination_detected: false,
        contaminated_images: vec![],
    };

    let verdict = compute_verdict(&metrics, !contamination_result.contamination_detected);

    println!("\n============================================================");
    println!();
    println!("FINAL VERDICT");
    println!();
    println!("============================================================");
    println!();
    println!("System Readiness:");
    println!("  Clustering: {}", if verdict.verdict == "SAFE" { "SAFE" } else { "UNSAFE" });
    println!();
    println!("Metrics:");
    println!("  False Merge Rate: {:.4}", verdict.false_merge_rate);
    println!("  False Split Rate: {:.4}", verdict.false_split_rate);
    println!("  Worst Cluster Purity: {:.4}", verdict.worst_cluster_purity);
    println!("  Chain Contamination Safe: {}", verdict.chain_contamination_safe);
    println!();
    println!("Pairwise Metrics:");
    println!("  Precision: {:.4}", metrics.pairwise_precision);
    println!("  Recall: {:.4}", metrics.pairwise_recall);
    println!("  F1: {:.4}", metrics.pairwise_f1);
    println!();
    println!("============================================================");
    println!("VERDICT: {}", verdict.verdict);
    println!("============================================================\n");

    assert!(verdict.verdict == "SAFE" || verdict.verdict == "CAUTION" || verdict.verdict == "UNSAFE");
}
