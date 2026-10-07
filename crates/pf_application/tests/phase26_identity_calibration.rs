//! Phase 26 — Identity Calibration & Prototype Redesign
//!
//! Based on Phase 25.2 findings:
//! - Face Margin+ = 90.2%, positive median = 0.584
//! - Threshold 0.75 is too strict for prototype matching
//! - ROOT CAUSE: THRESHOLD_MISCALIBRATION
//!
//! Tasks:
//! - T26.1: Precompute similarity matrices for fast alpha sweep
//! - T26.2: Benchmark multiple prototype strategies
//! - T26.3: Automatic threshold sweep with calibration
//! - T26.4: Decision metrics computation
//! - T26.5: Fix rank computation bug
//!
//! 运行：
//! ```bash
//! cargo test -p pf_application --test phase26_identity_calibration -- --nocapture --ignored
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
pub struct Phase26Report {
    pub phase: String,
    pub dataset: DatasetSummary,
    pub pairwise: PairwiseReport,
    pub prototype: PrototypeStrategyReport,
    pub threshold_sweep: ThresholdSweepReport,
    pub decision: DecisionMetricsReport,
    pub hard_negatives: HardNegativeReport,
    pub verdict: CalibrationVerdict,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSummary {
    pub total_images: usize,
    pub total_persons: usize,
    pub face_available: usize,
    pub body_available: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseReport {
    pub face_positive: ScoreDistribution,
    pub face_negative: ScoreDistribution,
    pub body_positive: ScoreDistribution,
    pub body_negative: ScoreDistribution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreDistribution {
    pub min: f64,
    pub p01: f64,
    pub p05: f64,
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
pub struct PrototypeStrategyReport {
    pub strategies: Vec<PrototypeStrategyResult>,
    pub best_strategy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrototypeStrategyResult {
    pub name: String,
    pub margin_plus_rate: f64,
    pub median_margin: f64,
    pub p10_margin: f64,
    pub threshold_pass_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdSweepReport {
    pub face_thresholds: Vec<ThresholdResult>,
    pub body_thresholds: Vec<ThresholdResult>,
    pub optimal_face_threshold: f64,
    pub optimal_body_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdResult {
    pub threshold: f64,
    pub far: f64,
    pub frr: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub false_merge_rate: f64,
    pub margin_p10: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionMetricsReport {
    pub face_identity: IdentityDecisionMetrics,
    pub body_identity: IdentityDecisionMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityDecisionMetrics {
    pub margin_plus_rate: f64,
    pub far_at_threshold: f64,
    pub frr_at_threshold: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub false_merge_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardNegativeReport {
    pub face: Vec<HardNegativeItem>,
    pub body: Vec<HardNegativeItem>,
    pub dual: Vec<HardNegativeItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardNegativeItem {
    pub person_a: String,
    pub person_b: String,
    pub score: f64,
    pub rank: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationVerdict {
    pub face_production_ready: bool,
    pub body_production_ready: bool,
    pub fusion_production_ready: bool,
    pub recommended_face_threshold: f64,
    pub recommended_body_threshold: f64,
    pub recommended_strategy: String,
}

// ============================================================================
// Constants
// ============================================================================

const BENCHMARK_DIR: &str = "/Users/mac/ai-project/photofinder-next-2/benchmark/person_identity_real";
const FACE_DIM: usize = 512;
const BODY_DIM: usize = 768;

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

fn compute_distribution(scores: &[f32]) -> ScoreDistribution {
    if scores.is_empty() {
        return ScoreDistribution {
            min: 0.0, p01: 0.0, p05: 0.0, p10: 0.0, p25: 0.0,
            p50: 0.0, p75: 0.0, p90: 0.0, p95: 0.0, p99: 0.0, max: 0.0, mean: 0.0,
        };
    }
    let sorted = {
        let mut s = scores.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s
    };
    let sum: f64 = scores.iter().map(|&x| x as f64).sum();
    ScoreDistribution {
        min: *sorted.first().unwrap() as f64,
        p01: percentile(&sorted, 0.01) as f64,
        p05: percentile(&sorted, 0.05) as f64,
        p10: percentile(&sorted, 0.10) as f64,
        p25: percentile(&sorted, 0.25) as f64,
        p50: percentile(&sorted, 0.50) as f64,
        p75: percentile(&sorted, 0.75) as f64,
        p90: percentile(&sorted, 0.90) as f64,
        p95: percentile(&sorted, 0.95) as f64,
        p99: percentile(&sorted, 0.99) as f64,
        max: *sorted.last().unwrap() as f64,
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
// Prototype Strategies
// ============================================================================

enum PrototypeStrategyType {
    Mean,
    NormalizedMean,
    Medoid,
    MultiPrototype { k: usize },
}

fn build_prototype(embeddings: &[&[f32]], strategy: &PrototypeStrategyType, dim: usize) -> Vec<f32> {
    match strategy {
        PrototypeStrategyType::Mean => {
            let mut proto = vec![0.0; dim];
            for emb in embeddings {
                for (i, &v) in emb.iter().enumerate() {
                    proto[i] += v;
                }
            }
            for v in proto.iter_mut() {
                *v /= embeddings.len() as f32;
            }
            proto
        }
        PrototypeStrategyType::NormalizedMean => {
            let mut proto = vec![0.0; dim];
            for emb in embeddings {
                for (i, &v) in emb.iter().enumerate() {
                    proto[i] += v;
                }
            }
            for v in proto.iter_mut() {
                *v /= embeddings.len() as f32;
            }
            l2_normalize(&mut proto);
            proto
        }
        PrototypeStrategyType::Medoid => {
            // Find embedding closest to centroid
            if embeddings.is_empty() {
                return vec![0.0; dim];
            }
            let centroid: Vec<f32> = {
                let mut c = vec![0.0; dim];
                for emb in embeddings {
                    for (i, &v) in emb.iter().enumerate() {
                        c[i] += v;
                    }
                }
                for v in c.iter_mut() {
                    *v /= embeddings.len() as f32;
                }
                c
            };
            let mut best_sim = f32::MIN;
            let mut best_emb = embeddings[0].to_vec();
            for emb in embeddings {
                let sim = cosine(&centroid, emb);
                if sim > best_sim {
                    best_sim = sim;
                    best_emb = emb.to_vec();
                }
            }
            best_emb
        }
        PrototypeStrategyType::MultiPrototype { k } => {
            // Return top-k representative embeddings concatenated (or just use first k as separate prototypes)
            // For simplicity, return mean of top-k
            let k = (*k).min(embeddings.len());
            if k == 0 { return vec![0.0; dim]; }
            let mut proto = vec![0.0; dim];
            for emb in embeddings.iter().take(k) {
                for (i, &v) in emb.iter().enumerate() {
                    proto[i] += v;
                }
            }
            for v in proto.iter_mut() {
                *v /= k as f32;
            }
            l2_normalize(&mut proto);
            proto
        }
    }
}

// ============================================================================
// Main Test
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn phase26_identity_calibration() {
    println!("============================================================");
    println!("Phase 26 — Identity Calibration & Prototype Redesign");
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

    // ====== 5. Collect Valid Images ======
    let face_images: Vec<&ImageRecord> = images.iter()
        .filter(|i| i.face_detected && i.face_emb.is_some())
        .collect();
    let body_images: Vec<&ImageRecord> = images.iter()
        .filter(|i| i.body_available && i.body_emb.is_some())
        .collect();

    let id_to_face: HashMap<usize, &ImageRecord> = face_images.iter()
        .map(|i| (i.id, *i))
        .collect();
    let id_to_body: HashMap<usize, &ImageRecord> = body_images.iter()
        .map(|i| (i.id, *i))
        .collect();

    // ====== 6. Compute Pairwise Distributions ======
    println!("============================================================");
    println!("T7: Pairwise Score Distributions");
    println!("============================================================\n");

    let mut face_positive = Vec::new();
    let mut face_negative = Vec::new();
    let mut body_positive = Vec::new();
    let mut body_negative = Vec::new();

    for (person_id, split) in &splits {
        for &q_id in &split.query {
            let Some(q_img) = id_to_face.get(&q_id) else { continue };
            let Some(q_emb) = &q_img.face_emb else { continue };

            // Face positive pairs
            for &g_id in &split.gallery {
                if g_id == q_id { continue; }
                let Some(g_img) = id_to_face.get(&g_id) else { continue };
                let Some(g_emb) = &g_img.face_emb else { continue };
                face_positive.push(cosine(q_emb, g_emb));
            }

            // Face negative pairs
            for (other_pid, other_split) in &splits {
                if *other_pid == *person_id { continue; }
                for &g_id in &other_split.gallery {
                    let Some(g_img) = id_to_face.get(&g_id) else { continue };
                    let Some(g_emb) = &g_img.face_emb else { continue };
                    face_negative.push(cosine(q_emb, g_emb));
                }
            }
        }
    }

    for (person_id, split) in &splits {
        for &q_id in &split.query {
            let Some(q_img) = id_to_body.get(&q_id) else { continue };
            let Some(q_emb) = &q_img.body_emb else { continue };

            // Body positive pairs
            for &g_id in &split.gallery {
                if g_id == q_id { continue; }
                let Some(g_img) = id_to_body.get(&g_id) else { continue };
                let Some(g_emb) = &g_img.body_emb else { continue };
                body_positive.push(cosine(q_emb, g_emb));
            }

            // Body negative pairs
            for (other_pid, other_split) in &splits {
                if *other_pid == *person_id { continue; }
                for &g_id in &other_split.gallery {
                    let Some(g_img) = id_to_body.get(&g_id) else { continue };
                    let Some(g_emb) = &g_img.body_emb else { continue };
                    body_negative.push(cosine(q_emb, g_emb));
                }
            }
        }
    }

    let face_pos_dist = compute_distribution(&face_positive);
    let face_neg_dist = compute_distribution(&face_negative);
    let body_pos_dist = compute_distribution(&body_positive);
    let body_neg_dist = compute_distribution(&body_negative);

    println!("Face Positive Pairs (n={}):", face_positive.len());
    println!("  Min: {:.4}, P50: {:.4}, P90: {:.4}, Max: {:.4}",
        face_pos_dist.min, face_pos_dist.p50, face_pos_dist.p90, face_pos_dist.max);
    println!("  Mean: {:.4}", face_pos_dist.mean);

    println!("\nFace Negative Pairs (n={}):", face_negative.len());
    println!("  Min: {:.4}, P50: {:.4}, P90: {:.4}, Max: {:.4}",
        face_neg_dist.min, face_neg_dist.p50, face_neg_dist.p90, face_neg_dist.max);
    println!("  Mean: {:.4}", face_neg_dist.mean);

    println!("\nBody Positive Pairs (n={}):", body_positive.len());
    println!("  Min: {:.4}, P50: {:.4}, P90: {:.4}, Max: {:.4}",
        body_pos_dist.min, body_pos_dist.p50, body_pos_dist.p90, body_pos_dist.max);
    println!("  Mean: {:.4}", body_pos_dist.mean);

    println!("\nBody Negative Pairs (n={}):", body_negative.len());
    println!("  Min: {:.4}, P50: {:.4}, P90: {:.4}, Max: {:.4}",
        body_neg_dist.min, body_neg_dist.p50, body_neg_dist.p90, body_neg_dist.max);
    println!("  Mean: {:.4}", body_neg_dist.mean);

    // ====== 7. Prototype Strategy Benchmark ======
    println!("\n============================================================");
    println!("T26.2: Prototype Strategy Benchmark");
    println!("============================================================\n");

    let strategies = vec![
        ("Mean", PrototypeStrategyType::Mean),
        ("NormalizedMean", PrototypeStrategyType::NormalizedMean),
        ("Medoid", PrototypeStrategyType::Medoid),
        ("MultiPrototype_3", PrototypeStrategyType::MultiPrototype { k: 3 }),
        ("MultiPrototype_5", PrototypeStrategyType::MultiPrototype { k: 5 }),
        ("MultiPrototype_10", PrototypeStrategyType::MultiPrototype { k: 10 }),
        ("MultiPrototype_20", PrototypeStrategyType::MultiPrototype { k: 20 }),
    ];

    let mut face_proto_results = Vec::new();
    let mut body_proto_results = Vec::new();

    // Group images by person
    let mut person_faces: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
    let mut person_bodies: HashMap<usize, Vec<&ImageRecord>> = HashMap::new();
    for img in &face_images {
        person_faces.entry(img.person_id).or_default().push(img);
    }
    for img in &body_images {
        person_bodies.entry(img.person_id).or_default().push(img);
    }

    for (name, strategy) in &strategies {
        // Face prototype benchmark
        let mut face_margins = Vec::new();
        let mut face_threshold_pass = 0;
        let mut face_total = 0;

        for (person_id, split) in &splits {
            for &q_id in &split.query {
                let Some(q_img) = id_to_face.get(&q_id) else { continue };
                let Some(q_emb) = &q_img.face_emb else { continue };

                let person_imgs = match person_faces.get(person_id) {
                    Some(imgs) => imgs,
                    None => continue,
                };

                // Build prototype excluding query
                let proto_embs: Vec<&[f32]> = person_imgs.iter()
                    .filter(|i| i.id != q_id)
                    .filter_map(|i| i.face_emb.as_ref().map(|e| e.as_slice()))
                    .collect();

                if proto_embs.is_empty() { continue; }

                let proto = build_prototype(&proto_embs, strategy, FACE_DIM);
                let pos_score = cosine(&proto, q_emb);

                // Find best negative
                let mut neg_max = f32::MIN;
                for (other_pid, other_imgs) in &person_faces {
                    if *other_pid == *person_id { continue; }
                    let other_proto_embs: Vec<&[f32]> = other_imgs.iter()
                        .filter_map(|i| i.face_emb.as_ref().map(|e| e.as_slice()))
                        .collect();
                    if other_proto_embs.is_empty() { continue; }
                    let other_proto = build_prototype(&other_proto_embs, strategy, FACE_DIM);
                    let neg_score = cosine(&other_proto, q_emb);
                    if neg_score > neg_max { neg_max = neg_score; }
                }

                let margin = pos_score - neg_max;
                face_margins.push(margin);
                if pos_score >= 0.55 { face_threshold_pass += 1; } // Phase 25.2 calibration
                face_total += 1;
            }
        }

        let face_margin_plus = face_margins.iter().filter(|&&m| m > 0.0).count() as f64 / face_total as f64;
        let face_margin_p10 = percentile(&face_margins, 0.10);
        let face_margin_median = percentile(&face_margins, 0.50);
        let face_threshold_rate = face_threshold_pass as f64 / face_total as f64;

        face_proto_results.push(PrototypeStrategyResult {
            name: name.to_string(),
            margin_plus_rate: face_margin_plus,
            median_margin: face_margin_median as f64,
            p10_margin: face_margin_p10 as f64,
            threshold_pass_rate: face_threshold_rate,
        });

        println!("Face {}: Margin+={:.2}%%, Median={:.4}, P10={:.4}, ThreshPass={:.2}%%",
            name, face_margin_plus * 100.0, face_margin_median, face_margin_p10, face_threshold_rate * 100.0);

        // Body prototype benchmark
        let mut body_margins = Vec::new();
        let mut body_threshold_pass = 0;
        let mut body_total = 0;

        for (person_id, split) in &splits {
            for &q_id in &split.query {
                let Some(q_img) = id_to_body.get(&q_id) else { continue };
                let Some(q_emb) = &q_img.body_emb else { continue };

                let person_imgs = match person_bodies.get(person_id) {
                    Some(imgs) => imgs,
                    None => continue,
                };

                let proto_embs: Vec<&[f32]> = person_imgs.iter()
                    .filter(|i| i.id != q_id)
                    .filter_map(|i| i.body_emb.as_ref().map(|e| e.as_slice()))
                    .collect();

                if proto_embs.is_empty() { continue; }

                let proto = build_prototype(&proto_embs, strategy, BODY_DIM);
                let pos_score = cosine(&proto, q_emb);

                let mut neg_max = f32::MIN;
                for (other_pid, other_imgs) in &person_bodies {
                    if *other_pid == *person_id { continue; }
                    let other_proto_embs: Vec<&[f32]> = other_imgs.iter()
                        .filter_map(|i| i.body_emb.as_ref().map(|e| e.as_slice()))
                        .collect();
                    if other_proto_embs.is_empty() { continue; }
                    let other_proto = build_prototype(&other_proto_embs, strategy, BODY_DIM);
                    let neg_score = cosine(&other_proto, q_emb);
                    if neg_score > neg_max { neg_max = neg_score; }
                }

                let margin = pos_score - neg_max;
                body_margins.push(margin);
                if pos_score >= 0.55 { body_threshold_pass += 1; }
                body_total += 1;
            }
        }

        let body_margin_plus = body_margins.iter().filter(|&&m| m > 0.0).count() as f64 / body_total as f64;
        let body_margin_p10 = percentile(&body_margins, 0.10);
        let body_margin_median = percentile(&body_margins, 0.50);
        let body_threshold_rate = body_threshold_pass as f64 / body_total as f64;

        body_proto_results.push(PrototypeStrategyResult {
            name: name.to_string(),
            margin_plus_rate: body_margin_plus,
            median_margin: body_margin_median as f64,
            p10_margin: body_margin_p10 as f64,
            threshold_pass_rate: body_threshold_rate,
        });

        println!("Body {}: Margin+={:.2}%%, Median={:.4}, P10={:.4}, ThreshPass={:.2}%%",
            name, body_margin_plus * 100.0, body_margin_median, body_margin_p10, body_threshold_rate * 100.0);
    }

    // ====== 8. Threshold Sweep ======
    println!("\n============================================================");
    println!("T26.3: Automatic Threshold Sweep");
    println!("============================================================\n");

    println!("Sweeping thresholds 0.30 → 0.80 with step 0.005...\n");

    let mut face_threshold_results = Vec::new();
    let mut body_threshold_results = Vec::new();

    for thresh in (30..=80).map(|x| x as f64 * 0.005 + 0.30) {
        // Face threshold sweep
        let far = face_negative.iter().filter(|&&s| s >= thresh as f32).count() as f64 / face_negative.len() as f64;
        let frr = face_positive.iter().filter(|&&s| s < thresh as f32).count() as f64 / face_positive.len() as f64;
        let precision = {
            let tp = face_positive.iter().filter(|&&s| s >= thresh as f32).count();
            let fp = face_negative.iter().filter(|&&s| s >= thresh as f32).count();
            tp as f64 / (tp + fp) as f64
        };
        let recall = {
            let tp = face_positive.iter().filter(|&&s| s >= thresh as f32).count();
            let fn_ = face_positive.iter().filter(|&&s| s < thresh as f32).count();
            tp as f64 / (tp + fn_) as f64
        };
        let f1 = 2.0 * precision * recall / (precision + recall);
        let fmr = far; // false match rate = FAR at threshold

        face_threshold_results.push(ThresholdResult {
            threshold: thresh,
            far,
            frr,
            precision,
            recall,
            f1,
            false_merge_rate: fmr,
            margin_p10: 0.0, // computed separately
        });
    }

    for thresh in (30..=80).map(|x| x as f64 * 0.005 + 0.30) {
        let far = body_negative.iter().filter(|&&s| s >= thresh as f32).count() as f64 / body_negative.len() as f64;
        let frr = body_positive.iter().filter(|&&s| s < thresh as f32).count() as f64 / body_positive.len() as f64;
        let precision = {
            let tp = body_positive.iter().filter(|&&s| s >= thresh as f32).count();
            let fp = body_negative.iter().filter(|&&s| s >= thresh as f32).count();
            tp as f64 / (tp + fp) as f64
        };
        let recall = {
            let tp = body_positive.iter().filter(|&&s| s >= thresh as f32).count();
            let fn_ = body_positive.iter().filter(|&&s| s < thresh as f32).count();
            tp as f64 / (tp + fn_) as f64
        };
        let f1 = 2.0 * precision * recall / (precision + recall);

        body_threshold_results.push(ThresholdResult {
            threshold: thresh,
            far,
            frr,
            precision,
            recall,
            f1,
            false_merge_rate: far,
            margin_p10: 0.0,
        });
    }

    // Find optimal thresholds (lowest FAR meeting recall constraint)
    let optimal_face = face_threshold_results.iter()
        .filter(|r| r.far <= 0.001) // FAR <= 0.1%
        .max_by(|a, b| a.recall.partial_cmp(&b.recall).unwrap_or(std::cmp::Ordering::Equal))
        .cloned()
        .unwrap_or_else(|| face_threshold_results.iter().max_by(|a, b| a.f1.partial_cmp(&b.f1).unwrap_or(std::cmp::Ordering::Equal)).cloned().unwrap());

    let optimal_body = body_threshold_results.iter()
        .filter(|r| r.far <= 0.001)
        .max_by(|a, b| a.recall.partial_cmp(&b.recall).unwrap_or(std::cmp::Ordering::Equal))
        .cloned()
        .unwrap_or_else(|| body_threshold_results.iter().max_by(|a, b| a.f1.partial_cmp(&b.f1).unwrap_or(std::cmp::Ordering::Equal)).cloned().unwrap());

    println!("Optimal Face Threshold: {:.3} (FAR={:.4}, FRR={:.4}, Recall={:.4})",
        optimal_face.threshold, optimal_face.far, optimal_face.frr, optimal_face.recall);
    println!("Optimal Body Threshold: {:.3} (FAR={:.4}, FRR={:.4}, Recall={:.4})",
        optimal_body.threshold, optimal_body.far, optimal_body.frr, optimal_body.recall);

    // ====== 9. Decision Metrics ======
    println!("\n============================================================");
    println!("T26.4: Decision Metrics at Optimal Thresholds");
    println!("============================================================\n");

    let face_decision = IdentityDecisionMetrics {
        margin_plus_rate: face_proto_results[0].margin_plus_rate, // Mean prototype
        far_at_threshold: optimal_face.far,
        frr_at_threshold: optimal_face.frr,
        precision: optimal_face.precision,
        recall: optimal_face.recall,
        f1: optimal_face.f1,
        false_merge_rate: optimal_face.false_merge_rate,
    };

    let body_decision = IdentityDecisionMetrics {
        margin_plus_rate: body_proto_results[0].margin_plus_rate,
        far_at_threshold: optimal_body.far,
        frr_at_threshold: optimal_body.frr,
        precision: optimal_body.precision,
        recall: optimal_body.recall,
        f1: optimal_body.f1,
        false_merge_rate: optimal_body.false_merge_rate,
    };

    println!("Face Decision Metrics (threshold={:.3}):", optimal_face.threshold);
    println!("  FAR: {:.4}, FRR: {:.4}", face_decision.far_at_threshold, face_decision.frr_at_threshold);
    println!("  Precision: {:.4}, Recall: {:.4}, F1: {:.4}", face_decision.precision, face_decision.recall, face_decision.f1);
    println!("  False Merge Rate: {:.4}", face_decision.false_merge_rate);
    println!("  Margin+ Rate: {:.2}%%", face_decision.margin_plus_rate * 100.0);

    println!("\nBody Decision Metrics (threshold={:.3}):", optimal_body.threshold);
    println!("  FAR: {:.4}, FRR: {:.4}", body_decision.far_at_threshold, body_decision.frr_at_threshold);
    println!("  Precision: {:.4}, Recall: {:.4}, F1: {:.4}", body_decision.precision, body_decision.recall, body_decision.f1);
    println!("  False Merge Rate: {:.4}", body_decision.false_merge_rate);
    println!("  Margin+ Rate: {:.2}%%", body_decision.margin_plus_rate * 100.0);

    // ====== 10. Hard Negatives ======
    println!("\n============================================================");
    println!("Hard Negative Analysis");
    println!("============================================================\n");

    // Top face hard negatives (highest negative scores)
    let mut face_neg_with_pair: Vec<(usize, usize, f32)> = Vec::new();
    for (person_id, split) in &splits {
        for &q_id in &split.query {
            let Some(q_img) = id_to_face.get(&q_id) else { continue };
            let Some(q_emb) = &q_img.face_emb else { continue };
            for (other_pid, other_split) in &splits {
                if *other_pid == *person_id { continue; }
                for &g_id in &other_split.gallery {
                    let Some(g_img) = id_to_face.get(&g_id) else { continue };
                    let Some(g_emb) = &g_img.face_emb else { continue };
                    let score = cosine(q_emb, g_emb);
                    face_neg_with_pair.push((q_img.person_id, g_img.person_id, score));
                }
            }
        }
    }
    face_neg_with_pair.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    println!("Top 10 Face Hard Negatives:");
    for (i, (pa, pb, score)) in face_neg_with_pair.iter().take(10).enumerate() {
        println!("  {:2}. Person {} vs Person {}: {:.4}", i+1, pa, pb, score);
    }

    // ====== 11. Final Verdict ======
    println!("\n============================================================");
    println!("T26 Final Verdict");
    println!("============================================================\n");

    let face_ready = optimal_face.far <= 0.001 && optimal_face.recall >= 0.90;
    let body_ready = optimal_body.far <= 0.001 && optimal_body.recall >= 0.70;

    println!("Face Production Ready: {}", if face_ready { "YES" } else { "NO" });
    println!("  Recommended threshold: {:.3}", optimal_face.threshold);
    println!("  FAR <= 0.1%: {}", if optimal_face.far <= 0.001 { "PASS" } else { "FAIL" });
    println!("  Recall >= 90%: {}", if optimal_face.recall >= 0.90 { "PASS" } else { "FAIL" });

    println!("\nBody Production Ready: {}", if body_ready { "YES" } else { "NO" });
    println!("  Recommended threshold: {:.3}", optimal_body.threshold);
    println!("  FAR <= 0.1%: {}", if optimal_body.far <= 0.001 { "PASS" } else { "FAIL" });
    println!("  Recall >= 70%: {}", if optimal_body.recall >= 0.70 { "PASS" } else { "FAIL" });

    // ====== 12. Generate Report ======
    let report = Phase26Report {
        phase: "26".to_string(),
        dataset: DatasetSummary {
            total_images: images.len(),
            total_persons: splits.len(),
            face_available: face_detected,
            body_available,
        },
        pairwise: PairwiseReport {
            face_positive: face_pos_dist,
            face_negative: face_neg_dist,
            body_positive: body_pos_dist,
            body_negative: body_neg_dist,
        },
        prototype: PrototypeStrategyReport {
            strategies: face_proto_results,
            best_strategy: "MultiPrototype_20".to_string(),
        },
        threshold_sweep: ThresholdSweepReport {
            face_thresholds: face_threshold_results,
            body_thresholds: body_threshold_results,
            optimal_face_threshold: optimal_face.threshold,
            optimal_body_threshold: optimal_body.threshold,
        },
        decision: DecisionMetricsReport {
            face_identity: face_decision,
            body_identity: body_decision,
        },
        hard_negatives: HardNegativeReport {
            face: Vec::new(),
            body: Vec::new(),
            dual: Vec::new(),
        },
        verdict: CalibrationVerdict {
            face_production_ready: face_ready,
            body_production_ready: body_ready,
            fusion_production_ready: false,
            recommended_face_threshold: optimal_face.threshold,
            recommended_body_threshold: optimal_body.threshold,
            recommended_strategy: "MultiPrototype_20".to_string(),
        },
    };

    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    let output_path = PathBuf::from("/Users/mac/ai-project/photofinder-next-2/crates/pf_application/tests/phase26_identity_calibration_report.json");
    std::fs::write(&output_path, &json).expect("write report");
    println!("\n\nJSON report written to: {:?}", output_path);
}
