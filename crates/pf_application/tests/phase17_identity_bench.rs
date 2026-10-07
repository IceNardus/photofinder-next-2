//! Phase 17 — Face + Body Identity Production Benchmark
//!
//! Comprehensive benchmark for Face + Body person identity verification.
//!
//! Runs:
//! ```bash
//! cargo test -p pf_application --test phase17_identity_bench -- --nocapture --ignored
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use pf_ai::{
    ArcFaceEmbedder, BodyCropStrategy, BodyPipeline, FacePipeline, QualityFilter,
    ScrfdDetector, SimpleAligner,
};
use pf_ai::body::BODY_MODEL_NAME;
use pf_core::{FACE_MODEL_NAME, BBox};
use pf_application::{BodyPrototypeService, PrototypeService};
use pf_database::{builtin_migrations, Database, NewPerson};
use pf_platform::{FileSystemPhotoProvider, PhotoProvider};
use pf_vector::{HnswIndex, VectorIndex};

// ============================================================================
// Constants
// ============================================================================

const DOWNLOADS: &str = "/Users/mac/Downloads";

// ============================================================================
// Test Data
// ============================================================================

struct TestPerson {
    name: &'static str,
    images: &'static [&'static str],
}

const TEST_PERSONS: &[TestPerson] = &[
    // Person A - jaqor (only 3 images)
    TestPerson {
        name: "jaqor",
        images: &[
            "pexels-jaqor-33601811.jpg",
            "pexels-jaqor-33601831.jpg",
            "pexels-jaqor-33601835.jpg",
        ],
    },
    // Person B - cottonbro (check if faces detected)
    TestPerson {
        name: "cottonbro",
        images: &[
            "pexels-cottonbro-5900525.jpg",
            "pexels-cottonbro-7609197.jpg",
            "pexels-cottonbro-7609199.jpg",
        ],
    },
    // Person C - daria-voronkov
    TestPerson {
        name: "daria-voronkov",
        images: &[
            "pexels-daria-voronkov-381938591-14723650.jpg",
            "pexels-daria-voronkov-381938591-14723672.jpg",
        ],
    },
    // Person D - others
    TestPerson {
        name: "yi-ren",
        images: &["pexels-yi-ren-57040649-33026322.jpg"],
    },
    TestPerson {
        name: "joelle",
        images: &["pexels-joelle-s-2162497381-38263248.jpg"],
    },
    TestPerson {
        name: "peterdanthy",
        images: &["pexels-peterdanthy-33692605.jpg"],
    },
    TestPerson {
        name: "soc-nang",
        images: &["pexels-soc-nang-d-ng-2150345854-38142867.jpg"],
    },
];

// ============================================================================
// Utility Functions
// ============================================================================

fn resolve_model(name: &str) -> PathBuf {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    [
        workspace.join("models").join(name),
        PathBuf::from("/Users/mac/Library/Application Support/PhotoFinderNext/resources/models").join(name),
        PathBuf::from("/Users/mac/Library/Application Support/PhotoFinderNext/models").join(name),
    ]
    .into_iter()
    .find(|p| p.exists())
    .unwrap_or_else(|| workspace.join("models").join(name))
}

fn tmpdir(prefix: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("pf-phase17-{}-{}", prefix,
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    fs::create_dir_all(&p).unwrap();
    p
}

fn cosine(v1: &[f32], v2: &[f32]) -> f32 {
    let dot = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum::<f32>();
    let n1 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let n2 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n1 < 1e-8 || n2 < 1e-8 {
        return 0.0;
    }
    dot / (n1 * n2)
}

fn mean_prototype<T: AsRef<[f32]>>(embeddings: &[T]) -> Option<Vec<f32>> {
    if embeddings.is_empty() {
        return None;
    }
    let dim = embeddings[0].as_ref().len();
    let mut sum = vec![0.0f32; dim];
    for emb in embeddings {
        let emb = emb.as_ref();
        for (i, v) in emb.iter().enumerate() {
            sum[i] += v;
        }
    }
    let n = embeddings.len() as f32;
    let norm = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-8 {
        return Some(sum);
    }
    Some(sum.iter().map(|x| x / norm).collect())
}

fn percentile(data: &[f32], p: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (p * (sorted.len() - 1) as f32).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn roc_auc(scores: &[(bool, f32)]) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let n_pos = sorted.iter().filter(|(is_pos, _)| *is_pos).count() as f32;
    let n_neg = sorted.iter().filter(|(is_pos, _)| !*is_pos).count() as f32;

    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.0;
    }

    let mut tpr_list = Vec::new();
    let mut fpr_list = Vec::new();

    for threshold in sorted.iter().map(|(_, s)| *s) {
        let tp = sorted.iter().filter(|(is_pos, s)| *is_pos && *s >= threshold).count() as f32;
        let fp = sorted.iter().filter(|(is_pos, s)| !*is_pos && *s >= threshold).count() as f32;
        tpr_list.push(tp / n_pos);
        fpr_list.push(fp / n_neg);
    }

    // Simple AUC calculation
    let mut auc = 0.0;
    for i in 1..tpr_list.len() {
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2.0;
    }
    auc.abs()
}

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone)]
struct ImageData {
    id: i64,
    name: String,
    person: &'static str,
    face_emb: Vec<f32>,
    body_emb: Vec<f32>,
    has_face: bool,
    has_body: bool,
}

#[derive(Debug)]
struct LooResult {
    query: String,
    person: &'static str,
    face_score: f32,
    body_score: f32,
    face_margin: f32,
    body_margin: f32,
    fusion_margin: f32,
    face_correct: bool,
    body_correct: bool,
    fusion_correct: bool,
}

#[derive(Debug)]
struct PairResult {
    img1: String,
    img2: String,
    same_person: bool,
    face_score: f32,
    body_score: f32,
}

#[derive(Debug)]
struct HardNegResult {
    query: String,
    target_person: &'static str,
    rank: usize,
    is_positive: bool,
    face_score: f32,
    body_score: f32,
    fusion_margin: f32,
}

// ============================================================================
// Phase 17 Test
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn phase17_identity_benchmark() {
    println!("\n========================================");
    println!("Phase 17: Face + Body Production Benchmark");
    println!("========================================\n");

    // ====== Setup ======
    let tmp = tmpdir("phase17");
    let photos_dir = tmp.join("photos");
    fs::create_dir(&photos_dir).unwrap();

    // Load models
    println!("Loading models...");
    let scrfd = resolve_model("scrfd_500m_bnkps.onnx");
    let arcface = resolve_model("w600k_r50.onnx");
    let ytu = resolve_model("person_reid_youtu_2021nov.onnx");

    let detector = ScrfdDetector::load(&scrfd).expect("SCRFD failed");
    let aligner = Arc::new(SimpleAligner::new());
    let embedder = ArcFaceEmbedder::load(&arcface).expect("ArcFace failed");
    let qf = QualityFilter::from_config(0.30, 20, 0.45, 30.0);
    let face_pipeline = Arc::new(FacePipeline::new(detector, aligner, embedder, qf));

    let ytu_embedder = pf_ai::body::YouTuReIdEmbedder::load(&ytu).expect("YouTu failed");
    let body_pipeline = Arc::new(BodyPipeline::new(ytu_embedder, BodyCropStrategy::HeadBody));

    // Setup DB
    let db_dir = tmp.join("db");
    fs::create_dir(&db_dir).unwrap();
    let db = Arc::new(Database::open(&db_dir.join("phase17.db"), builtin_migrations()).unwrap());

    let hnsw_dir = tmp.join("hnsw");
    fs::create_dir(&hnsw_dir).unwrap();
    let face_index: Arc<dyn VectorIndex> = Arc::new(HnswIndex::new(512, hnsw_dir.clone(), "face"));
    let body_index: Arc<dyn VectorIndex> = Arc::new(HnswIndex::new(768, hnsw_dir.clone(), "body"));

    // ====== Extract Embeddings ======
    println!("\n=== Extracting Embeddings ===\n");

    let mut all_images: Vec<ImageData> = Vec::new();
    let downloads = PathBuf::from(DOWNLOADS);

    for person in TEST_PERSONS {
        for img_name in person.images {
            let src = downloads.join(img_name);
            if !src.exists() {
                println!("  SKIP {}: not found", img_name);
                continue;
            }

            // Copy to photos_dir
            let dst = photos_dir.join(img_name);
            if !dst.exists() {
                fs::copy(&src, &dst).unwrap();
            }

            // Insert image record
            let bytes = fs::read(&dst).unwrap();
            let img_data = pf_ai::ImageData::from_bytes(&bytes).unwrap();

            let image_id = db.transaction(|tx| {
                tx.images().insert(&pf_database::NewImage {
                    path: dst.to_string_lossy().to_string(),
                    hash: "placeholder".to_string(),
                    size: bytes.len() as u64,
                    modified_time: 0,
                    width: img_data.width(),
                    height: img_data.height(),
                    captured_at: None,
                })
            }).unwrap();

            // Extract face
            let face_features = face_pipeline.process(&img_data).await.unwrap();

            if face_features.is_empty() {
                println!("  {}: NO FACE detected", img_name);
                // Still record as image but no face
                all_images.push(ImageData {
                    id: image_id,
                    name: img_name.to_string(),
                    person: person.name,
                    face_emb: vec![],
                    body_emb: vec![],
                    has_face: false,
                    has_body: false,
                });
                continue;
            }

            let face_emb = face_features[0].embedding.values.clone();
            let face_bbox = &face_features[0].detection.bbox;
            let face_quality = face_features[0].detection.score;

            // Extract body
            let body_result = body_pipeline.process(&img_data, face_bbox, face_quality);
            let (body_emb, has_body) = match body_result {
                Ok(b) => (b.embedding.values, true),
                Err(_) => (vec![], false),
            };

            println!("  {}: face={} body={} ({})",
                img_name, face_emb.len(), body_emb.len(), person.name);

            all_images.push(ImageData {
                id: image_id,
                name: img_name.to_string(),
                person: person.name,
                face_emb,
                body_emb,
                has_face: true,
                has_body,
            });
        }
    }

    // ====== Build Dataset ======
    println!("\n=== Dataset Summary ===\n");

    let persons: Vec<(&&str, &(&'static [&'static str]))> = TEST_PERSONS.iter()
        .map(|p| (&p.name, &p.images))
        .collect();

    let mut valid_images: Vec<_> = all_images.iter()
        .filter(|img| img.has_face && img.has_body)
        .collect();

    println!("Total images with face+body: {}", valid_images.len());

    // Group by person
    let mut person_groups: HashMap<&str, Vec<_>> = HashMap::new();
    for img in &valid_images {
        person_groups.entry(img.person).or_default().push(img);
    }

    for (name, imgs) in &person_groups {
        println!("  {}: {} images", name, imgs.len());
    }

    if valid_images.len() < 10 {
        println!("\n  INSUFFICIENT_DATA: Need at least 10 valid face+body images");
        println!("  Have: {} images across {} persons",
            valid_images.len(), person_groups.len());
        println!("\n  Phase 17 requires:");
        println!("    - 10+ persons");
        println!("    - 5+ images per person");
        println!("    - Hard negatives");
        println!("\n  Current dataset is insufficient for full benchmark.");
        return;
    }

    // ====== Phase 17.1: Pair Verification ======
    println!("\n========================================");
    println!("Phase 17.1: Pair Verification");
    println!("========================================\n");

    let mut pairs: Vec<PairResult> = Vec::new();

    for i in 0..valid_images.len() {
        for j in (i+1)..valid_images.len() {
            let img1 = &valid_images[i];
            let img2 = &valid_images[j];
            let same_person = img1.person == img2.person;

            let face_score = cosine(&img1.face_emb, &img2.face_emb);
            let body_score = cosine(&img1.body_emb, &img2.body_emb);

            pairs.push(PairResult {
                img1: img1.name.clone(),
                img2: img2.name.clone(),
                same_person,
                face_score,
                body_score,
            });
        }
    }

    // Compute metrics
    let face_scores: Vec<(bool, f32)> = pairs.iter()
        .map(|p| (p.same_person, p.face_score))
        .collect();
    let body_scores: Vec<(bool, f32)> = pairs.iter()
        .map(|p| (p.same_person, p.body_score))
        .collect();

    let face_auc = roc_auc(&face_scores);
    let body_auc = roc_auc(&body_scores);

    println!("  Total pairs: {}", pairs.len());
    println!("  Positive pairs: {}", pairs.iter().filter(|p| p.same_person).count());
    println!("  Negative pairs: {}", pairs.iter().filter(|p| !p.same_person).count());
    println!("  Face ROC-AUC: {:.4}", face_auc);
    println!("  Body ROC-AUC: {:.4}", body_auc);

    // ====== Phase 17.2: LOO Test ======
    println!("\n========================================");
    println!("Phase 17.2: Leave-One-Out Verification");
    println!("========================================\n");

    let mut loo_results: Vec<LooResult> = Vec::new();

    for (person_name, person_images) in &person_groups {
        if person_images.len() < 2 {
            continue;
        }

        for query_img in person_images {
            // Build LOO prototype (exclude query)
            let gallery: Vec<_> = person_images.iter()
                .filter(|img| img.name != query_img.name)
                .collect();

            if gallery.is_empty() {
                continue;
            }

            let face_gallery: Vec<_> = gallery.iter().map(|img| &img.face_emb[..]).collect();
            let body_gallery: Vec<_> = gallery.iter().map(|img| &img.body_emb[..]).collect();

            let face_proto = mean_prototype(&face_gallery);
            let body_proto = mean_prototype(&body_gallery);

            let face_score = face_proto.as_ref()
                .map(|p| cosine(&query_img.face_emb, p))
                .unwrap_or(0.0);

            let body_score = body_proto.as_ref()
                .map(|p| cosine(&query_img.body_emb, p))
                .unwrap_or(0.0);

            // Compute margins vs all negatives
            let neg_images: Vec<_> = valid_images.iter()
                .filter(|img| img.person != *person_name)
                .collect();

            let face_neg_max = neg_images.iter()
                .map(|neg| cosine(&query_img.face_emb, &neg.face_emb))
                .fold(0.0f32, |a, b| a.max(b));

            let body_neg_max = neg_images.iter()
                .map(|neg| cosine(&query_img.body_emb, &neg.body_emb))
                .fold(0.0f32, |a, b| a.max(b));

            let face_margin = face_score - face_neg_max;
            let body_margin = body_score - body_neg_max;
            let fusion_margin = 0.5 * face_margin + 0.5 * body_margin;

            loo_results.push(LooResult {
                query: query_img.name.clone(),
                person: person_name,
                face_score,
                body_score,
                face_margin,
                body_margin,
                fusion_margin,
                face_correct: face_margin > 0.0,
                body_correct: body_margin > 0.0,
                fusion_correct: fusion_margin > 0.0,
            });
        }
    }

    let face_correct = loo_results.iter().filter(|r| r.face_correct).count();
    let body_correct = loo_results.iter().filter(|r| r.body_correct).count();
    let fusion_correct = loo_results.iter().filter(|r| r.fusion_correct).count();

    println!("  LOO Results: {} queries", loo_results.len());
    println!("  Face correct: {}/{} ({:.1}%)",
        face_correct, loo_results.len(),
        100.0 * face_correct as f32 / loo_results.len() as f32);
    println!("  Body correct: {}/{} ({:.1}%)",
        body_correct, loo_results.len(),
        100.0 * body_correct as f32 / loo_results.len() as f32);
    println!("  Fusion correct: {}/{} ({:.1}%)",
        fusion_correct, loo_results.len(),
        100.0 * fusion_correct as f32 / loo_results.len() as f32);

    // Margin statistics
    if !loo_results.is_empty() {
        let face_margins: Vec<f32> = loo_results.iter().map(|r| r.face_margin).collect();
        let body_margins: Vec<f32> = loo_results.iter().map(|r| r.body_margin).collect();
        let fusion_margins: Vec<f32> = loo_results.iter().map(|r| r.fusion_margin).collect();

        println!("\n  Face margin stats:");
        println!("    min: {:.4}, P10: {:.4}, median: {:.4}, P90: {:.4}, max: {:.4}",
            face_margins.iter().cloned().fold(f32::INFINITY, f32::min),
            percentile(&face_margins, 0.10),
            percentile(&face_margins, 0.50),
            percentile(&face_margins, 0.90),
            face_margins.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

        println!("  Body margin stats:");
        println!("    min: {:.4}, P10: {:.4}, median: {:.4}, P90: {:.4}, max: {:.4}",
            body_margins.iter().cloned().fold(f32::INFINITY, f32::min),
            percentile(&body_margins, 0.10),
            percentile(&body_margins, 0.50),
            percentile(&body_margins, 0.90),
            body_margins.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

        println!("  Fusion margin stats:");
        println!("    min: {:.4}, P10: {:.4}, median: {:.4}, P90: {:.4}, max: {:.4}",
            fusion_margins.iter().cloned().fold(f32::INFINITY, f32::min),
            percentile(&fusion_margins, 0.10),
            percentile(&fusion_margins, 0.50),
            percentile(&fusion_margins, 0.90),
            fusion_margins.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    }

    // ====== Phase 17.3: Hard Negative Ranking ======
    println!("\n========================================");
    println!("Phase 17.3: Hard Negative Ranking");
    println!("========================================\n");

    let mut hard_neg_results: Vec<HardNegResult> = Vec::new();

    for (person_name, person_images) in &person_groups {
        if person_images.len() < 2 {
            continue;
        }

        for query_img in person_images {
            // LOO prototype
            let gallery: Vec<_> = person_images.iter()
                .filter(|img| img.name != query_img.name)
                .collect();

            let face_proto = mean_prototype(&gallery.iter().map(|img| &img.face_emb[..]).collect::<Vec<_>>());
            let body_proto = mean_prototype(&gallery.iter().map(|img| &img.body_emb[..]).collect::<Vec<_>>());

            // Score all images
            let mut candidates: Vec<_> = valid_images.iter()
                .filter(|img| img.name != query_img.name)
                .map(|img| {
                    let fs = face_proto.as_ref()
                        .map(|p| cosine(&img.face_emb, p))
                        .unwrap_or(0.0);
                    let bs = body_proto.as_ref()
                        .map(|p| cosine(&img.body_emb, p))
                        .unwrap_or(0.0);
                    let fm = fs - gallery.iter()
                        .map(|g| cosine(&img.face_emb, &g.face_emb))
                        .fold(0.0f32, |a, b| a.max(b));
                    let bm = bs - gallery.iter()
                        .map(|g| cosine(&img.body_emb, &g.body_emb))
                        .fold(0.0f32, |a, b| a.max(b));
                    let fusion = 0.5 * fm + 0.5 * bm;
                    (img, fs, bs, fusion)
                })
                .collect();

            candidates.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

            for (rank, (img, fs, bs, fusion)) in candidates.iter().enumerate().take(10) {
                hard_neg_results.push(HardNegResult {
                    query: query_img.name.clone(),
                    target_person: img.person,
                    rank: rank + 1,
                    is_positive: img.person == *person_name,
                    face_score: *fs,
                    body_score: *bs,
                    fusion_margin: *fusion,
                });
            }
        }
    }

    // Check for wrong top-1
    let top1_wrong = hard_neg_results.iter()
        .filter(|r| r.rank == 1 && !r.is_positive)
        .count();
    let top3_wrong = hard_neg_results.iter()
        .filter(|r| r.rank <= 3 && !r.is_positive)
        .count();
    let top5_wrong = hard_neg_results.iter()
        .filter(|r| r.rank <= 5 && !r.is_positive)
        .count();

    let total_queries = person_groups.values()
        .filter(|imgs| imgs.len() >= 2)
        .map(|imgs| imgs.len())
        .sum::<usize>();

    println!("  Total LOO queries: {}", total_queries);
    println!("  Top-1 wrong rate: {}/{} ({:.1}%)", top1_wrong, total_queries,
        100.0 * top1_wrong as f32 / total_queries as f32);
    println!("  Top-3 wrong rate: {}/{} ({:.1}%)", top3_wrong, total_queries,
        100.0 * top3_wrong as f32 / total_queries as f32);
    println!("  Top-5 wrong rate: {}/{} ({:.1}%)", top5_wrong, total_queries,
        100.0 * top5_wrong as f32 / total_queries as f32);

    // ====== Phase 17.4: Error Complementarity ======
    println!("\n========================================");
    println!("Phase 17.4: Error Complementarity");
    println!("========================================\n");

    let face_body_correct = loo_results.iter()
        .filter(|r| r.face_correct && r.body_correct)
        .count();
    let face_only = loo_results.iter()
        .filter(|r| r.face_correct && !r.body_correct)
        .count();
    let body_only = loo_results.iter()
        .filter(|r| !r.face_correct && r.body_correct)
        .count();
    let both_wrong = loo_results.iter()
        .filter(|r| !r.face_correct && !r.body_correct)
        .count();

    println!("  Face correct + Body correct: {} ({:.1}%)",
        face_body_correct, 100.0 * face_body_correct as f32 / loo_results.len() as f32);
    println!("  Face correct + Body wrong: {} ({:.1}%)",
        face_only, 100.0 * face_only as f32 / loo_results.len() as f32);
    println!("  Face wrong + Body correct: {} ({:.1}%)",
        body_only, 100.0 * body_only as f32 / loo_results.len() as f32);
    println!("  Face wrong + Body wrong: {} ({:.1}%)",
        both_wrong, 100.0 * both_wrong as f32 / loo_results.len() as f32);

    if body_only > 0 {
        println!("\n  NOTE: Body corrects {} cases where Face fails!", body_only);
        println!("  This confirms Body provides complementary signal.");
    }

    // ====== Phase 17.5: Fusion Weight Sweep ======
    println!("\n========================================");
    println!("Phase 17.5: Fusion Weight Sweep");
    println!("========================================\n");

    println!("  alpha | FAR   | FRR   | EER   | margin");
    println!("  ------+-------+-------+-------+--------");

    for alpha_i in 0..=4 {
        let alpha = alpha_i as f32 / 4.0_f32;

        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_ = 0;

        for result in &loo_results {
            let fusion = alpha * result.face_margin + (1.0 - alpha) * result.body_margin;
            let threshold = 0.0;

            if fusion > threshold {
                if result.face_correct || result.body_correct {
                    tp += 1;
                } else {
                    fp += 1;
                }
            } else {
                if result.face_correct || result.body_correct {
                    fn_ += 1;
                } else {
                    tn += 1;
                }
            }
        }

        let far = fp as f32 / (fp + tn).max(1) as f32;
        let frr = fn_ as f32 / (tp + fn_).max(1) as f32;
        let eer = (far + frr) / 2.0;
        let margin_avg = loo_results.iter()
            .map(|r| alpha * r.face_margin + (1.0 - alpha) * r.body_margin)
            .sum::<f32>() / loo_results.len() as f32;

        println!("  {:.2}   | {:.3} | {:.3} | {:.3} | {:.4}",
            alpha, far, frr, eer, margin_avg);
    }

    // ====== Final Verdict ======
    println!("\n========================================");
    println!("Phase 17 Final Verdict");
    println!("========================================\n");

    let face_loo_rate = face_correct as f32 / loo_results.len().max(1) as f32;
    let body_loo_rate = body_correct as f32 / loo_results.len().max(1) as f32;
    let fusion_loo_rate = fusion_correct as f32 / loo_results.len().max(1) as f32;
    let top1_error_rate = top1_wrong as f32 / total_queries.max(1) as f32;

    let far_at_50 = {
        let alpha = 0.5;
        let threshold = 0.0;
        let fp = loo_results.iter()
            .filter(|r| {
                let fusion = alpha * r.face_margin + (1.0 - alpha) * r.body_margin;
                fusion > threshold && !r.face_correct && !r.body_correct
            })
            .count();
        fp as f32 / loo_results.len().max(1) as f32
    };

    println!("Dataset: {} persons, {} images", person_groups.len(), valid_images.len());
    println!("LOO queries: {}", loo_results.len());
    println!();
    println!("Metrics:");
    println!("  Face LOO accuracy: {:.1}%", 100.0 * face_loo_rate);
    println!("  Body LOO accuracy: {:.1}%", 100.0 * body_loo_rate);
    println!("  Fusion LOO accuracy: {:.1}%", 100.0 * fusion_loo_rate);
    println!("  Top-1 error rate: {:.1}%", 100.0 * top1_error_rate);
    println!("  FAR (alpha=0.5): {:.3}", far_at_50);
    println!();

    let verdict = if far_at_50 <= 0.01 && fusion_loo_rate >= 0.95 && top1_error_rate <= 0.01 {
        "PRODUCTION_READY"
    } else if face_loo_rate >= 0.75 && body_loo_rate >= 0.75 {
        "FACE_PASS_BODY_PASS"
    } else if face_loo_rate >= 0.75 {
        "FACE_PASS"
    } else if body_loo_rate >= 0.75 {
        "BODY_AUXILIARY"
    } else {
        "INSUFFICIENT_DATA"
    };

    println!("VERDICT: {}", verdict);

    println!("\n========================================");
    println!("Phase 17 Complete");
    println!("========================================\n");
}
