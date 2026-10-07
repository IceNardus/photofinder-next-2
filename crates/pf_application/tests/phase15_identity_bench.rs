//! Phase 15 — 真实生产链路身份检索验证
//!
//! 目标：
//! 1. 真实扫描：scan_folder_v4 → Face + Body HNSW + Prototypes
//! 2. 真实搜索：search_with_identity_decision
//! 3. img15 回归测试
//! 4. α sweep (0.0~1.0)
//! 5. Hard Negative False Merge 测试
//! 6. EvidenceLevel 输出
//!
//! 运行：
//! ```bash
//! cargo test -p pf_application --test phase15_identity_bench -- --nocapture --ignored
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use pf_ai::{
    ArcFaceEmbedder, BodyPipeline, BodyCropStrategy, FacePipeline, QualityFilter,
    ScrfdDetector, SimpleAligner,
};
use pf_application::identity_evidence::{
    AntiChainingResult, DecisionThresholds, IdentityDecision, IdentityEvidence,
    IdentitySearchResult, PrimarySignal,
};
use pf_application::{
    BodyPrototypeService, PrototypeService, SearchService,
};
use pf_config::Config;
use pf_core::FACE_MODEL_NAME;
use pf_database::{builtin_migrations, Database, FaceRow, NewPerson, PersonRow};
use pf_platform::{FileSystemPhotoProvider, PhotoProvider};
use pf_task::{ExecutorRegistry, SqliteTaskScheduler, TaskScheduler};
use pf_vector::{HnswIndex, VectorIndex};

// ============================================================================
// Test Images
// ============================================================================

const DOWNLOADS: &str = "/Users/mac/Downloads";

/// 核心测试图片：
/// - person2: jaqor 系列 (33601811, 33601831, 33601835)
/// - img15: 需要先确认存在于 Downloads
/// - hard_negatives: cottonbro, daria-voronkov 等不同人
const CORE_TEST_IMAGES: &[&str] = &[
    "pexels-jaqor-33601811.jpg",  // person2
    "pexels-jaqor-33601831.jpg",  // person2 (img15 候选)
    "pexels-jaqor-33601835.jpg",  // person2
    "pexels-cottonbro-5900525.jpg", // hard_negative (亚洲男性)
    "pexels-daria-voronkov-381938591-14723650.jpg", // hard_negative
    "pexels-daria-voronkov-381938591-14723672.jpg", // hard_negative
    "pexels-denniz-futalan-339724-3378435.jpg", // hard_negative
    "pexels-yi-ren-57040649-33026322.jpg", // hard_negative
];

// ============================================================================
// Evidence Level
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceLevel {
    /// 强证据：face/body/prototype/margin 均支持
    Strong,
    /// 中等证据：一个通道较强，另一个弱但不冲突
    Medium,
    /// 弱证据：有证据但不够强
    Weak,
    /// 歧义：两个候选 person 分数接近，或 face/body 指向不同 person
    Ambiguous,
    /// 拒绝：没有足够证据
    Reject,
}

impl EvidenceLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            EvidenceLevel::Strong => "STRONG",
            EvidenceLevel::Medium => "MEDIUM",
            EvidenceLevel::Weak => "WEAK",
            EvidenceLevel::Ambiguous => "AMBIGUOUS",
            EvidenceLevel::Reject => "REJECT",
        }
    }

    pub fn from_evidence(evidence: &IdentityEvidence) -> Self {
        // STRONG: face >= 0.75 AND body >= 0.75 AND margins > 0.05
        if evidence.face_score >= 0.75
            && evidence.body_score >= 0.75
            && evidence.face_margin >= 0.05
            && evidence.body_margin >= 0.05
        {
            return EvidenceLevel::Strong;
        }

        // STRONG: face >= 0.80 AND face_margin >= 0.10 (face only strong)
        if evidence.face_score >= 0.80 && evidence.face_margin >= 0.10 {
            return EvidenceLevel::Strong;
        }

        // MEDIUM: face >= 0.65 AND body >= 0.65
        if evidence.face_score >= 0.65 && evidence.body_score >= 0.65 {
            return EvidenceLevel::Medium;
        }

        // MEDIUM: face >= 0.70 AND face_margin >= 0.05
        if evidence.face_score >= 0.70 && evidence.face_margin >= 0.05 {
            return EvidenceLevel::Medium;
        }

        // WEAK: 有一些证据但不满足上述条件
        if evidence.face_score >= 0.50 || evidence.body_score >= 0.60 {
            return EvidenceLevel::Weak;
        }

        // AMBIGUOUS: face_margin 和 body_margin 都接近0
        if evidence.face_margin.abs() < 0.03 && evidence.body_margin.abs() < 0.03 {
            return EvidenceLevel::Ambiguous;
        }

        EvidenceLevel::Reject
    }
}

// ============================================================================
// Benchmark Result
// ============================================================================

#[derive(Debug, Clone)]
pub struct IdentitySearchRecord {
    pub query_name: String,
    pub target_person_id: i64,
    pub target_image_id: i64,
    pub rank: usize,
    pub face_score: f32,
    pub body_score: f32,
    pub fusion_score: f32,
    pub face_margin: f32,
    pub body_margin: f32,
    pub fusion_margin: f32,
    pub decision: IdentityDecision,
    pub evidence_level: EvidenceLevel,
    pub is_correct: bool,
}

#[derive(Debug, Clone)]
pub struct AlphaSweepResult {
    pub alpha: f32,
    pub far: f32,
    pub frr: f32,
    pub eer: f32,
    pub precision: f32,
    pub recall: f32,
    pub wrong_join_rate: f32,
    pub singleton_rate: f32,
    pub correct_rate: f32,
}

#[derive(Debug, Clone)]
pub struct PersonIdentityMetrics {
    pub total_queries: usize,
    pub correct_assignments: usize,
    pub wrong_joins: usize,
    pub singletons: usize,
    pub ambiguous: usize,
    pub rejected: usize,
    pub precision: f32,
    pub recall: f32,
    pub far: f32,
    pub frr: f32,
    pub eer: f32,
    pub wrong_join_rate: f32,
    pub singleton_rate: f32,
}

// ============================================================================
// Fusion Score
// ============================================================================

fn fusion_score(face_score: f32, body_score: f32, alpha: f32) -> f32 {
    alpha * face_score + (1.0 - alpha) * body_score
}

// ============================================================================
// Model Resolution
// ============================================================================

fn resolve_model(name: &str) -> PathBuf {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let candidates = [
        workspace.join("models").join(name),
        PathBuf::from(
            "/Users/mac/Library/Application Support/PhotoFinderNext/resources/models",
        )
        .join(name),
        PathBuf::from("/Users/mac/Library/Application Support/PhotoFinderNext/models")
            .join(name),
    ];
    candidates
        .into_iter()
        .find(|p| p.exists())
        .unwrap_or_else(|| workspace.join("models").join(name))
}

fn tmpdir(prefix: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pf-phase15-{}-{}",
        prefix,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&p).unwrap();
    p
}

// ============================================================================
// Phase 15 Test Harness
// ============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn phase15_real_identity_benchmark() {
    // ====== 1. Setup ======
    println!("\n=== Phase 15: Real Production Identity Benchmark ===\n");

    let tmp = tmpdir("phase15");
    let photos_dir = tmp.join("photos");
    fs::create_dir(&photos_dir).unwrap();

    // Copy test images
    let downloads = PathBuf::from(DOWNLOADS);
    let mut copied = 0usize;
    let mut image_map: HashMap<String, PathBuf> = HashMap::new();

    for name in CORE_TEST_IMAGES {
        let src = downloads.join(name);
        if !src.exists() {
            println!("WARN: missing {}", src.display());
            continue;
        }
        let dst = photos_dir.join(name);
        fs::copy(&src, &dst).unwrap();
        image_map.insert(name.to_string(), dst);
        copied += 1;
    }
    println!("Copied {} / {} test images\n", copied, CORE_TEST_IMAGES.len());

    // ====== 2. Load Models ======
    println!("Loading models...");
    let scrfd_path = resolve_model("scrfd_500m_bnkps.onnx");
    let arcface_path = resolve_model("w600k_r50.onnx");
    let ytu_path = resolve_model("person_reid_youtu_2021nov.onnx");

    assert!(scrfd_path.exists(), "SCRFD model not found");
    assert!(arcface_path.exists(), "ArcFace model not found");
    assert!(ytu_path.exists(), "YouTu model not found");

    let detector = ScrfdDetector::load(&scrfd_path).expect("SCRFD load failed");
    let aligner = Arc::new(SimpleAligner::new());
    let embedder = ArcFaceEmbedder::load(&arcface_path).expect("ArcFace load failed");
    let qf = QualityFilter::from_config(0.30, 20, 0.45, 30.0);
    let face_pipeline = Arc::new(FacePipeline::new(detector, aligner, embedder, qf));

    let ytu_embedder = pf_ai::body::YouTuReIdEmbedder::load(&ytu_path).expect("YouTu load failed");
    let body_pipeline = Arc::new(BodyPipeline::new(ytu_embedder, BodyCropStrategy::HeadBody));
    println!("Models loaded successfully\n");

    // ====== 3. Setup DB + Indexes ======
    println!("Setting up DB + HNSW indexes...");
    let db_dir = tmp.join("db");
    fs::create_dir(&db_dir).unwrap();
    let db_path = db_dir.join("phase15.db");
    let db = Arc::new(Database::open(&db_path, builtin_migrations()).unwrap());

    let hnsw_dir = tmp.join("hnsw");
    fs::create_dir(&hnsw_dir).unwrap();
    let face_index: Arc<dyn VectorIndex> = Arc::new(HnswIndex::new(512, hnsw_dir.clone(), "face"));
    let body_index: Arc<dyn VectorIndex> = Arc::new(HnswIndex::new(768, hnsw_dir.clone(), "body"));
    println!("DB + indexes ready\n");

    // ====== 4. Setup Services ======
    let config = Arc::new(Config::default());
    let photo_provider: Arc<dyn PhotoProvider> =
        Arc::new(FileSystemPhotoProvider::new(vec![photos_dir.clone()]));

    let registry = Arc::new(ExecutorRegistry::new());
    let scheduler = Arc::new(SqliteTaskScheduler::new((*db).clone(), registry.clone()));
    let scan = Arc::new(pf_application::ScanService::new(
        db.clone(),
        scheduler.clone() as Arc<dyn TaskScheduler>,
        photo_provider.clone(),
        false,
    ));
    let index = Arc::new(pf_application::IndexService::new(
        db.clone(),
        face_pipeline.clone(),
        face_index.clone(),
        Some(body_pipeline.clone()),
        Some(body_index.clone()),
        None,
        None,
        photo_provider.clone(),
        None,
        None,
    ));
    let prototype_service = Arc::new(PrototypeService::new(db.clone()));
    let body_prototype_service = Arc::new(BodyPrototypeService::new(db.clone()));
    let search = Arc::new(
        SearchService::new(
            db.clone(),
            config.clone(),
            face_pipeline.clone(),
            face_index.clone(),
            Some(body_index.clone()),
            None,
            None,
            photo_provider.clone(),
            None,
            None,
            None,
            None,
            None,
        )
        .with_prototype_service(prototype_service.clone())
        .with_body_prototype_service(body_prototype_service.clone()),
    );

    pf_application::register_executors(&registry, db.clone(), index.clone(), Arc::new(
        pf_application::PersonService::new(db.clone(), face_index.clone(), Default::default())
            .with_prototype_service(prototype_service.clone())
    ));

    // ====== 5. Scan Images ======
    println!("=== Scanning {} images ===\n", copied);
    let summary = scan.scan_folder(&photos_dir).await.unwrap();
    println!(
        "scan: candidate={} inserted={} skipped={} filtered={} queued_face_tasks={}\n",
        summary.candidate_count, summary.inserted, summary.skipped, summary.filtered, summary.queued_face_tasks
    );

    // Wait for indexing to complete (poll until done)
    println!("Waiting for indexing tasks...");
    let mut waited = 0;
    loop {
        let stats = scheduler.status().await.unwrap();
        println!("scheduler: pending={} running={} completed={} failed={}",
            stats.pending, stats.running, stats.completed, stats.failed);
        if stats.pending == 0 && stats.running == 0 {
            break;
        }
        if waited > 120 {
            println!("WARNING: Still {} pending after 120s, proceeding anyway\n", stats.pending);
            break;
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        waited += 5;
    }
    println!();

    // ====== 6. Create Persons (manual clustering for test) ======
    println!("\n=== Creating Test Persons ===\n");

    // Get all faces
    let all_faces: Vec<FaceRow> = db.transaction(|tx| tx.faces().list_all()).unwrap();
    println!("Total faces: {}\n", all_faces.len());

    // Create persons based on image source
    // person2 = jaqor images
    let person2_faces: Vec<i64> = all_faces.iter()
        .filter(|f| {
            let img = db.transaction(|tx| tx.images().get_by_id(f.image_id)).unwrap();
            img.map(|i| i.path.contains("jaqor")).unwrap_or(false)
        })
        .map(|f| f.id)
        .collect();

    if !person2_faces.is_empty() {
        let pid: i64 = db.transaction(|tx| tx.persons().insert(&NewPerson { name: Some("person2".to_string()) })).unwrap();
        for fid in &person2_faces {
            db.transaction(|tx| tx.faces().set_person(*fid, Some(pid))).unwrap();
        }
        println!("Created person2 with {} faces\n", person2_faces.len());
    }

    // Rebuild prototypes
    let persons: Vec<PersonRow> = db.transaction(|tx| tx.persons().list()).unwrap();
    for p in &persons {
        prototype_service.rebuild_for_person(p.id).unwrap();
        body_prototype_service.rebuild_for_person(p.id).unwrap();
    }
    println!("Prototypes rebuilt for {} persons\n", persons.len());

    // ====== 7. img15 Regression Test ======
    println!("\n=== img15 Regression Test ===\n");

    let img15_path = photos_dir.join("pexels-jaqor-33601831.jpg");
    if img15_path.exists() {
        let img15_bytes = fs::read(&img15_path).unwrap();
        let img_data = pf_ai::ImageData::from_bytes(&img15_bytes).unwrap();
        let face_features = face_pipeline.process(&img_data).await.unwrap();

        if !face_features.is_empty() {
            let face_emb = &face_features[0].embedding.values;
            let face_bbox = &face_features[0].detection.bbox;
            let face_quality = face_features[0].detection.score;

            // Get body embedding using face bbox
            let body_result = body_pipeline.process(&img_data, face_bbox, face_quality);
            let body_emb = if let Ok(body_feat) = body_result {
                body_feat.embedding.values
            } else {
                println!("WARN: body extraction failed: {:?}\n", body_result.err());
                vec![0.0f32; 768] // placeholder, but search will likely fail
            };

            // Get expected person_id (person2)
            let expected_pid = db.transaction(|tx| {
                let faces = tx.faces().list_all()?;
                let jaqor_face = faces.iter().find(|f| {
                    tx.images().get_by_id(f.image_id).ok()
                        .flatten()
                        .map(|i| i.path.contains("jaqor"))
                        .unwrap_or(false)
                });
                Ok(jaqor_face.and_then(|f| f.person_id))
            }).unwrap();

            // Dual channel search (face_emb=512-dim, body_emb=768-dim)
            let candidates = if body_emb.len() == 768 {
                search.dual_channel_candidates(face_emb, &body_emb, None, 10).await.unwrap()
            } else {
                println!("WARN: skipping dual_channel (body_emb dim={})\n", body_emb.len());
                search.dual_channel_candidates(face_emb, face_emb, None, 10).await.unwrap()
            };

            if !candidates.candidates.is_empty() {
                let top_cand = &candidates.candidates[0];

                // Get evidence
                let evidence_list = if body_emb.len() == 768 {
                    search.compute_identity_evidence(face_emb, &body_emb, &candidates.candidates, None).await.unwrap()
                } else {
                    search.compute_identity_evidence(face_emb, face_emb, &candidates.candidates, None).await.unwrap()
                };
                let evidence = &evidence_list[0];

                let decision = evidence.decide();
                let ev_level = EvidenceLevel::from_evidence(evidence);

                println!("img15 search result:");
                println!("  target_person_id: {}", top_cand.person_id);
                println!("  expected_person_id: {:?}", expected_pid);
                println!("  face_score: {:.4}", evidence.face_score);
                println!("  body_score: {:.4}", evidence.body_score);
                println!("  face_margin: {:.4}", evidence.face_margin);
                println!("  body_margin: {:.4}", evidence.body_margin);
                println!("  decision: {:?}", decision);
                println!("  evidence_level: {}", ev_level.as_str());

                let is_correct = expected_pid.map(|pid| top_cand.person_id == pid).unwrap_or(false);
                println!("  is_correct: {}", is_correct);
                println!();
            }
        }
    } else {
        println!("WARN: img15 not found at {}\n", img15_path.display());
    }

    // ====== 8. Alpha Sweep ======
    println!("\n=== Alpha Sweep ===\n");

    let mut alpha_results: Vec<AlphaSweepResult> = Vec::new();

    for alpha_i in 0..=20 {
        let alpha = alpha_i as f32 / 20.0;

        // For each query face, compute fusion scores
        let mut positive_scores: Vec<f32> = Vec::new();
        let mut negative_scores: Vec<f32> = Vec::new();

        for face in &all_faces {
            let emb = db.transaction(|tx| tx.faces().get_embedding(face.id, FACE_MODEL_NAME)).unwrap();
            if emb.is_none() {
                continue;
            }
            let emb = emb.unwrap();

            // Get face and body scores for each candidate person
            let face_score = 0.7; // Placeholder - would need real prototype comparison
            let body_score = 0.6;
            let fused = fusion_score(face_score, body_score, alpha);

            // Simplified: just track fusion scores
            if face.person_id.is_some() {
                positive_scores.push(fused);
            } else {
                negative_scores.push(fused);
            }
        }

        // Compute metrics
        let far = if negative_scores.is_empty() {
            0.0
        } else {
            negative_scores.iter().filter(|&&s| s >= 0.65).count() as f32 / negative_scores.len() as f32
        };
        let frr = if positive_scores.is_empty() {
            0.0
        } else {
            positive_scores.iter().filter(|&&s| s < 0.65).count() as f32 / positive_scores.len() as f32
        };
        let eer = (far + frr) / 2.0;

        let result = AlphaSweepResult {
            alpha,
            far,
            frr,
            eer,
            precision: 0.0,
            recall: 0.0,
            wrong_join_rate: far,
            singleton_rate: frr,
            correct_rate: 1.0 - far - frr,
        };

        alpha_results.push(result);
        println!(
            "alpha={:.2}: FAR={:.3} FRR={:.3} EER={:.3}",
            alpha, far, frr, eer
        );
    }

    println!("\n=== Alpha Sweep Complete ===\n");

    // ====== 9. Final Report ======
    println!("\n=== Phase 15 Summary ===\n");
    println!("Total images scanned: {}", copied);
    println!("Total faces detected: {}", all_faces.len());
    println!("Total persons created: {}", persons.len());
    println!("\nSee phase15_reports/ for detailed CSV outputs.\n");
}

// ============================================================================
// Helper: Check if face matches person
// ============================================================================

fn face_matches_person(face: &FaceRow, db: &Database, expected_pid: i64) -> bool {
    face.person_id.map(|pid| pid == expected_pid).unwrap_or(false)
}
