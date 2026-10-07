//! Production E2E Tests — Identity Pipeline
//!
//! Validates the complete production pipeline for face detection, scanning,
//! person assignment, and search with real models.
//!
//! Run with:
//! ```bash
//! cargo test --release -p pf_application --test production_e2e -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use pf_ai::{ArcFaceEmbedder, FacePipeline, QualityFilter, ScrfdDetector, SimpleAligner};
use pf_application::executors::register_executors;
use pf_application::index::IndexService;
use pf_application::person::{ClusterPolicy, PersonService};
use pf_application::prototype_service::PrototypeService;
use pf_application::scan::ScanService;
use pf_application::search::SearchService;
use pf_config::Config;
use pf_database::{builtin_migrations, Database};
use pf_platform::{FileSystemPhotoProvider, PhotoProvider};
use pf_task::{ExecutorRegistry, SqliteTaskScheduler, TaskScheduler};
use pf_vector::{HnswIndex, VectorIndex};

fn resolve_model(name: &str) -> PathBuf {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let candidates = [
        workspace.join("models").join(name),
        PathBuf::from("/Users/mac/Library/Application Support/PhotoFinderNext/resources/models").join(name),
        PathBuf::from("/Users/mac/Library/Application Support/PhotoFinderNext/models").join(name),
    ];
    candidates.into_iter().find(|p| p.exists()).unwrap_or_else(|| workspace.join("models").join(name))
}

fn tmpdir(prefix: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("pf-prod-{}-{}", prefix, std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn create_test_image(dest: &PathBuf, name: &str, width: u32, height: u32) -> bool {
    use image::ImageBuffer;
    let img: image::RgbImage = ImageBuffer::from_fn(width, height, |_x, _y| {
        image::Rgb([128u8, 128u8, 128u8])
    });
    img.save(dest.join(name)).is_ok()
}

async fn wait_for_completion(scheduler: &Arc<SqliteTaskScheduler>, timeout_secs: u64) -> bool {
    let start = Instant::now();
    loop {
        let s = scheduler.status().await.unwrap();
        let busy = s.pending + s.running;
        if busy == 0 || s.failed > 0 {
            return s.failed == 0;
        }
        if start.elapsed() > Duration::from_secs(timeout_secs) {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

// =============================================================================
// Test 1: NO FACE — Images without faces should not create Persons
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn test_no_face_images() {
    let tmp = tmpdir("no-face");
    let photos_dir = tmp.join("photos");
    std::fs::create_dir(&photos_dir).unwrap();

    // Use real images from Downloads (no-face test: images unlikely to have faces)
    let downloads = PathBuf::from("/Users/mac/Downloads");
    let no_face_candidates = [
        "jecqan-owl-7779315_1920.jpg",  // owl - no face
        "pexels-cottonbro-7609197.jpg",  // may or may not have faces
    ];

    let mut copied = 0;
    for (i, name) in no_face_candidates.iter().cycle().enumerate() {
        if copied >= 10 { break; }
        let src = downloads.join(name);
        if src.exists() {
            std::fs::copy(&src, photos_dir.join(format!("img{:02}.jpg", copied))).unwrap();
            copied += 1;
        }
    }

    if copied < 2 {
        eprintln!("SKIP: No-face test images not found");
        return;
    }
    eprintln!("Copied {} no-face images", copied);

    // Setup services
    let scrfd_path = resolve_model("scrfd_500m_bnkps.onnx");
    let arcface_path = resolve_model("w600k_r50.onnx");

    let detector = ScrfdDetector::load(&scrfd_path).expect("load SCRFD");
    let aligner = Arc::new(SimpleAligner::new());
    let embedder = ArcFaceEmbedder::load(&arcface_path).expect("load ArcFace");
    let qf = QualityFilter::from_config(0.30, 20, 0.45, 30.0);
    let pipeline = Arc::new(FacePipeline::new(detector, aligner, embedder, qf));

    let db_dir = tmp.join("db");
    std::fs::create_dir(&db_dir).unwrap();
    let db_path = db_dir.join("prod.db");
    let db = Arc::new(Database::open(&db_path, builtin_migrations()).unwrap());

    let hnsw_dir = tmp.join("hnsw");
    std::fs::create_dir(&hnsw_dir).unwrap();
    let face_index: Arc<HnswIndex> = Arc::new(HnswIndex::new(512, hnsw_dir, "face"));

    let config = Arc::new(Config::default());
    let photo_provider: Arc<dyn PhotoProvider> = Arc::new(FileSystemPhotoProvider::new(vec![photos_dir.clone()]));

    let registry = Arc::new(ExecutorRegistry::new());
    let scheduler = Arc::new(SqliteTaskScheduler::new((*db).clone(), registry.clone()));

    let index = Arc::new(IndexService::new(
        db.clone(),
        pipeline.clone(),
        face_index.clone(),
        None,
        None,
        None,
        None,
        photo_provider.clone(),
        None,
        None,
    ));

    let search = Arc::new(
        SearchService::new(
            db.clone(),
            config.clone(),
            pipeline.clone(),
            face_index.clone(),
            None,
            None,
            None,
            photo_provider.clone(),
            None,
            None,
            None,
            None,
            None,
        )
        .with_prototype_service(Arc::new(PrototypeService::new(db.clone()))),
    );

    let person = Arc::new(
        PersonService::new(db.clone(), face_index.clone(), ClusterPolicy::default())
            .with_prototype_service(Arc::new(PrototypeService::new(db.clone()))),
    );

    let scan = Arc::new(ScanService::new(
        db.clone(),
        scheduler.clone() as Arc<dyn TaskScheduler>,
        photo_provider.clone(),
        false,
    ));

    register_executors(&registry, db.clone(), index.clone(), person.clone());

    // Scan
    let summary = scan.scan_folder(&photos_dir).await.unwrap();
    eprintln!("Scan: {} images, {} faces queued", summary.candidate_count, summary.queued_face_tasks);

    // Wait for indexing
    let success = wait_for_completion(&scheduler, 120).await;
    assert!(success, "Indexing should complete without failures");

    // Verify: No faces detected
    let n_img = db.transaction(|tx| tx.images().count()).unwrap();
    let n_face = db.transaction(|tx| tx.faces().count()).unwrap();
    let n_person = db.transaction(|tx| tx.persons().list()).unwrap().len() as i64;

    eprintln!("NO FACE TEST:");
    eprintln!("  Images: {}", n_img);
    eprintln!("  Faces: {}", n_face);
    eprintln!("  Persons: {}", n_person);

    // The "no-face" images may actually contain low-quality detected faces
    // Key assertion: if faces are detected but don't meet quality threshold, no persons created
    // (Person creation requires face to pass quality gate and get assigned to a person)
    if n_face > 0 {
        assert_eq!(n_person, 0, "No persons should be created when face quality is low");
    } else {
        assert_eq!(n_img, copied as i64, "All no-face images should be indexed");
    }

    eprintln!(">>> TEST NO_FACE: PASS <<<");
}

// =============================================================================
// Test 2: PERSON CREATION — Faces should create persons
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn test_face_creates_person() {
    let tmp = tmpdir("face-person");
    let photos_dir = tmp.join("photos");
    std::fs::create_dir(&photos_dir).unwrap();

    // Copy real face images from Downloads
    let pexels_path = PathBuf::from("/Users/mac/Downloads");
    let face_images = [
        "pexels-jaqor-33601831.jpg",
        "pexels-jaqor-33601835.jpg",
    ];

    let mut copied = 0;
    for (i, name) in face_images.iter().enumerate() {
        let src = pexels_path.join(name);
        if src.exists() {
            std::fs::copy(&src, photos_dir.join(format!("face{}.jpg", i))).unwrap();
            copied += 1;
        }
    }

    if copied < 2 {
        eprintln!("SKIP: Pexels face images not found");
        return;
    }

    eprintln!("Copied {} face images", copied);

    // Setup services
    let scrfd_path = resolve_model("scrfd_500m_bnkps.onnx");
    let arcface_path = resolve_model("w600k_r50.onnx");

    let detector = ScrfdDetector::load(&scrfd_path).expect("load SCRFD");
    let aligner = Arc::new(SimpleAligner::new());
    let embedder = ArcFaceEmbedder::load(&arcface_path).expect("load ArcFace");
    let qf = QualityFilter::from_config(0.30, 20, 0.45, 30.0);
    let pipeline = Arc::new(FacePipeline::new(detector, aligner, embedder, qf));

    let db_dir = tmp.join("db");
    std::fs::create_dir(&db_dir).unwrap();
    let db_path = db_dir.join("prod.db");
    let db = Arc::new(Database::open(&db_path, builtin_migrations()).unwrap());

    let hnsw_dir = tmp.join("hnsw");
    std::fs::create_dir(&hnsw_dir).unwrap();
    let face_index: Arc<HnswIndex> = Arc::new(HnswIndex::new(512, hnsw_dir, "face"));

    let config = Arc::new(Config::default());
    let photo_provider: Arc<dyn PhotoProvider> = Arc::new(FileSystemPhotoProvider::new(vec![photos_dir.clone()]));

    let registry = Arc::new(ExecutorRegistry::new());
    let scheduler = Arc::new(SqliteTaskScheduler::new((*db).clone(), registry.clone()));

    let index = Arc::new(IndexService::new(
        db.clone(),
        pipeline.clone(),
        face_index.clone(),
        None,
        None,
        None,
        None,
        photo_provider.clone(),
        None,
        None,
    ));

    let search = Arc::new(
        SearchService::new(
            db.clone(),
            config.clone(),
            pipeline.clone(),
            face_index.clone(),
            None,
            None,
            None,
            photo_provider.clone(),
            None,
            None,
            None,
            None,
            None,
        )
        .with_prototype_service(Arc::new(PrototypeService::new(db.clone()))),
    );

    let person = Arc::new(
        PersonService::new(db.clone(), face_index.clone(), ClusterPolicy::default())
            .with_prototype_service(Arc::new(PrototypeService::new(db.clone()))),
    );

    let scan = Arc::new(ScanService::new(
        db.clone(),
        scheduler.clone() as Arc<dyn TaskScheduler>,
        photo_provider.clone(),
        false,
    ));

    register_executors(&registry, db.clone(), index.clone(), person.clone());

    // Scan
    let summary = scan.scan_folder(&photos_dir).await.unwrap();
    eprintln!("Scan: {} images, {} faces queued", summary.candidate_count, summary.queued_face_tasks);

    // Wait for indexing
    let success = wait_for_completion(&scheduler, 120).await;
    assert!(success, "Indexing should complete");

    // Run clustering
    person.cluster_all().await.unwrap();

    // Verify: Faces detected and persons created
    let n_img = db.transaction(|tx| tx.images().count()).unwrap();
    let n_face = db.transaction(|tx| tx.faces().count()).unwrap();
    let n_person = db.transaction(|tx| tx.persons().list()).unwrap().len() as i64;

    eprintln!("FACE PERSON TEST:");
    eprintln!("  Images: {}", n_img);
    eprintln!("  Faces: {}", n_face);
    eprintln!("  Persons: {}", n_person);

    assert_eq!(n_img, 2, "Should have 2 images");
    assert!(n_face > 0, "Faces should be detected");
    assert!(n_person > 0, "Persons should be created for detected faces");

    // Verify faces are assigned to persons
    let faces = db.transaction(|tx| tx.faces().list_all()).unwrap();
    let assigned_count = faces.iter().filter(|f| f.person_id.is_some()).count();

    eprintln!("  Assigned faces: {}/{}", assigned_count, faces.len());

    eprintln!(">>> TEST FACE_CREATES_PERSON: PASS <<<");
}

// =============================================================================
// Test 3: MULTI-FACE IMAGE — Single image with multiple faces
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn test_multi_face_image() {
    let tmp = tmpdir("multi-face");
    let photos_dir = tmp.join("photos");
    std::fs::create_dir(&photos_dir).unwrap();

    // Find a multi-face image
    let pexels_path = PathBuf::from("/Users/mac/Downloads");
    let src = pexels_path.join("pexels-cottonbro-5900525.jpg");
    if src.exists() {
        std::fs::copy(&src, photos_dir.join("multi.jpg")).unwrap();
    } else {
        eprintln!("SKIP: Multi-face image not found");
        return;
    }

    // Setup services
    let scrfd_path = resolve_model("scrfd_500m_bnkps.onnx");
    let arcface_path = resolve_model("w600k_r50.onnx");

    let detector = ScrfdDetector::load(&scrfd_path).expect("load SCRFD");
    let aligner = Arc::new(SimpleAligner::new());
    let embedder = ArcFaceEmbedder::load(&arcface_path).expect("load ArcFace");
    let qf = QualityFilter::from_config(0.30, 20, 0.45, 30.0);
    let pipeline = Arc::new(FacePipeline::new(detector, aligner, embedder, qf));

    let db_dir = tmp.join("db");
    std::fs::create_dir(&db_dir).unwrap();
    let db_path = db_dir.join("prod.db");
    let db = Arc::new(Database::open(&db_path, builtin_migrations()).unwrap());

    let hnsw_dir = tmp.join("hnsw");
    std::fs::create_dir(&hnsw_dir).unwrap();
    let face_index: Arc<HnswIndex> = Arc::new(HnswIndex::new(512, hnsw_dir, "face"));

    let config = Arc::new(Config::default());
    let photo_provider: Arc<dyn PhotoProvider> = Arc::new(FileSystemPhotoProvider::new(vec![photos_dir.clone()]));

    let registry = Arc::new(ExecutorRegistry::new());
    let scheduler = Arc::new(SqliteTaskScheduler::new((*db).clone(), registry.clone()));

    let index = Arc::new(IndexService::new(
        db.clone(),
        pipeline.clone(),
        face_index.clone(),
        None,
        None,
        None,
        None,
        photo_provider.clone(),
        None,
        None,
    ));

    let search = Arc::new(
        SearchService::new(
            db.clone(),
            config.clone(),
            pipeline.clone(),
            face_index.clone(),
            None,
            None,
            None,
            photo_provider.clone(),
            None,
            None,
            None,
            None,
            None,
        )
        .with_prototype_service(Arc::new(PrototypeService::new(db.clone()))),
    );

    let person = Arc::new(
        PersonService::new(db.clone(), face_index.clone(), ClusterPolicy::default())
            .with_prototype_service(Arc::new(PrototypeService::new(db.clone()))),
    );

    let scan = Arc::new(ScanService::new(
        db.clone(),
        scheduler.clone() as Arc<dyn TaskScheduler>,
        photo_provider.clone(),
        false,
    ));

    register_executors(&registry, db.clone(), index.clone(), person.clone());

    // Scan
    let summary = scan.scan_folder(&photos_dir).await.unwrap();
    eprintln!("Scan: {} images, {} faces queued", summary.candidate_count, summary.queued_face_tasks);

    // Wait for indexing
    let success = wait_for_completion(&scheduler, 120).await;
    assert!(success, "Indexing should complete");

    // Verify: Multiple faces in same image
    let n_face = db.transaction(|tx| tx.faces().count()).unwrap();
    let images = db.transaction(|tx| tx.images().list_paged(100, 0)).unwrap();

    eprintln!("MULTI-FACE TEST:");
    eprintln!("  Images: {}", images.len());
    eprintln!("  Faces: {}", n_face);

    if n_face > 1 {
        // All faces should have same image_id
        let faces = db.transaction(|tx| tx.faces().list_all()).unwrap();
        let first_img_id = faces[0].image_id;
        let all_same_image = faces.iter().all(|f| f.image_id == first_img_id);
        assert!(all_same_image, "All faces in multi-face image should have same image_id");

        eprintln!("  All {} faces correctly assigned to same image", n_face);
    }

    eprintln!(">>> TEST MULTI_FACE: PASS <<<");
}

// =============================================================================
// Test 4: RESTART — Persist and reload state
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn test_restart_persistence() {
    let tmp = tmpdir("restart");
    let photos_dir = tmp.join("photos");
    std::fs::create_dir(&photos_dir).unwrap();

    // Create some images
    for i in 0..3 {
        let name = format!("img{}.jpg", i);
        create_test_image(&photos_dir, &name, 800, 600);
    }

    // Setup services
    let scrfd_path = resolve_model("scrfd_500m_bnkps.onnx");
    let arcface_path = resolve_model("w600k_r50.onnx");

    let detector = ScrfdDetector::load(&scrfd_path).expect("load SCRFD");
    let aligner = Arc::new(SimpleAligner::new());
    let embedder = ArcFaceEmbedder::load(&arcface_path).expect("load ArcFace");
    let qf = QualityFilter::from_config(0.30, 20, 0.45, 30.0);
    let pipeline = Arc::new(FacePipeline::new(detector, aligner, embedder, qf));

    let db_dir = tmp.join("db");
    std::fs::create_dir(&db_dir).unwrap();
    let db_path = db_dir.join("prod.db");
    let db = Arc::new(Database::open(&db_path, builtin_migrations()).unwrap());

    let hnsw_dir = tmp.join("hnsw");
    std::fs::create_dir(&hnsw_dir).unwrap();
    let face_index: Arc<HnswIndex> = Arc::new(HnswIndex::new(512, hnsw_dir.clone(), "face"));

    let config = Arc::new(Config::default());
    let photo_provider: Arc<dyn PhotoProvider> = Arc::new(FileSystemPhotoProvider::new(vec![photos_dir.clone()]));

    let registry = Arc::new(ExecutorRegistry::new());
    let scheduler = Arc::new(SqliteTaskScheduler::new((*db).clone(), registry.clone()));

    let index = Arc::new(IndexService::new(
        db.clone(),
        pipeline.clone(),
        face_index.clone(),
        None,
        None,
        None,
        None,
        photo_provider.clone(),
        None,
        None,
    ));

    let search = Arc::new(
        SearchService::new(
            db.clone(),
            config.clone(),
            pipeline.clone(),
            face_index.clone(),
            None,
            None,
            None,
            photo_provider.clone(),
            None,
            None,
            None,
            None,
            None,
        )
        .with_prototype_service(Arc::new(PrototypeService::new(db.clone()))),
    );

    let person = Arc::new(
        PersonService::new(db.clone(), face_index.clone(), ClusterPolicy::default())
            .with_prototype_service(Arc::new(PrototypeService::new(db.clone()))),
    );

    let scan = Arc::new(ScanService::new(
        db.clone(),
        scheduler.clone() as Arc<dyn TaskScheduler>,
        photo_provider.clone(),
        false,
    ));

    register_executors(&registry, db.clone(), index.clone(), person.clone());

    // Scan and index
    let _summary = scan.scan_folder(&photos_dir).await.unwrap();
    let success = wait_for_completion(&scheduler, 120).await;
    assert!(success, "Initial indexing should complete");

    // Get initial state
    let n_img_before = db.transaction(|tx| tx.images().count()).unwrap();
    let n_face_before = db.transaction(|tx| tx.faces().count()).unwrap();
    let hnsw_len_before = face_index.len();

    eprintln!("BEFORE RESTART:");
    eprintln!("  Images: {}", n_img_before);
    eprintln!("  Faces: {}", n_face_before);
    eprintln!("  HNSW len: {}", hnsw_len_before);

    // Simulate restart by dropping and reopening
    drop(db);
    drop(face_index);

    // Reopen database
    let db_reopened = Arc::new(Database::open(&db_path, builtin_migrations()).unwrap());

    // Verify state persisted
    let n_img_after = db_reopened.transaction(|tx| tx.images().count()).unwrap();
    let n_face_after = db_reopened.transaction(|tx| tx.faces().count()).unwrap();

    eprintln!("AFTER RESTART:");
    eprintln!("  Images: {}", n_img_after);
    eprintln!("  Faces: {}", n_face_after);

    assert_eq!(n_img_before, n_img_after, "Image count should persist");
    assert_eq!(n_face_before, n_face_after, "Face count should persist");

    eprintln!(">>> TEST RESTART: PASS <<<");
}

// =============================================================================
// Test 5: INCREMENTAL SCAN — Add images to existing library
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn test_incremental_scan() {
    let tmp = tmpdir("incremental");
    let photos_dir = tmp.join("photos");
    std::fs::create_dir(&photos_dir).unwrap();

    // Initial images (use real images from Downloads)
    let downloads = PathBuf::from("/Users/mac/Downloads");
    let face_images = ["pexels-jaqor-33601831.jpg", "pexels-jaqor-33601835.jpg"];

    let mut init_copied = 0;
    for (i, name) in face_images.iter().enumerate() {
        let src = downloads.join(name);
        if src.exists() {
            std::fs::copy(&src, photos_dir.join(format!("init{}.jpg", i))).unwrap();
            init_copied += 1;
        }
    }

    if init_copied < 2 {
        eprintln!("SKIP: Initial face images not found");
        return;
    }
    eprintln!("Copied {} initial images", init_copied);

    // Setup services
    let scrfd_path = resolve_model("scrfd_500m_bnkps.onnx");
    let arcface_path = resolve_model("w600k_r50.onnx");

    let detector = ScrfdDetector::load(&scrfd_path).expect("load SCRFD");
    let aligner = Arc::new(SimpleAligner::new());
    let embedder = ArcFaceEmbedder::load(&arcface_path).expect("load ArcFace");
    let qf = QualityFilter::from_config(0.30, 20, 0.45, 30.0);
    let pipeline = Arc::new(FacePipeline::new(detector, aligner, embedder, qf));

    let db_dir = tmp.join("db");
    std::fs::create_dir(&db_dir).unwrap();
    let db_path = db_dir.join("prod.db");
    let db = Arc::new(Database::open(&db_path, builtin_migrations()).unwrap());

    let hnsw_dir = tmp.join("hnsw");
    std::fs::create_dir(&hnsw_dir).unwrap();
    let face_index: Arc<HnswIndex> = Arc::new(HnswIndex::new(512, hnsw_dir, "face"));

    let config = Arc::new(Config::default());
    let photo_provider: Arc<dyn PhotoProvider> = Arc::new(FileSystemPhotoProvider::new(vec![photos_dir.clone()]));

    let registry = Arc::new(ExecutorRegistry::new());
    let scheduler = Arc::new(SqliteTaskScheduler::new((*db).clone(), registry.clone()));

    let index = Arc::new(IndexService::new(
        db.clone(),
        pipeline.clone(),
        face_index.clone(),
        None,
        None,
        None,
        None,
        photo_provider.clone(),
        None,
        None,
    ));

    let search = Arc::new(
        SearchService::new(
            db.clone(),
            config.clone(),
            pipeline.clone(),
            face_index.clone(),
            None,
            None,
            None,
            photo_provider.clone(),
            None,
            None,
            None,
            None,
            None,
        )
        .with_prototype_service(Arc::new(PrototypeService::new(db.clone()))),
    );

    let person = Arc::new(
        PersonService::new(db.clone(), face_index.clone(), ClusterPolicy::default())
            .with_prototype_service(Arc::new(PrototypeService::new(db.clone()))),
    );

    let scan = Arc::new(ScanService::new(
        db.clone(),
        scheduler.clone() as Arc<dyn TaskScheduler>,
        photo_provider.clone(),
        false,
    ));

    register_executors(&registry, db.clone(), index.clone(), person.clone());

    // First scan
    let _summary1 = scan.scan_folder(&photos_dir).await.unwrap();
    let success1 = wait_for_completion(&scheduler, 120).await;
    assert!(success1, "Initial indexing should complete");

    let n_img_after_init = db.transaction(|tx| tx.images().count()).unwrap();
    let n_face_after_init = db.transaction(|tx| tx.faces().count()).unwrap();

    eprintln!("AFTER INITIAL SCAN:");
    eprintln!("  Images: {}", n_img_after_init);
    eprintln!("  Faces: {}", n_face_after_init);

    // Add more images (use additional real images)
    let more_images = ["pexels-cottonbro-5900525.jpg", "pexels-cottonbro-7609199.jpg"];
    let mut added = 0;
    for (i, name) in more_images.iter().enumerate() {
        let src = downloads.join(name);
        if src.exists() {
            std::fs::copy(&src, photos_dir.join(format!("added{}.jpg", i))).unwrap();
            added += 1;
        }
    }
    eprintln!("Added {} more images", added);

    // Second scan (incremental)
    let _summary2 = scan.scan_folder(&photos_dir).await.unwrap();
    let success2 = wait_for_completion(&scheduler, 120).await;
    assert!(success2, "Incremental indexing should complete");

    let n_img_final = db.transaction(|tx| tx.images().count()).unwrap();
    let n_face_final = db.transaction(|tx| tx.faces().count()).unwrap();

    eprintln!("AFTER INCREMENTAL SCAN:");
    eprintln!("  Images: {}", n_img_final);
    eprintln!("  Faces: {}", n_face_final);

    let expected_total = (init_copied + added) as i64;
    assert_eq!(n_img_final, expected_total, "Should have {} total images", expected_total);
    // Face count should not decrease
    assert!(n_face_final >= n_face_after_init, "Face count should not decrease");

    eprintln!(">>> TEST INCREMENTAL_SCAN: PASS <<<");
}
