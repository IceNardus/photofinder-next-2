//! Phase 31 T2 — Shadow Cluster Integration Test
//!
//! Tests the full flow:
//! scan → index → cluster_all → assign_face_with_shadow → IdentityPipeline → shadow_records
//!
//! Run with:
//! ```bash
//! cargo test -p pf_application --test phase31_shadow_cluster_integration -- --nocapture --ignored
//! ```

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use pf_ai::{
        ArcFaceEmbedder, FacePipeline, QualityFilter, ScrfdDetector, SimpleAligner,
    };
    use pf_application::executors::register_executors;
    use pf_application::identity_evidence::{IdentityExecutionMode, IdentityPipeline};
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

    const DOWNLOADS: &str = "/Users/mac/Downloads";

    const PEXELS: &[&str] = &[
        "pexels-cottonbro-5900525.jpg",
        "pexels-daria-voronkov-381938591-14723650.jpg",
        "pexels-daria-voronkov-381938591-14723672.jpg",
        "pexels-denniz-futalan-339724-3378435.jpg",
        "pexels-jaqor-33601811.jpg",
        "pexels-jaqor-33601831.jpg",
        "pexels-jaqor-33601835.jpg",
        "pexels-joelle-s-2162497381-38263248.jpg",
        "pexels-peterdanthy-33692605.jpg",
        "pexels-soc-nang-d-ng-2150345854-38142867.jpg",
        "pexels-yi-ren-57040649-33026322.jpg",
    ];

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
        p.push(format!("pf-shadow-cluster-{}-{}", prefix, std::process::id()));
        // Clean up any existing directory first
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn copy_images(n: usize, dest: &PathBuf) -> Vec<PathBuf> {
        let mut images = Vec::new();
        let downloads = PathBuf::from(DOWNLOADS);

        for name in PEXELS {
            if images.len() >= n {
                break;
            }
            let src = downloads.join(name);
            if src.exists() {
                let dst = dest.join(name);
                let _ = fs::copy(&src, &dst);
                images.push(dst);
            }
        }
        images
    }

    fn make_config(execution_mode: IdentityExecutionMode) -> Config {
        use pf_config::IdentityExecutionMode as ConfigExecMode;

        let mut config = Config::default();
        config.identity.execution_mode = match execution_mode {
            IdentityExecutionMode::Legacy => ConfigExecMode::Legacy,
            IdentityExecutionMode::Shadow => ConfigExecMode::Shadow,
            IdentityExecutionMode::NewPipeline => ConfigExecMode::NewPipeline,
        };
        config
    }

    /// Setup common services, returns (db, face_index, person, scheduler, photos_dir)
    fn setup_services(
        tmp: &PathBuf,
        execution_mode: IdentityExecutionMode,
    ) -> (
        Arc<Database>,
        Arc<dyn VectorIndex>,
        Arc<PersonService>,
        Arc<SqliteTaskScheduler>,
        PathBuf,
    ) {
        let scrfd_path = resolve_model("scrfd_500m_bnkps.onnx");
        let arcface_path = resolve_model("w600k_r50.onnx");

        let detector = ScrfdDetector::load(&scrfd_path).expect("load SCRFD");
        let aligner = Arc::new(SimpleAligner::new());
        let embedder = ArcFaceEmbedder::load(&arcface_path).expect("load ArcFace");
        let qf = QualityFilter::from_config(0.30, 20, 0.45, 30.0);
        let pipeline = Arc::new(FacePipeline::new(detector, aligner, embedder, qf));

        let db_dir = tmp.join("db");
        fs::create_dir_all(&db_dir).expect("create db dir");
        let db_path = db_dir.join("smoke.db");
        let db = Arc::new(
            Database::open(&db_path, builtin_migrations()).expect("open db"),
        );

        let hnsw_dir = tmp.join("hnsw");
        fs::create_dir_all(&hnsw_dir).expect("create hnsw dir");
        let face_index: Arc<dyn VectorIndex> =
            Arc::new(HnswIndex::new(512, hnsw_dir.clone(), "face"));

        let photos_dir = tmp.join("photos");
        fs::create_dir_all(&photos_dir).expect("create photos dir");
        copy_images(30, &photos_dir);

        let config = Arc::new(make_config(execution_mode));
        let photo_provider: Arc<dyn PhotoProvider> =
            Arc::new(FileSystemPhotoProvider::new(vec![photos_dir.clone()]));

        let registry = Arc::new(ExecutorRegistry::new());
        let scheduler = Arc::new(SqliteTaskScheduler::new(
            (*db).clone(),
            registry.clone(),
        ));
        let scan = Arc::new(ScanService::new(
            db.clone(),
            scheduler.clone() as Arc<dyn TaskScheduler>,
            photo_provider.clone(),
            false,
        ));
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

        let identity_pipeline = IdentityPipeline::new(
            face_index.clone(),
            None,
            db.clone(),
            Some(Arc::new(PrototypeService::new(db.clone()))),
        );

        let person = Arc::new(
            PersonService::new(db.clone(), face_index.clone(), ClusterPolicy::default())
                .with_prototype_service(Arc::new(PrototypeService::new(db.clone())))
                .with_identity_pipeline(Arc::new(identity_pipeline))
                .with_execution_mode(execution_mode.clone()),
        );

        register_executors(&registry, db.clone(), index.clone(), person.clone());

        (db, face_index, person, scheduler, photos_dir)
    }

    /// Run scan + wait for indexing to complete
    async fn run_scan_and_index(
        scan: &ScanService,
        scheduler: &SqliteTaskScheduler,
        photos_dir: &PathBuf,
    ) {
        let scan_start = Instant::now();
        let _summary = scan.scan_folder(photos_dir).await.expect("scan failed");

        // Wait for all IndexImage tasks to complete
        loop {
            let s = scheduler.status().await.expect("scheduler status");
            let busy = s.pending + s.running;
            eprintln!(
                "  t={:>6?}  pending={} running={} completed={} failed={}",
                scan_start.elapsed(),
                s.pending,
                s.running,
                s.completed,
                s.failed
            );
            if busy == 0 || s.failed > 0 {
                break;
            }
            if scan_start.elapsed() > Duration::from_secs(600) {
                panic!("索引超时 (600s)");
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    /// Collect stats from DB
    fn collect_stats(db: &Database) -> (i64, i64, i64, i64, i64) {
        db.transaction(|tx| {
            let n_images = tx.images().count().unwrap_or(0);
            let n_faces = tx.faces().count().unwrap_or(0);
            let n_persons = tx.persons().list().unwrap_or_default().len() as i64;
            let n_shadow = tx.shadow_records().count().unwrap_or(0);
            let n_disagreements = tx.shadow_records().count_disagreements().unwrap_or(0);
            Ok((n_images, n_faces, n_persons, n_shadow, n_disagreements))
        })
        .unwrap()
    }

    /// Print detailed shadow records for analysis
    fn print_shadow_records(db: &Database) {
        let records = db.transaction(|tx| tx.shadow_records().list_recent(100)).unwrap();
        eprintln!("\n========== SHADOW RECORDS DETAIL ==========");
        for r in &records {
            let is_agree = r.legacy_decision == r.shadow_decision;
            eprintln!(
                "face_id={:>3} | legacy_pid={:>12?} ({}) | shadow_pid={:>12?} ({}) | disagree_type={} | {}",
                r.face_id.unwrap_or(0),
                r.legacy_person_id,
                r.legacy_decision,
                r.shadow_person_id,
                r.shadow_decision,
                r.disagreement_type,
                if is_agree { "AGREE" } else { "DISAGREE" }
            );
        }
        eprintln!("==========================================\n");
    }

    // =========================================================================
    // TEST: Legacy mode - baseline
    // =========================================================================

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn test_legacy_cluster_baseline() {
        let tmp = tmpdir("legacy-baseline");
        eprintln!("\n=== TEST: Legacy Cluster Baseline ===");

        let (db, _face_index, person, scheduler, photos_dir) =
            setup_services(&tmp, IdentityExecutionMode::Legacy);

        let scan = ScanService::new(
            db.clone(),
            scheduler.clone() as Arc<dyn TaskScheduler>,
            Arc::new(FileSystemPhotoProvider::new(vec![photos_dir.clone()])),
            false,
        );

        // Run scan + index
        run_scan_and_index(&scan, &scheduler, &photos_dir).await;

        // Now trigger cluster_all manually
        eprintln!("\n--- Running cluster_all ---");
        let cluster_result = person.cluster_all().await.expect("cluster_all failed");
        eprintln!(
            "cluster_all: assigned={} created={}",
            cluster_result.assigned, cluster_result.created
        );

        let stats = collect_stats(&db);
        eprintln!(
            "Legacy stats: images={} faces={} persons={} shadow_records={} disagreements={}",
            stats.0, stats.1, stats.2, stats.3, stats.4
        );

        assert!(stats.0 > 0, "Should have images");
        assert!(stats.1 > 0, "Should have faces");
        // Persons may be 0 if faces are few/diverse

        eprintln!("\n>>> LEGACY BASELINE: PASS <<<");
        fs::remove_dir_all(&tmp).ok();
    }

    // =========================================================================
    // TEST: Shadow mode - should produce shadow records
    // =========================================================================

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn test_shadow_cluster_produces_records() {
        let tmp = tmpdir("shadow-cluster");
        eprintln!("\n=== TEST: Shadow Cluster Produces Records ===");

        let (db, _face_index, person, scheduler, photos_dir) =
            setup_services(&tmp, IdentityExecutionMode::Shadow);

        let scan = ScanService::new(
            db.clone(),
            scheduler.clone() as Arc<dyn TaskScheduler>,
            Arc::new(FileSystemPhotoProvider::new(vec![photos_dir.clone()])),
            false,
        );

        // Run scan + index
        run_scan_and_index(&scan, &scheduler, &photos_dir).await;

        // Now trigger cluster_all - this should call assign_face_with_shadow
        eprintln!("\n--- Running cluster_all in SHADOW mode ---");
        let cluster_result = person.cluster_all().await.expect("cluster_all failed");
        eprintln!(
            "cluster_all: assigned={} created={}",
            cluster_result.assigned, cluster_result.created
        );

        let stats = collect_stats(&db);
        eprintln!(
            "Shadow stats: images={} faces={} persons={} shadow_records={} disagreements={}",
            stats.0, stats.1, stats.2, stats.3, stats.4
        );

        // Print detailed shadow records for analysis
        print_shadow_records(&db);

        // Core assertions
        assert!(stats.0 > 0, "Should have images");
        assert!(stats.1 > 0, "Should have faces");

        // THIS IS THE KEY TEST: shadow_records must be > 0
        // If this fails, cluster_all is not calling assign_face_with_shadow properly
        assert!(
            stats.3 > 0,
            "SHADOW RECORDS MUST BE > 0! Got {}, expected > 0. cluster_all may not be calling assign_face_with_shadow.",
            stats.3
        );

        eprintln!(
            "\n>>> SHADOW RECORDS TEST: PASS ({} records) <<<",
            stats.3
        );
        fs::remove_dir_all(&tmp).ok();
    }

    // =========================================================================
    // TEST: Legacy Integrity - Shadow must NOT modify Legacy decisions
    // =========================================================================

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn test_legacy_integrity_preserved_in_shadow_mode() {
        let tmp = tmpdir("integrity-check");
        eprintln!("\n=== TEST: Legacy Integrity Preserved in Shadow Mode ===");

        // ===== RUN 1: Legacy mode =====
        eprintln!("\n--- RUN 1: Legacy Mode ---");
        let (db1, _face_index1, person1, scheduler1, photos_dir1) =
            setup_services(&tmp, IdentityExecutionMode::Legacy);

        let scan1 = ScanService::new(
            db1.clone(),
            scheduler1.clone() as Arc<dyn TaskScheduler>,
            Arc::new(FileSystemPhotoProvider::new(vec![photos_dir1.clone()])),
            false,
        );

        run_scan_and_index(&scan1, &scheduler1, &photos_dir1).await;
        let _cluster1 = person1.cluster_all().await.expect("cluster1 failed");
        let legacy_stats = collect_stats(&db1);
        eprintln!(
            "Legacy: images={} faces={} persons={}",
            legacy_stats.0, legacy_stats.1, legacy_stats.2
        );

        // ===== CLEANUP RUN 1 =====
        drop(person1);
        drop(scheduler1);
        drop(scan1);
        drop(db1);
        drop(_face_index1);

        // Remove DB and HNSW to start fresh
        let db_dir = tmp.join("db");
        let hnsw_dir = tmp.join("hnsw");
        fs::remove_file(db_dir.join("smoke.db")).ok();
        fs::remove_file(db_dir.join("smoke.db-wal")).ok();
        fs::remove_file(db_dir.join("smoke.db-shm")).ok();
        for entry in fs::read_dir(&hnsw_dir).unwrap() {
            let entry = entry.unwrap();
            if entry
                .path()
                .extension()
                .map(|e| e == "bin" || e == "lock" || e == "txt")
                .unwrap_or(false)
            {
                fs::remove_file(entry.path()).ok();
            }
        }

        // ===== RUN 2: Shadow mode =====
        eprintln!("\n--- RUN 2: Shadow Mode ---");
        let (db2, _face_index2, person2, scheduler2, photos_dir2) =
            setup_services(&tmp, IdentityExecutionMode::Shadow);

        let scan2 = ScanService::new(
            db2.clone(),
            scheduler2.clone() as Arc<dyn TaskScheduler>,
            Arc::new(FileSystemPhotoProvider::new(vec![photos_dir2.clone()])),
            false,
        );

        run_scan_and_index(&scan2, &scheduler2, &photos_dir2).await;
        let _cluster2 = person2.cluster_all().await.expect("cluster2 failed");
        let shadow_stats = collect_stats(&db2);
        eprintln!(
            "Shadow: images={} faces={} persons={} shadow_records={}",
            shadow_stats.0, shadow_stats.1, shadow_stats.2, shadow_stats.3
        );

        // ===== LEGACY INTEGRITY CHECK =====
        eprintln!("\n========== LEGACY INTEGRITY CHECK ==========");

        let face_match = legacy_stats.1 == shadow_stats.1;
        let person_match = legacy_stats.2 == shadow_stats.2;

        eprintln!(
            "Face count: legacy={} shadow={} -> {}",
            legacy_stats.1,
            shadow_stats.1,
            if face_match { "OK" } else { "MISMATCH!" }
        );

        eprintln!(
            "Person count: legacy={} shadow={} -> {}",
            legacy_stats.2,
            shadow_stats.2,
            if person_match { "OK" } else { "MISMATCH!" }
        );

        let integrity_pass = face_match && person_match;

        eprintln!("\n===========================================");
        eprintln!(
            "LEGACY_INTEGRITY: {}",
            if integrity_pass { "PASS" } else { "FAIL" }
        );
        eprintln!("===========================================");

        assert!(
            integrity_pass,
            "LEGACY_INTEGRITY: FAIL - Shadow mode modified Legacy decisions!"
        );

        if shadow_stats.3 > 0 {
            let agreement_rate =
                1.0 - (shadow_stats.4 as f64 / shadow_stats.1 as f64);
            eprintln!(
                "Shadow records: {}, Disagreements: {}, Agreement rate: {:.1}%",
                shadow_stats.3, shadow_stats.4, agreement_rate * 100.0
            );
        }

        eprintln!("\n>>> LEGACY INTEGRITY: PASS <<<");
        fs::remove_dir_all(&tmp).ok();
    }

    // =========================================================================
    // TEST: T14 - Performance Benchmark (<30% overhead target)
    // =========================================================================

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn test_shadow_performance_benchmark() {
        let tmp = tmpdir("perf-benchmark");
        eprintln!("\n=== TEST: Shadow Mode Performance Benchmark ===");

        // ===== RUN 1: Legacy mode (baseline) =====
        eprintln!("\n--- RUN 1: Legacy Mode (baseline) ---");
        let (db1, face_index1, person1, scheduler1, photos_dir1) =
            setup_services(&tmp, IdentityExecutionMode::Legacy);

        let scan1 = ScanService::new(
            db1.clone(),
            scheduler1.clone() as Arc<dyn TaskScheduler>,
            Arc::new(FileSystemPhotoProvider::new(vec![photos_dir1.clone()])),
            false,
        );

        // Run scan + index
        run_scan_and_index(&scan1, &scheduler1, &photos_dir1).await;

        // Time cluster_all in Legacy mode
        let legacy_start = Instant::now();
        let _legacy_result = person1.cluster_all().await.expect("legacy cluster_all failed");
        let legacy_cluster_time = legacy_start.elapsed();

        let legacy_stats = collect_stats(&db1);
        eprintln!(
            "Legacy: cluster_time={:?}, faces={}, persons={}",
            legacy_cluster_time, legacy_stats.1, legacy_stats.2
        );

        // Cleanup RUN 1
        drop(person1);
        drop(scheduler1);
        drop(scan1);
        drop(db1);
        drop(face_index1);

        // Reset DB and HNSW
        let db_dir = tmp.join("db");
        let hnsw_dir = tmp.join("hnsw");
        fs::remove_file(db_dir.join("smoke.db")).ok();
        fs::remove_file(db_dir.join("smoke.db-wal")).ok();
        fs::remove_file(db_dir.join("smoke.db-shm")).ok();
        for entry in fs::read_dir(&hnsw_dir).unwrap() {
            let entry = entry.unwrap();
            if entry
                .path()
                .extension()
                .map(|e| e == "bin" || e == "lock" || e == "txt")
                .unwrap_or(false)
            {
                fs::remove_file(entry.path()).ok();
            }
        }

        // ===== RUN 2: Shadow mode =====
        eprintln!("\n--- RUN 2: Shadow Mode ---");
        let (db2, face_index2, person2, scheduler2, photos_dir2) =
            setup_services(&tmp, IdentityExecutionMode::Shadow);

        let scan2 = ScanService::new(
            db2.clone(),
            scheduler2.clone() as Arc<dyn TaskScheduler>,
            Arc::new(FileSystemPhotoProvider::new(vec![photos_dir2.clone()])),
            false,
        );

        // Run scan + index
        run_scan_and_index(&scan2, &scheduler2, &photos_dir2).await;

        // Time cluster_all in Shadow mode
        let shadow_start = Instant::now();
        let _shadow_result = person2.cluster_all().await.expect("shadow cluster_all failed");
        let shadow_cluster_time = shadow_start.elapsed();

        let shadow_stats = collect_stats(&db2);
        eprintln!(
            "Shadow: cluster_time={:?}, faces={}, persons={}, shadow_records={}",
            shadow_cluster_time, shadow_stats.1, shadow_stats.2, shadow_stats.3
        );

        // ===== PERFORMANCE ANALYSIS =====
        eprintln!("\n========== PERFORMANCE ANALYSIS ==========");
        let overhead_pct = if legacy_cluster_time.as_secs_f64() > 0.0 {
            ((shadow_cluster_time.as_secs_f64() - legacy_cluster_time.as_secs_f64())
                / legacy_cluster_time.as_secs_f64())
                * 100.0
        } else {
            0.0
        };

        eprintln!("Legacy cluster time: {:?}", legacy_cluster_time);
        eprintln!("Shadow cluster time: {:?}", shadow_cluster_time);
        eprintln!("Overhead: {:.1}%", overhead_pct);
        eprintln!("==========================================");

        // Target: <30% overhead
        let overhead_ok = overhead_pct < 30.0;
        eprintln!("Overhead target (<30%): {}", if overhead_ok { "PASS" } else { "FAIL" });

        assert!(overhead_ok, "Shadow mode overhead {}% exceeds 30% target", overhead_pct);

        eprintln!("\n>>> PERFORMANCE BENCHMARK: PASS <<<");
        fs::remove_dir_all(&tmp).ok();
    }
}
