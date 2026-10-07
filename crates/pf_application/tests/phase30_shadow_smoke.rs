//! Phase 30 T12 — Shadow Mode Smoke Test
//!
//! Validates Shadow Mode in a real scan workflow.
//!
//! Run with:
//! ```bash
//! cargo test -p pf_application --test phase30_shadow_smoke -- --nocapture --ignored
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
        p.push(format!("pf-shadow-{}-{}", prefix, std::process::id()));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn select_images(n: usize) -> Vec<PathBuf> {
        let mut images = Vec::new();
        let downloads = PathBuf::from(DOWNLOADS);

        for name in PEXELS {
            let img_path = downloads.join(name);
            if img_path.exists() {
                images.push(img_path);
                if images.len() >= n {
                    return images;
                }
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn shadow_smoke_legacy_vs_shadow() {
        let tmp = tmpdir("shadow-smoke");
        eprintln!("=== Phase 30 Shadow Smoke Test ===");
        eprintln!("tmpdir: {}", tmp.display());

        // Select images
        let images = select_images(30);
        eprintln!("Selected {} images", images.len());
        assert!(images.len() >= 8, "Need at least 8 images");

        // Copy images to temp photos dir
        let photos_dir = tmp.join("photos");
        fs::create_dir(&photos_dir).unwrap();
        for (i, src) in images.iter().enumerate() {
            let dst = photos_dir.join(format!("img_{:03}.jpg", i));
            let _ = fs::copy(src, &dst);
        }

        // Setup models
        let scrfd_path = resolve_model("scrfd_500m_bnkps.onnx");
        let arcface_path = resolve_model("w600k_r50.onnx");
        eprintln!("SCRFD   = {}", scrfd_path.display());
        eprintln!("ArcFace = {}", arcface_path.display());
        assert!(scrfd_path.exists(), "SCRFD model not found");
        assert!(arcface_path.exists(), "ArcFace model not found");

        let detector = ScrfdDetector::load(&scrfd_path).expect("load SCRFD");
        let aligner = Arc::new(SimpleAligner::new());
        let embedder = ArcFaceEmbedder::load(&arcface_path).expect("load ArcFace");
        let qf = QualityFilter::from_config(0.30, 20, 0.45, 30.0);
        let pipeline = Arc::new(FacePipeline::new(detector, aligner, embedder, qf));

        // Setup DB
        let db_dir = tmp.join("db");
        fs::create_dir(&db_dir).unwrap();
        let db_path = db_dir.join("smoke.db");
        let db = Arc::new(
            Database::open(&db_path, builtin_migrations()).expect("open db"),
        );

        // Setup HNSW
        let hnsw_dir = tmp.join("hnsw");
        fs::create_dir(&hnsw_dir).unwrap();
        let face_index: Arc<dyn VectorIndex> =
            Arc::new(HnswIndex::new(512, hnsw_dir.clone(), "face"));

        let config = Arc::new(make_config(IdentityExecutionMode::Legacy));
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
                .with_execution_mode(IdentityExecutionMode::Legacy),
        );
        register_executors(&registry, db.clone(), index.clone(), person.clone());

        // ===== RUN 1: Legacy mode =====
        eprintln!("\n========== RUN 1: Legacy Mode ==========");
        let scan_start = Instant::now();
        let summary = scan.scan_folder(&photos_dir).await.expect("scan failed");
        eprintln!(
            "scan: candidate={} inserted={} skipped={} filtered={} filtered_bpp={} queued_face_tasks={}",
            summary.candidate_count,
            summary.inserted,
            summary.skipped,
            summary.filtered,
            summary.filtered_bpp,
            summary.queued_face_tasks
        );

        // Wait for tasks
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

        let final_stats = scheduler.status().await.expect("final stats");
        eprintln!(
            "Legacy final: completed={} failed={}",
            final_stats.completed, final_stats.failed
        );

        // Collect Legacy results
        let legacy_result = db
            .transaction(|tx| {
                let n_images = tx.images().count().unwrap_or(0);
                let n_faces = tx.faces().count().unwrap_or(0);
                let n_persons = tx.persons().list().unwrap_or_default().len() as i64;
                Ok((n_images, n_faces, n_persons))
            })
            .expect("query legacy");

        eprintln!(
            "Legacy: images={} faces={} persons={}",
            legacy_result.0, legacy_result.1, legacy_result.2
        );

        // ===== RUN 2: Shadow mode =====
        // Reset DB and HNSW
        drop(legacy_result);
        drop(scan);
        drop(scheduler);
        drop(person);
        drop(search);
        drop(index);
        drop(db);
        drop(face_index);

        fs::remove_file(&db_path).ok();
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

        // Re-create services with Shadow mode
        let db = Arc::new(
            Database::open(&db_path, builtin_migrations()).expect("open db2"),
        );
        let face_index: Arc<dyn VectorIndex> =
            Arc::new(HnswIndex::new(512, hnsw_dir.clone(), "face"));

        let shadow_config = Arc::new(make_config(IdentityExecutionMode::Shadow));
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
        let search = Arc::new(
            SearchService::new(
                db.clone(),
                shadow_config.clone(),
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
                .with_execution_mode(IdentityExecutionMode::Shadow),
        );
        register_executors(&registry, db.clone(), index.clone(), person.clone());

        eprintln!("\n========== RUN 2: Shadow Mode ==========");
        let scan_start = Instant::now();
        let summary = scan.scan_folder(&photos_dir).await.expect("scan2 failed");

        // Wait for tasks
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

        let final_stats = scheduler.status().await.expect("final stats");
        eprintln!(
            "Shadow final: completed={} failed={}",
            final_stats.completed, final_stats.failed
        );

        // Collect Shadow results
        let shadow_result = db
            .transaction(|tx| {
                let n_images = tx.images().count().unwrap_or(0);
                let n_faces = tx.faces().count().unwrap_or(0);
                let n_persons = tx.persons().list().unwrap_or_default().len() as i64;
                let n_shadow = tx.shadow_records().count().unwrap_or(0);
                let n_disagreements = tx.shadow_records().count_disagreements().unwrap_or(0);
                Ok((n_images, n_faces, n_persons, n_shadow, n_disagreements))
            })
            .expect("query shadow");

        eprintln!(
            "Shadow: images={} faces={} persons={} shadow_records={} disagreements={}",
            shadow_result.0, shadow_result.1, shadow_result.2, shadow_result.3, shadow_result.4
        );

        // ===== LEGACY INTEGRITY CHECK =====
        eprintln!("\n========== LEGACY INTEGRITY CHECK ==========");

        let integrity_pass = legacy_result.1 == shadow_result.1 // faces must match
            && legacy_result.2 == shadow_result.2; // persons must match

        eprintln!(
            "Face count: legacy={} shadow={} -> {}",
            legacy_result.1,
            shadow_result.1,
            if legacy_result.1 == shadow_result.1 {
                "OK"
            } else {
                "MISMATCH!"
            }
        );

        eprintln!(
            "Person count: legacy={} shadow={} -> {}",
            legacy_result.2,
            shadow_result.2,
            if legacy_result.2 == shadow_result.2 {
                "OK"
            } else {
                "MISMATCH!"
            }
        );

        eprintln!("\n===========================================");
        eprintln!(
            "LEGACY_INTEGRITY: {}",
            if integrity_pass { "PASS" } else { "FAIL" }
        );
        eprintln!("===========================================");

        if shadow_result.3 > 0 {
            let agreement_rate =
                1.0 - (shadow_result.4 as f64 / shadow_result.1 as f64);
            eprintln!("Shadow records: {}", shadow_result.3);
            eprintln!("Disagreements: {}", shadow_result.4);
            eprintln!("Agreement rate: {:.1}%", agreement_rate * 100.0);
        }

        if integrity_pass {
            eprintln!("\n>>> SHADOW_SMOKE_TEST: PASS <<<");
        } else {
            eprintln!("\n>>> SHADOW_SMOKE_TEST: FAIL <<<");
            panic!("LEGACY_INTEGRITY: FAIL");
        }

        fs::remove_dir_all(&tmp).ok();
    }
}
