//! Phase 31 T1 — Large Scale Shadow Analysis
//!
//! Expands shadow testing to 100+ faces for meaningful disagreement analysis.
//!
//! Run with:
//! ```bash
//! cargo test -p pf_application --test phase31_shadow_large_scale -- --nocapture --ignored
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
    use pf_config::Config;
    use pf_database::{builtin_migrations, Database};
    use pf_platform::{FileSystemPhotoProvider, PhotoProvider};
    use pf_task::{ExecutorRegistry, SqliteTaskScheduler, TaskScheduler};
    use pf_vector::{HnswIndex, VectorIndex};

    const LFW_ROOT: &str = "/Users/mac/Library/Caches/PhotoFinder/lfw-eval/lfw-deepfunneled/lfw-deepfunneled";

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
        p.push(format!("pf-shadow-large-{}-{}", prefix, std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    /// Copy a subset of LFW images to temp directory, resized to 256x256 for ImageClassifier
    /// Returns (copied_count, person_count)
    fn copy_lfw_subset(dest: &PathBuf, max_images: usize) -> (usize, usize) {
        use image::imageops::FilterType;

        let lfw_root = PathBuf::from(LFW_ROOT);
        let mut images = Vec::new();
        let mut persons = 0;

        if !lfw_root.exists() {
            eprintln!("LFW root not found: {}", LFW_ROOT);
            return (0, 0);
        }

        // Collect image paths
        let entries = fs::read_dir(&lfw_root).unwrap();
        for entry in entries {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                persons += 1;
                let img_entries = fs::read_dir(&path).unwrap();
                for img_entry in img_entries {
                    let img_entry = img_entry.unwrap();
                    let img_path = img_entry.path();
                    if img_path.extension().map(|e| e == "jpg").unwrap_or(false) {
                        images.push(img_path);
                        if images.len() >= max_images {
                            break;
                        }
                    }
                }
                if images.len() >= max_images {
                    break;
                }
            }
        }

        // Copy and resize images to 256x256 (min_width_height requirement)
        let mut copied = 0;
        for (i, src) in images.iter().enumerate() {
            let dst = dest.join(format!("{:06}_{}", i, src.file_name().unwrap().to_str().unwrap()));
            if let Ok(img) = image::open(src) {
                // Resize to 256x256 using Lanczos3 (high quality)
                let resized = img.resize_exact(256, 256, FilterType::Lanczos3);
                if resized.save(&dst).is_ok() {
                    copied += 1;
                }
            }
        }

        eprintln!("Copied {} images (resized to 256x256) from {} persons", copied, persons);
        (copied, persons)
    }

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

        let config = Arc::new(Config::default());
        let photo_provider: Arc<dyn PhotoProvider> =
            Arc::new(FileSystemPhotoProvider::new(vec![photos_dir.clone()]));

        let registry = Arc::new(ExecutorRegistry::new());
        let scheduler = Arc::new(SqliteTaskScheduler::new(
            (*db).clone(),
            registry.clone(),
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
                .with_execution_mode(execution_mode),
        );

        register_executors(&registry, db.clone(), index.clone(), person.clone());

        (db, face_index, person, scheduler, photos_dir)
    }

    async fn run_scan_and_index(
        scan: &ScanService,
        scheduler: &SqliteTaskScheduler,
        photos_dir: &PathBuf,
        timeout_secs: u64,
    ) {
        let scan_start = Instant::now();
        let _summary = scan.scan_folder(photos_dir).await.expect("scan failed");

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
            if scan_start.elapsed() > Duration::from_secs(timeout_secs) {
                panic!("索引超时 ({}s)", timeout_secs);
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

    /// Print shadow records analysis
    fn analyze_shadow_records(db: &Database) {
        let records = db.transaction(|tx| tx.shadow_records().list_recent(1000)).unwrap();

        let total = records.len();
        if total == 0 {
            eprintln!("No shadow records found");
            return;
        }

        let mut agree = 0;
        let mut different_person = 0;
        let mut legacy_new_shadow_assign = 0;
        let mut legacy_assign_shadow_unknown = 0;
        let mut other = 0;

        for r in &records {
            match r.disagreement_type.as_str() {
                "Agree" => agree += 1,
                "LegacyAssignShadowDifferentPerson" => different_person += 1,
                "LegacyNewPersonShadowAssign" => legacy_new_shadow_assign += 1,
                "LegacyAssignShadowUnknown" => legacy_assign_shadow_unknown += 1,
                _ => other += 1,
            }
        }

        eprintln!("\n========== SHADOW RECORDS ANALYSIS ==========");
        eprintln!("Total shadow records: {}", total);
        eprintln!("----------------------------------------");
        eprintln!("Agreement:           {:>4} ({:>5.1}%)", agree, (agree as f64 / total as f64) * 100.0);
        eprintln!("Different Person:   {:>4} ({:>5.1}%)", different_person, (different_person as f64 / total as f64) * 100.0);
        eprintln!("LegacyNewShadow:    {:>4} ({:>5.1}%)", legacy_new_shadow_assign, (legacy_new_shadow_assign as f64 / total as f64) * 100.0);
        eprintln!("LegacyUnknown:      {:>4} ({:>5.1}%)", legacy_assign_shadow_unknown, (legacy_assign_shadow_unknown as f64 / total as f64) * 100.0);
        eprintln!("Other:              {:>4} ({:>5.1}%)", other, (other as f64 / total as f64) * 100.0);
        eprintln!("==========================================\n");
    }

    // =========================================================================
    // TEST: Large Scale Shadow Analysis (100+ faces)
    // =========================================================================

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore]
    async fn test_large_scale_shadow_analysis() {
        let tmp = tmpdir("large-scale");
        eprintln!("\n=== TEST: Large Scale Shadow Analysis ===");
        eprintln!("tmpdir: {}", tmp.display());

        // Copy LFW subset (aim for 100+ images)
        let photos_dir = tmp.join("photos");
        fs::create_dir_all(&photos_dir).unwrap();
        let (img_count, person_count) = copy_lfw_subset(&photos_dir, 200);
        eprintln!("Copied {} images from {} LFW persons", img_count, person_count);

        if img_count < 50 {
            eprintln!("Not enough images for meaningful test, skipping");
            fs::remove_dir_all(&tmp).ok();
            return;
        }

        // Setup Shadow mode
        let (db, _face_index, person, scheduler, _photos_dir) =
            setup_services(&tmp, IdentityExecutionMode::Shadow);

        let scan = ScanService::new(
            db.clone(),
            scheduler.clone() as Arc<dyn TaskScheduler>,
            Arc::new(FileSystemPhotoProvider::new(vec![photos_dir.clone()])),
            false,
        );

        // Run scan + index
        eprintln!("\n--- Phase 1: Scan + Index ---");
        let start = Instant::now();
        run_scan_and_index(&scan, &scheduler, &photos_dir, 600).await;
        eprintln!("Index completed in {:?}", start.elapsed());

        // Run cluster_all in Shadow mode
        eprintln!("\n--- Phase 2: Cluster All (Shadow Mode) ---");
        let cluster_start = Instant::now();
        let cluster_result = person.cluster_all().await.expect("cluster_all failed");
        eprintln!(
            "cluster_all: assigned={} created={} in {:?}",
            cluster_result.assigned, cluster_result.created, cluster_start.elapsed()
        );

        // Collect and analyze stats
        let stats = collect_stats(&db);
        eprintln!(
            "\nShadow stats: images={} faces={} persons={} shadow_records={} disagreements={}",
            stats.0, stats.1, stats.2, stats.3, stats.4
        );

        // Detailed analysis
        analyze_shadow_records(&db);

        // Assertions
        assert!(stats.0 > 0, "Should have indexed images");
        assert!(stats.1 > 0, "Should have detected faces");
        assert!(stats.3 > 0, "Should have shadow records");

        // Disagreement rate should be meaningful with larger sample
        let disagreement_rate = if stats.3 > 0 {
            stats.4 as f64 / stats.3 as f64
        } else {
            0.0
        };

        eprintln!("\n>>> LARGE SCALE SHADOW ANALYSIS: {} shadow records, {:.1}% disagreement <<<",
            stats.3, disagreement_rate * 100.0);

        fs::remove_dir_all(&tmp).ok();
    }
}
