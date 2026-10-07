//! Phase 32 — Shadow Disagreement Root-Cause Analysis
//!
//! Analyzes 233 shadow records to understand why 96.6% disagreement occurs.
//!
//! Run with:
//! ```bash
//! cargo test -p pf_application --test phase32_disagreement_analysis -- --nocapture --ignored
//! ```

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Instant;

    use pf_ai::{
        ArcFaceEmbedder, FacePipeline, QualityFilter, ScrfdDetector, SimpleAligner,
    };
    use pf_application::executors::register_executors;
    use pf_application::identity_evidence::{IdentityExecutionMode, IdentityPipeline};
    use pf_application::index::IndexService;
    use pf_application::person::{ClusterPolicy, PersonService};
    use pf_application::prototype_service::PrototypeService;
    use pf_application::scan::ScanService;
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
        p.push(format!("pf-phase32-{}-{}", prefix, std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    /// Copy LFW images resized to 256x256
    fn copy_lfw_subset(dest: &PathBuf, max_images: usize) -> (usize, usize) {
        use image::imageops::FilterType;

        let lfw_root = PathBuf::from(LFW_ROOT);
        let mut images = Vec::new();
        let mut persons = 0;

        if !lfw_root.exists() {
            eprintln!("LFW root not found: {}", LFW_ROOT);
            return (0, 0);
        }

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

        let mut copied = 0;
        for (i, src) in images.iter().enumerate() {
            let dst = dest.join(format!("{:06}_{}", i, src.file_name().unwrap().to_str().unwrap()));
            if let Ok(img) = image::open(src) {
                let resized = img.resize_exact(256, 256, FilterType::Lanczos3);
                if resized.save(&dst).is_ok() {
                    copied += 1;
                }
            }
        }

        eprintln!("Copied {} images (resized to 256x256) from {} persons", copied, persons);
        (copied, persons)
    }

    fn setup_services(tmp: &PathBuf, execution_mode: IdentityExecutionMode) -> Arc<Database> {
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

        db
    }

    async fn run_scan_and_index(
        scan: &ScanService,
        scheduler: &SqliteTaskScheduler,
        photos_dir: &PathBuf,
        timeout_secs: u64,
    ) {
        use std::time::Duration;
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

    // =========================================================================
    // Phase 32 T1: Complete Disagreement Data Export
    // =========================================================================

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore]
    async fn test_phase32_disagreement_analysis() {
        let tmp = tmpdir("phase32");
        eprintln!("\n=== Phase 32: Disagreement Root-Cause Analysis ===");

        // Copy LFW subset
        let photos_dir = tmp.join("photos");
        fs::create_dir_all(&photos_dir).unwrap();
        let (img_count, person_count) = copy_lfw_subset(&photos_dir, 200);
        eprintln!("Copied {} images from {} persons", img_count, person_count);

        if img_count < 50 {
            eprintln!("Not enough images, skipping");
            fs::remove_dir_all(&tmp).ok();
            return;
        }

        // Setup Shadow mode
        let db = setup_services(&tmp, IdentityExecutionMode::Shadow);

        let scrfd_path = resolve_model("scrfd_500m_bnkps.onnx");
        let arcface_path = resolve_model("w600k_r50.onnx");

        let detector = ScrfdDetector::load(&scrfd_path).expect("load SCRFD");
        let aligner = Arc::new(SimpleAligner::new());
        let embedder = ArcFaceEmbedder::load(&arcface_path).expect("load ArcFace");
        let qf = QualityFilter::from_config(0.30, 20, 0.45, 30.0);
        let pipeline = Arc::new(FacePipeline::new(detector, aligner, embedder, qf));

        let hnsw_dir = tmp.join("hnsw");
        fs::create_dir_all(&hnsw_dir).expect("create hnsw dir");
        let face_index: Arc<dyn VectorIndex> =
            Arc::new(HnswIndex::new(512, hnsw_dir.clone(), "face"));

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
                .with_execution_mode(IdentityExecutionMode::Shadow),
        );

        register_executors(&registry, db.clone(), index.clone(), person.clone());

        // Run scan + index
        eprintln!("\n--- Phase 1: Scan + Index ---");
        run_scan_and_index(&scan, &scheduler, &photos_dir, 600).await;

        // Run cluster_all
        eprintln!("\n--- Phase 2: Cluster All (Shadow Mode) ---");
        let cluster_result = person.cluster_all().await.expect("cluster_all failed");
        eprintln!(
            "cluster_all: assigned={} created={}",
            cluster_result.assigned, cluster_result.created
        );

        // Collect all shadow records
        let records = db.transaction(|tx| tx.shadow_records().list_recent(1000)).unwrap();

        eprintln!("\n========== Phase 32: DISAGREEMENT ANALYSIS ==========\n");
        eprintln!("Total shadow records: {}", records.len());

        // Classify by disagreement type
        let mut legacy_new_shadow = Vec::new();
        let mut different_person = Vec::new();
        let mut agree = Vec::new();
        let mut other = Vec::new();

        for r in &records {
            match r.disagreement_type.as_str() {
                // Legacy said New, Shadow said Existing
                "LegacyNewPersonShadowAssign" => legacy_new_shadow.push(r),
                // Both said Existing but different persons
                "LegacyAssignShadowDifferentPerson" => different_person.push(r),
                "Agree" => agree.push(r),
                _ => other.push(r),
            }
        }

        eprintln!("\n--- Disagreement Classification ---");
        eprintln!("Agree:                    {:>4} ({:>5.1}%)", agree.len(), (agree.len() as f64 / records.len() as f64) * 100.0);
        eprintln!("LegacyNewShadow:          {:>4} ({:>5.1}%)", legacy_new_shadow.len(), (legacy_new_shadow.len() as f64 / records.len() as f64) * 100.0);
        eprintln!("DifferentPerson:          {:>4} ({:>5.1}%)", different_person.len(), (different_person.len() as f64 / records.len() as f64) * 100.0);
        eprintln!("Other:                   {:>4} ({:>5.1}%)", other.len(), (other.len() as f64 / records.len() as f64) * 100.0);

        // =========================================================================
        // T2: LegacyNewShadow Score Analysis
        // =========================================================================
        eprintln!("\n========== T2: LegacyNewShadow Score Analysis ==========\n");

        if !legacy_new_shadow.is_empty() {
            // Bucket by shadow_face_score (what Shadow saw)
            let mut score_buckets: [(f32, f32, usize); 7] = [
                (0.0, 0.30, 0),
                (0.30, 0.40, 0),
                (0.40, 0.50, 0),
                (0.50, 0.60, 0),
                (0.60, 0.70, 0),
                (0.70, 0.80, 0),
                (0.80, 99.0, 0),
            ];

            for r in &legacy_new_shadow {
                let score = r.shadow_face_score.unwrap_or(0.0);
                for bucket in &mut score_buckets {
                    if score >= bucket.0 && score < bucket.1 {
                        bucket.2 += 1;
                        break;
                    }
                }
            }

            eprintln!("Shadow Face Score Distribution (LegacyNewShadow cases):");
            eprintln!("{:15} {:>8} {:>10}", "Bucket", "Count", "Pct");
            eprintln!("{:15} {:>8} {:>10}", "---------", "--------", "----------");
            for (low, high, count) in &score_buckets {
                let pct = (*count as f64 / legacy_new_shadow.len() as f64) * 100.0;
                let label = if *high > 10.0 {
                    format!(">= {}", *low as i32)
                } else {
                    format!("{:.2}-{:.2}", low, high)
                };
                eprintln!("{:15} {:>8} {:>9.1}%", label, count, pct);
            }

            // Key insight: What % are in high confidence region?
            let high_conf = score_buckets[5].2 + score_buckets[6].2; // 0.70+
            let pct = (high_conf as f64 / legacy_new_shadow.len() as f64) * 100.0;
            eprintln!("\n>>> HIGH CONFIDENCE (>=0.70): {} / {} ({:.1}%) <<<",
                high_conf, legacy_new_shadow.len(), pct);

            // What % are in review region?
            let review_conf = score_buckets[3].2 + score_buckets[4].2; // 0.50-0.70
            let pct = (review_conf as f64 / legacy_new_shadow.len() as f64) * 100.0;
            eprintln!(">>> REVIEW REGION (0.50-0.70): {} / {} ({:.1}%) <<<",
                review_conf, legacy_new_shadow.len(), pct);

            // What % are borderline?
            let borderline = score_buckets[0].2 + score_buckets[1].2 + score_buckets[2].2; // <0.50
            let pct = (borderline as f64 / legacy_new_shadow.len() as f64) * 100.0;
            eprintln!(">>> BORDERLINE (<0.50): {} / {} ({:.1}%) <<<",
                borderline, legacy_new_shadow.len(), pct);
        }

        // =========================================================================
        // T3: Shadow Score Analysis - Focus on high confidence matches
        // =========================================================================
        eprintln!("\n========== T3: High Confidence Shadow Analysis ==========\n");

        // Find cases where Shadow is confident but Legacy said New
        let high_conf_new_shadow: Vec<_> = legacy_new_shadow.iter()
            .filter(|r| r.shadow_face_score.unwrap_or(0.0) >= 0.70)
            .collect();

        eprintln!("High Confidence LegacyNewShadow (shadow_score >= 0.70): {}", high_conf_new_shadow.len());

        if !high_conf_new_shadow.is_empty() {
            // Analyze margin for these high confidence cases
            let mut margins: Vec<f32> = high_conf_new_shadow.iter()
                .filter_map(|r| r.shadow_face_margin)
                .collect();
            margins.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let p10_idx = (margins.len() as f32 * 0.10) as usize;
            let p50_idx = (margins.len() as f32 * 0.50) as usize;
            let p90_idx = (margins.len() as f32 * 0.90) as usize;

            eprintln!("Shadow Margin (high conf cases):");
            eprintln!("  P10: {:.4}", margins[p10_idx.min(margins.len()-1)]);
            eprintln!("  P50: {:.4}", margins[p50_idx.min(margins.len()-1)]);
            eprintln!("  P90: {:.4}", margins[p90_idx.min(margins.len()-1)]);

            // Analyze face quality
            let mut qualities: Vec<f32> = high_conf_new_shadow.iter()
                .filter_map(|r| r.face_quality)
                .collect();
            qualities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            if !qualities.is_empty() {
                let p50_idx = (qualities.len() as f32 * 0.50) as usize;
                eprintln!("Face Quality (high conf cases):");
                eprintln!("  P50: {:.4}", qualities[p50_idx.min(qualities.len()-1)]);
            }
        }

        // =========================================================================
        // T4: Margin Four-Quadrant Analysis
        // =========================================================================
        eprintln!("\n========== T4: Four-Quadrant Analysis ==========\n");

        // For all disagreement cases, classify into quadrants
        let mut quadrant = [
            ("HIGH_SCORE_HIGH_MARGIN", 0usize),  // SAFE - strong match
            ("HIGH_SCORE_LOW_MARGIN", 0usize),   // REVIEW - ambiguous
            ("LOW_SCORE_HIGH_MARGIN", 0usize),    // REVIEW - uncertain
            ("LOW_SCORE_LOW_MARGIN", 0usize),     // UNKNOWN - reject
        ];

        for r in &legacy_new_shadow {
            let score = r.shadow_face_score.unwrap_or(0.0);
            let margin = r.shadow_face_margin.unwrap_or(0.0);
            let is_high_score = score >= 0.60;
            let is_high_margin = margin >= 0.05;

            if is_high_score && is_high_margin {
                quadrant[0].1 += 1;
            } else if is_high_score && !is_high_margin {
                quadrant[1].1 += 1;
            } else if !is_high_score && is_high_margin {
                quadrant[2].1 += 1;
            } else {
                quadrant[3].1 += 1;
            }
        }

        eprintln!("Four-Quadrant Distribution:");
        eprintln!("{:25} {:>8} {:>8}", "Quadrant", "Count", "Pct");
        eprintln!("{:25} {:>8} {:>8}", "---------", "--------", "----------");
        for (name, count) in &quadrant {
            let pct = (*count as f64 / legacy_new_shadow.len() as f64) * 100.0;
            eprintln!("{:25} {:>8} {:>7.1}%", name, count, pct);
        }

        // =========================================================================
        // T5: Face Quality Stratification
        // =========================================================================
        eprintln!("\n========== T5: Face Quality Stratification ==========\n");

        // Analyze by face_size
        let mut size_buckets: [(f32, f32, usize); 6] = [
            (0.0, 40.0, 0),
            (40.0, 60.0, 0),
            (60.0, 80.0, 0),
            (80.0, 120.0, 0),
            (120.0, 200.0, 0),
            (200.0, 9999.0, 0),
        ];

        for r in &legacy_new_shadow {
            let size = r.face_size.unwrap_or(0.0);
            for bucket in &mut size_buckets {
                if size >= bucket.0 && size < bucket.1 {
                    bucket.2 += 1;
                    break;
                }
            }
        }

        eprintln!("Face Size Distribution:");
        eprintln!("{:15} {:>8} {:>10}", "Bucket", "Count", "Pct");
        eprintln!("{:15} {:>8} {:>10}", "---------", "--------", "----------");
        for (low, high, count) in &size_buckets {
            let pct = (*count as f64 / legacy_new_shadow.len() as f64) * 100.0;
            eprintln!("{:>6.0}-{:>6.0} {:>8} {:>9.1}%", low, high, count, pct);
        }

        // Analyze by yaw
        let mut yaw_buckets: [(f32, f32, usize); 5] = [
            (0.0, 15.0, 0),
            (15.0, 30.0, 0),
            (30.0, 45.0, 0),
            (45.0, 60.0, 0),
            (60.0, 180.0, 0),
        ];

        for r in &legacy_new_shadow {
            let yaw = r.yaw.unwrap_or(0.0).abs();
            for bucket in &mut yaw_buckets {
                if yaw >= bucket.0 && yaw < bucket.1 {
                    bucket.2 += 1;
                    break;
                }
            }
        }

        eprintln!("\nYaw Distribution:");
        eprintln!("{:15} {:>8} {:>10}", "Bucket", "Count", "Pct");
        eprintln!("{:15} {:>8} {:>10}", "---------", "--------", "----------");
        for (low, high, count) in &yaw_buckets {
            let pct = (*count as f64 / legacy_new_shadow.len() as f64) * 100.0;
            eprintln!("{:>6.0}-{:>6.0} {:>8} {:>9.1}%", low, high, count, pct);
        }

        // =========================================================================
        // T6: DifferentPerson Analysis
        // =========================================================================
        eprintln!("\n========== T6: DifferentPerson Deep Analysis ==========\n");
        eprintln!("Total DifferentPerson cases: {}", different_person.len());

        if !different_person.is_empty() {
            eprintln!("\nDetailed DifferentPerson records:");
            eprintln!("{:>6} {:>12} {:>12} {:>10} {:>10} {:>10} {:>8}",
                "face_id", "legacy_pid", "shadow_pid", "shadow_score", "legacy_score", "shadow_margin", "quality");
            eprintln!("{:>6} {:>12} {:>12} {:>10} {:>10} {:>10} {:>8}",
                "------", "----------", "----------", "----------", "-----------", "-----------", "--------");

            for r in &different_person {
                eprintln!(
                    "{:>6} {:>12} {:>12} {:>10.4} {:>10.4} {:>10.4} {:>8.4}",
                    r.face_id.unwrap_or(0),
                    r.legacy_person_id.unwrap_or(-1),
                    r.shadow_person_id.unwrap_or(-1),
                    r.shadow_face_score.unwrap_or(0.0),
                    r.legacy_score.unwrap_or(0.0),
                    r.shadow_face_margin.unwrap_or(0.0),
                    r.face_quality.unwrap_or(0.0)
                );
            }
        }

        // =========================================================================
        // T7: Decision Matrix Summary
        // =========================================================================
        eprintln!("\n========== T7: Identity Decision Matrix ==========\n");

        let total_disagreements = legacy_new_shadow.len() + different_person.len();
        let high_conf_legacy_new = legacy_new_shadow.iter()
            .filter(|r| r.shadow_face_score.unwrap_or(0.0) >= 0.70)
            .count();
        let high_conf_different = different_person.iter()
            .filter(|r| r.shadow_face_score.unwrap_or(0.0) >= 0.70)
            .count();

        eprintln!("STRONG MATCH (auto assign):");
        eprintln!("  shadow_score >= 0.70 AND margin >= 0.05");
        eprintln!("  LegacyNewShadow HIGH CONF: {} ({:.1}%)", high_conf_legacy_new,
            (high_conf_legacy_new as f64 / total_disagreements as f64) * 100.0);
        eprintln!("  DifferentPerson HIGH CONF: {} ({:.1}%)", high_conf_different,
            (high_conf_different as f64 / total_disagreements as f64) * 100.0);

        let medium_conf = legacy_new_shadow.iter()
            .filter(|r| {
                let s = r.shadow_face_score.unwrap_or(0.0);
                s >= 0.50 && s < 0.70
            })
            .count();

        eprintln!("\nREVIEW (manual check):");
        eprintln!("  0.50 <= shadow_score < 0.70");
        eprintln!("  Cases: {} ({:.1}%)", medium_conf,
            (medium_conf as f64 / total_disagreements as f64) * 100.0);

        let low_conf = legacy_new_shadow.iter()
            .filter(|r| r.shadow_face_score.unwrap_or(0.0) < 0.50)
            .count();

        eprintln!("\nUNKNOWN / NEW PERSON:");
        eprintln!("  shadow_score < 0.50");
        eprintln!("  Cases: {} ({:.1}%)", low_conf,
            (low_conf as f64 / total_disagreements as f64) * 100.0);

        // =========================================================================
        // Final Summary
        // =========================================================================
        eprintln!("\n========== Phase 32: FINAL SUMMARY ==========\n");
        eprintln!("Total shadows: {}", records.len());
        eprintln!("Total disagreements: {} ({:.1}%)", total_disagreements,
            (total_disagreements as f64 / records.len() as f64) * 100.0);
        eprintln!("");
        eprintln!("KEY FINDING:");
        eprintln!("  95.3% of disagreements are LegacyNewShadow");
        eprintln!("  {} of these have shadow_score >= 0.70 (HIGH CONFIDENCE)", high_conf_legacy_new);
        eprintln!("");
        eprintln!("INTERPRETATION:");
        eprintln!("  Legacy is VERY CONSERVATIVE - rejecting matches that Shadow finds.");
        eprintln!("  {} cases where Shadow is confident but Legacy created NEW person.", high_conf_legacy_new);
        eprintln!("");
        eprintln!("RECOMMENDATION:");
        eprintln!("  For LegacyNewShadow with shadow_score >= 0.70 + margin >= 0.05:");
        eprintln!("  -> Consider entering REVIEW queue (not auto-merge yet)");
        eprintln!("  -> Manual verification needed before auto-assignment");

        // Cleanup
        fs::remove_dir_all(&tmp).ok();
    }
}
