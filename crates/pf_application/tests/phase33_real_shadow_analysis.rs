//! Phase 33 — Real Shadow Comparison Analysis
//!
//! Analyzes Legacy vs Shadow decisions on real production data.
//!
//! Goals:
//! - T1: Large-scale shadow (500+ images, 1000+ faces)
//! - T2: Disagreement statistics
//! - T3: Decision Matrix
//! - T4: DifferentPerson analysis (top 50 by margin)
//! - T5: Shadow aggressiveness/conservativeness analysis
//! - T6: Anti-Chaining verification
//! - T7: Prototype Pollution test
//! - T8: Face Quality bucketing
//! - T9: Decision Policy validation
//! - T10: Production Gate
//!
//! Run with:
//! ```bash
//! cargo test -p pf_application --test phase33_real_shadow_analysis -- --nocapture --ignored
//! ```

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Instant;

    use pf_ai::{
        ArcFaceEmbedder, FacePipeline, QualityFilter, ScrfdDetector, SimpleAligner,
    };
    use pf_application::executors::register_executors;
    use pf_application::identity_evidence::{
        IdentityDecision, IdentityExecutionMode, IdentityPipeline, ShadowDisagreementType,
    };
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
        p.push(format!("pf-phase33-{}-{}", prefix, std::process::id()));
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

        (copied, persons)
    }

    fn setup_services(tmp: &PathBuf, _execution_mode: IdentityExecutionMode) -> Arc<Database> {
        let db_path = tmp.join("photo.db");
        let db = Database::open(&db_path, builtin_migrations()).expect("open db");
        db.reset().expect("reset db");
        Arc::new(db)
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
    // Phase 33: Real Shadow Analysis
    // =========================================================================

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore]
    async fn test_phase33_real_shadow_analysis() {
        let tmp = tmpdir("phase33");
        eprintln!("\n=== Phase 33: Real Shadow Comparison Analysis ===");

        // T1: Copy LFW subset - 500+ images for meaningful statistics
        let photos_dir = tmp.join("photos");
        fs::create_dir_all(&photos_dir).unwrap();
        let target_images = 500; // Target for Phase 33
        let (img_count, person_count) = copy_lfw_subset(&photos_dir, target_images);
        eprintln!("Copied {} images from {} persons", img_count, person_count);

        if img_count < 100 {
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
        let cluster_start = Instant::now();
        let cluster_result = person.cluster_all().await.expect("cluster_all failed");
        let cluster_time = cluster_start.elapsed();
        eprintln!(
            "cluster_all: assigned={} created={} in {:.1}s",
            cluster_result.assigned, cluster_result.created, cluster_time.as_secs_f32()
        );

        // Collect all shadow records
        let records = db.transaction(|tx| tx.shadow_records().list_recent(5000)).unwrap();

        eprintln!("\n========== Phase 33: REAL SHADOW ANALYSIS ==========\n");
        eprintln!("Total shadow records: {}", records.len());

        // T1 Summary Statistics
        let total_faces = records.len();
        let face_available = records.iter().filter(|r| r.face_id.is_some()).count();
        let body_available = records.iter().filter(|r| r.body_id.is_some()).count();

        eprintln!("\n--- T1: Dataset Statistics ---");
        eprintln!("Total faces:        {}", total_faces);
        eprintln!("Face available:     {} ({:.1}%)", face_available,
            (face_available as f64 / total_faces as f64) * 100.0);
        eprintln!("Body available:     {} ({:.1}%)", body_available,
            (body_available as f64 / total_faces as f64) * 100.0);

        // T2: Classify all records by disagreement type
        let mut agree = Vec::new();
        let mut legacy_new_shadow = Vec::new();
        let mut shadow_new_legacy = Vec::new();  // LegacyAssignShadowNew
        let mut different_person = Vec::new();
        let mut score_only = Vec::new();
        let mut other = Vec::new();

        for r in &records {
            match r.disagreement_type.as_str() {
                "Agree" => agree.push(r),
                "LegacyNewPersonShadowAssign" => legacy_new_shadow.push(r),
                "LegacyAssignShadowUnknown" => shadow_new_legacy.push(r),
                "LegacyAssignShadowDifferentPerson" => different_person.push(r),
                "ScoreOnlyDifference" => score_only.push(r),
                _ => other.push(r),
            }
        }

        eprintln!("\n--- T2: Disagreement Classification ---");
        let n = records.len() as f64;
        eprintln!("{:25} {:>6} {:>8}", "Type", "Count", "Percent");
        eprintln!("{:25} {:>6} {:>8}", "-----", "-----", "-------");
        eprintln!("{:25} {:>6} {:>7.1}%", "Agree", agree.len(), (agree.len() as f64 / n) * 100.0);
        eprintln!("{:25} {:>6} {:>7.1}%", "LegacyNewShadow", legacy_new_shadow.len(), (legacy_new_shadow.len() as f64 / n) * 100.0);
        eprintln!("{:25} {:>6} {:>7.1}%", "ShadowNewLegacy", shadow_new_legacy.len(), (shadow_new_legacy.len() as f64 / n) * 100.0);
        eprintln!("{:25} {:>6} {:>7.1}%", "DifferentPerson", different_person.len(), (different_person.len() as f64 / n) * 100.0);
        eprintln!("{:25} {:>6} {:>7.1}%", "ScoreOnly", score_only.len(), (score_only.len() as f64 / n) * 100.0);
        eprintln!("{:25} {:>6} {:>7.1}%", "Other", other.len(), (other.len() as f64 / n) * 100.0);

        // T3: Decision Matrix
        eprintln!("\n--- T3: Shadow Decision Matrix ---\n");

        // Build decision matrix
        let mut matrix: HashMap<(String, String), Vec<_>> = HashMap::new();
        for r in &records {
            let legacy = r.legacy_decision.clone();
            let shadow = r.shadow_decision.clone();
            matrix.entry((legacy, shadow)).or_default().push(r);
        }

        eprintln!("{:20} {:>15} {:>15} {:>10}", "Legacy \\ Shadow", "Person (same)", "New Person", "Total");
        eprintln!("{:20} {:>15} {:>15} {:>10}", "---------------", "-------------", "----------", "-----");

        // Count matrix entries
        let legacy_decisions = ["confirmed", "probable", "conflict", "unknown"];
        let shadow_decisions = ["confirmed", "probable", "conflict", "unknown"];

        for ld in &legacy_decisions {
            let mut same_person_count = 0usize;
            let mut new_person_count = 0usize;

            for sd in &shadow_decisions {
                let key = (ld.to_string(), sd.to_string());
                if let Some(v) = matrix.get(&key) {
                    // Count how many have same person_id vs different
                    for r in v {
                        let same = r.legacy_person_id == r.shadow_person_id;
                        if same { same_person_count += 1; } else { new_person_count += 1; }
                    }
                }
            }

            let total = same_person_count + new_person_count;
            if total > 0 {
                eprintln!("{:20} {:>15} {:>15} {:>10}", ld, same_person_count, new_person_count, total);
            }
        }

        // T4: DifferentPerson Analysis - Top 50 by margin ascending
        eprintln!("\n--- T4: DifferentPerson Deep Analysis (Top 50 by margin) ---\n");

        let mut different_person_details: Vec<_> = different_person.iter()
            .filter(|r| r.shadow_person_id.is_some() && r.legacy_person_id.is_some())
            .collect();

        // Sort by margin ascending (most problematic first)
        different_person_details.sort_by(|a, b| {
            let ma = a.shadow_face_margin.unwrap_or(0.0);
            let mb = b.shadow_face_margin.unwrap_or(0.0);
            ma.partial_cmp(&mb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let top50: Vec<_> = different_person_details.iter().take(50).collect();

        eprintln!("Top 50 DifferentPerson cases (sorted by margin ascending):");
        eprintln!("{:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8} {:>8}",
            "face_id", "legacy_pid", "shadow_pid", "l_score", "s_score", "margin", "size", "yaw", "blur");
        eprintln!("{:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8} {:>8}",
            "------", "---------", "---------", "-------", "-------", "-----", "----", "---", "----");

        for r in &top50 {
            eprintln!(
                "{:>6} {:>10} {:>10} {:>10.4} {:>10.4} {:>10.4} {:>8.0} {:>8.1} {:>8.2}",
                r.face_id.unwrap_or(0),
                r.legacy_person_id.unwrap_or(-1),
                r.shadow_person_id.unwrap_or(-1),
                r.legacy_score.unwrap_or(0.0),
                r.shadow_face_score.unwrap_or(0.0),
                r.shadow_face_margin.unwrap_or(0.0),
                r.face_size.unwrap_or(0.0),
                r.yaw.unwrap_or(0.0).abs(),
                r.blur_score.unwrap_or(0.0)
            );
        }

        // T5: Shadow Aggressiveness/Conservativeness Analysis
        eprintln!("\n--- T5: Shadow Behavior Analysis ---\n");

        let total_disagreements = legacy_new_shadow.len() + shadow_new_legacy.len() + different_person.len();

        // Shadow is MORE AGGRESSIVE: finds matches Legacy rejected
        let shadow_more_aggressive = legacy_new_shadow.len();
        // Shadow is MORE CONSERVATIVE: rejects matches Legacy found
        let shadow_more_conservative = shadow_new_legacy.len();
        // Identity conflicts
        let identity_conflicts = different_person.len();

        eprintln!("Shadow Behavior Profile:");
        eprintln!("  Shadow MORE AGGRESSIVE (found match Legacy missed): {:>4} ({:.1}%)",
            shadow_more_aggressive,
            (shadow_more_aggressive as f64 / total_disagreements as f64) * 100.0);
        eprintln!("  Shadow MORE CONSERVATIVE (rejected valid Legacy match): {:>4} ({:.1}%)",
            shadow_more_conservative,
            (shadow_more_conservative as f64 / total_disagreements as f64) * 100.0);
        eprintln!("  Identity Conflicts (different person assigned): {:>4} ({:.1}%)",
            identity_conflicts,
            (identity_conflicts as f64 / total_disagreements as f64) * 100.0);

        // Analyze LegacyNewShadow cases in detail
        if !legacy_new_shadow.is_empty() {
            eprintln!("\n  LegacyNewShadow Breakdown:");
            let mut high_conf = 0usize;
            let mut medium_conf = 0usize;
            let mut low_conf = 0usize;

            for r in &legacy_new_shadow {
                let score = r.shadow_face_score.unwrap_or(0.0);
                if score >= 0.70 { high_conf += 1; }
                else if score >= 0.50 { medium_conf += 1; }
                else { low_conf += 1; }
            }

            eprintln!("    HIGH confidence (>=0.70): {} ({:.1}%)",
                high_conf, (high_conf as f64 / legacy_new_shadow.len() as f64) * 100.0);
            eprintln!("    MEDIUM confidence (0.50-0.70): {} ({:.1}%)",
                medium_conf, (medium_conf as f64 / legacy_new_shadow.len() as f64) * 100.0);
            eprintln!("    LOW confidence (<0.50): {} ({:.1}%)",
                low_conf, (low_conf as f64 / legacy_new_shadow.len() as f64) * 100.0);
        }

        // T6: Anti-Chaining Verification
        eprintln!("\n--- T6: Anti-Chaining Verification ---\n");

        let anti_chain_stats = records.iter()
            .filter_map(|r| {
                match r.anti_chain_status.as_str() {
                    "passed" => Some(true),
                    "blocked" => Some(false),
                    _ => None,
                }
            })
            .fold((0usize, 0usize), |(pass, fail), passed| {
                if passed { (pass + 1, fail) } else { (pass, fail + 1) }
            });

        eprintln!("Anti-Chain Status:");
        eprintln!("  Passed: {} ({:.1}%)", anti_chain_stats.0,
            (anti_chain_stats.0 as f64 / (anti_chain_stats.0 + anti_chain_stats.1) as f64) * 100.0);
        eprintln!("  Blocked: {} ({:.1}%)", anti_chain_stats.1,
            (anti_chain_stats.1 as f64 / (anti_chain_stats.0 + anti_chain_stats.1) as f64) * 100.0);

        // T7: Prototype Pollution Analysis
        eprintln!("\n--- T7: Prototype Pollution Analysis ---\n");

        // Analyze by candidate_count (proxy for prototype pollution risk)
        let mut candidate_buckets: [(i32, i32, usize); 5] = [
            (0, 1, 0),
            (2, 3, 0),
            (4, 5, 0),
            (6, 10, 0),
            (11, 999, 0),
        ];

        for r in &records {
            let cc = r.candidate_count.unwrap_or(0);
            for bucket in &mut candidate_buckets {
                if cc >= bucket.0 && cc <= bucket.1 {
                    bucket.2 += 1;
                    break;
                }
            }
        }

        eprintln!("Candidate Count Distribution (proxy for prototype pollution risk):");
        eprintln!("{:>12} {:>10} {:>10}", "Candidates", "Count", "Percent");
        eprintln!("{:>12} {:>10} {:>10}", "----------", "-----", "-------");
        for (low, high, count) in &candidate_buckets {
            let pct = (*count as f64 / records.len() as f64) * 100.0;
            eprintln!("{:>5}-{:>5} {:>10} {:>9.1}%", low, high, count, pct);
        }

        // T8: Face Quality Bucketing
        eprintln!("\n--- T8: Face Quality Bucketing ---\n");

        // Size buckets
        let mut size_buckets: [(f32, f32, usize, usize, usize); 5] = [
            (0.0, 40.0, 0, 0, 0),    // (min, max, total, high_conf, low_conf)
            (40.0, 60.0, 0, 0, 0),
            (60.0, 80.0, 0, 0, 0),
            (80.0, 120.0, 0, 0, 0),
            (120.0, 9999.0, 0, 0, 0),
        ];

        for r in &records {
            let size = r.face_size.unwrap_or(0.0);
            for bucket in &mut size_buckets {
                if size >= bucket.0 && size < bucket.1 {
                    bucket.2 += 1;
                    let score = r.shadow_face_score.unwrap_or(0.0);
                    if score >= 0.70 { bucket.3 += 1; }
                    else if score < 0.50 { bucket.4 += 1; }
                    break;
                }
            }
        }

        eprintln!("Face Size Buckets (with confidence distribution):");
        eprintln!("{:>10} {:>8} {:>10} {:>10} {:>10}", "Size", "Total", "HighConf", "LowConf", "HighPct");
        eprintln!("{:>10} {:>8} {:>10} {:>10} {:>10}", "----", "-----", "--------", "--------", "-------");
        for (low, high, total, high_conf, low_conf) in &size_buckets {
            let high_pct = if *total > 0 { (*high_conf as f64 / *total as f64) * 100.0 } else { 0.0 };
            eprintln!("{:>5.0}-{:>5.0} {:>8} {:>10} {:>10} {:>9.1}%", low, high, total, high_conf, low_conf, high_pct);
        }

        // T9: Decision Policy Validation
        eprintln!("\n--- T9: Decision Policy Validation ---\n");

        // Simulate AUTO / REVIEW / UNKNOWN policy
        let mut auto_assign = 0usize;
        let mut review = 0usize;
        let mut unknown = 0usize;

        for r in &records {
            let score = r.shadow_face_score.unwrap_or(0.0);
            let margin = r.shadow_face_margin.unwrap_or(0.0);
            let quality = r.face_quality.unwrap_or(0.0);

            // Policy: AUTO if score >= 0.70 AND margin >= 0.05 AND quality >= 0.3
            //         REVIEW if score >= 0.50 AND margin >= 0.02
            //         UNKNOWN otherwise
            if score >= 0.70 && margin >= 0.05 && quality >= 0.3 {
                auto_assign += 1;
            } else if score >= 0.50 && margin >= 0.02 {
                review += 1;
            } else {
                unknown += 1;
            }
        }

        let total_decisions = auto_assign + review + unknown;
        eprintln!("Decision Policy Simulation:");
        eprintln!("  AUTO_ASSIGN (score>=0.70, margin>=0.05, quality>=0.3): {:>4} ({:.1}%)",
            auto_assign, (auto_assign as f64 / total_decisions as f64) * 100.0);
        eprintln!("  REVIEW (score>=0.50, margin>=0.02): {:>4} ({:.1}%)",
            review, (review as f64 / total_decisions as f64) * 100.0);
        eprintln!("  UNKNOWN (below thresholds): {:>4} ({:.1}%)",
            unknown, (unknown as f64 / total_decisions as f64) * 100.0);

        // T10: Production Gate
        eprintln!("\n--- T10: Production Gate Evaluation ---\n");

        let agreement_rate = agree.len() as f64 / n;
        let different_person_rate = different_person.len() as f64 / n;
        let auto_assign_rate = auto_assign as f64 / total_decisions as f64;
        let review_rate = review as f64 / total_decisions as f64;

        eprintln!("PRODUCTION GATE CRITERIA:");
        eprintln!();
        eprintln!("  SAFETY METRICS:");
        eprintln!("    Agreement Rate:           {:.1}% (target: >90%)", agreement_rate * 100.0);
        eprintln!("    DifferentPerson Rate:      {:.1}% (target: <5%)", different_person_rate * 100.0);
        eprintln!("    Anti-Chain Pass Rate:     {:.1}% (target: >95%)",
            (anti_chain_stats.0 as f64 / (anti_chain_stats.0 + anti_chain_stats.1) as f64) * 100.0);
        eprintln!();
        eprintln!("  EFFECTIVENESS METRICS:");
        eprintln!("    Auto-Assign Rate:         {:.1}%", auto_assign_rate * 100.0);
        eprintln!("    Review Rate:              {:.1}%", review_rate * 100.0);
        eprintln!("    Shadow Aggressive Rate:   {:.1}% (Shadow found more than Legacy)",
            (shadow_more_aggressive as f64 / total_disagreements as f64) * 100.0);
        eprintln!();

        // Determine gate status
        let safety_pass = agreement_rate > 0.90 && different_person_rate < 0.05;
        let anti_chain_pass = anti_chain_stats.0 as f64 / (anti_chain_stats.0 + anti_chain_stats.1) as f64 > 0.95;

        eprintln!("GATE STATUS:");
        if safety_pass && anti_chain_pass {
            eprintln!("  >>> SHADOW CANDIDATE FOR PRODUCTION <<<");
            eprintln!("  - Safety metrics PASS");
            eprintln!("  - Anti-chaining PASS");
            if auto_assign_rate > 0.5 {
                eprintln!("  - High auto-assign rate ({:.1}%) - efficient", auto_assign_rate * 100.0);
            }
        } else {
            eprintln!("  >>> SHADOW NOT READY FOR PRODUCTION <<<");
            if !safety_pass {
                eprintln!("  - FAILED: Safety metrics");
                eprintln!("    DifferentPerson rate ({:.1}%) exceeds 5% threshold", different_person_rate * 100.0);
            }
            if !anti_chain_pass {
                eprintln!("  - FAILED: Anti-chaining protection");
            }
        }

        // =========================================================================
        // Final Summary
        // =========================================================================
        eprintln!("\n========== Phase 33: FINAL SUMMARY ==========\n");
        eprintln!("Dataset: {} images, {} faces, {} LFW persons", img_count, total_faces, person_count);
        eprintln!("Processing time: {:.1}s", cluster_time.as_secs_f32());
        eprintln!();
        eprintln!("KEY FINDINGS:");
        eprintln!("  - Agreement rate: {:.1}%", agreement_rate * 100.0);
        eprintln!("  - Shadow aggressiveness: {:.1}% of disagreements",
            (shadow_more_aggressive as f64 / total_disagreements.max(1) as f64) * 100.0);
        eprintln!("  - DifferentPerson conflicts: {}", different_person.len());
        eprintln!();
        eprintln!("RECOMMENDATION:");
        if safety_pass && anti_chain_pass {
            eprintln!("  PROCEED to Phase 34: Threshold Calibration");
            eprintln!("  Shadow pipeline shows promise for production use.");
        } else {
            eprintln!("  DO NOT PROCEED - address safety issues first.");
            eprintln!("  Focus on reducing DifferentPerson rate before production.");
        }

        // Cleanup
        fs::remove_dir_all(&tmp).ok();
    }
}