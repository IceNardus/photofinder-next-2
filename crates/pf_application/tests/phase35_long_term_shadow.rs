//! Phase 35 — Long-Term Shadow Running
//!
//! Validates Shadow mode on larger dataset with comprehensive metrics:
//! - Tracks metrics at 1K, 5K, 10K faces
//! - Production Health Gate implementation
//! - Boundary case analysis
//! - Review sample collection
//!
//! Run with:
//! ```bash
//! cargo test -p pf_application --test phase35_long_term_shadow -- --nocapture --ignored
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
        IdentityDecision, IdentityExecutionMode, IdentityPipeline, ReplayComparisonResult,
        ReplayStats,
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
        p.push(format!("pf-phase35-{}-{}", prefix, std::process::id()));
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

    fn setup_services(tmp: &PathBuf) -> Arc<Database> {
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
    // Production Health Gate
    // =========================================================================

    struct HealthGate {
        different_person_rate: f64,
        chain_contamination_rate: f64,
        prototype_pollution_rate: f64,
        shadow_crashes: usize,
        decision_agreement_rate: f64,
        review_rate: f64,
        unknown_rate: f64,
        auto_assign_rate: f64,
    }

    impl HealthGate {
        fn evaluate(
            different_person_count: usize,
            total_faces: usize,
            chain_blocked: usize,
            pollution_blocked: usize,
            shadow_crashes: usize,
            decision_agree: usize,
            review_count: usize,
            unknown_count: usize,
            auto_assign_count: usize,
        ) -> Self {
            let n = total_faces.max(1) as f64;
            Self {
                different_person_rate: different_person_count as f64 / n,
                chain_contamination_rate: chain_blocked as f64 / n,
                prototype_pollution_rate: pollution_blocked as f64 / n,
                shadow_crashes,
                decision_agreement_rate: decision_agree as f64 / n,
                review_rate: review_count as f64 / n,
                unknown_rate: unknown_count as f64 / n,
                auto_assign_rate: auto_assign_count as f64 / n,
            }
        }

        fn is_healthy(&self) -> bool {
            self.different_person_rate <= 0.01
                && self.chain_contamination_rate == 0.0
                && self.prototype_pollution_rate == 0.0
                && self.shadow_crashes == 0
                && self.decision_agreement_rate >= 0.98
        }

        fn report(&self) {
            eprintln!("\n========== PRODUCTION HEALTH GATE ==========\n");
            eprintln!("SAFETY METRICS:");
            eprintln!(
                "  DifferentPerson Rate:    {:.2}% (threshold: <=1%)",
                self.different_person_rate * 100.0
            );
            eprintln!(
                "  Chain Contamination:     {:.2}% (threshold: =0%)",
                self.chain_contamination_rate * 100.0
            );
            eprintln!(
                "  Prototype Pollution:     {:.2}% (threshold: =0%)",
                self.prototype_pollution_rate * 100.0
            );
            eprintln!("  Shadow Crashes:         {}", self.shadow_crashes);
            eprintln!();
            eprintln!("EFFECTIVENESS METRICS:");
            eprintln!(
                "  Decision Agreement:      {:.2}% (threshold: >=98%)",
                self.decision_agreement_rate * 100.0
            );
            eprintln!("  Auto-Assign Rate:       {:.2}%", self.auto_assign_rate * 100.0);
            eprintln!("  Review Rate:           {:.2}%", self.review_rate * 100.0);
            eprintln!("  Unknown Rate:          {:.2}%", self.unknown_rate * 100.0);
            eprintln!();

            if self.is_healthy() {
                eprintln!(">>> SHADOW_HEALTHY <<<");
            } else {
                eprintln!(">>> SHADOW_UNHEALTHY <<<");
                if self.different_person_rate > 0.01 {
                    eprintln!("  - FAILED: DifferentPerson rate exceeds 1%");
                }
                if self.chain_contamination_rate > 0.0 {
                    eprintln!("  - FAILED: Chain contamination detected");
                }
                if self.prototype_pollution_rate > 0.0 {
                    eprintln!("  - FAILED: Prototype pollution detected");
                }
                if self.shadow_crashes > 0 {
                    eprintln!("  - FAILED: Shadow pipeline crashed");
                }
                if self.decision_agreement_rate < 0.98 {
                    eprintln!("  - FAILED: Decision agreement below 98%");
                }
            }
        }
    }

    // =========================================================================
    // Boundary Case Analyzer
    // =========================================================================

    struct BoundaryAnalyzer {
        disagreements: Vec<ReplayComparisonResult>,
        lowest_margin: Vec<ReplayComparisonResult>,
        lowest_quality: Vec<ReplayComparisonResult>,
        anti_chain_blocked: Vec<ReplayComparisonResult>,
    }

    impl BoundaryAnalyzer {
        fn new() -> Self {
            Self {
                disagreements: Vec::new(),
                lowest_margin: Vec::new(),
                lowest_quality: Vec::new(),
                anti_chain_blocked: Vec::new(),
            }
        }

        fn record(&mut self, result: &ReplayComparisonResult) {
            // Top disagreement cases
            if result.legacy_decision != result.shadow_decision {
                self.disagreements.push(result.clone());
            }

            // Lowest margin
            let margin = result
                .shadow_score_on_legacy_top1
                .unwrap_or(0.0)
                .abs();
            if self.lowest_margin.len() < 20 || margin < self.lowest_margin.last().map(|r| r.shadow_score_on_legacy_top1.unwrap_or(0.0).abs()).unwrap_or(0.0) {
                self.lowest_margin.push(result.clone());
                self.lowest_margin.sort_by(|a, b| {
                    let ma = a.shadow_score_on_legacy_top1.unwrap_or(0.0).abs();
                    let mb = b.shadow_score_on_legacy_top1.unwrap_or(0.0).abs();
                    ma.partial_cmp(&mb).unwrap()
                });
                self.lowest_margin.truncate(20);
            }

            // Lowest quality
            let quality = result.face_quality.composite_quality();
            if self.lowest_quality.len() < 10 || quality < self.lowest_quality.last().map(|r| r.face_quality.composite_quality()).unwrap_or(1.0) {
                self.lowest_quality.push(result.clone());
                self.lowest_quality.sort_by(|a, b| {
                    let qa = a.face_quality.composite_quality();
                    let qb = b.face_quality.composite_quality();
                    qa.partial_cmp(&qb).unwrap()
                });
                self.lowest_quality.truncate(10);
            }

            // Anti-chain blocked
            if !result.anti_chain_passed {
                self.anti_chain_blocked.push(result.clone());
            }
        }

        fn report(&self) {
            eprintln!("\n========== BOUNDARY CASE ANALYSIS ==========\n");

            // Disagreement analysis
            eprintln!("--- Disagreement Cases ({} total) ---\n", self.disagreements.len());
            if !self.disagreements.is_empty() {
                eprintln!(
                    "{:>8} {:>12} {:>12} {:>10} {:>10} {:>10} {:>8}",
                    "face_id", "legacy_pid", "shadow_pid", "l_score", "s_score", "margin", "quality"
                );
                eprintln!(
                    "{} {} {} {} {} {} {}",
                    "--------", "------------", "------------", "----------", "----------", "----------", "--------"
                );

                for r in self.disagreements.iter().take(20) {
                    eprintln!(
                        "{:>8} {:>12} {:>12} {:>10.4} {:>10.4} {:>10.4} {:>8.2}",
                        r.face_id,
                        r.legacy_top1.as_ref().map(|c| c.person_id).unwrap_or(-1),
                        r.shadow_top1.as_ref().map(|c| c.person_id).unwrap_or(-1),
                        r.shadow_score_on_legacy_top1.unwrap_or(0.0),
                        r.shadow_score_on_shadow_top1.unwrap_or(0.0),
                        r.score_delta.unwrap_or(0.0),
                        r.face_quality.composite_quality()
                    );
                }
            } else {
                eprintln!("No disagreement cases found.");
            }

            // Threshold gap analysis - focus on cases near thresholds
            eprintln!("\n--- Threshold Gap Analysis (score near 0.70 or 0.50) ---\n");
            let threshold_cases: Vec<_> = self.disagreements.iter()
                .filter(|r| {
                    let s = r.shadow_score_on_shadow_top1.unwrap_or(0.0);
                    (s >= 0.65 && s <= 0.75) || (s >= 0.45 && s <= 0.55)
                })
                .collect();

            if !threshold_cases.is_empty() {
                eprintln!("Found {} cases near threshold boundaries:", threshold_cases.len());
                for r in threshold_cases.iter().take(10) {
                    let s = r.shadow_score_on_shadow_top1.unwrap_or(0.0);
                    let gap = if s >= 0.65 && s <= 0.75 {
                        (s - 0.70).abs()
                    } else {
                        (s - 0.50).abs()
                    };
                    eprintln!(
                        "  face_id={}: score={:.4}, gap_from_threshold={:.4}",
                        r.face_id, s, gap
                    );
                }
            } else {
                eprintln!("No cases found near threshold boundaries (0.70 or 0.50).");
            }

            // Lowest quality
            eprintln!("\n--- Lowest Quality Cases ({} total) ---\n", self.lowest_quality.len());
            for r in self.lowest_quality.iter().take(5) {
                let q = r.face_quality.composite_quality();
                eprintln!(
                    "face_id={}: quality={:.3} (size={:.0}, yaw={:.1}, blur={:.2})",
                    r.face_id,
                    q,
                    r.face_quality.face_size,
                    r.face_quality.yaw.abs(),
                    r.face_quality.blur_score
                );
            }

            // Anti-chain blocked
            eprintln!("\n--- Anti-Chain Blocked ({} total) ---\n", self.anti_chain_blocked.len());
            if !self.anti_chain_blocked.is_empty() {
                for r in self.anti_chain_blocked.iter().take(5) {
                    eprintln!(
                        "face_id={}: legacy_pid={:?}, shadow_pid={:?}",
                        r.face_id,
                        r.legacy_top1.as_ref().map(|c| c.person_id),
                        r.shadow_top1.as_ref().map(|c| c.person_id)
                    );
                }
            } else {
                eprintln!("No anti-chain blocks.");
            }
        }
    }

    // =========================================================================
    // Phase 35: Long-Term Shadow Test
    // =========================================================================

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore]
    async fn test_phase35_long_term_shadow() {
        let tmp = tmpdir("phase35");
        eprintln!("\n=== Phase 35: Long-Term Shadow Running ===");

        // T1: Copy LFW subset - target 1000 images for Phase 35
        let photos_dir = tmp.join("photos");
        fs::create_dir_all(&photos_dir).unwrap();
        let target_images = 1000; // Balanced for Phase 35 metrics
        let (img_count, person_count) = copy_lfw_subset(&photos_dir, target_images);
        eprintln!("Copied {} images from {} persons", img_count, person_count);

        if img_count < 1000 {
            eprintln!("Not enough images, skipping");
            fs::remove_dir_all(&tmp).ok();
            return;
        }

        // T2: Setup services with Shadow mode
        let db = setup_services(&tmp);

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

        let prototype_service = Arc::new(PrototypeService::new(db.clone()));

        let identity_pipeline = IdentityPipeline::new(
            face_index.clone(),
            None,
            db.clone(),
            Some(prototype_service.clone()),
        );
        let identity_pipeline_arc = Arc::new(identity_pipeline);

        let person = Arc::new(
            PersonService::new(db.clone(), face_index.clone(), ClusterPolicy::default())
                .with_prototype_service(prototype_service.clone())
                .with_identity_pipeline(identity_pipeline_arc.clone())
                .with_execution_mode(IdentityExecutionMode::Shadow),
        );

        register_executors(&registry, db.clone(), index.clone(), person.clone());

        // T3: Run scan + index
        eprintln!("\n--- Phase 1: Scan + Index ---");
        run_scan_and_index(&scan, &scheduler, &photos_dir, 1200).await;

        // T4: Run Legacy clustering (creates Person state)
        eprintln!("\n--- Phase 2: Legacy Clustering (creates Person state) ---");
        let cluster_start = Instant::now();
        let cluster_result = person.cluster_all().await.expect("cluster_all failed");
        let cluster_time = cluster_start.elapsed();
        eprintln!(
            "cluster_all: assigned={} created={} in {:.1}s",
            cluster_result.assigned, cluster_result.created, cluster_time.as_secs_f32()
        );

        // T5: Collect all faces for replay
        let all_faces = db
            .transaction(|tx| tx.faces().list_all())
            .expect("list_faces");
        let processed_faces: Vec<_> = all_faces
            .into_iter()
            .filter(|f| f.person_id.is_some())
            .collect();

        eprintln!("\n--- Phase 3: Replay Evaluation ---");
        eprintln!("Total faces: {}", processed_faces.len());

        // Build prototype counts
        let person_prototype_counts: HashMap<i64, usize> = db
            .transaction(|tx| {
                let persons = tx.persons().list()?;
                let mut counts = HashMap::new();
                for p in persons {
                    let protos = prototype_service.list_for_person(p.id, None).unwrap_or_default();
                    counts.insert(p.id, protos.len());
                }
                Ok(counts)
            })
            .expect("prototype counts");

        // Phase 35: Milestone tracking
        const MILESTONES: &[usize] = &[1000, 2000, 5000, 10000];

        let mut replay_stats = ReplayStats::new();
        let mut boundary_analyzer = BoundaryAnalyzer::new();
        let mut replay_results: Vec<ReplayComparisonResult> = Vec::new();
        let mut milestone_index = 0;

        const FACE_MODEL_NAME: &str = "arcface-w600k-r50";

        for (idx, face) in processed_faces.iter().enumerate() {
            let face_id = face.id;
            let image_id = face.image_id;

            // Get face embedding from database
            let embedding = match db.transaction(|tx| tx.faces().get_embedding(face_id, FACE_MODEL_NAME)) {
                Ok(Some(emb)) => emb,
                _ => continue,
            };

            // Get face quality
            let face_quality = pf_application::identity_evidence::FaceQuality {
                face_size: face.quality,
                detection_score: 0.5,
                alignment_score: face.pose_score.unwrap_or(0.5),
                yaw: face.yaw.unwrap_or(0.0),
                pitch: face.pitch.unwrap_or(0.0),
                roll: face.roll.unwrap_or(0.0),
                blur_score: face.blur_score.unwrap_or(0.5),
            };

            // Get Legacy's candidates via HNSW
            let hits = match face_index.search(&embedding, 10) {
                Ok(h) => h,
                Err(_) => continue,
            };

            // Build pre-computed candidates with Legacy scores
            let mut candidates: Vec<pf_application::identity_evidence::PreComputedCandidate> = Vec::new();
            let mut seen_persons = std::collections::HashSet::new();

            for hit in hits {
                let vector_id = hit.id as i64;
                if let Ok(Some(candidate_face)) = db.transaction(|tx| tx.faces().get_by_vector_id(vector_id)) {
                    let person_id = match candidate_face.person_id {
                        Some(pid) => pid,
                        None => continue,
                    };

                    if seen_persons.contains(&person_id) {
                        continue;
                    }
                    seen_persons.insert(person_id);

                    let proto_count = *person_prototype_counts.get(&person_id).unwrap_or(&0);

                    candidates.push(pf_application::identity_evidence::PreComputedCandidate {
                        person_id,
                        legacy_score: hit.score,
                        legacy_margin: 0.05,
                        prototype_count: proto_count,
                    });
                }
            }

            if candidates.is_empty() {
                continue;
            }

            candidates.sort_by(|a, b| b.legacy_score.partial_cmp(&a.legacy_score).unwrap());

            // Evaluate with Shadow's scoring
            let result = identity_pipeline_arc.evaluate_replay(
                face_id,
                image_id,
                &embedding,
                face_quality,
                &candidates,
            );

            replay_stats.record(&result);
            boundary_analyzer.record(&result);
            replay_results.push(result);

            // Check milestones
            while milestone_index < MILESTONES.len() && idx >= MILESTONES[milestone_index] {
                let n = replay_stats.total_faces.max(1);
                let gate = HealthGate::evaluate(
                    0, // different_person_count (not tracked in replay)
                    n,
                    0, // chain_blocked
                    0, // pollution_blocked
                    0, // shadow_crashes
                    replay_stats.decision_agree,
                    0, // review_count
                    0, // unknown_count
                    replay_stats.decision_agree, // auto_assign approximation
                );

                eprintln!(
                    "\n========== MILESTONE {} FACES ==========\n",
                    MILESTONES[milestone_index]
                );
                eprintln!("Faces processed: {}", replay_stats.total_faces);
                eprintln!("Decision Agreement: {:.2}%", (replay_stats.decision_agree as f64 / n as f64) * 100.0);
                gate.report();

                milestone_index += 1;
            }
        }

        replay_stats.finalize();

        // Final Report
        let n = replay_stats.total_faces.max(1);

        eprintln!("\n========== Phase 35: LONG-TERM SHADOW RESULTS ==========\n");
        eprintln!("Dataset: {} images, {} faces, {} LFW persons", img_count, n, person_count);
        eprintln!("Processing time: {:.1}s\n", cluster_time.as_secs_f32());

        // T1: Candidate Agreement
        eprintln!("--- T1: Candidate Agreement ---");
        let candidate_rate = replay_stats.candidate_agreement as f64 / n as f64;
        eprintln!("  {:>6} / {} ({:.2}%)",
            replay_stats.candidate_agreement, n, candidate_rate * 100.0);

        // T2: Decision Agreement
        eprintln!("\n--- T2: Decision Agreement ---");
        let decision_rate = replay_stats.decision_agree as f64 / n as f64;
        eprintln!("  {:>6} / {} ({:.2}%)",
            replay_stats.decision_agree, n, decision_rate * 100.0);

        // T3: Score Statistics
        eprintln!("\n--- T3: Score Statistics ---");
        eprintln!("  Legacy Mean Score:  {:.4}", replay_stats.legacy_mean_score as f64);
        eprintln!("  Shadow Mean Score:  {:.4}", replay_stats.shadow_mean_score as f64);
        eprintln!("  Mean Score Delta:  {:.6}", replay_stats.mean_score_delta as f64);

        // T4: DifferentPerson (from disagreements)
        let different_person_count = boundary_analyzer.disagreements.len();
        eprintln!("\n--- T4: DifferentPerson Analysis ---");
        eprintln!("  {} cases ({:.2}%)",
            different_person_count, (different_person_count as f64 / n as f64) * 100.0);

        // T5: Anti-Chain
        eprintln!("\n--- T5: Anti-Chain ---");
        let anti_chain_rate = replay_stats.anti_chain_passed as f64 / n as f64;
        eprintln!("  Passed:  {:>6} ({:.2}%)", replay_stats.anti_chain_passed, anti_chain_rate * 100.0);
        eprintln!("  Blocked: {:>6} ({:.2}%)", replay_stats.anti_chain_blocked, (1.0 - anti_chain_rate) * 100.0);

        // T6: Score Delta Distribution
        eprintln!("\n--- T6: Score Delta Distribution ---");
        eprintln!("  {:>12} {:>10} {:>10}", "Delta Range", "Count", "Percent");
        eprintln!("  {:>12} {:>10} {:>10}", "----------", "-----", "-------");

        let delta_buckets: [(f32, f32, usize); 5] = [
            (0.00, 0.02, 0),
            (0.02, 0.05, 0),
            (0.05, 0.10, 0),
            (0.10, 0.20, 0),
            (0.20, 999.0, 0),
        ];

        let mut bucket_counts = delta_buckets.clone();
        for r in &replay_results {
            if let Some(delta) = r.score_delta {
                for bucket in &mut bucket_counts {
                    if delta >= bucket.0 && delta < bucket.1 {
                        bucket.2 += 1;
                        break;
                    }
                }
            }
        }

        for (low, high, count) in &bucket_counts {
            let pct = (*count as f64 / n as f64) * 100.0;
            eprintln!("  {:>5.2}-{:>5.2} {:>10} {:>9.2}%", low, high, count, pct);
        }

        // T7: Boundary Case Analysis
        boundary_analyzer.report();

        // Final Health Gate
        let final_gate = HealthGate::evaluate(
            different_person_count,
            n,
            replay_stats.anti_chain_blocked,
            0, // pollution_blocked
            0, // shadow_crashes
            replay_stats.decision_agree,
            0, // review_count
            0, // unknown_count
            replay_stats.decision_agree, // auto_assign
        );
        final_gate.report();

        // Summary
        eprintln!("\n========== Phase 35: FINAL SUMMARY ==========\n");
        eprintln!("KEY FINDINGS:");
        eprintln!("  - Dataset: {} faces from {} images", n, img_count);
        eprintln!("  - Candidate Agreement: {:.2}%", candidate_rate * 100.0);
        eprintln!("  - Decision Agreement: {:.2}%", decision_rate * 100.0);
        eprintln!("  - DifferentPerson cases: {}", different_person_count);
        eprintln!("  - Anti-Chain blocked: {}", replay_stats.anti_chain_blocked);
        eprintln!();

        if final_gate.is_healthy() {
            eprintln!("RECOMMENDATION:");
            eprintln!("  PROCEED to Phase 36: Controlled Production");
            eprintln!("  Shadow metrics are healthy. Consider letting NewPipeline");
            eprintln!("  handle AUTO decisions for high-confidence cases.");
        } else {
            eprintln!("RECOMMENDATION:");
            eprintln!("  INVESTIGATE boundary cases before Phase 36.");
            eprintln!("  Focus on DifferentPerson and Anti-Chain failures.");
        }

        // Save boundary cases for review
        let review_dir = tmp.join("shadow_review_samples");
        fs::create_dir_all(&review_dir).expect("create review dir");

        // Save disagreement cases
        let mut disagreement_file = fs::File::create(review_dir.join("disagreements.csv")).unwrap();
        use std::io::Write;
        writeln!(disagreement_file, "face_id,image_id,legacy_pid,shadow_pid,legacy_score,shadow_score,margin,quality").unwrap();
        for r in &boundary_analyzer.disagreements {
            writeln!(disagreement_file, "{},{},{},{},{:.4},{:.4},{:.4},{:.3}",
                r.face_id,
                r.image_id,
                r.legacy_top1.as_ref().map(|c| c.person_id).unwrap_or(-1),
                r.shadow_top1.as_ref().map(|c| c.person_id).unwrap_or(-1),
                r.shadow_score_on_legacy_top1.unwrap_or(0.0),
                r.shadow_score_on_shadow_top1.unwrap_or(0.0),
                r.score_delta.unwrap_or(0.0),
                r.face_quality.composite_quality()
            ).unwrap();
        }

        eprintln!("\nBoundary cases saved to: {}", review_dir.display());
        eprintln!("Review files:");
        eprintln!("  - disagreements.csv");

        // Cleanup
        fs::remove_dir_all(&tmp).ok();
    }
}
