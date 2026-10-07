//! Phase 34 — Candidate-Set Replay Test
//!
//! Core insight from Phase 33:
//! The previous "Shadow vs Legacy" comparison was INVALID because:
//! - Legacy maintains its own Person/Prototype state
//! - Shadow creates its OWN Person/Prototype state independently
//! - They were comparing two DIFFERENT "worlds"
//!
//! Phase 34 Solution:
//! 1. Run Legacy clustering first -> creates Person/Prototype state
//! 2. FREEZE that state
//! 3. For each face, use the SAME candidate set
//! 4. Score with BOTH Legacy and Shadow algorithms
//! 5. Compare decisions on EQUAL footing
//!
//! This allows us to answer:
//! "In the EXACT same Person/Prototype/Candidate state,
//!  does Shadow score/decision differ from Legacy?"
//!
//! Run with:
//! ```bash
//! cargo test -p pf_application --test phase34_candidate_replay -- --nocapture --ignored
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
        IdentityExecutionMode, IdentityPipeline, IdentityQuery, PreComputedCandidate,
        ReplayComparisonResult, ReplayStats,
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
        p.push(format!("pf-phase34-{}-{}", prefix, std::process::id()));
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
    // Phase 34: Candidate-Set Replay Test
    // =========================================================================

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore]
    async fn test_phase34_candidate_replay() {
        let tmp = tmpdir("phase34");
        eprintln!("\n=== Phase 34: Candidate-Set Replay Test ===");
        eprintln!("\nKEY INSIGHT from Phase 33:");
        eprintln!("  Legacy and Shadow maintain DIFFERENT Person states");
        eprintln!("  -> Comparison was INVALID");
        eprintln!("\nPhase 34 Solution:");
        eprintln!("  1. Run Legacy clustering first -> creates Person state");
        eprintln!("  2. FREEZE that state");
        eprintln!("  3. Use SAME candidate set for both");
        eprintln!("  4. Compare decisions on EQUAL footing\n");

        // T1: Copy LFW subset
        let photos_dir = tmp.join("photos");
        fs::create_dir_all(&photos_dir).unwrap();
        let target_images = 500;
        let (img_count, person_count) = copy_lfw_subset(&photos_dir, target_images);
        eprintln!("Copied {} images from {} persons", img_count, person_count);

        if img_count < 100 {
            eprintln!("Not enough images, skipping");
            fs::remove_dir_all(&tmp).ok();
            return;
        }

        // T2: Setup services
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
        run_scan_and_index(&scan, &scheduler, &photos_dir, 600).await;

        // T4: Run Legacy clustering (creates Person state)
        eprintln!("\n--- Phase 2: Legacy Clustering (creates Person state) ---");
        let cluster_start = Instant::now();
        let cluster_result = person.cluster_all().await.expect("cluster_all failed");
        let cluster_time = cluster_start.elapsed();
        eprintln!(
            "cluster_all: assigned={} created={} in {:.1}s",
            cluster_result.assigned, cluster_result.created, cluster_time.as_secs_f32()
        );

        // T5: Freeze Person state - get all persons and their prototypes
        eprintln!("\n--- Phase 3: Freeze Person State ---");
        let persons = person.list_all().expect("list persons");
        eprintln!("Total persons: {}", persons.len());

        // Build candidate map: person_id -> prototype_count
        let mut person_prototype_counts: HashMap<i64, usize> = HashMap::new();
        for p in &persons {
            let count = prototype_service
                .count_active_for_person(p.id)
                .unwrap_or(0) as usize;
            person_prototype_counts.insert(p.id, count);
        }

        // T6: For each face, get its embedding and build candidate set
        eprintln!("\n--- Phase 4: Build Candidate Sets for Each Face ---");

        // Get all indexed faces with embeddings
        let faces = db
            .transaction(|tx| tx.faces().list_all())
            .expect("list_all faces");

        eprintln!("Total faces: {}", faces.len());

        // Filter to only faces that have been assigned (have person_id)
        let processed_faces: Vec<_> = faces
            .into_iter()
            .filter(|f| f.person_id.is_some())
            .collect();

        eprintln!("Processed faces: {}", processed_faces.len());

        // T7: Replay evaluation - for each face:
        // - Get Legacy's candidates (from HNSW)
        // - Score with Shadow
        // - Compare decisions
        eprintln!("\n--- Phase 5: Replay Evaluation ---");

        let mut replay_stats = ReplayStats::new();
        let mut replay_results: Vec<ReplayComparisonResult> = Vec::new();

        const FACE_MODEL_NAME: &str = "arcface-w600k-r50";

        for face in &processed_faces {
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
            let mut candidates: Vec<PreComputedCandidate> = Vec::new();
            let mut seen_persons = std::collections::HashSet::new();

            for hit in hits {
                let vector_id = hit.id as i64;
                if let Ok(Some(candidate_face)) = db.transaction(|tx| tx.faces().get_by_vector_id(vector_id)) {
                    let person_id = match candidate_face.person_id {
                        Some(pid) => pid,
                        None => continue,
                    };

                    // Skip if same person already added
                    if seen_persons.contains(&person_id) {
                        continue;
                    }
                    seen_persons.insert(person_id);

                    // Get prototype count for this person
                    let proto_count = *person_prototype_counts.get(&person_id).unwrap_or(&0);

                    candidates.push(PreComputedCandidate {
                        person_id,
                        legacy_score: hit.score,
                        legacy_margin: 0.05, // Placeholder
                        prototype_count: proto_count,
                    });
                }
            }

            if candidates.is_empty() {
                continue;
            }

            // Sort candidates by legacy score (descending) - Legacy's order
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
            replay_results.push(result);
        }

        replay_stats.finalize();

        // =========================================================================
        // T8: Report Results
        // =========================================================================
        eprintln!("\n========== Phase 34: CANDIDATE-SET REPLAY RESULTS ==========\n");
        eprintln!("Dataset: {} images, {} faces, {} LFW persons", img_count, processed_faces.len(), person_count);
        eprintln!("Processing time: {:.1}s", cluster_time.as_secs_f32());

        // T9: Candidate Agreement
        eprintln!("\n--- T1: Candidate Agreement ---");
        let n = replay_stats.total_faces.max(1);
        eprintln!("Candidate Agreement (Legacy Top1 == Shadow Top1):");
        eprintln!("  {:>4} / {} ({:.1}%)",
            replay_stats.candidate_agreement, n,
            (replay_stats.candidate_agreement as f64 / n as f64) * 100.0);

        // T10: Decision Agreement
        eprintln!("\n--- T2: Decision Agreement ---");
        eprintln!("Decision Agreement:");
        eprintln!("  {:>4} / {} ({:.1}%)",
            replay_stats.decision_agree, n,
            (replay_stats.decision_agree as f64 / n as f64) * 100.0);

        // T11: Score Statistics
        eprintln!("\n--- T3: Score Statistics ---");
        eprintln!("Legacy Mean Score: {:.4}", replay_stats.legacy_mean_score);
        eprintln!("Shadow Mean Score:  {:.4}", replay_stats.shadow_mean_score);
        eprintln!("Mean Score Delta:  {:.4}", replay_stats.mean_score_delta);

        // T12: Anti-Chain
        eprintln!("\n--- T4: Anti-Chain ---");
        let total_anti = replay_stats.anti_chain_passed + replay_stats.anti_chain_blocked;
        eprintln!("Anti-Chain Passed: {:>4} ({:.1}%)",
            replay_stats.anti_chain_passed,
            if total_anti > 0 { (replay_stats.anti_chain_passed as f64 / total_anti as f64) * 100.0 } else { 0.0 });
        eprintln!("Anti-Chain Blocked: {:>4} ({:.1}%)",
            replay_stats.anti_chain_blocked,
            if total_anti > 0 { (replay_stats.anti_chain_blocked as f64 / total_anti as f64) * 100.0 } else { 0.0 });

        // T13: Detailed comparison - look at score deltas
        eprintln!("\n--- T5: Score Delta Distribution ---");
        let mut delta_buckets: [(f32, f32, usize); 5] = [
            (0.0, 0.02, 0),
            (0.02, 0.05, 0),
            (0.05, 0.10, 0),
            (0.10, 0.20, 0),
            (0.20, 999.0, 0),
        ];

        for result in &replay_results {
            if let Some(delta) = result.score_delta {
                for bucket in &mut delta_buckets {
                    if delta >= bucket.0 && delta < bucket.1 {
                        bucket.2 += 1;
                        break;
                    }
                }
            }
        }

        eprintln!("Score Delta Distribution:");
        eprintln!("{:>12} {:>8} {:>10}", "Delta", "Count", "Percent");
        eprintln!("{:>12} {:>8} {:>10}", "------", "-----", "-------");
        for (low, high, count) in &delta_buckets {
            let pct = (*count as f64 / n as f64) * 100.0;
            eprintln!("{:.2}-{:.2}  {:>8} {:>9.1}%", low, high, count, pct);
        }

        // T14: Different Person Analysis
        eprintln!("\n--- T6: DifferentPerson Analysis (Top 20 by delta) ---");
        let mut different_person_results: Vec<_> = replay_results
            .iter()
            .filter(|r| {
                let legacy_pid = r.legacy_top1.as_ref().map(|c| c.person_id);
                let shadow_pid = r.shadow_top1.as_ref().map(|c| c.person_id);
                legacy_pid != shadow_pid && legacy_pid.is_some() && shadow_pid.is_some()
            })
            .collect();

        different_person_results.sort_by(|a, b| {
            let da = a.score_delta.unwrap_or(999.0);
            let db = b.score_delta.unwrap_or(999.0);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });

        eprintln!("Total DifferentPerson cases: {}", different_person_results.len());
        eprintln!("\nTop 20 (sorted by delta ascending - most problematic first):");
        eprintln!("{:>6} {:>10} {:>10} {:>10} {:>10} {:>8}",
            "face_id", "legacy_pid", "shadow_pid", "l_score", "s_score", "delta");
        eprintln!("{:>6} {:>10} {:>10} {:>10} {:>10} {:>8}",
            "------", "---------", "---------", "-------", "-------", "-----");

        for result in different_person_results.iter().take(20) {
            eprintln!(
                "{:>6} {:>10} {:>10} {:>10.4} {:>10.4} {:>8.4}",
                result.face_id,
                result.legacy_top1.as_ref().map(|c| c.person_id).unwrap_or(-1),
                result.shadow_top1.as_ref().map(|c| c.person_id).unwrap_or(-1),
                result.shadow_score_on_legacy_top1.unwrap_or(0.0),
                result.shadow_score_on_shadow_top1.unwrap_or(0.0),
                result.score_delta.unwrap_or(0.0)
            );
        }

        // T15: Final Gate Evaluation
        eprintln!("\n========== Phase 34: FINAL GATE ==========\n");

        let candidate_agreement_rate = replay_stats.candidate_agreement as f64 / n as f64;
        let decision_agreement_rate = replay_stats.decision_agree as f64 / n as f64;
        let different_person_count = different_person_results.len();

        eprintln!("CANDIDATE-SET REPLAY GATE:");
        eprintln!();
        eprintln!("  Candidate Agreement:  {:.1}% (target: >80%)", candidate_agreement_rate * 100.0);
        eprintln!("  Decision Agreement:  {:.1}% (target: >90%)", decision_agreement_rate * 100.0);
        eprintln!("  DifferentPerson:     {} ({:.1}%)", different_person_count, (different_person_count as f64 / n as f64) * 100.0);
        eprintln!();

        // Gate determination
        let candidate_agree_pass = candidate_agreement_rate > 0.80;
        let decision_agree_pass = decision_agreement_rate > 0.90;
        let different_person_pass = (different_person_count as f64 / n as f64) < 0.05;

        if candidate_agree_pass && decision_agree_pass && different_person_pass {
            eprintln!("  >>> SHADOW CANDIDATE FOR PRODUCTION <<<");
            eprintln!("  - Candidate agreement PASS (>80%)");
            eprintln!("  - Decision agreement PASS (>90%)");
            eprintln!("  - DifferentPerson rate acceptable (<5%)");
        } else {
            eprintln!("  >>> SHADOW NEEDS IMPROVEMENT <<<");
            if !candidate_agree_pass {
                eprintln!("  - FAILED: Candidate agreement ({:.1}%) < 80%", candidate_agreement_rate * 100.0);
            }
            if !decision_agree_pass {
                eprintln!("  - FAILED: Decision agreement ({:.1}%) < 90%", decision_agreement_rate * 100.0);
            }
            if !different_person_pass {
                eprintln!("  - FAILED: DifferentPerson rate ({:.1}%) >= 5%",
                    (different_person_count as f64 / n as f64) * 100.0);
            }
        }

        // =========================================================================
        // Final Summary
        // =========================================================================
        eprintln!("\n========== Phase 34: FINAL SUMMARY ==========\n");
        eprintln!("KEY FINDINGS:");
        eprintln!("  - Candidate Agreement: {:.1}%", candidate_agreement_rate * 100.0);
        eprintln!("  - Decision Agreement: {:.1}%", decision_agreement_rate * 100.0);
        eprintln!("  - DifferentPerson cases: {}", different_person_count);
        eprintln!();
        eprintln!("INTERPRETATION:");
        if candidate_agreement_rate > 0.90 {
            eprintln!("  EXCELLENT: Shadow selects the same top candidate as Legacy");
        } else if candidate_agreement_rate > 0.70 {
            eprintln!("  GOOD: Shadow and Legacy largely agree on candidates");
        } else {
            eprintln!("  NEEDS WORK: Significant candidate disagreement");
        }
        eprintln!();
        eprintln!("RECOMMENDATION:");
        if candidate_agree_pass && decision_agree_pass {
            eprintln!("  PROCEED to Phase 35: Long-term Shadow Running");
        } else {
            eprintln!("  INVESTIGATE differences before proceeding");
            eprintln!("  Focus on high-delta DifferentPerson cases");
        }

        // Cleanup
        fs::remove_dir_all(&tmp).ok();
    }
}
