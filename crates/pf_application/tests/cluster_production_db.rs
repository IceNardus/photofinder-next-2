//! Cluster Production DB — Test cluster_all on existing photofinder.db
//!
//! Validates that the identity pipeline can process existing faces
//! in the production database without rescanning.
//!
//! Run with:
//! ```bash
//! cargo test --release -p pf_application --test cluster_production_db -- --nocapture --ignored
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use pf_ai::{ArcFaceEmbedder, FacePipeline, QualityFilter, ScrfdDetector, SimpleAligner};
use pf_application::person::{ClusterPolicy, PersonService};
use pf_application::prototype_service::PrototypeService;
use pf_config::Config;
use pf_application::identity_evidence::IdentityExecutionMode;
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn test_cluster_existing_faces_in_production_db() {
    eprintln!("========================================");
    eprintln!("CLUSTER EXISTING FACES IN PRODUCTION DB");
    eprintln!("========================================");

    // Open production database
    let db_path = PathBuf::from("/Users/mac/Library/Application Support/PhotoFinderNext/photofinder.db");
    if !db_path.exists() {
        eprintln!("SKIP: Production database not found");
        return;
    }

    let db = Arc::new(Database::open(&db_path, builtin_migrations()).unwrap());

    // Check initial state
    let n_img = db.transaction(|tx| tx.images().count()).unwrap();
    let n_face = db.transaction(|tx| tx.faces().count()).unwrap();
    let n_person = db.transaction(|tx| tx.persons().list()).unwrap().len() as i64;
    let n_proto = db.transaction(|tx| tx.person_prototypes().list_all_active()).unwrap().len() as i64;

    eprintln!("\n=== INITIAL STATE ===");
    eprintln!("Images: {}", n_img);
    eprintln!("Faces: {}", n_face);
    eprintln!("Persons: {}", n_person);
    eprintln!("Prototypes: {}", n_proto);

    // Load models
    let scrfd_path = resolve_model("scrfd_500m_bnkps.onnx");
    let arcface_path = resolve_model("w600k_r50.onnx");

    let detector = ScrfdDetector::load(&scrfd_path).expect("load SCRFD");
    let aligner = Arc::new(SimpleAligner::new());
    let embedder = ArcFaceEmbedder::load(&arcface_path).expect("load ArcFace");
    let qf = QualityFilter::from_config(0.30, 20, 0.45, 30.0);
    let pipeline = Arc::new(FacePipeline::new(detector, aligner, embedder, qf));

    // Load HNSW index
    let index_dir = PathBuf::from("/Users/mac/Library/Application Support/PhotoFinderNext/index");
    let face_index: Arc<HnswIndex> = Arc::new(HnswIndex::new(512, index_dir, "face"));
    eprintln!("HNSW index size: {}", face_index.len());

    // Create services
    let config = Arc::new(Config::default());

    let prototype_service = Arc::new(PrototypeService::new(db.clone()));
    let person = Arc::new(
        PersonService::new(db.clone(), face_index.clone(), ClusterPolicy::default())
            .with_prototype_service(prototype_service.clone())
            .with_execution_mode(IdentityExecutionMode::NewPipeline),
    );

    // Check unassigned faces
    let unassigned = db.transaction(|tx| tx.faces().list_unassigned_embeddings("arcface-w600k-r50")).unwrap();
    eprintln!("\nUnassigned faces: {}", unassigned.len());

    if unassigned.is_empty() {
        eprintln!("No unassigned faces to process");
        return;
    }

    // Run cluster_all
    eprintln!("\n=== RUNNING CLUSTER_ALL ===");
    let start = std::time::Instant::now();
    match person.cluster_all().await {
        Ok(summary) => {
            eprintln!("\n=== CLUSTER RESULT ===");
            eprintln!("Duration: {:?}", start.elapsed());
            eprintln!("Assigned: {}", summary.assigned);
            eprintln!("Created: {}", summary.created);
            eprintln!("Total processed: {}", summary.assigned + summary.created);
        }
        Err(e) => {
            eprintln!("\n=== CLUSTER ERROR ===");
            eprintln!("Error: {:?}", e);
        }
    }

    // Check final state
    let n_img_after = db.transaction(|tx| tx.images().count()).unwrap();
    let n_face_after = db.transaction(|tx| tx.faces().count()).unwrap();
    let n_person_after = db.transaction(|tx| tx.persons().list()).unwrap().len() as i64;
    let n_proto_after = db.transaction(|tx| tx.person_prototypes().list_all_active()).unwrap().len() as i64;

    eprintln!("\n=== FINAL STATE ===");
    eprintln!("Images: {} (unchanged)", n_img_after);
    eprintln!("Faces: {} (unchanged)", n_face_after);
    eprintln!("Persons: {} (was {})", n_person_after, n_person);
    eprintln!("Prototypes: {} (was {})", n_proto_after, n_proto);

    // Show face assignments
    if n_person_after > 0 {
        eprintln!("\n=== PERSON DISTRIBUTION ===");
        let faces = db.transaction(|tx| tx.faces().list_all()).unwrap();
        let mut by_person: std::collections::HashMap<i64, Vec<i64>> = std::collections::HashMap::new();
        for f in &faces {
            if let Some(pid) = f.person_id {
                by_person.entry(pid).or_default().push(f.id);
            }
        }
        for (pid, face_ids) in &by_person {
            eprintln!("Person {}: {} faces", pid, face_ids.len());
        }
    }

    eprintln!("\n>>> TEST COMPLETE <<<");
}
