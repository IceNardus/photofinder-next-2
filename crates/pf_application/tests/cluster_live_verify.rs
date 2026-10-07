//! 临时诊断（非生产）：在生产 DB+index 的**只读副本**上跑 `PersonService::cluster_all`，
//! 验证聚类算法触发后能否正确产出 person_id 分配。
//!
//! 背景：#163 调查"face.person_id 全为 None"。根因已定位 = ClusterFaces 任务从未被调度
//! （scan_folder 完成后不 enqueue ClusterFaces，只有 PersonsView 手动按钮 / reindex_all 触发）。
//! 本文件验证"若触发，算法是否工作"——在副本上运行，不碰生产数据。
//!
//! 用法：
//!   PF_DB=/tmp/pf-cluster-test/photofinder.db \
//!   PF_INDEX_DIR=/tmp/pf-cluster-test/index \
//!   cargo test --release -p pf_application --test cluster_live_verify -- --ignored --nocapture

use std::path::PathBuf;
use std::sync::Arc;

use pf_application::person::ClusterPolicy;
use pf_application::{PersonService, PrototypeService};
use pf_database::{builtin_migrations, Database};
use pf_vector::{HnswIndex, VectorIndex};

#[tokio::test]
#[ignore]
async fn cluster_all_produces_persons_on_live_copy() {
    let db_path = std::env::var("PF_DB").expect("set PF_DB to live photofinder.db copy");
    let index_dir = std::env::var("PF_INDEX_DIR").expect("set PF_INDEX_DIR to live index dir copy");

    let db = Arc::new(Database::open(std::path::Path::new(&db_path), builtin_migrations()).unwrap());
    let index = HnswIndex::new(512, PathBuf::from(&index_dir), "face");
    let mut index = index;
    index.load().expect("load face.hnsw from copy");
    eprintln!("before cluster: index.len = {}", index.len());

    let svc = PersonService::new(db.clone(), Arc::new(index), ClusterPolicy::default())
        .with_prototype_service(Arc::new(PrototypeService::new(db.clone())));

    let summary = svc.cluster_all().await.expect("cluster_all");
    eprintln!("cluster summary: assigned={} created={}", summary.assigned, summary.created);

    let persons = svc.list_all().expect("list persons");
    eprintln!("persons count = {}", persons.len());
    for p in &persons {
        let faces = svc.faces_of(p.id).expect("faces_of");
        let names: Vec<(i64, i64, f32)> = faces
            .iter()
            .map(|f| (f.id, f.image_id, f.quality))
            .collect();
        eprintln!("  person {} face_count={} faces={:?}", p.id, faces.len(), names);
    }

    let unassigned = db
        .transaction(|tx| tx.faces().list_unassigned_embeddings(pf_core::FACE_MODEL_NAME))
        .expect("list unassigned");
    eprintln!("faces still unassigned after cluster_all = {}", unassigned.len());

    assert!(!persons.is_empty(), "cluster_all should create persons");
    assert!(
        unassigned.is_empty(),
        "all indexed faces should be assigned after cluster_all"
    );
}
