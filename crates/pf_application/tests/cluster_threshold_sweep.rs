//! 临时诊断（非生产）：扫描聚类阈值，定位 stock 数据上错并（asian-man + young-asian-girl）
//! 的分离点。在 DB+index 的只读副本上跑，每个阈值清空聚类状态后重跑 `cluster_all`。
//!
//! 用法：
//!   PF_DB=/tmp/pf-sweep/photofinder.db PF_INDEX_DIR=/tmp/pf-sweep/index \
//!   cargo test --release -p pf_application --test cluster_threshold_sweep -- --ignored --nocapture

use std::path::PathBuf;
use std::sync::Arc;

use pf_application::person::{ClusterPolicy, PersonService};
use pf_application::PrototypeService;
use pf_database::{builtin_migrations, Database};
use pf_vector::{HnswIndex, VectorIndex};

const THRESHOLDS: &[f32] = &[0.65, 0.70, 0.75, 0.80, 0.85];

#[tokio::test]
#[ignore]
async fn sweep_cluster_threshold_on_live_copy() {
    let db_path = std::env::var("PF_DB").expect("set PF_DB to copy");
    let index_dir = std::env::var("PF_INDEX_DIR").expect("set PF_INDEX_DIR to copy");
    let db_path = std::path::PathBuf::from(&db_path);
    let index_dir = PathBuf::from(&index_dir);

    for &t in THRESHOLDS {
        // 1) 清空聚类状态
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute("DELETE FROM face_person_assignments", []).unwrap();
        conn.execute("DELETE FROM person_prototypes", []).unwrap();
        conn.execute("DELETE FROM persons", []).unwrap();
        conn.execute(
            "UPDATE faces SET person_id = NULL, cluster_method = NULL, cluster_score = NULL",
            [],
        )
        .unwrap();
        drop(conn);

        // 2) 重聚类
        let db = Arc::new(Database::open(&db_path, builtin_migrations()).unwrap());
        let mut index = HnswIndex::new(512, index_dir.clone(), "face");
        index.load().unwrap();
        let policy = ClusterPolicy {
            similarity_threshold: t,
            ..Default::default()
        };
        let svc = PersonService::new(db.clone(), Arc::new(index), policy)
            .with_prototype_service(Arc::new(PrototypeService::new(db.clone())));
        let summary = svc.cluster_all().await.unwrap();

        // 3) 报告：每 person 的 image 集合
        let persons = svc.list_all().unwrap();
        eprintln!(
            "=== threshold {t:.2}: persons={} assigned={} created={} ===",
            persons.len(),
            summary.assigned,
            summary.created
        );
        for p in &persons {
            let faces = svc.faces_of(p.id).unwrap();
            let imgs: Vec<i64> = faces.iter().map(|f| f.image_id).collect();
            let flags: Vec<String> = imgs
                .iter()
                .map(|&i| {
                    if i == 2 {
                        "MAN!".to_string()
                    } else if i == 20 {
                        "GIRL!".to_string()
                    } else if i == 9 {
                        "cottonbro".to_string()
                    } else if i == 17 {
                        "peterdanthy".to_string()
                    } else {
                        i.to_string()
                    }
                })
                .collect();
            eprintln!("    person {}: {:?}", p.id, flags);
        }
    }
}
