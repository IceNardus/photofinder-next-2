//! PrototypeService 集成测试(in-memory DB)。

use std::sync::Arc;

use pf_application::prototype_service::PrototypeService;
use pf_core::{BBox, FACE_MODEL_NAME};
use pf_database::{
    builtin_migrations, Database, FaceStatus, NewFace, NewImage, NewPerson,
};

fn make_db() -> Arc<Database> {
    Arc::new(Database::open_in_memory(builtin_migrations()).unwrap())
}

fn seed_image(db: &Database) -> i64 {
    db.transaction(|tx| {
        tx.images().insert(&NewImage {
            path: "/tmp/test.jpg".into(),
            hash: "hash-1".into(),
            size: 1024,
            modified_time: 0,
            width: 800,
            height: 600,
            captured_at: None,
        })
    })
    .unwrap()
}

fn seed_person_with_faces(db: &Database, person_id: i64, n: usize) -> Vec<i64> {
    let image_id = seed_image(db);
    let mut face_ids = Vec::with_capacity(n);
    for i in 0..n {
        let fid = db
            .transaction(|tx| {
                let fid = tx
                    .faces()
                    .insert(&NewFace {
                        image_id,
                        bbox: BBox {
                            x: i as f32 * 10.0,
                            y: 0.0,
                            w: 100.0,
                            h: 100.0,
                        },
                        detector_score: 0.9,
                        detector_model: "scrfd-500m-bnkps".into(),
                        keypoints_json: None,
                        yaw_pitch_roll: None,
                        quality: 0.7,
                        blur_score: None,
                        pose_score: None,
                        face_area_score: None,
                        alignment_version: Some("v1".into()),
                        embedding_model: Some(FACE_MODEL_NAME.into()),
                        model_version: "arcface@v1".into(),
                        vector_id: Some((i + 1) as i64),
                        hnsw_handle: Some((i + 1) as i64),
                        status: FaceStatus::Indexed,
                        index_generation: 0,
                        cluster_score: None,
                        cluster_method: None,
                    })
                    .unwrap();
                let vec = vec![0.0f32; 512];
                tx.faces()
                    .insert_embedding(
                        fid,
                        FACE_MODEL_NAME,
                        "v1",
                        512,
                        &vec,
                        Some((i + 1) as i64),
                        true,
                    )
                    .unwrap();
                Ok::<_, pf_database::DatabaseError>(fid)
            })
            .unwrap();
        face_ids.push(fid);
    }

    db.transaction(|tx| {
        for fid in &face_ids {
            tx.faces().set_person(*fid, Some(person_id)).unwrap();
        }
        Ok::<_, pf_database::DatabaseError>(())
    })
    .unwrap();

    face_ids
}

#[test]
fn rebuild_creates_expected_prototypes_for_no_yaw() {
    let db = make_db();
    let person_id = db
        .transaction(|tx| tx.persons().insert(&NewPerson { name: None }))
        .unwrap();
    let _face_ids = seed_person_with_faces(&db, person_id, 3);

    let ps = PrototypeService::new(db.clone());
    let n = ps.rebuild_for_person(person_id).unwrap();

    // 没有 yaw → 3 个 face 全进 frontal 桶;零向量 embedding 不去重 → 3 个 frontal
    // prototype(item 21/22:桶内去重 + 每 cluster 一个代表,无 general/high_quality 重复)
    assert_eq!(n, 3);

    let protos = ps.list_rows_for_person(person_id).unwrap();
    let types: Vec<_> = protos.iter().map(|p| p.prototype_type).collect();
    use pf_database::PrototypeType::*;
    assert!(types.iter().all(|t| *t == Frontal));
    assert!(!types.contains(&HighQuality));
    assert!(!types.contains(&General));
    assert!(!types.contains(&LeftProfile));
    assert!(!types.contains(&RightProfile));
    assert!(!types.contains(&Glasses));
}

#[test]
fn rebuild_is_idempotent() {
    let db = make_db();
    let person_id = db
        .transaction(|tx| tx.persons().insert(&NewPerson { name: None }))
        .unwrap();
    let _face_ids = seed_person_with_faces(&db, person_id, 3);

    let ps = PrototypeService::new(db.clone());
    let n1 = ps.rebuild_for_person(person_id).unwrap();
    let n2 = ps.rebuild_for_person(person_id).unwrap();

    assert_eq!(n1, n2);
    assert_eq!(n2, 3);

    // 014: 重建走软删,active 数应保持 3;list_active_rows_for_person 只看 active
    let protos = ps.list_active_rows_for_person(person_id).unwrap();
    assert_eq!(protos.len(), 3, "重建后 active 数量不变(软删旧 + 插新)");
}

#[test]
fn rebuild_with_no_faces_returns_zero() {
    let db = make_db();
    let person_id = db
        .transaction(|tx| tx.persons().insert(&NewPerson { name: None }))
        .unwrap();

    let ps = PrototypeService::new(db.clone());
    let n = ps.rebuild_for_person(person_id).unwrap();
    assert_eq!(n, 0);
}

#[test]
fn list_for_person_returns_embeddings() {
    let db = make_db();
    let person_id = db
        .transaction(|tx| tx.persons().insert(&NewPerson { name: None }))
        .unwrap();
    let _face_ids = seed_person_with_faces(&db, person_id, 3);

    let ps = PrototypeService::new(db.clone());
    ps.rebuild_for_person(person_id).unwrap();

    let result = ps.list_for_person(person_id, None).unwrap();
    assert_eq!(result.len(), 3);
    for (_, vec) in &result {
        assert_eq!(vec.len(), 512);
    }
}

#[test]
fn delete_for_person_removes_all() {
    let db = make_db();
    let person_id = db
        .transaction(|tx| tx.persons().insert(&NewPerson { name: None }))
        .unwrap();
    let _face_ids = seed_person_with_faces(&db, person_id, 3);

    let ps = PrototypeService::new(db.clone());
    ps.rebuild_for_person(person_id).unwrap();
    let n_deleted = ps.delete_for_person(person_id).unwrap();
    assert_eq!(n_deleted, 3);

    // 014: 软删后 active 应为 0(全量 list_rows_for_person 仍可能有 is_active=0 的行)
    let active = ps.list_active_rows_for_person(person_id).unwrap();
    assert!(active.is_empty());
}