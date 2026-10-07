//! Phase 1.3: `FaceRepository::distinct_model_versions` + `Database::open` 启动检查。
//!
//! 覆盖:
//! - 空 face_embeddings → empty
//! - 单版本 → 一项
//! - 多版本 → 多项,触发 warn log(不阻断启动)
//! - 排序字典序 asc(便于断言)

use pf_database::{builtin_migrations, Database, FaceStatus, NewFace, NewImage, NewPerson};
use pf_core::BBox;

fn open_fresh_db() -> Database {
    Database::open_in_memory(builtin_migrations()).expect("open in-memory")
}

fn make_new_face(image_id: i64, model_version: &str, model_name: &str) -> NewFace {
    NewFace {
        image_id,
        bbox: BBox {
            x: 0.0,
            y: 0.0,
            w: 100.0,
            h: 100.0,
        },
        detector_score: 0.9,
        detector_model: "scrfd-500m-bnkps".into(),
        keypoints_json: None,
        yaw_pitch_roll: None,
        quality: 0.8,
        blur_score: None,
        pose_score: None,
        face_area_score: None,
        alignment_version: Some("v1".into()),
        embedding_model: Some(model_name.into()),
        model_version: model_version.into(),
        vector_id: None,
        hnsw_handle: None,
        status: FaceStatus::Indexed,
        index_generation: 0,
        cluster_score: None,
        cluster_method: None,
    }
}

fn make_new_image(path: &str) -> NewImage {
    NewImage {
        path: path.into(),
        hash: "abc".into(),
        size: 1024,
        modified_time: 0,
        width: 800,
        height: 600,
        captured_at: Some(chrono::Utc::now()),
    }
}

fn insert_face_with_embedding(
    db: &Database,
    image_path: &str,
    model_name: &str,
    model_version: &str,
) {
    db.transaction(|tx| {
        let img_id = tx.images().insert(&make_new_image(image_path)).expect("img");
        let face_id = tx
            .faces()
            .insert(&make_new_face(img_id, model_version, model_name))
            .expect("face");
        let v = vec![0.5f32; 512];
        tx.faces()
            .insert_embedding(face_id, model_name, model_version, 512, &v, None, true)
            .expect("embedding");
        Ok(())
    })
    .expect("tx");
}

#[test]
fn distinct_versions_empty_on_fresh_db() {
    let db = open_fresh_db();
    let v = db
        .transaction(|tx| tx.faces().distinct_model_versions())
        .expect("distinct");
    assert!(v.is_empty(), "fresh DB should have no embeddings, got {:?}", v);
}

#[test]
fn distinct_versions_single_version() {
    let db = open_fresh_db();
    insert_face_with_embedding(&db, "/tmp/single.jpg", "arcface", "w600k_r50_v1");

    let versions = db
        .transaction(|tx| tx.faces().distinct_model_versions())
        .expect("distinct");
    assert_eq!(versions, vec!["w600k_r50_v1".to_string()]);
}

#[test]
fn distinct_versions_multiple_versions() {
    // 同一张 image 一个 face,塞两条不同 model_name+version 的 embedding
    // (UNIQUE(face_id, model_name, model_version) 允许这样)
    let db = open_fresh_db();
    db.transaction(|tx| {
        let img_id = tx
            .images()
            .insert(&make_new_image("/tmp/multi.jpg"))
            .expect("img");
        let face_id = tx
            .faces()
            .insert(&make_new_face(img_id, "w600k_r50_v1", "arcface"))
            .expect("face");
        let v = vec![0.5f32; 512];
        tx.faces()
            .insert_embedding(face_id, "arcface", "w600k_r50_v1", 512, &v, None, true)
            .expect("v1");
        tx.faces()
            .insert_embedding(face_id, "arcface-v2", "w600k_r50_v2", 512, &v, None, true)
            .expect("v2");
        Ok(())
    })
    .expect("tx");

    let versions = db
        .transaction(|tx| tx.faces().distinct_model_versions())
        .expect("distinct");
    assert_eq!(
        versions,
        vec!["w600k_r50_v1".to_string(), "w600k_r50_v2".to_string()],
        "sorted asc, both present"
    );
}

#[test]
fn distinct_versions_sorted_ascending() {
    // 故意按 desc 顺序插入,验证 distinct + sort 不依赖插入顺序
    let db = open_fresh_db();
    db.transaction(|tx| {
        let img_id = tx
            .images()
            .insert(&make_new_image("/tmp/sorted.jpg"))
            .expect("img");
        for (name, ver) in [("arcface", "z_v9"), ("arcface", "a_v1"), ("arcface", "m_v5")] {
            let face_id = tx
                .faces()
                .insert(&make_new_face(img_id, "w600k_r50_v1", name))
                .expect("face");
            let v = vec![0.5f32; 512];
            tx.faces()
                .insert_embedding(face_id, name, ver, 512, &v, None, true)
                .expect("emb");
        }
        Ok(())
    })
    .expect("tx");

    let versions = db
        .transaction(|tx| tx.faces().distinct_model_versions())
        .expect("distinct");
    assert_eq!(
        versions,
        vec![
            "a_v1".to_string(),
            "m_v5".to_string(),
            "z_v9".to_string(),
        ]
    );
}

#[test]
fn open_with_multiple_versions_does_not_panic() {
    // Phase 1.3: Database::open 应该 warn 而不 panic/abort
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("multi.db");

    // 第一次 open + 插入多版本 embedding
    let db = Database::open(&db_path, builtin_migrations()).expect("first open");
    db.transaction(|tx| {
        let img_id = tx
            .images()
            .insert(&make_new_image("/tmp/m.jpg"))
            .expect("img");
        let face_id = tx
            .faces()
            .insert(&make_new_face(img_id, "w600k_r50_v1", "arcface"))
            .expect("face");
        let v = vec![0.5f32; 512];
        tx.faces()
            .insert_embedding(face_id, "arcface", "w600k_r50_v1", 512, &v, None, true)
            .expect("v1");
        tx.faces()
            .insert_embedding(face_id, "arcface-v2", "w600k_r50_v2", 512, &v, None, true)
            .expect("v2");
        Ok(())
    })
    .expect("tx");
    drop(db);

    // 第二次 open(模拟重启) — 应能成功打开(warn, 但不 abort)
    let db2 = Database::open(&db_path, builtin_migrations()).expect("second open must succeed");
    let conn = db2.connection().expect("conn");
    let n: i64 = conn
        .query_row("SELECT COUNT(*) FROM face_embeddings", [], |r| r.get(0))
        .expect("count");
    assert_eq!(n, 2, "both v1 and v2 embeddings persist across restart");
}

// 保证 NewPerson 类型引用(供后续 Phase 5 person lifecycle 使用,本测试不用)
#[allow(dead_code)]
fn _unused_new_person() -> NewPerson {
    NewPerson { name: None }
}