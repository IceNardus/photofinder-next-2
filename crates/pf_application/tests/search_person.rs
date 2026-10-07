//! Phase 5 — Search V2 (face → persons via prototype) 集成测试。
//!
//! 流程:
//! 1. in-memory DB + HnswIndex(tempdir)
//! 2. 插入 person + 多个 face + arcface embedding
//! 3. 重建 prototypes
//! 4. 调 `search_by_person_impl` 验证结果

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use pf_application::l2_normalize;
use pf_application::search_v2::search_by_person_impl;
use pf_core::BBox;
use pf_database::{
    builtin_migrations, Database, FaceStatus, NewFace, NewImage, NewPerson, PrototypeType,
};
use pf_vector::{HnswIndex, VectorIndex};

const EMBEDDING_DIM: usize = 16; // 小维度,加快 HNSW 测试

fn make_db() -> Arc<Database> {
    Arc::new(Database::open_in_memory(builtin_migrations()).unwrap())
}

fn tmpdir(prefix: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pf-app-search-v2-{}-{}-{}",
        prefix,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&p).unwrap();
    p
}

fn seed_image(db: &Database, suffix: &str) -> i64 {
    db.transaction(|tx| {
        tx.images().insert(&NewImage {
            path: format!("/tmp/{suffix}.jpg"),
            hash: format!("hash-{suffix}-{}", std::process::id()),
            size: 1024,
            modified_time: 0,
            width: 800,
            height: 600,
            captured_at: None,
        })
    })
    .unwrap()
}

/// 插入 face + arcface embedding + HNSW + 关联 person。
/// 返回 face_id。
fn insert_face(
    db: &Database,
    index: &dyn VectorIndex,
    image_id: i64,
    vector_id: i64,
    embedding: &[f32],
    person_id: i64,
    quality: f32,
) -> i64 {
    let emb = l2_normalize(embedding);
    let face_id = db
        .transaction(|tx| {
            let fid = tx
                .faces()
                .insert(&NewFace {
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
                    quality,
                    blur_score: None,
                    pose_score: None,
                    face_area_score: None,
                    alignment_version: Some("v1".into()),
                    embedding_model: Some("arcface".into()),
                    model_version: "arcface@v1".into(),
                    vector_id: Some(vector_id),
                    hnsw_handle: Some(vector_id),
                    status: FaceStatus::Indexed,
                    index_generation: 0,
                    cluster_score: None,
                    cluster_method: None,
                })
                .unwrap();
            tx.faces()
                .insert_embedding(
                    fid,
                    "arcface",
                    "v1",
                    EMBEDDING_DIM as i64,
                    &emb,
                    Some(vector_id),
                    true,
                )
                .unwrap();
            tx.faces().set_person(fid, Some(person_id)).unwrap();
            Ok::<_, pf_database::DatabaseError>(fid)
        })
        .unwrap();
    index.insert(vector_id, &emb).unwrap();
    face_id
}

fn make_person(db: &Database) -> i64 {
    db.transaction(|tx| tx.persons().insert(&NewPerson { name: None }))
        .unwrap()
}

#[test]
fn returns_images_with_prototype_match() {
    let db = make_db();
    let dir = tmpdir("match");
    let index = Arc::new(HnswIndex::new(EMBEDDING_DIM, dir, "test"));

    let pid = make_person(&db);
    let img1 = seed_image(&db, "img1");
    let img2 = seed_image(&db, "img2");

    // person P 出现在 img1 和 img2
    let mut v = vec![0.0_f32; EMBEDDING_DIM];
    v[0] = 1.0;
    let _fid1 = insert_face(&db, &*index, img1, 1, &v, pid, 0.9);
    let _fid2 = insert_face(&db, &*index, img2, 2, &v, pid, 0.9);

    // 一个 prototype embedding(指向 v)
    let protos = vec![(PrototypeType::Frontal, l2_normalize(&v))];

    let results = search_by_person_impl(
        &*index,
        &db,
        &protos,
        pid,
        10,
        0.50, // threshold
    )
    .unwrap();

    assert_eq!(results.len(), 2, "should return both images");
    let image_ids: std::collections::HashSet<i64> =
        results.iter().map(|r| r.image_id).collect();
    assert!(image_ids.contains(&img1));
    assert!(image_ids.contains(&img2));
}

#[test]
fn aggregates_across_multiple_prototypes() {
    let db = make_db();
    let dir = tmpdir("multi");
    let index = Arc::new(HnswIndex::new(EMBEDDING_DIM, dir, "test"));

    let pid = make_person(&db);
    let img1 = seed_image(&db, "img1");
    let img2 = seed_image(&db, "img2");

    // img1 含 face with embedding v_frontal
    // img2 含 face with embedding v_left(完全不同方向)
    let mut v_front = vec![0.0_f32; EMBEDDING_DIM];
    v_front[0] = 1.0;
    let mut v_left = vec![0.0_f32; EMBEDDING_DIM];
    v_left[1] = 1.0;
    let _fid1 = insert_face(&db, &*index, img1, 1, &v_front, pid, 0.9);
    let _fid2 = insert_face(&db, &*index, img2, 2, &v_left, pid, 0.9);

    // 两个 prototype:frontal 和 left_profile
    let protos = vec![
        (PrototypeType::Frontal, l2_normalize(&v_front)),
        (PrototypeType::LeftProfile, l2_normalize(&v_left)),
    ];

    let results = search_by_person_impl(&*index, &db, &protos, pid, 10, 0.50).unwrap();

    assert_eq!(
        results.len(),
        2,
        "should find both images via different prototypes"
    );
    let image_ids: std::collections::HashSet<i64> = results.iter().map(|r| r.image_id).collect();
    assert!(image_ids.contains(&img1), "img1 found via Frontal prototype");
    assert!(image_ids.contains(&img2), "img2 found via LeftProfile prototype");
}

#[test]
fn filters_by_threshold() {
    let db = make_db();
    let dir = tmpdir("thresh");
    let index = Arc::new(HnswIndex::new(EMBEDDING_DIM, dir, "test"));

    let pid = make_person(&db);
    let img1 = seed_image(&db, "img1");

    // face with embedding v(orthogonal to prototype)
    let mut v = vec![0.0_f32; EMBEDDING_DIM];
    v[0] = 1.0;
    let _fid1 = insert_face(&db, &*index, img1, 1, &v, pid, 0.9);

    // prototype 完全不同方向
    let mut v_ortho = vec![0.0_f32; EMBEDDING_DIM];
    v_ortho[5] = 1.0;
    let protos = vec![(PrototypeType::Frontal, l2_normalize(&v_ortho))];

    // threshold=0.50 → 正交脸 score < 0.50 → 被过滤
    let results = search_by_person_impl(&*index, &db, &protos, pid, 10, 0.50).unwrap();
    assert!(results.is_empty(), "orthogonal face should be filtered out");

    // threshold=-1.0 → 不过滤
    let results = search_by_person_impl(&*index, &db, &protos, pid, 10, -1.0).unwrap();
    assert_eq!(results.len(), 1, "negative threshold accepts everything");
}

#[test]
fn returns_empty_when_no_prototypes() {
    let db = make_db();
    let dir = tmpdir("empty_proto");
    let index = Arc::new(HnswIndex::new(EMBEDDING_DIM, dir, "test"));

    let protos: Vec<(PrototypeType, Vec<f32>)> = Vec::new();
    let results = search_by_person_impl(&*index, &db, &protos, 999, 10, 0.50).unwrap();
    assert!(results.is_empty());
}

#[test]
fn only_includes_faces_of_target_person() {
    let db = make_db();
    let dir = tmpdir("only_target");
    let index = Arc::new(HnswIndex::new(EMBEDDING_DIM, dir, "test"));

    let pid_target = make_person(&db);
    let pid_other = make_person(&db);
    let img1 = seed_image(&db, "img1");

    // 同一 image 含两张脸:一张属于 target,一张属于 other
    let mut v = vec![0.0_f32; EMBEDDING_DIM];
    v[0] = 1.0;
    let _fid_target = insert_face(&db, &*index, img1, 1, &v, pid_target, 0.9);
    let mut v_other = vec![0.0_f32; EMBEDDING_DIM];
    v_other[1] = 0.05; // 接近 target
    v_other[0] = 1.0;
    let _fid_other = insert_face(&db, &*index, img1, 2, &v_other, pid_other, 0.9);

    let protos = vec![(PrototypeType::Frontal, l2_normalize(&v))];

    let results = search_by_person_impl(&*index, &db, &protos, pid_target, 10, 0.50).unwrap();

    // 应只返回 target 的 face,不返回 other 的
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].target_id, _fid_target);
}

#[test]
fn respects_top_k() {
    let db = make_db();
    let dir = tmpdir("topk");
    let index = Arc::new(HnswIndex::new(EMBEDDING_DIM, dir, "test"));

    let pid = make_person(&db);

    // 5 个 image,各 1 个 face
    let mut v = vec![0.0_f32; EMBEDDING_DIM];
    v[0] = 1.0;
    let mut image_ids = Vec::new();
    for i in 0..5 {
        let img_id = seed_image(&db, &format!("img{i}"));
        insert_face(&db, &*index, img_id, i as i64 + 1, &v, pid, 0.9);
        image_ids.push(img_id);
    }

    let protos = vec![(PrototypeType::Frontal, l2_normalize(&v))];

    // top_k=3 → 只返回 3 个
    let results = search_by_person_impl(&*index, &db, &protos, pid, 3, 0.50).unwrap();
    assert_eq!(results.len(), 3);
    // rank 1..3
    assert_eq!(results[0].rank, 1);
    assert_eq!(results[2].rank, 3);
}

#[test]
fn keeps_highest_score_per_image() {
    let db = make_db();
    let dir = tmpdir("max_per_img");
    let index = Arc::new(HnswIndex::new(EMBEDDING_DIM, dir, "test"));

    let pid = make_person(&db);
    let img1 = seed_image(&db, "img1");

    // 同一 image 含两个 target face:HNSW 都命中,但 score 不同
    let mut v_high = vec![0.0_f32; EMBEDDING_DIM];
    v_high[0] = 1.0;
    let mut v_low = vec![0.0_f32; EMBEDDING_DIM];
    v_low[0] = 0.95;
    v_low[1] = 0.31; // similar but lower score
    let fid_high = insert_face(&db, &*index, img1, 1, &v_high, pid, 0.9);
    let _fid_low = insert_face(&db, &*index, img1, 2, &v_low, pid, 0.9);

    let protos = vec![(PrototypeType::Frontal, l2_normalize(&v_high))];

    let results = search_by_person_impl(&*index, &db, &protos, pid, 10, 0.0).unwrap();
    assert_eq!(results.len(), 1, "dedup by image_id");
    assert_eq!(results[0].target_id, fid_high, "keep highest-score face_id");
}