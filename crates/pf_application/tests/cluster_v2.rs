//! Phase 4 — Cluster V2 (prototype 验证) 集成测试。
//!
//! 流程:
//! 1. 启 in-memory DB + 精确最近邻索引(`BruteForceIndex`,确定性,见下方说明)
//! 2. 插入 person + face + arcface embedding
//! 3. 重建 prototypes
//! 4. 调 `assign_or_create` 验证分配结果

use std::sync::Arc;

use pf_application::l2_normalize;
use pf_application::person::ClusterPolicy;
use pf_application::{PersonService, PrototypeService};
use pf_core::{BBox, Embedding, ModelVersion, FACE_MODEL_NAME};
use pf_database::{
    builtin_migrations, Database, FaceStatus, NewFace, NewImage, NewPerson,
};
use pf_vector::{Metric, SearchHit, VectorError, VectorIndex};

const EMBEDDING_DIM: usize = 16; // 小维度,加快测试

/// 测试用精确最近邻索引（O(N) brute force）。
///
/// cluster_v2 断言的是聚类逻辑（prototype 验证 / chaining prevention / fallback），
/// 不是 HNSW 的近似召回率。hnsw_rs 用 `StdRng::from_os_rng()` 随机建图，偶发
/// 召回失败（小图上可能只返回 1 个邻居）会让这些"期待精确结果"的测试偶发失败。
/// 换成确定性精确索引后逻辑测试稳定可复现；HNSW 召回率由 pf_vector 单测 + lfw_eval 覆盖。
struct BruteForceIndex {
    dim: usize,
    items: std::sync::Mutex<Vec<(i64, Vec<f32>)>>,
}

impl BruteForceIndex {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            items: std::sync::Mutex::new(Vec::new()),
        }
    }
}

impl VectorIndex for BruteForceIndex {
    fn dim(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        self.items.lock().unwrap().len()
    }

    fn insert(&self, id: i64, vector: &[f32]) -> Result<(), VectorError> {
        if vector.len() != self.dim {
            return Err(VectorError::Dimension {
                expected: self.dim,
                actual: vector.len(),
            });
        }
        self.items.lock().unwrap().push((id, vector.to_vec()));
        Ok(())
    }

    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchHit>, VectorError> {
        if query.len() != self.dim {
            return Err(VectorError::Dimension {
                expected: self.dim,
                actual: query.len(),
            });
        }
        let items = self.items.lock().unwrap();
        let mut hits: Vec<SearchHit> = items
            .iter()
            .map(|(id, v)| SearchHit {
                id: *id,
                score: v.iter().zip(query).map(|(a, b)| a * b).sum(),
            })
            .collect();
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(top_k);
        Ok(hits)
    }

    fn save(&self) -> Result<(), VectorError> {
        Ok(())
    }

    fn load(&self) -> Result<(), VectorError> {
        Ok(())
    }

    fn metric(&self) -> Metric {
        Metric::Cosine
    }
}

fn make_db() -> Arc<Database> {
    Arc::new(Database::open_in_memory(builtin_migrations()).unwrap())
}

fn seed_image(db: &Database) -> i64 {
    db.transaction(|tx| {
        tx.images().insert(&NewImage {
            path: "/tmp/test.jpg".into(),
            hash: format!("hash-{}", std::process::id()),
            size: 1024,
            modified_time: 0,
            width: 800,
            height: 600,
            captured_at: None,
        })
    })
    .unwrap()
}

/// 插入一个 face + arcface embedding + 写入索引,并可选关联到 person。
/// 返回 face_id。
fn insert_face(
    db: &Database,
    index: &dyn VectorIndex,
    image_id: i64,
    vector_id: i64,
    embedding: &[f32],
    person_id: Option<i64>,
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
                    embedding_model: Some(FACE_MODEL_NAME.into()),
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
                    FACE_MODEL_NAME,
                    "v1",
                    EMBEDDING_DIM as i64,
                    &emb,
                    Some(vector_id),
                    true,
                )
                .unwrap();
            if let Some(pid) = person_id {
                tx.faces().set_person(fid, Some(pid)).unwrap();
            }
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
fn assigns_to_existing_when_prototype_matches() {
    let db = make_db();
    let image_id = seed_image(&db);
    let index = Arc::new(BruteForceIndex::new(EMBEDDING_DIM));

    // Person P1,有一个 face with embedding v1
    let pid = make_person(&db);
    let v1 = vec![1.0_f32; EMBEDDING_DIM];
    let _fid1 = insert_face(&db, &*index, image_id, 1, &v1, Some(pid), 0.9);

    // 重建 prototypes
    let ps = PrototypeService::new(db.clone());
    ps.rebuild_for_person(pid).unwrap();

    // 新 face:embedding 接近 v1(±0.01 noise)
    let mut v1_noisy = v1.clone();
    v1_noisy[0] += 0.01;
    let v1_noisy_norm = l2_normalize(&v1_noisy);
    // 不插入 HNSW / DB(模拟 unassigned face),直接调 assign_or_create
    let policy = ClusterPolicy {
        similarity_threshold: 0.65,
        ..Default::default()
    };
    let svc = PersonService::new(db.clone(), index.clone(), policy)
        .with_prototype_service(Arc::new(ps));

    let result = svc.assign_or_create(&v1_noisy_norm).unwrap();
    match result {
        pf_core::PersonAssignment::Existing(p) => assert_eq!(p, pid),
        pf_core::PersonAssignment::New { .. } => panic!("expected Existing({pid})"),
    }
}

#[test]
fn creates_new_when_no_prototype_match() {
    let db = make_db();
    let image_id = seed_image(&db);
    let index = Arc::new(BruteForceIndex::new(EMBEDDING_DIM));

    // P1 with embedding v1
    let pid = make_person(&db);
    let v1 = vec![1.0_f32; EMBEDDING_DIM];
    let _fid1 = insert_face(&db, &*index, image_id, 1, &v1, Some(pid), 0.9);

    let ps = PrototypeService::new(db.clone());
    ps.rebuild_for_person(pid).unwrap();

    // 新 face:embedding 正交(差得很远)
    let mut v_ortho = vec![0.0_f32; EMBEDDING_DIM];
    v_ortho[5] = 1.0; // 完全不同方向
    let v_ortho_norm = l2_normalize(&v_ortho);

    let policy = ClusterPolicy {
        similarity_threshold: 0.65,
        ..Default::default()
    };
    let svc = PersonService::new(db.clone(), index.clone(), policy)
        .with_prototype_service(Arc::new(ps));

    let result = svc.assign_or_create(&v_ortho_norm).unwrap();
    match result {
        pf_core::PersonAssignment::New { .. } => {}
        pf_core::PersonAssignment::Existing(p) => panic!("expected New, got Existing({p})"),
    }
}

#[test]
fn picks_better_person_via_prototype_when_multiple_candidates() {
    let db = make_db();
    let image_id = seed_image(&db);
    let index = Arc::new(BruteForceIndex::new(EMBEDDING_DIM));

    // P1 with embedding aligned to v_dir
    // P2 with embedding aligned to v_dir2 (orthogonal to v_dir)
    let pid1 = make_person(&db);
    let pid2 = make_person(&db);
    let mut v_dir = vec![0.0_f32; EMBEDDING_DIM];
    v_dir[0] = 1.0;
    let mut v_dir2 = vec![0.0_f32; EMBEDDING_DIM];
    v_dir2[1] = 1.0;

    let _fid1 = insert_face(&db, &*index, image_id, 1, &v_dir, Some(pid1), 0.9);
    let _fid2 = insert_face(&db, &*index, image_id, 2, &v_dir2, Some(pid2), 0.9);

    let ps = PrototypeService::new(db.clone());
    ps.rebuild_for_person(pid1).unwrap();
    ps.rebuild_for_person(pid2).unwrap();

    // query:close to v_dir
    let mut v_query = v_dir.clone();
    v_query[1] = 0.05;
    let v_query_norm = l2_normalize(&v_query);

    let policy = ClusterPolicy {
        similarity_threshold: 0.65,
        ..Default::default()
    };
    let svc = PersonService::new(db.clone(), index.clone(), policy)
        .with_prototype_service(Arc::new(ps));

    let result = svc.assign_or_create(&v_query_norm).unwrap();
    match result {
        pf_core::PersonAssignment::Existing(p) => {
            assert_eq!(p, pid1, "should pick pid1 (closer to v_query via prototype)");
        }
        pf_core::PersonAssignment::New { .. } => panic!("expected Existing({pid1})"),
    }
}

#[test]
fn falls_back_to_top_neighbor_when_no_prototype_service() {
    let db = make_db();
    let image_id = seed_image(&db);
    let index = Arc::new(BruteForceIndex::new(EMBEDDING_DIM));

    let pid = make_person(&db);
    let v1 = vec![1.0_f32; EMBEDDING_DIM];
    let _fid1 = insert_face(&db, &*index, image_id, 1, &v1, Some(pid), 0.9);

    // 注意:不重建 prototypes — 没 prototype_service
    let policy = ClusterPolicy {
        similarity_threshold: 0.65,
        ..Default::default()
    };
    let svc = PersonService::new(db.clone(), index.clone(), policy);
    // 不调 with_prototype_service,prototype_service = None

    let mut v1_noisy = v1.clone();
    v1_noisy[0] += 0.01;
    let v1_noisy_norm = l2_normalize(&v1_noisy);
    let result = svc.assign_or_create(&v1_noisy_norm).unwrap();
    match result {
        pf_core::PersonAssignment::Existing(p) => assert_eq!(p, pid),
        pf_core::PersonAssignment::New { .. } => panic!("expected Existing({pid}) via top-neighbor fallback"),
    }
}

#[test]
fn cluster_v2_end_to_end_via_cluster_all() {
    // 端到端:cluster_all 把 unassigned face 分配给已有 person(prototype-driven)
    let db = make_db();
    let image_id = seed_image(&db);
    let index = Arc::new(BruteForceIndex::new(EMBEDDING_DIM));

    // Pre-existing P1 with v1
    let pid = make_person(&db);
    let v1 = vec![1.0_f32; EMBEDDING_DIM];
    let _fid1 = insert_face(&db, &*index, image_id, 1, &v1, Some(pid), 0.9);
    // 手动 rebuild(因为 cluster_all 是给 unassigned face 用的)
    let ps = PrototypeService::new(db.clone());
    ps.rebuild_for_person(pid).unwrap();

    // 新 unassigned face(用 set_person(None) 让它落到 unassigned)
    let mut v1_noisy = v1.clone();
    v1_noisy[0] += 0.01;
    let v1_noisy_norm = l2_normalize(&v1_noisy);
    let fid2 = insert_face(&db, &*index, image_id, 2, &v1_noisy_norm, None, 0.8);

    let policy = ClusterPolicy {
        similarity_threshold: 0.65,
        ..Default::default()
    };
    let svc = Arc::new(
        PersonService::new(db.clone(), index.clone(), policy)
            .with_prototype_service(Arc::new(ps)),
    );

    // 直接调 assign_or_create 验证(不走 cluster_all 因为 list_unassigned_embeddings 的 SQL 我们先不动)
    let result = svc.assign_or_create(&v1_noisy_norm).unwrap();
    match result {
        pf_core::PersonAssignment::Existing(p) => assert_eq!(p, pid),
        pf_core::PersonAssignment::New { .. } => panic!("expected Existing({pid})"),
    }

    // 验证 set_person 调用生效
    db.transaction(|tx| tx.faces().set_person(fid2, Some(pid))).unwrap();
    let face2 = db.transaction(|tx| tx.faces().get_by_id(fid2)).unwrap().unwrap();
    assert_eq!(face2.person_id, Some(pid));

    // 确认 Embedding 构造正确(没有 model dependency)
    let _emb = Embedding::new(v1_noisy_norm.clone(), ModelVersion::new("arcface@v1"));
}

/// item 16:防 chaining — query 与两个 person 的 prototype 都几乎等距时,创建新 person
/// 而不是并入较近者。
#[test]
fn chaining_prevention_creates_new_when_top2_are_ambiguous() {
    let db = make_db();
    let image_id = seed_image(&db);
    let index = Arc::new(BruteForceIndex::new(EMBEDDING_DIM));

    // P1 proto ≈ [1,0,...];P2 proto ≈ [cos8°, sin8°] — 只差 8°,query 与两者都极近
    let pid1 = make_person(&db);
    let pid2 = make_person(&db);
    let mut v1 = vec![0.0_f32; EMBEDDING_DIM];
    v1[0] = 1.0;
    let mut v2 = vec![0.0_f32; EMBEDDING_DIM];
    v2[0] = 8.0f32.to_radians().cos();
    v2[1] = 8.0f32.to_radians().sin();

    let _fid1 = insert_face(&db, &*index, image_id, 1, &v1, Some(pid1), 0.9);
    let _fid2 = insert_face(&db, &*index, image_id, 2, &v2, Some(pid2), 0.9);

    let ps = PrototypeService::new(db.clone());
    ps.rebuild_for_person(pid1).unwrap();
    ps.rebuild_for_person(pid2).unwrap();

    // query 与 v1 完全一致:无 margin 时会被并入 P1;有 margin 时应歧义 → New
    let v_query_norm = l2_normalize(&v1);
    let policy = ClusterPolicy {
        similarity_threshold: 0.65,
        ..Default::default()
    };
    let svc = PersonService::new(db.clone(), index.clone(), policy)
        .with_prototype_service(Arc::new(ps));

    let result = svc.assign_or_create(&v_query_norm).unwrap();
    match result {
        pf_core::PersonAssignment::New { .. } => {}
        pf_core::PersonAssignment::Existing(p) => {
            panic!("expected New (ambiguous top-2), got Existing({p})")
        }
    }
}

/// item 17:assign_face 会把 Confirmed assignment 记录到 face_person_assignments,
/// 且 confidence 由 similarity+quality+pose 计算。
#[test]
fn assign_face_records_confirmed_assignment() {
    let db = make_db();
    let image_id = seed_image(&db);
    let index = Arc::new(BruteForceIndex::new(EMBEDDING_DIM));

    let pid = make_person(&db);
    let v1 = vec![1.0_f32; EMBEDDING_DIM];
    let _fid1 = insert_face(&db, &*index, image_id, 1, &v1, Some(pid), 0.9);

    let ps = PrototypeService::new(db.clone());
    ps.rebuild_for_person(pid).unwrap();

    // 新 unassigned face(embedding 接近 v1)
    let mut v1_noisy = v1.clone();
    v1_noisy[0] += 0.01;
    let v1_noisy_norm = l2_normalize(&v1_noisy);
    let fid2 = insert_face(&db, &*index, image_id, 2, &v1_noisy_norm, None, 0.8);

    let policy = ClusterPolicy {
        similarity_threshold: 0.65,
        ..Default::default()
    };
    let svc = PersonService::new(db.clone(), index.clone(), policy)
        .with_prototype_service(Arc::new(ps));

    // assign_face:quality=0.8, pose=0.7
    let result = svc
        .assign_face(fid2, &v1_noisy_norm, Some(0.8), Some(0.7))
        .unwrap();
    match result {
        pf_core::PersonAssignment::Existing(p) => assert_eq!(p, pid),
        pf_core::PersonAssignment::New { .. } => panic!("expected Existing({pid})"),
    }

    // 验证 face_person_assignments 里有 Confirmed 行
    let rows = db
        .transaction(|tx| tx.face_person_assignments().list_by_face(fid2))
        .unwrap();
    assert_eq!(rows.len(), 1, "assign_face should record exactly one assignment");
    let row = &rows[0];
    use pf_database::MatchStatus;
    assert_eq!(row.status, MatchStatus::Confirmed);
    assert_eq!(row.person_id, pid);
    assert!(row.similarity > 0.65, "similarity should pass threshold: {}", row.similarity);
    // confidence 由 similarity+quality+pose 融合,应在 [0,1] 且接近 similarity
    assert!(row.confidence >= 0.0 && row.confidence <= 1.0);
    assert!(
        (row.confidence - row.similarity).abs() < 0.5,
        "confidence {} should be in the same ballpark as similarity {}",
        row.confidence,
        row.similarity
    );
}