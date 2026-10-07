//! 诊断：人脸搜索是否会返回「无人脸」图片？
//!
//! 加载真实 DB + face HNSW index，用库里的**每一张**真实 face embedding 搜索，
//! 聚合所有返回命中映射到的 image_id，检查是否有任何命中落到「无人脸」图片上。
//! 只读诊断，`#[ignore]`，不随正常测试跑。
//!
//! 用法：
//!   PF_DB=/Users/mac/Library/Application\ Support/PhotoFinderNext/photofinder.db \
//!   PF_INDEX_DIR="/Users/mac/Library/Application Support/PhotoFinderNext/index" \
//!   cargo test -p pf_application --test face_search_live -- --ignored --nocapture

use std::collections::HashMap;

use pf_core::FACE_MODEL_NAME;
use pf_database::{builtin_migrations, Database};
use pf_vector::{HnswIndex, VectorIndex};

#[test]
#[ignore]
fn face_search_returns_only_images_with_faces() {
    let db_path =
        std::env::var("PF_DB").expect("set PF_DB to live photofinder.db path");
    let index_dir =
        std::env::var("PF_INDEX_DIR").expect("set PF_INDEX_DIR to live index dir");

    let db = Database::open(std::path::Path::new(&db_path), builtin_migrations()).unwrap();
    let mut index = HnswIndex::new(512, std::path::PathBuf::from(&index_dir), "face");
    index.load().expect("load face.hnsw");

    // DB 中的全部 face 行
    let all_faces = db
        .transaction(|tx| tx.faces().list_all())
        .expect("list all faces");
    let faces_with_vid: Vec<_> = all_faces
        .iter()
        .filter(|f| f.vector_id.is_some())
        .collect();

    eprintln!(
        "faces_in_db={} faces_with_vid={} index.len={} (index-len vs DB: +{})",
        all_faces.len(),
        faces_with_vid.len(),
        index.len(),
        index.len() as i64 - all_faces.len() as i64
    );

    // 每个 face 作为 query，聚合所有返回的 image_id
    let mut returned_images: HashMap<i64, (f32, i64)> = HashMap::new(); // image_id -> (best score, from face_id)
    let mut unmapped_hits = 0usize;
    let mut total_hits = 0usize;

    for f in &faces_with_vid {
        let fid = f.id;
        let emb = db
            .transaction(|tx| {
                Ok::<_, pf_database::DatabaseError>(
                    tx.faces()
                        .get_embedding(fid, FACE_MODEL_NAME)?
                        .expect("embedding exists"),
                )
            })
            .unwrap();

        let hits = index.search(&emb, 40).expect("search");
        total_hits += hits.len();

        let vector_ids: Vec<i64> = hits.iter().map(|h| h.id).collect();
        let face_by_vid: HashMap<i64, pf_database::FaceRow> = db
            .transaction(|tx| tx.faces().list_by_vector_ids(&vector_ids))
            .unwrap()
            .into_iter()
            .filter_map(|r| r.vector_id.map(|vid| (vid, r)))
            .collect();

        for h in &hits {
            match face_by_vid.get(&h.id) {
                Some(face) => {
                    let entry = returned_images.entry(face.image_id).or_insert((h.score, fid));
                    if h.score > entry.0 {
                        *entry = (h.score, fid);
                    }
                }
                None => {
                    unmapped_hits += 1;
                }
            }
        }
    }

    eprintln!(
        "TOTAL query_faces={} total_hits={} unmapped_stale_hits={} unique_images_returned={}",
        faces_with_vid.len(),
        total_hits,
        unmapped_hits,
        returned_images.len()
    );

    let mut sorted: Vec<_> = returned_images.into_iter().collect();
    sorted.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());

    // 每个返回的 image_id 都必须有至少一条 face 行（用 faces().list_by_image 验证）
    let mut bad: Vec<i64> = Vec::new();
    for (image_id, (score, fid)) in &sorted {
        let face_rows = db
            .transaction(|tx| tx.faces().list_by_image(*image_id))
            .expect("list_by_image");
        let has_face = !face_rows.is_empty();
        if !has_face {
            bad.push(*image_id);
        }
        eprintln!(
            "  image_id={} best_score={:.3} from_face={} faces_on_image={}{}",
            image_id,
            score,
            fid,
            face_rows.len(),
            if has_face { "" } else { " *** NO FACE ROW ***" }
        );
    }

    assert_eq!(
        bad,
        Vec::<i64>::new(),
        "face search returned no-face images: {bad:?}"
    );

    // 结构 sanity
    assert!(!all_faces.is_empty(), "no faces in DB");
    eprintln!("RESULT: all {} returned images have face rows — OK", sorted.len());
}
