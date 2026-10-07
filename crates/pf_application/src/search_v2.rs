//! Phase 5 — Search V2 (face → persons via prototype) 的纯算法逻辑。
//!
//! 把 `SearchService::search_by_person` 的核心算法抽出来便于测试:
//! - 不依赖 SearchService / FacePipeline / Config
//! - 仅依赖 `&Database`, `&dyn VectorIndex`, `&[(PrototypeType, Vec<f32>)]`
//! - 在 `search.rs::search_by_person` 里被调用,负责 prototype 拿到后
//!   multi-prototype HNSW 搜索 + per-image 聚合 + 阈值过滤 + top_k 排序
//!
//! 单元测试在 tests/search_person.rs(用 in-memory DB + tempdir HNSW)。
//! 本模块不放单测,因为没有 DB / HNSW 难以构造有意义用例。

use std::collections::HashMap;

use pf_core::SearchResult;
use pf_database::{Database, FaceRow, PrototypeType};
use pf_vector::{SearchHit, VectorIndex};
use tracing::debug;

use crate::error::ApplicationError;

/// Multi-prototype HNSW 搜索 + per-image 聚合 + 阈值过滤 + top_k。
///
/// 参数:
/// - `face_index`: face 向量索引(HNSW;`Arc<dyn VectorIndex>` deref to `&dyn`)
/// - `db`: 数据库句柄
/// - `prototypes`: person 的 prototype embeddings(Phase 3 PrototypeService 产出)
/// - `person_id`: 仅保留 person_id == person_id 的 hit(避免跨 person 串扰)
/// - `top_k`: 最多返回多少 image
/// - `threshold`: cosine 相似度阈值(>= 即视为命中)
///
/// 聚合规则:同一 image_id 多个 prototype 命中时,保留 score 最高的 (face_id, score)。
/// 返回 `SearchResult { image_id, target_id=face_id, score, rank }`。
pub fn search_by_person_impl(
    face_index: &dyn VectorIndex,
    db: &Database,
    prototypes: &[(PrototypeType, Vec<f32>)],
    person_id: i64,
    top_k: usize,
    threshold: f32,
) -> Result<Vec<SearchResult>, ApplicationError> {
    if prototypes.is_empty() {
        return Ok(Vec::new());
    }
    let fetch_k = top_k.max(1).saturating_mul(4);

    // 先搜所有 prototype,收集全部 hits
    let mut all_hits: Vec<SearchHit> = Vec::new();
    for (_ptype, proto_emb) in prototypes {
        let hits: Vec<SearchHit> = face_index
            .search(proto_emb, fetch_k)
            .map_err(ApplicationError::Vector)?;
        all_hits.extend(hits);
    }

    // Audit B3:一次 IN-clause 批量反查所有 hit 的 face(此前每个 hit 单独开事务,
    // ≤5 prototype × 4×top_k hit → top_k=50 时最多 ~1000 次事务)。
    let vector_ids: Vec<i64> = all_hits.iter().map(|h| h.id).collect();
    let face_by_vector: HashMap<i64, FaceRow> = db
        .transaction(|tx| tx.faces().list_by_vector_ids(&vector_ids))
        .map_err(|e| ApplicationError::Internal(format!("face lookup: {e}")))?
        .into_iter()
        .filter_map(|f| f.vector_id.map(|vid| (vid, f)))
        .collect();

    // 聚合: image_id → (best_face_id, best_score)
    let mut best_per_image: HashMap<i64, (i64, f32)> = HashMap::new();
    for hit in all_hits {
        let Some(face) = face_by_vector.get(&hit.id) else { continue };
        if face.person_id != Some(person_id) {
            continue;
        }
        let entry = best_per_image.entry(face.image_id);
        match entry {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert((face.id, hit.score));
            }
            std::collections::hash_map::Entry::Occupied(mut e) => {
                if hit.score > e.get().1 {
                    e.insert((face.id, hit.score));
                }
            }
        }
    }

    let mut scored: Vec<(i64, i64, f32)> = best_per_image
        .into_iter()
        .map(|(image_id, (face_id, score))| (image_id, face_id, score))
        .filter(|(_, _, s)| *s >= threshold)
        .collect();
    scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);

    debug!(
        person_id,
        n_prototypes = prototypes.len(),
        n_results = scored.len(),
        "search_by_person done"
    );

    let mut out = Vec::with_capacity(scored.len());
    for (rank, (image_id, face_id, score)) in scored.into_iter().enumerate() {
        out.push(SearchResult {
            image_id,
            target_id: face_id,
            score,
            rank: rank + 1,
        });
    }
    Ok(out)
}