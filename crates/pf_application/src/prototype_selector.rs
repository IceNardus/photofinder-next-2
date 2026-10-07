//! Prototype selection — 从一个 person 的 faces 中挑选代表性 face 作为 prototype。
//!
//! 设计(P2 item 21/22):
//! - item 21: 按 pose 分桶 — frontal / left / right / high_quality。
//!   `classify_face_to_prototype_type` 用 yaw 分桶;high_quality 桶 = 全局 quality 最高 face。
//! - item 22: 桶内去冗余 — 每个 pose 桶内对 embedding 做 greedy single-linkage 聚类,
//!   与已选代表 cosine ≥ `DEDUP_COSINE_THRESHOLD` 的近重复 face 只保留 quality 最高者;
//!   总 prototype 数上限 `MAX_PROTOTYPES`(cap 8)。
//! - 桶优先级 frontal > left > right > high_quality:cap 截断时保证 pose 覆盖。
//!   Glasses 类型暂不产出(等接入 glasses 检测)。
//! - 过滤 quality < ACCEPTABLE_QUALITY 的 face;yaw=None 时按 frontal 处理。
//!
//! 014 后:`PrototypeCandidate` 直接携带 embedding (Vec<f32>) — 不再依赖
//! `face_embeddings` 表做 FK roundtrip;`PrototypeService::rebuild_for_person`
//! 拿到 candidate 后直接 `INSERT person_prototypes(embedding=...)`。
//!
//! 函数入口:
//! - `classify_face_to_prototype_type`: 单 face 分类(测试用)
//! - `select_prototypes_for_person`: 给定 faces + embeddings,产出 PrototypeCandidate 列表

use std::collections::HashMap;

use crate::cluster_v2::cosine_similarity;
use pf_ai::quality::ACCEPTABLE_QUALITY;
use pf_database::{FaceRow, PrototypeType};

/// yaw 阈值(度),用于区分正脸 / 侧脸。
///
/// 正脸: |yaw| < FRONTAL_YAW_MAX_DEG
/// 侧脸: |yaw| > PROFILE_YAW_MIN_DEG
const FRONTAL_YAW_MAX_DEG: f32 = 15.0;
const PROFILE_YAW_MIN_DEG: f32 = 30.0;

/// 桶内去重阈值(item 22):同一 pose 桶内两个 face 的 embedding cosine ≥ 该值时
/// 视为近重复(同图重复检测 / 几乎相同的 crop),只保留 quality 更高者。
///
/// 取值依据:同人不同照片的 ArcFace cosine 通常在 0.5–0.7(diagnostics 实测),
/// 只有真正近重复的 crop 才超过 0.90 → 不会被误合并。属于 #123 可调参数。
const DEDUP_COSINE_THRESHOLD: f32 = 0.90;

/// 每个 person 的 prototype 数量上限(item 22 cap 8)。
const MAX_PROTOTYPES: usize = 8;

/// Prototype 候选 — selector 输出。
///
/// 014 后:embedding 内嵌进 candidate,不再走 face_embeddings 表。
/// PrototypeService::rebuild_for_person 直接 INSERT 新 prototype,
/// embedding 来自此处,无需 face_embeddings FK。
#[derive(Debug, Clone)]
pub struct PrototypeCandidate {
    /// 选中的 face_id
    pub face_id: i64,
    /// 该 face 的 prototype_type
    pub prototype_type: PrototypeType,
    /// 该 face 的 embedding(供写 person_prototypes.embedding BLOB)
    pub embedding: Vec<f32>,
    /// 来源 face 的 pose_score(供 person_prototypes.pose_score)
    pub pose_score: f32,
    /// 来源 face 的 yaw(供 person_prototypes.pose_yaw)
    pub pose_yaw: f32,
    /// 来源 face 的 quality_score(供 person_prototypes.quality_score)
    pub quality_score: f32,
    /// 投票权重(由 selector 基于 prototype_type 计算:
    /// frontal=1.0,profile=0.8,high_quality=1.2,general=0.6)
    pub weight: f32,
    /// embedding 模型版本(供 person_prototypes.model_version)
    pub model_version: String,
}

/// 将单 face 按 yaw 分类到 prototype_type(测试用)。
///
/// 注意:Glasses 永不从此函数返回(Phase 3 不产出 glasses prototype)。
/// 返回 None 表示该 face 不适合做任何 prototype(quality 过低)。
pub fn classify_face_to_prototype_type(face: &FaceRow) -> Option<PrototypeType> {
    if face.quality < ACCEPTABLE_QUALITY {
        return None;
    }
    match face.yaw {
        None => Some(PrototypeType::Frontal), // yaw 未知 → 视作正脸
        Some(y) => {
            let abs_y = y.abs();
            if abs_y < FRONTAL_YAW_MAX_DEG {
                Some(PrototypeType::Frontal)
            } else if y < -PROFILE_YAW_MIN_DEG {
                Some(PrototypeType::LeftProfile)
            } else if y > PROFILE_YAW_MIN_DEG {
                Some(PrototypeType::RightProfile)
            } else {
                // 15° ≤ |yaw| ≤ 30°:过渡区域,暂归 frontal
                Some(PrototypeType::Frontal)
            }
        }
    }
}

/// 给定 person 的所有 faces + 它们的 embedding,产出 prototype 候选列表。
///
/// 规则(item 21/22):
/// 1. 过滤 quality < ACCEPTABLE_QUALITY 且必须有 embedding 的 face
/// 2. item 21:按 pose 分桶 — frontal / left_profile / right_profile
/// 3. item 22:桶内 greedy single-linkage 聚类去重(cosine ≥ DEDUP_COSINE_THRESHOLD
///    视为近重复,只保留 quality 最高者),每个 cluster 产出 1 个代表
/// 4. high_quality 桶:全局 quality 最高 face;若其已是某 pose 桶代表则跳过(避免
///    embedding 重复 — greedy 下几乎总是如此,该分支作为兜底)
/// 5. cap 8:按桶优先级 frontal > left > right > high_quality 填充,超出截断。
///    不产出 Glasses prototype(Phase 3 范围外)
///
/// `embeddings` key = face_id, value = (Vec<f32>, String) = (embedding, model_version)。
/// 若 face_id 不在 embeddings 中,该 face 在 selector 中被忽略(必须有 embedding 才能做 prototype)。
pub fn select_prototypes_for_person(
    faces: &[FaceRow],
    embeddings: &HashMap<i64, (Vec<f32>, String)>,
) -> Vec<PrototypeCandidate> {
    // 1. 过滤
    let eligible: Vec<&FaceRow> = faces
        .iter()
        .filter(|f| f.quality >= ACCEPTABLE_QUALITY)
        .filter(|f| embeddings.contains_key(&f.id))
        .collect();
    if eligible.is_empty() {
        return Vec::new();
    }

    let mk_candidate = |face: &FaceRow, pt: PrototypeType| -> PrototypeCandidate {
        let (vec, model_version) = embeddings.get(&face.id).expect("pre-filtered").clone();
        PrototypeCandidate {
            face_id: face.id,
            prototype_type: pt,
            embedding: vec,
            pose_score: face.pose_score.unwrap_or(0.0),
            pose_yaw: face.yaw.unwrap_or(0.0),
            quality_score: face.quality,
            weight: default_weight_for(pt),
            model_version,
        }
    };

    // 2. item 21:按 pose 分桶(high_quality 单独处理)
    let mut frontal: Vec<&FaceRow> = Vec::new();
    let mut left: Vec<&FaceRow> = Vec::new();
    let mut right: Vec<&FaceRow> = Vec::new();
    for face in &eligible {
        match classify_face_to_prototype_type(face) {
            Some(PrototypeType::LeftProfile) => left.push(face),
            Some(PrototypeType::RightProfile) => right.push(face),
            Some(PrototypeType::Frontal) | _ => frontal.push(face),
        }
    }

    // 3+5. 桶内去重聚类 + cap 8;桶优先级保证 pose 覆盖
    let mut out: Vec<PrototypeCandidate> = Vec::with_capacity(MAX_PROTOTYPES);
    let mut selected_ids: std::collections::HashSet<i64> = std::collections::HashSet::new();
    for (bucket, pt) in [
        (frontal.as_slice(), PrototypeType::Frontal),
        (left.as_slice(), PrototypeType::LeftProfile),
        (right.as_slice(), PrototypeType::RightProfile),
    ] {
        for rep in dedup_cluster_reps(bucket, embeddings) {
            if out.len() >= MAX_PROTOTYPES {
                break;
            }
            selected_ids.insert(rep.id);
            out.push(mk_candidate(rep, pt));
        }
    }

    // 4. high_quality 兜底:全局最佳 face 未被 pose 桶选走时才补发
    if out.len() < MAX_PROTOTYPES {
        if let Some(best) = best_quality_face(&eligible) {
            if !selected_ids.contains(&best.id) {
                out.push(mk_candidate(best, PrototypeType::HighQuality));
            }
        }
    }

    out
}

/// prototype 默认投票权重(Phase 3 经验值,后续可基于实际数据调优)。
fn default_weight_for(pt: PrototypeType) -> f32 {
    match pt {
        PrototypeType::Frontal => 1.0,
        PrototypeType::LeftProfile | PrototypeType::RightProfile => 0.8,
        PrototypeType::HighQuality => 1.2,
        PrototypeType::General => 0.6,
        PrototypeType::Glasses => 0.0, // Phase 3 不产出
    }
}

/// 桶内去重(item 22):按 quality 降序贪婪扫描,凡是 embedding 与任一已选代表
/// cosine ≥ `DEDUP_COSINE_THRESHOLD` 的 face 都视为同一 cluster 的近重复,跳过。
/// 返回每个 cluster 的代表(quality 最高者优先)。这是 single-linkage 分层聚类的
/// 确定性贪婪近似 — 近重复 crop 合并,真正不同的 look 保留。
fn dedup_cluster_reps<'a>(
    bucket: &[&'a FaceRow],
    embeddings: &HashMap<i64, (Vec<f32>, String)>,
) -> Vec<&'a FaceRow> {
    let mut sorted: Vec<&FaceRow> = bucket.to_vec();
    sorted.sort_by(|a, b| b.quality.partial_cmp(&a.quality).unwrap_or(std::cmp::Ordering::Equal));
    let mut reps: Vec<&FaceRow> = Vec::new();
    'outer: for face in &sorted {
        let emb = &embeddings.get(&face.id).expect("pre-filtered").0;
        for rep in &reps {
            let rep_emb = &embeddings.get(&rep.id).expect("pre-filtered").0;
            if cosine_similarity(emb, rep_emb) >= DEDUP_COSINE_THRESHOLD {
                continue 'outer;
            }
        }
        reps.push(face);
    }
    reps
}

/// eligible 中 quality 最高的 face。
fn best_quality_face<'a>(faces: &[&'a FaceRow]) -> Option<&'a FaceRow> {
    faces
        .iter()
        .max_by(|a, b| a.quality.partial_cmp(&b.quality).unwrap_or(std::cmp::Ordering::Equal))
        .copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use pf_core::BBox;

    fn make_face(id: i64, quality: f32, yaw: Option<f32>) -> FaceRow {
        FaceRow {
            id,
            image_id: 1,
            person_id: Some(1),
            bbox: BBox { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            detector_score: 0.9,
            detector_model: "scrfd-500m-bnkps".into(),
            keypoints_json: None,
            yaw,
            pitch: None,
            roll: None,
            quality,
            blur_score: None,
            pose_score: Some(0.7),
            face_area_score: None,
            alignment_version: Some("v1".into()),
            embedding_model: Some("arcface".into()),
            model_version: "arcface@v1".into(),
            vector_id: Some(id),
            hnsw_handle: Some(id),
            status: pf_database::FaceStatus::Indexed,
            index_status: pf_database::FaceIndexStatus::Indexed,
            indexed_at: Some(Utc::now()),
            error_message: None,
            last_attempt_at: Some(Utc::now()),
            index_generation: 0,
            cluster_score: None,
            cluster_method: None,
            created_at: Utc::now(),
        }
    }

    fn embs_for(ids: &[i64]) -> HashMap<i64, (Vec<f32>, String)> {
        let mut m = HashMap::new();
        for id in ids {
            m.insert(*id, (vec![0.0; 512], "arcface@v1".into()));
        }
        m
    }

    /// 显式指定 (face_id, embedding) 的 embeddings 表。
    fn embs(assignments: &[(i64, Vec<f32>)]) -> HashMap<i64, (Vec<f32>, String)> {
        let mut m = HashMap::new();
        for (id, v) in assignments {
            m.insert(*id, (v.clone(), "arcface@v1".into()));
        }
        m
    }

    /// 维度 dim 的第 idx 个标准基向量(e_i,两两正交 → cosine=0)。
    fn basis(dim: usize, idx: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[idx] = 1.0;
        v
    }

    #[test]
    fn classify_no_yaw_is_frontal() {
        let f = make_face(1, 0.7, None);
        assert_eq!(
            classify_face_to_prototype_type(&f),
            Some(PrototypeType::Frontal)
        );
    }

    #[test]
    fn classify_low_quality_is_none() {
        let f = make_face(1, 0.3, None);
        assert_eq!(classify_face_to_prototype_type(&f), None);
    }

    #[test]
    fn classify_profile_by_yaw() {
        let left = make_face(1, 0.7, Some(-45.0));
        assert_eq!(
            classify_face_to_prototype_type(&left),
            Some(PrototypeType::LeftProfile)
        );
        let right = make_face(2, 0.7, Some(45.0));
        assert_eq!(
            classify_face_to_prototype_type(&right),
            Some(PrototypeType::RightProfile)
        );
        let mid = make_face(3, 0.7, Some(20.0)); // 15° < |yaw| < 30°
        assert_eq!(
            classify_face_to_prototype_type(&mid),
            Some(PrototypeType::Frontal)
        );
    }

    #[test]
    fn select_only_frontal_when_no_yaw() {
        let faces = vec![
            make_face(1, 0.7, None),
            make_face(2, 0.8, None),
            make_face(3, 0.9, None),
        ];
        let embs = embs_for(&[1, 2, 3]);
        let out = select_prototypes_for_person(&faces, &embs);
        // 没有 yaw → 全进 frontal 桶;零向量 embedding cosine=0,不去重 → 3 个 frontal
        // high_quality 兜底:最佳 face(id=3)已是 frontal 代表 → 跳过,不重复
        let types: Vec<_> = out.iter().map(|p| p.prototype_type).collect();
        assert_eq!(types.len(), 3);
        assert!(types.iter().all(|t| *t == PrototypeType::Frontal));
        assert!(!types.contains(&PrototypeType::HighQuality));
        assert!(!types.contains(&PrototypeType::LeftProfile));
        assert!(!types.contains(&PrototypeType::RightProfile));
        assert!(!types.contains(&PrototypeType::Glasses));
    }

    #[test]
    fn select_with_profile_when_yaw_present() {
        let faces = vec![
            make_face(1, 0.7, None),
            make_face(2, 0.8, Some(-45.0)),  // left profile
            make_face(3, 0.9, Some(50.0)),   // right profile, 最高 quality
            make_face(4, 0.6, Some(0.0)),    // frontal
        ];
        let embs = embs_for(&[1, 2, 3, 4]);
        let out = select_prototypes_for_person(&faces, &embs);
        // frontal 桶: face 1,4(零向量不去重)→ 2 个代表;left: face 2;right: face 3
        let types: Vec<_> = out.iter().map(|p| p.prototype_type).collect();
        assert!(types.contains(&PrototypeType::LeftProfile));
        assert!(types.contains(&PrototypeType::RightProfile));
        assert_eq!(types.iter().filter(|t| **t == PrototypeType::Frontal).count(), 2);
        assert_eq!(types.len(), 4);
        assert!(!types.contains(&PrototypeType::HighQuality));
        assert!(!types.contains(&PrototypeType::General));
    }

    #[test]
    fn select_filters_low_quality() {
        let faces = vec![
            make_face(1, 0.3, None), // too low
            make_face(2, 0.4, None), // too low
        ];
        let embs = embs_for(&[1, 2]);
        let out = select_prototypes_for_person(&faces, &embs);
        assert!(out.is_empty());
    }

    #[test]
    fn select_skips_face_without_embedding() {
        let faces = vec![
            make_face(1, 0.7, None), // 没有 embedding
            make_face(2, 0.8, None), // 有
        ];
        let embs = embs_for(&[2]);
        let out = select_prototypes_for_person(&faces, &embs);
        // face 1 被排除
        assert!(out.iter().all(|p| p.face_id != 1));
    }

    #[test]
    fn dedup_collapses_near_duplicate_faces() {
        // 3 frontal faces: face1(q=0.9, e1) 为代表;face2(q=0.85) 的 embedding 与 e1
        // 近重复(cos≈0.995 ≥ 阈值)→ 应被合并丢弃;face3(q=0.8, e2 正交) 保留
        let near = vec![0.995f32, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let faces = vec![
            make_face(1, 0.9, None),
            make_face(2, 0.85, None),
            make_face(3, 0.8, None),
        ];
        let embs = embs(&[
            (1, basis(10, 0)),
            (2, near),
            (3, basis(10, 1)),
        ]);
        let out = select_prototypes_for_person(&faces, &embs);
        let ids: Vec<_> = out.iter().map(|p| p.face_id).collect();
        assert!(ids.contains(&1), "e1 代表保留: {ids:?}");
        assert!(ids.contains(&3), "正交 face 保留: {ids:?}");
        assert!(!ids.contains(&2), "近重复 face2 应被合并丢弃: {ids:?}");
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn cap_limits_total_prototypes_to_eight() {
        // 10 个两两正交的 frontal face → 不去重,但 cap 8 截断保留 quality 最高 8 个
        let faces: Vec<FaceRow> = (1..=10)
            .map(|i| make_face(i, 0.5 + i as f32 * 0.04, None))
            .collect();
        let embs = embs(
            &(1..=10)
                .map(|i| (i, basis(10, i as usize - 1)))
                .collect::<Vec<_>>(),
        );
        let out = select_prototypes_for_person(&faces, &embs);
        let ids: Vec<_> = out.iter().map(|p| p.face_id).collect();
        assert_eq!(out.len(), 8, "cap 8: {ids:?}");
        for high_id in 3..=10 {
            assert!(ids.contains(&high_id), "quality 前 8 应保留 face{high_id}: {ids:?}");
        }
        assert!(!ids.contains(&1) && !ids.contains(&2), "quality 最低 2 个被截断: {ids:?}");
    }

    #[test]
    fn cap_preserves_pose_diversity() {
        // 7 frontal(e1..e7)+ 2 left(e8,e9)→ cap 8:frontal 7 个全保留,left 保留 1 个
        let mut faces: Vec<FaceRow> = (1..=7).map(|i| make_face(i, 0.7, None)).collect();
        faces.push(make_face(8, 0.9, Some(-45.0)));
        faces.push(make_face(9, 0.8, Some(-40.0)));
        let mut assignments: Vec<(i64, Vec<f32>)> = Vec::new();
        for i in 1..=7 {
            assignments.push((i, basis(10, i as usize - 1)));
        }
        assignments.push((8, basis(10, 7)));
        assignments.push((9, basis(10, 8)));
        let embs = embs(&assignments);
        let out = select_prototypes_for_person(&faces, &embs);
        let types: Vec<_> = out.iter().map(|p| p.prototype_type).collect();
        assert_eq!(out.len(), 8);
        assert_eq!(types.iter().filter(|t| **t == PrototypeType::Frontal).count(), 7);
        assert_eq!(types.iter().filter(|t| **t == PrototypeType::LeftProfile).count(), 1);
    }

    #[test]
    fn candidate_carries_inline_embedding_and_pose() {
        let faces = vec![make_face(1, 0.9, Some(0.0))];
        let embs = embs_for(&[1]);
        let out = select_prototypes_for_person(&faces, &embs);
        let frontal = out
            .iter()
            .find(|p| p.prototype_type == PrototypeType::Frontal)
            .unwrap();
        assert_eq!(frontal.embedding.len(), 512, "candidate carries inline embedding");
        assert_eq!(frontal.model_version, "arcface@v1");
        assert!(frontal.pose_score > 0.0);
        assert_eq!(frontal.pose_yaw, 0.0);
        assert!((frontal.quality_score - 0.9).abs() < 1e-6);
        assert!(frontal.weight > 0.0, "frontal weight should be positive");
    }

    #[test]
    fn default_weight_varies_by_type() {
        assert!((default_weight_for(PrototypeType::Frontal) - 1.0).abs() < 1e-6);
        assert!(default_weight_for(PrototypeType::HighQuality) > default_weight_for(PrototypeType::Frontal));
        assert!(default_weight_for(PrototypeType::General) < default_weight_for(PrototypeType::Frontal));
        assert!(default_weight_for(PrototypeType::Glasses) == 0.0);
    }
}