//! PrototypeService — 维护 person ↔ prototypes 一致性。
//!
//! Phase 3 责任:
//! - `rebuild_for_person(person_id)`: 列 faces + embeddings → selector → 删旧 + 插新
//! - `list_for_person(person_id)`: 给 Phase 5 search 用,返回 (PrototypeType, Vec<f32>)
//! - `delete_for_person(person_id)`: 走 FK 级联;提供显式方法便于测试
//!
//! 写入事务:
//! 1. SELECT person 全部 face rows
//! 2. SELECT 各 face 的 arcface embedding → HashMap
//! 3. 调 select_prototypes_for_person 拿到 candidates
//! 4. 单事务: DELETE FROM person_prototypes WHERE person_id=? + 逐个 candidate
//!    INSERT face_embeddings (拿 embedding_id) + INSERT person_prototypes

use std::collections::HashMap;
use std::sync::Arc;

use pf_core::FACE_MODEL_NAME;
use pf_database::{
    Database, FaceRow, NewPersonPrototype, PersonPrototypeRow, PrototypeType,
};
use tracing::{debug, warn};

use crate::error::ApplicationError;
use crate::prototype_selector::{
    select_prototypes_for_person, PrototypeCandidate,
};

/// Phase 3 Prototype CRUD。
#[derive(Clone)]
pub struct PrototypeService {
    db: Arc<Database>,
}

impl PrototypeService {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// 重建一个 person 的所有 prototypes。
    ///
    /// 014 后:embedding 内嵌进 person_prototypes,不再需要 face_embeddings 表。
    /// 单事务:列该 person 全部 face → 拉 embedding → 选 → 软删旧 → 插新。
    /// 出错时整个事务回滚,DB 状态不变。
    pub fn rebuild_for_person(&self, person_id: i64) -> Result<usize, ApplicationError> {
        self.db
            .transaction_with_retry(|tx| {
                // 1) 列 faces
                let faces: Vec<FaceRow> = tx.faces().list_by_person(person_id)?;
                if faces.is_empty() {
                    debug!(person_id, "no faces, skip rebuild");
                    return Ok(0);
                }

                // 2) 拉各 face 的 arcface embedding(走 face_embeddings JOIN)
                //    014: dim 派生自 embedding.len(),selector 不再需要单独字段
                let mut embeddings: HashMap<i64, (Vec<f32>, String)> = HashMap::new();
                for face in &faces {
                    if let Some(vec) = tx.faces().get_embedding(face.id, FACE_MODEL_NAME)? {
                        let model_version = face.model_version.clone();
                        embeddings.insert(face.id, (vec, model_version));
                    }
                }

                // 3) selector
                let candidates: Vec<PrototypeCandidate> =
                    select_prototypes_for_person(&faces, &embeddings);

                // 4) 软删旧(保留行,is_active=0,审计可查)
                tx.person_prototypes().soft_delete_for_person(person_id)?;

                // 5) 写新(每 candidate 直接内嵌 embedding)
                let mut n = 0;
                for c in &candidates {
                    let new_proto = NewPersonPrototype {
                        person_id,
                        face_id: c.face_id,
                        embedding: c.embedding.clone(),
                        prototype_type: c.prototype_type,
                        pose_score: Some(c.pose_score),
                        pose_yaw: Some(c.pose_yaw),
                        quality_score: Some(c.quality_score),
                        weight: c.weight,
                        model_version: c.model_version.clone(),
                    };
                    tx.person_prototypes().insert(&new_proto)?;
                    n += 1;
                }

                Ok::<_, pf_database::DatabaseError>(n)
            })
            .map_err(|e| {
                warn!(person_id, error = %e, "PrototypeService::rebuild_for_person failed");
                ApplicationError::Database(e)
            })
    }

    /// 列出 person 的 active prototypes(prototype_type + embedding)。
    ///
    /// Phase 5 search 会用:每条都是候选 query 向量。
    ///
    /// `exclude_image_id` 用于 LOO 测试时排除 query 自身的 prototypes。
    pub fn list_for_person(
        &self,
        person_id: i64,
        exclude_image_id: Option<i64>,
    ) -> Result<Vec<(PrototypeType, Vec<f32>)>, ApplicationError> {
        let protos = self
            .db
            .transaction(|tx| {
                match exclude_image_id {
                    Some(img_id) => tx.person_prototypes().list_active_by_person_exclude_image(person_id, img_id),
                    None => tx.person_prototypes().list_active_by_person(person_id),
                }
            })
            .map_err(|e| ApplicationError::Database(e))?;
        Ok(protos
            .into_iter()
            .map(|p| (p.prototype_type, p.embedding))
            .collect())
    }

    /// 显式软删某 person 的所有 prototypes(测试 / 管理工具用)。
    pub fn delete_for_person(&self, person_id: i64) -> Result<u64, ApplicationError> {
        self.db
            .transaction(|tx| tx.person_prototypes().soft_delete_for_person(person_id))
            .map_err(ApplicationError::Database)
    }

    /// 列某 person 的所有 prototype rows(诊断用,包含软删)。
    pub fn list_rows_for_person(
        &self,
        person_id: i64,
    ) -> Result<Vec<PersonPrototypeRow>, ApplicationError> {
        self.db
            .transaction(|tx| tx.person_prototypes().list_by_person(person_id))
            .map_err(ApplicationError::Database)
    }

    /// 列某 person 的 active prototype rows(标准用法)。
    ///
    /// 014 后 `rebuild_for_person` 走软删,旧 rows 保留作审计;调用方做常规统计时应使用本方法。
    pub fn list_active_rows_for_person(
        &self,
        person_id: i64,
    ) -> Result<Vec<PersonPrototypeRow>, ApplicationError> {
        self.db
            .transaction(|tx| tx.person_prototypes().list_active_by_person(person_id))
            .map_err(ApplicationError::Database)
    }

    /// 统计某 person 的 active prototype 数。
    pub fn count_active_for_person(&self, person_id: i64) -> Result<i64, ApplicationError> {
        self.db
            .transaction(|tx| tx.person_prototypes().count_active_by_person(person_id))
            .map_err(ApplicationError::Database)
    }
}

// =============================================================================
// Welford 在线更新 / L2 归一化工具
// =============================================================================

/// L2 归一化向量（原地修改）。
pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for val in v.iter_mut() {
        *val /= norm;
    }
}

/// Welford 在线均值更新 + L2 归一化。
///
/// 用于增量更新 prototype embedding：
/// - `old_mean`: 当前 prototype embedding
/// - `n`: 当前 prototype 覆盖的 face 数
/// - `new_face`: 新增 face 的 embedding
///
/// 返回 (new_mean, new_n)。
///
/// 注意：Phase 19 当前使用全量 rebuild 策略，本函数为未来增量更新预留。
pub fn welford_update_prototype(
    old_mean: &[f32],
    n: usize,
    new_face: &[f32],
) -> (Vec<f32>, usize) {
    let dim = old_mean.len();
    let new_n = n + 1;
    let mut new_mean = vec![0.0f32; dim];

    for i in 0..dim {
        // new_mean = (old_mean * n + new_face) / (n + 1)
        new_mean[i] = (old_mean[i] * n as f32 + new_face[i]) / new_n as f32;
    }

    // L2 normalize
    l2_normalize(&mut new_mean);

    (new_mean, new_n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_normalize_unit_vector() {
        let mut v = vec![0.5f32, 0.5, 0.5, 0.5];
        l2_normalize(&mut v);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_preserves_direction() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_welford_update_single_face() {
        // First face: new_mean = new_face
        let face = vec![0.5f32, 0.5, 0.5, 0.5];
        let (mean, n) = welford_update_prototype(&[0.0f32; 4], 0, &face);
        let norm = mean.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert_eq!(n, 1);
    }

    #[test]
    fn test_welford_update_blends_faces() {
        // Two orthogonal unit vectors should blend to a 45-degree unit vector
        let face1 = vec![1.0f32, 0.0];
        let face2 = vec![0.0f32, 1.0];

        // After first face
        let (mean1, n1) = welford_update_prototype(&[0.0f32; 2], 0, &face1);
        assert!((mean1[0] - 1.0).abs() < 1e-6);
        assert_eq!(n1, 1);

        // After second face: mean = (face1 + face2) / 2 = [0.5, 0.5], then normalize to [0.707, 0.707]
        let (mean2, n2) = welford_update_prototype(&mean1, n1, &face2);
        assert!((mean2[0] - 0.707).abs() < 1e-3);
        assert!((mean2[1] - 0.707).abs() < 1e-3);
        let norm = mean2.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert_eq!(n2, 2);
    }
}