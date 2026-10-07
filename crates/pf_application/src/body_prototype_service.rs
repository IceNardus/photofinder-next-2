//! BodyPrototypeService — 维护 person ↔ body prototypes 一致性.
//!
//! V2 职责：
//! - `rebuild_for_person(person_id)`：列 bodies + embeddings → selector → 删旧 + 插新
//! - `list_for_person(person_id)`：返回 (crop_type, Vec<f32>) 列表供 candidate retrieval
//! - `delete_for_person(person_id)`：软删所有 body prototypes

use std::collections::HashMap;
use std::sync::Arc;

use pf_ai::body::BODY_MODEL_NAME;
use pf_database::{Database, NewPersonBodyPrototype, PersonBodyPrototypeRow};
use tracing::{debug, warn};

use crate::body_prototype_selector::{select_body_prototypes_for_person, BodyPrototypeCandidate};
use crate::error::ApplicationError;

/// Phase V2 Body Prototype CRUD.
#[derive(Clone)]
pub struct BodyPrototypeService {
    db: Arc<Database>,
}

impl BodyPrototypeService {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// 重建一个 person 的所有 body prototypes.
    pub fn rebuild_for_person(&self, person_id: i64) -> Result<usize, ApplicationError> {
        self.db
            .transaction_with_retry(|tx| {
                // 1) 列 bodies
                let bodies = tx.bodies().list_by_person(person_id)?;
                if bodies.is_empty() {
                    debug!(person_id, "no bodies, skip rebuild");
                    return Ok(0);
                }

                // 2) 拉各 body 的 yutu_reid embedding
                let mut embeddings: HashMap<i64, (Vec<f32>, String)> = HashMap::new();
                for body in &bodies {
                    if let Some(vec) = tx.bodies().get_embedding(body.id, BODY_MODEL_NAME)? {
                        let model_version = body.model_version.clone();
                        embeddings.insert(body.id, (vec, model_version));
                    }
                }

                // 3) selector
                let candidates: Vec<BodyPrototypeCandidate> =
                    select_body_prototypes_for_person(&bodies, &embeddings);

                // 4) 软删旧
                tx.person_body_prototypes().soft_delete_for_person(person_id)?;

                // 5) 写新
                let mut n = 0;
                for c in &candidates {
                    let new_proto = NewPersonBodyPrototype {
                        person_id,
                        body_id: c.body_id,
                        embedding: c.embedding.clone(),
                        crop_type: c.crop_type.clone(),
                        quality_score: Some(c.quality_score),
                        weight: c.weight,
                        model_version: c.model_version.clone(),
                    };
                    tx.person_body_prototypes().insert(&new_proto)?;
                    n += 1;
                }

                Ok::<_, pf_database::DatabaseError>(n)
            })
            .map_err(|e| {
                warn!(person_id, error = %e, "BodyPrototypeService::rebuild_for_person failed");
                ApplicationError::Database(e)
            })
    }

    /// 列出 person 的 active body prototypes.
    ///
    /// `exclude_image_id` 用于 LOO 测试时排除 query 自身的 prototypes。
    pub fn list_for_person(
        &self,
        person_id: i64,
        exclude_image_id: Option<i64>,
    ) -> Result<Vec<(String, Vec<f32>)>, ApplicationError> {
        let protos = self
            .db
            .transaction(|tx| {
                match exclude_image_id {
                    Some(img_id) => tx.person_body_prototypes().list_active_by_person_exclude_image(person_id, img_id),
                    None => tx.person_body_prototypes().list_active_by_person(person_id),
                }
            })
            .map_err(|e| ApplicationError::Database(e))?;
        Ok(protos
            .into_iter()
            .map(|p| (p.crop_type.clone(), p.embedding))
            .collect())
    }

    /// 列出 person 的所有 body prototype rows.
    pub fn list_rows_for_person(
        &self,
        person_id: i64,
    ) -> Result<Vec<PersonBodyPrototypeRow>, ApplicationError> {
        self.db
            .transaction(|tx| tx.person_body_prototypes().list_by_person(person_id))
            .map_err(|e| ApplicationError::Database(e))
    }

    /// 显式软删某 person 的所有 body prototypes.
    pub fn delete_for_person(&self, person_id: i64) -> Result<u64, ApplicationError> {
        self.db
            .transaction(|tx| tx.person_body_prototypes().soft_delete_for_person(person_id))
            .map_err(ApplicationError::Database)
    }

    /// 统计某 person 的 active body prototype 数.
    pub fn count_active_for_person(&self, person_id: i64) -> Result<i64, ApplicationError> {
        self.db
            .transaction(|tx| tx.person_body_prototypes().count_active_by_person(person_id))
            .map_err(ApplicationError::Database)
    }
}
