//! PersonBodyPrototype repository (V2).
//!
//! 字段：
//!   id / person_id / body_id / embedding / dim / crop_type
//!   / quality_score / weight / model_version / is_active / created_at

use chrono::{DateTime, Utc};
use rusqlite::params;

use crate::error::DatabaseError;
use crate::Transaction;

/// 新插入的 body prototype（不含 id）。
/// crop_type 使用字符串: "head_body" / "upper_body" / "bbox_tight" / "bbox_p20"
#[derive(Debug, Clone)]
pub struct NewPersonBodyPrototype {
    pub person_id: i64,
    pub body_id: i64,
    pub embedding: Vec<f32>,
    pub crop_type: String,
    pub quality_score: Option<f32>,
    pub weight: f32,
    pub model_version: String,
}

/// 数据库行 → PersonBodyPrototypeRow。
#[derive(Debug, Clone)]
pub struct PersonBodyPrototypeRow {
    pub id: i64,
    pub person_id: i64,
    pub body_id: i64,
    pub embedding: Vec<f32>,
    pub dim: usize,
    pub crop_type: String,
    pub quality_score: Option<f32>,
    pub weight: f32,
    pub model_version: String,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
}

/// PersonBodyPrototype repository.
pub struct PersonBodyPrototypeRepository<'tx, 'db> {
    pub(crate) tx: &'tx mut Transaction<'db>,
}

impl<'tx, 'db> PersonBodyPrototypeRepository<'tx, 'db> {
    /// 插入新 body prototype。返回 id。
    pub fn insert(&mut self, p: &NewPersonBodyPrototype) -> Result<i64, DatabaseError> {
        let bytes = embedding_to_blob(&p.embedding);
        self.tx.execute(
            "INSERT INTO person_body_prototypes
             (person_id, body_id, embedding, dim, crop_type, quality_score, weight, model_version, is_active)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1)",
            params![
                p.person_id,
                p.body_id,
                bytes,
                p.embedding.len() as i64,
                p.crop_type,
                p.quality_score,
                p.weight,
                p.model_version,
            ],
        )?;
        Ok(self.tx.last_insert_rowid())
    }

    /// 软删：is_active = 0。
    pub fn soft_delete(&mut self, id: i64) -> Result<(), DatabaseError> {
        self.tx.execute(
            "UPDATE person_body_prototypes SET is_active = 0 WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    /// 列出 person 的激活 body prototypes.
    pub fn list_active_by_person(
        &self,
        person_id: i64,
    ) -> Result<Vec<PersonBodyPrototypeRow>, DatabaseError> {
        self.list_by_person_internal(person_id, true)
    }

    /// 列出 person 的激活 body prototypes，排除指定 image 的 body（用于 LOO 测试）。
    ///
    /// 通过 JOIN bodies 表过滤掉来自该 image 的 prototypes。
    pub fn list_active_by_person_exclude_image(
        &self,
        person_id: i64,
        exclude_image_id: i64,
    ) -> Result<Vec<PersonBodyPrototypeRow>, DatabaseError> {
        let sql = "SELECT pbp.id, pbp.person_id, pbp.body_id, pbp.embedding, pbp.dim,
                          pbp.crop_type, pbp.quality_score, pbp.weight, pbp.model_version,
                          pbp.is_active, pbp.created_at
                   FROM person_body_prototypes pbp
                   JOIN bodies b ON b.id = pbp.body_id
                   WHERE pbp.person_id = ?1 AND pbp.is_active = 1 AND b.image_id != ?2
                   ORDER BY pbp.quality_score DESC, pbp.id";
        let mut stmt = self.tx.prepare(sql)?;
        let mut rows = stmt.query(params![person_id, exclude_image_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_body_prototype(row)?);
        }
        Ok(out)
    }

    /// 列出 person 的所有 body prototypes（含软删）。
    pub fn list_by_person(
        &self,
        person_id: i64,
    ) -> Result<Vec<PersonBodyPrototypeRow>, DatabaseError> {
        self.list_by_person_internal(person_id, false)
    }

    fn list_by_person_internal(
        &self,
        person_id: i64,
        active_only: bool,
    ) -> Result<Vec<PersonBodyPrototypeRow>, DatabaseError> {
        let sql = if active_only {
            "SELECT id, person_id, body_id, embedding, dim, crop_type, quality_score,
                    weight, model_version, is_active, created_at
             FROM person_body_prototypes
             WHERE person_id = ?1 AND is_active = 1
             ORDER BY quality_score DESC, id"
        } else {
            "SELECT id, person_id, body_id, embedding, dim, crop_type, quality_score,
                    weight, model_version, is_active, created_at
             FROM person_body_prototypes
             WHERE person_id = ?1
             ORDER BY id"
        };
        let mut stmt = self.tx.prepare(sql)?;
        let mut rows = stmt.query(params![person_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_body_prototype(row)?);
        }
        Ok(out)
    }

    /// 按 person_id 软删所有 body prototype.
    pub fn soft_delete_for_person(&mut self, person_id: i64) -> Result<u64, DatabaseError> {
        let n = self.tx.execute(
            "UPDATE person_body_prototypes SET is_active = 0 WHERE person_id = ?1 AND is_active = 1",
            params![person_id],
        )?;
        Ok(n as u64)
    }

    /// 统计某 person 的 active body prototype 数.
    pub fn count_active_by_person(&self, person_id: i64) -> Result<i64, DatabaseError> {
        let n: i64 = self.tx.query_row(
            "SELECT COUNT(*) FROM person_body_prototypes
             WHERE person_id = ?1 AND is_active = 1",
            params![person_id],
            |r| r.get(0),
        )?;
        Ok(n)
    }
}

fn embedding_to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

fn blob_to_embedding(blob: &[u8], dim: usize) -> Result<Vec<f32>, DatabaseError> {
    if blob.len() != dim * 4 {
        return Err(DatabaseError::Conversion(format!(
            "body prototype embedding blob size mismatch: {} bytes for dim={}",
            blob.len(),
            dim
        )));
    }
    let mut out = Vec::with_capacity(dim);
    for chunk in blob.chunks_exact(4) {
        let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        out.push(f32::from_le_bytes(bytes));
    }
    Ok(out)
}

fn row_to_body_prototype(row: &rusqlite::Row<'_>) -> Result<PersonBodyPrototypeRow, DatabaseError> {
    let created_str: String = row.get(10)?;
    let created_at = DateTime::parse_from_rfc3339(&created_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| DatabaseError::Conversion(format!("created_at parse: {e}")))?;
    let dim: i64 = row.get(4)?;
    let blob: Vec<u8> = row.get(3)?;
    let embedding = blob_to_embedding(&blob, dim as usize)?;
    let is_active_int: i64 = row.get(9)?;
    Ok(PersonBodyPrototypeRow {
        id: row.get(0)?,
        person_id: row.get(1)?,
        body_id: row.get(2)?,
        embedding,
        dim: dim as usize,
        crop_type: row.get(5)?,
        quality_score: row.get(6)?,
        weight: row.get(7)?,
        model_version: row.get(8)?,
        is_active: is_active_int != 0,
        created_at,
    })
}
