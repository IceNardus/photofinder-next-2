//! Bodies repository (V2).
//!
//! 提供 body 表的 CRUD + body_embeddings 表的 embedding 查询。

use chrono::{DateTime, Utc};
use rusqlite::params;

use crate::error::DatabaseError;
use crate::Transaction;

/// Body row（对应 015_bodies.sql 的 bodies 表）。
#[derive(Debug, Clone)]
pub struct BodyRow {
    pub id: i64,
    pub image_id: i64,
    pub person_id: Option<i64>,
    pub bbox_x: f32,
    pub bbox_y: f32,
    pub bbox_w: f32,
    pub bbox_h: f32,
    pub crop_type: String,
    pub embedding_model: String,
    pub model_version: String,
    pub vector_id: Option<i64>,
    pub quality_score: f32,
    pub status: String,
    pub created_at: DateTime<Utc>,
}

/// New body to insert.
#[derive(Debug, Clone)]
pub struct NewBody {
    pub image_id: i64,
    pub bbox_x: f32,
    pub bbox_y: f32,
    pub bbox_w: f32,
    pub bbox_h: f32,
    pub crop_type: String,
    pub embedding_model: String,
    pub model_version: String,
    pub quality_score: f32,
}

/// Bodies repository.
pub struct BodiesRepository<'tx, 'db> {
    pub(crate) tx: &'tx mut Transaction<'db>,
}

impl<'tx, 'db> BodiesRepository<'tx, 'db> {
    /// 列出某 person 的所有 bodies.
    pub fn list_by_person(&self, person_id: i64) -> Result<Vec<BodyRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, image_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h,
                    crop_type, embedding_model, model_version, vector_id, quality_score,
                    status, created_at
             FROM bodies WHERE person_id = ?1 ORDER BY quality_score DESC",
        )?;
        let mut rows = stmt.query(params![person_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_body(row)?);
        }
        Ok(out)
    }

    /// 取一个 body 的 embedding（按 model_name 过滤）。
    pub fn get_embedding(
        &self,
        body_id: i64,
        model_name: &str,
    ) -> Result<Option<Vec<f32>>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT dimension, vector FROM body_embeddings
             WHERE body_id = ?1 AND model_name = ?2 LIMIT 1",
        )?;
        let mut rows = stmt.query(params![body_id, model_name])?;
        if let Some(row) = rows.next()? {
            let dim: i64 = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            Ok(Some(blob_to_vector(&blob, dim as usize)?))
        } else {
            Ok(None)
        }
    }

    /// 通过 vector_id 查找 body.
    pub fn get_by_vector_id(&self, vector_id: i64) -> Result<Option<BodyRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, image_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h,
                    crop_type, embedding_model, model_version, vector_id, quality_score,
                    status, created_at
             FROM bodies WHERE vector_id = ?1 LIMIT 1",
        )?;
        let mut rows = stmt.query(params![vector_id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row_to_body(row)?))
        } else {
            Ok(None)
        }
    }

    /// 批量通过 vector_id 查找 bodies（用于双通道 candidate retrieval）。
    pub fn list_by_vector_ids(&self, vector_ids: &[i64]) -> Result<Vec<BodyRow>, DatabaseError> {
        if vector_ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders: String = vector_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT id, image_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h,
                    crop_type, embedding_model, model_version, vector_id, quality_score,
                    status, created_at
             FROM bodies WHERE vector_id IN ({})",
            placeholders
        );
        let mut stmt = self.tx.prepare(&sql)?;
        let params: Vec<&dyn rusqlite::ToSql> = vector_ids.iter().map(|v| v as &dyn rusqlite::ToSql).collect();
        let mut rows = stmt.query(params.as_slice())?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_body(row)?);
        }
        Ok(out)
    }

    /// 插入新 body。返回 id。
    pub fn insert(&mut self, body: &NewBody) -> Result<i64, DatabaseError> {
        self.tx.execute(
            "INSERT INTO bodies
             (image_id, bbox_x, bbox_y, bbox_w, bbox_h,
              crop_type, embedding_model, model_version, quality_score, status)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'pending')",
            params![
                body.image_id,
                body.bbox_x,
                body.bbox_y,
                body.bbox_w,
                body.bbox_h,
                body.crop_type,
                body.embedding_model,
                body.model_version,
                body.quality_score,
            ],
        )?;
        Ok(self.tx.last_insert_rowid())
    }

    /// 标记 body 已索引（设置 vector_id）。
    pub fn mark_indexed(&mut self, body_id: i64, vector_id: i64) -> Result<(), DatabaseError> {
        self.tx.execute(
            "UPDATE bodies SET vector_id = ?1, status = 'indexed' WHERE id = ?2",
            params![vector_id, body_id],
        )?;
        Ok(())
    }

    /// 插入 body embedding。
    pub fn insert_embedding(
        &mut self,
        body_id: i64,
        model_name: &str,
        model_version: &str,
        vector: &[f32],
        vector_id: i64,
    ) -> Result<(), DatabaseError> {
        let dim = vector.len() as i64;
        let blob: Vec<u8> = vector
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        self.tx.execute(
            "INSERT INTO body_embeddings (body_id, model_name, model_version, dimension, vector, vector_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![body_id, model_name, model_version, dim, blob, vector_id],
        )?;
        Ok(())
    }

    /// 设置 body 的 person_id。
    pub fn set_person(&mut self, body_id: i64, person_id: Option<i64>) -> Result<(), DatabaseError> {
        self.tx.execute(
            "UPDATE bodies SET person_id = ?1 WHERE id = ?2",
            params![person_id, body_id],
        )?;
        Ok(())
    }

    /// 统计某 person 的 body 数。
    pub fn count_by_person(&self, person_id: i64) -> Result<i64, DatabaseError> {
        let n: i64 = self.tx.query_row(
            "SELECT COUNT(*) FROM bodies WHERE person_id = ?1",
            params![person_id],
            |r| r.get(0),
        )?;
        Ok(n)
    }

    /// 标记 body 失败。
    pub fn mark_failed(&mut self, body_id: i64, error: &str) -> Result<(), DatabaseError> {
        self.tx.execute(
            "UPDATE bodies SET status = 'failed' WHERE id = ?1",
            params![body_id],
        )?;
        Ok(())
    }
}

fn blob_to_vector(blob: &[u8], dim: usize) -> Result<Vec<f32>, DatabaseError> {
    if blob.len() != dim * 4 {
        return Err(DatabaseError::Conversion(format!(
            "body embedding blob size mismatch: {} bytes for dim={}",
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

fn row_to_body(row: &rusqlite::Row<'_>) -> Result<BodyRow, DatabaseError> {
    let created_str: String = row.get(13)?;
    let created_at = DateTime::parse_from_rfc3339(&created_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| DatabaseError::Conversion(format!("created_at parse: {e}")))?;
    Ok(BodyRow {
        id: row.get(0)?,
        image_id: row.get(1)?,
        person_id: row.get(2)?,
        bbox_x: row.get(3)?,
        bbox_y: row.get(4)?,
        bbox_w: row.get(5)?,
        bbox_h: row.get(6)?,
        crop_type: row.get(7)?,
        embedding_model: row.get(8)?,
        model_version: row.get(9)?,
        vector_id: row.get(10)?,
        quality_score: row.get(11)?,
        status: row.get(12)?,
        created_at,
    })
}
