//! PersonPrototype repository。
//!
//! 014 (face v3 identity): schema 改成内嵌 embedding blob,
//! 不再 FK 引用 face_embeddings。这样 rebuild_face_index 后 prototype 仍然保留,
//! 不依赖 face_embeddings 表存在。
//!
//! 字段(对齐 user spec §5):
//!   id / person_id / face_id / embedding / dim / pose_score / pose_yaw
//!   / quality_score / prototype_type / weight / model_version / face_count
//!   / is_active / created_at

use chrono::{DateTime, Utc};
use rusqlite::params;

use crate::error::DatabaseError;
use crate::Transaction;

/// Prototype 类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrototypeType {
    Frontal,
    LeftProfile,
    RightProfile,
    HighQuality,
    Glasses,
    General,
}

impl PrototypeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            PrototypeType::Frontal => "frontal",
            PrototypeType::LeftProfile => "left_profile",
            PrototypeType::RightProfile => "right_profile",
            PrototypeType::HighQuality => "high_quality",
            PrototypeType::Glasses => "glasses",
            PrototypeType::General => "general",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "frontal" => PrototypeType::Frontal,
            "left_profile" => PrototypeType::LeftProfile,
            "right_profile" => PrototypeType::RightProfile,
            "high_quality" => PrototypeType::HighQuality,
            "glasses" => PrototypeType::Glasses,
            _ => PrototypeType::General,
        }
    }

    /// 从 yaw 角度推导 prototype_type(P2 自动选型用)。
    /// |yaw| < 15° → frontal
    /// yaw > 15°  → right_profile (观察者左侧脸 = 被观察者右脸)
    /// yaw < -15° → left_profile
    pub fn from_yaw(yaw: f32) -> Self {
        if yaw > 15.0 {
            PrototypeType::RightProfile
        } else if yaw < -15.0 {
            PrototypeType::LeftProfile
        } else {
            PrototypeType::Frontal
        }
    }
}

/// 新插入的 prototype(不含 id)。
///
/// 014 后:`embedding` 直接传入 f32 slice,dim 自动计算;
/// 不再依赖 face_embeddings 表。
#[derive(Debug, Clone)]
pub struct NewPersonPrototype {
    /// 所属 person
    pub person_id: i64,
    /// 来源 face_id(用于审计 / 关联查询,可与删除 face 解耦)
    pub face_id: i64,
    /// embedding (小端 f32,内部存为 BLOB)
    pub embedding: Vec<f32>,
    /// 类型
    pub prototype_type: PrototypeType,
    /// 来源 face 的 pose_score(影响 prototype 投票权重)
    pub pose_score: Option<f32>,
    /// 来源 face 的 yaw(影响 prototype_type 分桶)
    pub pose_yaw: Option<f32>,
    /// 来源 face 的 quality_score
    pub quality_score: Option<f32>,
    /// 投票权重(由调用方根据 cluster 内 face 数量 / 总数计算)
    pub weight: f32,
    /// 嵌入模型版本
    pub model_version: String,
}

/// 数据库行 → PersonPrototypeRow。
#[derive(Debug, Clone)]
pub struct PersonPrototypeRow {
    /// id
    pub id: i64,
    /// person_id
    pub person_id: i64,
    /// 来源 face_id
    pub face_id: i64,
    /// embedding(已解码为 f32)
    pub embedding: Vec<f32>,
    /// dim
    pub dim: usize,
    /// 来源 face 的 pose_score
    pub pose_score: Option<f32>,
    /// 来源 face 的 yaw
    pub pose_yaw: Option<f32>,
    /// 来源 face 的 quality_score
    pub quality_score: Option<f32>,
    /// prototype 类型
    pub prototype_type: PrototypeType,
    /// 投票权重
    pub weight: f32,
    /// embedding 模型版本
    pub model_version: String,
    /// 该 prototype 覆盖的 face 数(审计)
    pub face_count: i32,
    /// 是否激活(0=软删)
    pub is_active: bool,
    /// 创建时间
    pub created_at: DateTime<Utc>,
}

/// PersonPrototype repository。
pub struct PersonPrototypeRepository<'tx, 'db> {
    pub(crate) tx: &'tx mut Transaction<'db>,
}

impl<'tx, 'db> PersonPrototypeRepository<'tx, 'db> {
    /// 插入新 prototype。返回 id。
    pub fn insert(&mut self, p: &NewPersonPrototype) -> Result<i64, DatabaseError> {
        let bytes = embedding_to_blob(&p.embedding);
        self.tx.execute(
            "INSERT INTO person_prototypes
             (person_id, face_id, embedding, dim, pose_score, pose_yaw,
              quality_score, prototype_type, weight, model_version, face_count, is_active)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 0, 1)",
            params![
                p.person_id,
                p.face_id,
                bytes,
                p.embedding.len() as i64,
                p.pose_score,
                p.pose_yaw,
                p.quality_score,
                p.prototype_type.as_str(),
                p.weight,
                p.model_version,
            ],
        )?;
        Ok(self.tx.last_insert_rowid())
    }

    /// 软删:is_active = 0。保留行用于审计 / 撤销。
    pub fn soft_delete(&mut self, id: i64) -> Result<(), DatabaseError> {
        self.tx.execute(
            "UPDATE person_prototypes SET is_active = 0 WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    /// 硬删(测试 / 管理用)。
    pub fn delete(&mut self, id: i64) -> Result<(), DatabaseError> {
        self.tx.execute(
            "DELETE FROM person_prototypes WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    /// 列出 person 的激活 prototypes(按 quality_score DESC)。
    pub fn list_active_by_person(
        &self,
        person_id: i64,
    ) -> Result<Vec<PersonPrototypeRow>, DatabaseError> {
        self.list_by_person_internal(person_id, true)
    }

    /// 列出 person 的激活 prototypes，排除指定 image 的 face（用于 LOO 测试）。
    ///
    /// 通过 JOIN faces 表过滤掉来自该 image 的 prototypes。
    pub fn list_active_by_person_exclude_image(
        &self,
        person_id: i64,
        exclude_image_id: i64,
    ) -> Result<Vec<PersonPrototypeRow>, DatabaseError> {
        let sql = "SELECT pp.id, pp.person_id, pp.face_id, pp.embedding, pp.dim,
                          pp.pose_score, pp.pose_yaw, pp.quality_score, pp.prototype_type,
                          pp.weight, pp.model_version, pp.face_count, pp.is_active, pp.created_at
                   FROM person_prototypes pp
                   JOIN faces f ON f.id = pp.face_id
                   WHERE pp.person_id = ?1 AND pp.is_active = 1 AND f.image_id != ?2
                   ORDER BY pp.quality_score DESC, pp.id";
        let mut stmt = self.tx.prepare(sql)?;
        let mut rows = stmt.query(params![person_id, exclude_image_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_prototype(row)?);
        }
        Ok(out)
    }

    /// 列出 person 的所有 prototypes(含软删)。
    pub fn list_by_person(
        &self,
        person_id: i64,
    ) -> Result<Vec<PersonPrototypeRow>, DatabaseError> {
        self.list_by_person_internal(person_id, false)
    }

    fn list_by_person_internal(
        &self,
        person_id: i64,
        active_only: bool,
    ) -> Result<Vec<PersonPrototypeRow>, DatabaseError> {
        let sql = if active_only {
            "SELECT id, person_id, face_id, embedding, dim, pose_score, pose_yaw,
                    quality_score, prototype_type, weight, model_version,
                    face_count, is_active, created_at
             FROM person_prototypes
             WHERE person_id = ?1 AND is_active = 1
             ORDER BY quality_score DESC, id"
        } else {
            "SELECT id, person_id, face_id, embedding, dim, pose_score, pose_yaw,
                    quality_score, prototype_type, weight, model_version,
                    face_count, is_active, created_at
             FROM person_prototypes
             WHERE person_id = ?1
             ORDER BY id"
        };
        let mut stmt = self.tx.prepare(sql)?;
        let mut rows = stmt.query(params![person_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_prototype(row)?);
        }
        Ok(out)
    }

    /// 按 person_id 软删所有 prototype(Phase 5 rebuild 时清理)。
    pub fn soft_delete_for_person(&mut self, person_id: i64) -> Result<u64, DatabaseError> {
        let n = self.tx.execute(
            "UPDATE person_prototypes SET is_active = 0 WHERE person_id = ?1 AND is_active = 1",
            params![person_id],
        )?;
        Ok(n as u64)
    }

    /// 统计某 person 的 active prototype 数。
    pub fn count_active_by_person(&self, person_id: i64) -> Result<i64, DatabaseError> {
        let n: i64 = self.tx.query_row(
            "SELECT COUNT(*) FROM person_prototypes
             WHERE person_id = ?1 AND is_active = 1",
            params![person_id],
            |r| r.get(0),
        )?;
        Ok(n)
    }

    /// 列出所有 active prototype(诊断用)。
    pub fn list_all_active(&self) -> Result<Vec<PersonPrototypeRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, person_id, face_id, embedding, dim, pose_score, pose_yaw,
                    quality_score, prototype_type, weight, model_version,
                    face_count, is_active, created_at
             FROM person_prototypes
             WHERE is_active = 1
             ORDER BY person_id, id",
        )?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_prototype(row)?);
        }
        Ok(out)
    }
}

/// f32 vector → 小端 BLOB。
fn embedding_to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// BLOB → f32 vector(要求长度匹配)。
fn blob_to_embedding(blob: &[u8], dim: usize) -> Result<Vec<f32>, DatabaseError> {
    if blob.len() != dim * 4 {
        return Err(DatabaseError::Conversion(format!(
            "prototype embedding blob size mismatch: {} bytes for dim={}",
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

fn row_to_prototype(row: &rusqlite::Row<'_>) -> Result<PersonPrototypeRow, DatabaseError> {
    let created_str: String = row.get(13)?;
    let created_at = DateTime::parse_from_rfc3339(&created_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| DatabaseError::Conversion(format!("created_at parse: {e}")))?;
    let pt_str: String = row.get(8)?;
    let dim: i64 = row.get(4)?;
    let blob: Vec<u8> = row.get(3)?;
    let embedding = blob_to_embedding(&blob, dim as usize)?;
    let is_active_int: i64 = row.get(12)?;
    Ok(PersonPrototypeRow {
        id: row.get(0)?,
        person_id: row.get(1)?,
        face_id: row.get(2)?,
        embedding,
        dim: dim as usize,
        pose_score: row.get(5)?,
        pose_yaw: row.get(6)?,
        quality_score: row.get(7)?,
        prototype_type: PrototypeType::from_str(&pt_str),
        weight: row.get(9)?,
        model_version: row.get(10)?,
        face_count: row.get(11)?,
        is_active: is_active_int != 0,
        created_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prototype_type_roundtrip() {
        for t in [
            PrototypeType::Frontal,
            PrototypeType::LeftProfile,
            PrototypeType::RightProfile,
            PrototypeType::HighQuality,
            PrototypeType::Glasses,
            PrototypeType::General,
        ] {
            assert_eq!(PrototypeType::from_str(t.as_str()), t);
        }
        assert_eq!(
            PrototypeType::from_str("unknown"),
            PrototypeType::General
        );
    }

    #[test]
    fn prototype_type_from_yaw() {
        assert_eq!(PrototypeType::from_yaw(0.0), PrototypeType::Frontal);
        assert_eq!(PrototypeType::from_yaw(14.9), PrototypeType::Frontal);
        assert_eq!(PrototypeType::from_yaw(15.1), PrototypeType::RightProfile);
        assert_eq!(PrototypeType::from_yaw(-15.1), PrototypeType::LeftProfile);
        assert_eq!(PrototypeType::from_yaw(45.0), PrototypeType::RightProfile);
        assert_eq!(PrototypeType::from_yaw(-45.0), PrototypeType::LeftProfile);
    }

    #[test]
    fn embedding_blob_roundtrip() {
        let v = vec![0.1f32, 0.2, 0.3, 0.4];
        let blob = embedding_to_blob(&v);
        assert_eq!(blob.len(), 16);
        let back = blob_to_embedding(&blob, 4).unwrap();
        for (a, b) in v.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}