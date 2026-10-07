//! FacePersonAssignment repository。
//!
//! 014 后:
//! - 表 rename: face_person_matches → face_person_assignments
//! - 类型重命名: `FacePersonMatch` → `FacePersonAssignment` (旧名用 `#[deprecated]` 标注)
//! - 新增字段: similarity / confidence / model_version
//!
//! 用于追踪"为什么这个 face 被认为属于这个 person":
//! - `Confirmed`: 高置信度,自动聚类直接确认
//! - `Candidate`: 中置信度,候选 assignment,等下轮升级或人工审核
//! - `Rejected`:  低置信度,显式拒绝(防被自动 recluster 误并)
//!
//! `similarity` (0..1 cosine) + `confidence` (0..1,综合相似度+质量+pose)
//! 是两个独立指标:
//! - similarity 越大 → 与该 person prototype 越像
//! - confidence 越大 → 该 assignment 越可靠
//! 在多级决策中,sorted by similarity,但 acceptance 由 confidence 决定。

use chrono::{DateTime, Utc};
use rusqlite::params;

use crate::error::DatabaseError;
use crate::Transaction;

/// 匹配状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchStatus {
    Candidate,
    Confirmed,
    Rejected,
}

impl MatchStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            MatchStatus::Candidate => "candidate",
            MatchStatus::Confirmed => "confirmed",
            MatchStatus::Rejected => "rejected",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "confirmed" => MatchStatus::Confirmed,
            "rejected" => MatchStatus::Rejected,
            _ => MatchStatus::Candidate,
        }
    }
}

/// 新插入的 assignment(不含 id)。
///
/// 014 后:similarity + confidence + model_version 三个新字段。
#[derive(Debug, Clone)]
pub struct NewFacePersonAssignment {
    pub face_id: i64,
    pub person_id: i64,
    /// cosine 相似度(0..1)。最终 cluster 决策用这个值。
    pub similarity: f32,
    /// 综合置信度(0..1),由 similarity + quality_score + pose_score 融合得出。
    pub confidence: f32,
    /// 聚类方法("prototype_max", "cumulative_voting" 等)
    pub method: String,
    pub status: MatchStatus,
    /// embedding 模型版本,用于多模型并存时区分
    pub model_version: String,
}

/// 数据库行 → FacePersonAssignmentRow。
#[derive(Debug, Clone)]
pub struct FacePersonAssignmentRow {
    pub id: i64,
    pub face_id: i64,
    pub person_id: i64,
    pub similarity: f32,
    pub confidence: f32,
    pub method: String,
    pub status: MatchStatus,
    pub model_version: String,
    pub created_at: DateTime<Utc>,
}

// ========================================================================
// 兼容层(临时):让旧调用方不立即破
// ========================================================================

/// 旧类型别名(014 之前),用 `#[deprecated]` 标注。
#[deprecated(
    since = "0.3.0",
    note = "use FacePersonAssignment / NewFacePersonAssignment / FacePersonAssignmentRow instead"
)]
pub type FacePersonMatch = FacePersonAssignmentRow;

#[deprecated(
    since = "0.3.0",
    note = "use NewFacePersonAssignment instead"
)]
pub type NewFacePersonMatch = NewFacePersonAssignment;

// 注意:`FacePersonMatchRepository` 是带 lifetime 参数的泛型,
// 不适合做 type alias。调用方请直接用 `FacePersonAssignmentRepository`。

/// FacePersonAssignment repository。
pub struct FacePersonAssignmentRepository<'tx, 'db> {
    pub(crate) tx: &'tx mut Transaction<'db>,
}

impl<'tx, 'db> FacePersonAssignmentRepository<'tx, 'db> {
    /// 插入新 assignment。
    pub fn insert(
        &mut self,
        m: &NewFacePersonAssignment,
    ) -> Result<i64, DatabaseError> {
        // 兼容层:旧 schema 也保留了 `score` 列,在迁移期间也写一份
        self.tx.execute(
            "INSERT INTO face_person_assignments
             (face_id, person_id, score, similarity, confidence, method, status, model_version)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                m.face_id,
                m.person_id,
                m.similarity,         // score 同步(similarity 是它的升级版)
                m.similarity,
                m.confidence,
                m.method,
                m.status.as_str(),
                m.model_version,
            ],
        )?;
        Ok(self.tx.last_insert_rowid())
    }

    /// 列出某 face 的所有 assignments(按 similarity DESC)。
    pub fn list_by_face(
        &self,
        face_id: i64,
    ) -> Result<Vec<FacePersonAssignmentRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, face_id, person_id, similarity, confidence, method, status, model_version, created_at
             FROM face_person_assignments WHERE face_id = ?1
             ORDER BY similarity DESC",
        )?;
        let mut rows = stmt.query(params![face_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_assignment(row)?);
        }
        Ok(out)
    }

    /// 列出某 face + status 的 assignments。
    pub fn list_by_face_and_status(
        &self,
        face_id: i64,
        status: MatchStatus,
    ) -> Result<Vec<FacePersonAssignmentRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, face_id, person_id, similarity, confidence, method, status, model_version, created_at
             FROM face_person_assignments
             WHERE face_id = ?1 AND status = ?2
             ORDER BY similarity DESC",
        )?;
        let mut rows = stmt.query(params![face_id, status.as_str()])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_assignment(row)?);
        }
        Ok(out)
    }

    /// 列出某 person 的所有 assignments。
    pub fn list_by_person(
        &self,
        person_id: i64,
    ) -> Result<Vec<FacePersonAssignmentRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, face_id, person_id, similarity, confidence, method, status, model_version, created_at
             FROM face_person_assignments WHERE person_id = ?1
             ORDER BY similarity DESC",
        )?;
        let mut rows = stmt.query(params![person_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_assignment(row)?);
        }
        Ok(out)
    }

    /// 列出某 person 的 confirmed assignments(快速拿到 person 的所有成员 face_id)。
    pub fn list_confirmed_by_person(
        &self,
        person_id: i64,
    ) -> Result<Vec<FacePersonAssignmentRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, face_id, person_id, similarity, confidence, method, status, model_version, created_at
             FROM face_person_assignments
             WHERE person_id = ?1 AND status = 'confirmed'
             ORDER BY similarity DESC",
        )?;
        let mut rows = stmt.query(params![person_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_assignment(row)?);
        }
        Ok(out)
    }

    /// 按 id 删除(管理用)。
    pub fn delete(&mut self, id: i64) -> Result<(), DatabaseError> {
        self.tx.execute(
            "DELETE FROM face_person_assignments WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    /// 删除某 face 的所有 assignments(re-cluster 前清理)。
    pub fn delete_by_face(&mut self, face_id: i64) -> Result<u64, DatabaseError> {
        let n = self.tx.execute(
            "DELETE FROM face_person_assignments WHERE face_id = ?1",
            params![face_id],
        )?;
        Ok(n as u64)
    }

    /// 删除某 person 的所有 assignments。
    pub fn delete_by_person(&mut self, person_id: i64) -> Result<u64, DatabaseError> {
        let n = self.tx.execute(
            "DELETE FROM face_person_assignments WHERE person_id = ?1",
            params![person_id],
        )?;
        Ok(n as u64)
    }

    /// 按 face_id + person_id 查找(用于 "rejected pair 是否被自动 recluster" 检查)。
    pub fn find(
        &self,
        face_id: i64,
        person_id: i64,
    ) -> Result<Option<FacePersonAssignmentRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, face_id, person_id, similarity, confidence, method, status, model_version, created_at
             FROM face_person_assignments
             WHERE face_id = ?1 AND person_id = ?2 LIMIT 1",
        )?;
        let mut rows = stmt.query(params![face_id, person_id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row_to_assignment(row)?))
        } else {
            Ok(None)
        }
    }

    /// 列出所有 assignments(调试用)。
    pub fn list_all(&self) -> Result<Vec<FacePersonAssignmentRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, face_id, person_id, similarity, confidence, method, status, model_version, created_at
             FROM face_person_assignments ORDER BY id",
        )?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_assignment(row)?);
        }
        Ok(out)
    }
}

fn row_to_assignment(
    row: &rusqlite::Row<'_>,
) -> Result<FacePersonAssignmentRow, DatabaseError> {
    let created_str: String = row.get(8)?;
    let created_at = DateTime::parse_from_rfc3339(&created_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| DatabaseError::Conversion(format!("created_at parse: {e}")))?;
    let status_str: String = row.get(6)?;
    Ok(FacePersonAssignmentRow {
        id: row.get(0)?,
        face_id: row.get(1)?,
        person_id: row.get(2)?,
        similarity: row.get(3)?,
        confidence: row.get(4)?,
        method: row.get(5)?,
        status: MatchStatus::from_str(&status_str),
        model_version: row.get(7)?,
        created_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_status_roundtrip() {
        for s in [MatchStatus::Candidate, MatchStatus::Confirmed, MatchStatus::Rejected] {
            assert_eq!(MatchStatus::from_str(s.as_str()), s);
        }
        assert_eq!(MatchStatus::from_str("nonsense"), MatchStatus::Candidate);
    }
}