//! Person repository。
//!
//! Phase 1 (plan §7): 加 `status` (`active` / `hidden` / `merged` / `deleted`)
//! 与 `updated_at` 列。`status` 默认 `active`。

use chrono::{DateTime, Utc};
use rusqlite::params;

use crate::error::DatabaseError;
use crate::Transaction;

/// Person 状态(plan §7)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersonStatus {
    Active,
    Hidden,
    Merged,
    Deleted,
}

/// Person identity confirmation status (V2 dual-prototype).
///
/// - CONFIRMED: face prototype + body prototype both confirmed, meets thresholds
/// - PROBABLE: face prototype matches but body weak/missing
/// - UNKNOWN: insufficient evidence
/// - CONFLICT: face and body signals disagree
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersonIdentityStatus {
    Confirmed,
    Probable,
    Unknown,
    Conflict,
}

impl PersonIdentityStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            PersonIdentityStatus::Confirmed => "confirmed",
            PersonIdentityStatus::Probable => "probable",
            PersonIdentityStatus::Unknown => "unknown",
            PersonIdentityStatus::Conflict => "conflict",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "confirmed" => PersonIdentityStatus::Confirmed,
            "probable" => PersonIdentityStatus::Probable,
            "conflict" => PersonIdentityStatus::Conflict,
            _ => PersonIdentityStatus::Unknown,
        }
    }
}

impl PersonStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            PersonStatus::Active => "active",
            PersonStatus::Hidden => "hidden",
            PersonStatus::Merged => "merged",
            PersonStatus::Deleted => "deleted",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "hidden" => PersonStatus::Hidden,
            "merged" => PersonStatus::Merged,
            "deleted" => PersonStatus::Deleted,
            _ => PersonStatus::Active,
        }
    }
}

/// 新插入的人物。
#[derive(Debug, Clone)]
pub struct NewPerson {
    /// 用户命名（可选）
    pub name: Option<String>,
}

/// 数据库行 → PersonRow。
#[derive(Debug, Clone)]
pub struct PersonRow {
    /// id
    pub id: i64,
    /// 用户命名
    pub name: Option<String>,
    /// 包含的 face 数
    pub face_count: i32,
    /// V2: 包含的 body 数
    pub body_count: i32,
    /// 状态
    pub status: PersonStatus,
    /// V2: Identity confirmation status
    pub identity_status: PersonIdentityStatus,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 更新时间
    pub updated_at: DateTime<Utc>,
}

/// Person repository。
pub struct PersonRepository<'tx, 'db> {
    pub(crate) tx: &'tx mut Transaction<'db>,
}

impl<'tx, 'db> PersonRepository<'tx, 'db> {
    /// 插入新 person（默认 active）。
    pub fn insert(&mut self, p: &NewPerson) -> Result<i64, DatabaseError> {
        self.tx.execute(
            "INSERT INTO persons (name) VALUES (?1)",
            params![p.name],
        )?;
        Ok(self.tx.last_insert_rowid())
    }

    /// 按 id 查询。
    pub fn get_by_id(&self, id: i64) -> Result<Option<PersonRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, name, face_count, body_count, status, identity_status, created_at, updated_at
             FROM persons WHERE id = ?1",
        )?;
        let mut rows = stmt.query(params![id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row_to_person(row)?))
        } else {
            Ok(None)
        }
    }

    /// 列出所有人（按 face_count DESC,默认排除 deleted 状态）。
    pub fn list(&self) -> Result<Vec<PersonRow>, DatabaseError> {
        self.list_with_status(None)
    }

    /// 列出指定 status 的人（None = 全部）。
    pub fn list_with_status(
        &self,
        status: Option<PersonStatus>,
    ) -> Result<Vec<PersonRow>, DatabaseError> {
        let (sql, has_filter) = match status {
            Some(_) => (
                "SELECT id, name, face_count, body_count, status, identity_status, created_at, updated_at
                 FROM persons WHERE status = ?1 ORDER BY face_count DESC",
                true,
            ),
            None => (
                "SELECT id, name, face_count, body_count, status, identity_status, created_at, updated_at
                 FROM persons ORDER BY face_count DESC",
                false,
            ),
        };
        let mut stmt = self.tx.prepare(sql)?;
        let mut rows = if has_filter {
            stmt.query(params![status.unwrap().as_str()])?
        } else {
            stmt.query([])?
        };
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_person(row)?);
        }
        Ok(out)
    }

    /// 设置 name。
    pub fn set_name(&mut self, id: i64, name: Option<&str>) -> Result<(), DatabaseError> {
        self.tx.execute(
            "UPDATE persons SET name = ?1,
             updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
             WHERE id = ?2",
            params![name, id],
        )?;
        Ok(())
    }

    /// 设置 status(plan §33: 用 status='merged' 替代物理删除)。
    pub fn set_status(&mut self, id: i64, status: PersonStatus) -> Result<(), DatabaseError> {
        self.tx.execute(
            "UPDATE persons SET status = ?1,
             updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
             WHERE id = ?2",
            params![status.as_str(), id],
        )?;
        Ok(())
    }

    /// 设置 identity_status（V2 dual-prototype）。
    pub fn set_identity_status(
        &mut self,
        id: i64,
        identity_status: PersonIdentityStatus,
    ) -> Result<(), DatabaseError> {
        self.tx.execute(
            "UPDATE persons SET identity_status = ?1,
             updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
             WHERE id = ?2",
            params![identity_status.as_str(), id],
        )?;
        Ok(())
    }

    /// 同步 face_count（SELECT COUNT 后写回）。
    pub fn refresh_face_count(&mut self, id: i64) -> Result<i64, DatabaseError> {
        let n: i64 = self.tx.query_row(
            "SELECT COUNT(*) FROM faces WHERE person_id = ?1",
            params![id],
            |r| r.get(0),
        )?;
        self.tx.execute(
            "UPDATE persons SET face_count = ?1,
             updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
             WHERE id = ?2",
            params![n, id],
        )?;
        Ok(n)
    }

    /// 物理删除 person（仅用于测试/迁移/管理工具）。
    /// 应用层应优先用 `set_status(.., Deleted)` 保留历史。
    /// 注：FK ON DELETE SET NULL 会自动把关联 face.person_id 置空。
    pub fn delete(&mut self, id: i64) -> Result<(), DatabaseError> {
        self.tx.execute("DELETE FROM persons WHERE id = ?1", params![id])?;
        Ok(())
    }
}

fn row_to_person(row: &rusqlite::Row<'_>) -> Result<PersonRow, DatabaseError> {
    let created_str: String = row.get(6)?;
    let created_at = DateTime::parse_from_rfc3339(&created_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| DatabaseError::Conversion(format!("created_at parse: {e}")))?;
    let updated_str: String = row.get(7)?;
    let updated_at = DateTime::parse_from_rfc3339(&updated_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| DatabaseError::Conversion(format!("updated_at parse: {e}")))?;
    let status_str: String = row.get(4)?;
    let identity_str: String = row.get(5)?;
    Ok(PersonRow {
        id: row.get(0)?,
        name: row.get(1)?,
        face_count: row.get(2)?,
        body_count: row.get(3)?,
        status: PersonStatus::from_str(&status_str),
        identity_status: PersonIdentityStatus::from_str(&identity_str),
        created_at,
        updated_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn person_status_roundtrip() {
        for s in [
            PersonStatus::Active,
            PersonStatus::Hidden,
            PersonStatus::Merged,
            PersonStatus::Deleted,
        ] {
            assert_eq!(PersonStatus::from_str(s.as_str()), s);
        }
        // unknown 默认 active
        assert_eq!(PersonStatus::from_str("nonsense"), PersonStatus::Active);
    }

    #[test]
    fn person_identity_status_roundtrip() {
        for s in [
            PersonIdentityStatus::Confirmed,
            PersonIdentityStatus::Probable,
            PersonIdentityStatus::Unknown,
            PersonIdentityStatus::Conflict,
        ] {
            assert_eq!(PersonIdentityStatus::from_str(s.as_str()), s);
        }
        // 未知字符串默认 unknown
        assert_eq!(PersonIdentityStatus::from_str("nonsense"), PersonIdentityStatus::Unknown);
    }
}
