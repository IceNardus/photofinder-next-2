//! Task repository。

use chrono::{DateTime, Utc};
use rusqlite::params;
use rusqlite::OptionalExtension;

use crate::error::DatabaseError;
use crate::Transaction;

/// Task 状态（DB 字符串）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// 待执行
    Pending,
    /// 正在执行
    Running,
    /// 已完成
    Completed,
    /// 已失败（可重试）
    Failed,
    /// 已取消
    Cancelled,
}

impl TaskStatus {
    /// 字符串表示。
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskStatus::Pending => "pending",
            TaskStatus::Running => "running",
            TaskStatus::Completed => "completed",
            TaskStatus::Failed => "failed",
            TaskStatus::Cancelled => "cancelled",
        }
    }

    /// 从字符串解析。
    pub fn parse(s: &str) -> Result<Self, DatabaseError> {
        match s {
            "pending" => Ok(TaskStatus::Pending),
            "running" => Ok(TaskStatus::Running),
            "completed" => Ok(TaskStatus::Completed),
            "failed" => Ok(TaskStatus::Failed),
            "cancelled" => Ok(TaskStatus::Cancelled),
            other => Err(DatabaseError::Conversion(format!(
                "invalid TaskStatus: {other}"
            ))),
        }
    }
}

/// 新插入的任务（不含 id / 时间戳）。
#[derive(Debug, Clone)]
pub struct NewTask {
    /// `TaskKind` 的 JSON 字符串
    pub kind_json: String,
    /// 优先级
    pub priority: i32,
    /// 最大重试次数
    pub max_retries: i32,
}

/// 数据库行 → TaskRow。
#[derive(Debug, Clone)]
pub struct TaskRow {
    /// id
    pub id: i64,
    /// TaskKind JSON
    pub kind_json: String,
    /// 优先级
    pub priority: i32,
    /// 状态
    pub status: TaskStatus,
    /// 已重试次数
    pub retry_count: i32,
    /// 最大重试次数
    pub max_retries: i32,
    /// 错误消息
    pub error: Option<String>,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 开始时间
    pub started_at: Option<DateTime<Utc>>,
    /// 完成时间
    pub completed_at: Option<DateTime<Utc>>,
}

/// Task repository。
pub struct TaskRepository<'tx, 'db> {
    pub(crate) tx: &'tx mut Transaction<'db>,
}

impl<'tx, 'db> TaskRepository<'tx, 'db> {
    /// 插入新任务。返回 id。
    pub fn insert(&mut self, task: &NewTask) -> Result<i64, DatabaseError> {
        self.tx.execute(
            "INSERT INTO tasks (kind_json, priority, max_retries) VALUES (?1, ?2, ?3)",
            params![task.kind_json, task.priority, task.max_retries],
        )?;
        Ok(self.tx.last_insert_rowid())
    }

    /// 按 id 查询。
    pub fn get_by_id(&self, id: i64) -> Result<Option<TaskRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, kind_json, priority, status, retry_count, max_retries, error,
                    created_at, started_at, completed_at
             FROM tasks WHERE id = ?1",
        )?;
        let mut rows = stmt.query(params![id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row_to_task(row)?))
        } else {
            Ok(None)
        }
    }

    /// 抢一条 pending → running（按 priority DESC, created_at ASC）。
    /// 必须在事务内调用，复合 UPDATE 锁定单行。
    pub fn claim_next(&mut self) -> Result<Option<TaskRow>, DatabaseError> {
        // 用 RETURNING 直接拿到刚 claim 的行 id(避免与历史 stale 'running' 混淆)。
        // 旧实现:`SELECT ... WHERE status='running' ORDER BY started_at DESC LIMIT 1`
        //   会优先返回之前崩溃留下的 stale running task,而不是本事务刚 claim 的 pending。
        //   这会导致 worker 反复 pick stale task,pending 永远排不上。
        let mut stmt = self.tx.prepare(
            "UPDATE tasks
             SET status = 'running', started_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
             WHERE id = (
                 SELECT id FROM tasks
                 WHERE status = 'pending'
                 ORDER BY priority DESC, created_at ASC
                 LIMIT 1
             )
             RETURNING id",
        )?;
        let claimed_id: Option<i64> = stmt
            .query_row([], |r| r.get::<_, i64>(0))
            .optional()
            .map_err(DatabaseError::Sqlite)?;
        let Some(id) = claimed_id else {
            return Ok(None);
        };

        // 取出刚 claim 的行
        let mut stmt = self.tx.prepare(
            "SELECT id, kind_json, priority, status, retry_count, max_retries, error,
                    created_at, started_at, completed_at
             FROM tasks
             WHERE id = ?1",
        )?;
        let row = stmt
            .query_row(params![id], |r| {
                row_to_task(r).map_err(|e| match e {
                    DatabaseError::Sqlite(se) => se,
                    other => rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        Box::<dyn std::error::Error + Send + Sync>::from(other.to_string()),
                    ),
                })
            })
            .optional()
            .map_err(DatabaseError::Sqlite)?;
        Ok(row)
    }

    /// 标记任务完成。
    pub fn mark_completed(&mut self, id: i64) -> Result<(), DatabaseError> {
        self.tx.execute(
            "UPDATE tasks
             SET status = 'completed', completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
             WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    /// 标记任务失败（若 retry_count < max_retries 则转回 pending）。
    pub fn mark_failed_or_retry(
        &mut self,
        id: i64,
        error: &str,
    ) -> Result<bool, DatabaseError> {
        // 先增加 retry_count 并读取
        self.tx.execute(
            "UPDATE tasks
             SET retry_count = retry_count + 1
             WHERE id = ?1",
            params![id],
        )?;
        let row = self
            .get_by_id(id)?
            .ok_or_else(|| DatabaseError::Conversion(format!("task {} vanished", id)))?;
        let should_retry = row.retry_count < row.max_retries;
        if should_retry {
            self.tx.execute(
                "UPDATE tasks
                 SET status = 'pending', error = ?1, started_at = NULL, completed_at = NULL
                 WHERE id = ?2",
                params![error, id],
            )?;
        } else {
            self.tx.execute(
                "UPDATE tasks
                 SET status = 'failed', error = ?1,
                 completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                 WHERE id = ?2",
                params![error, id],
            )?;
        }
        Ok(should_retry)
    }

    /// 标记任务取消。
    pub fn mark_cancelled(&mut self, id: i64) -> Result<(), DatabaseError> {
        self.tx.execute(
            "UPDATE tasks
             SET status = 'cancelled', completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
             WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    /// 取消任务（Pending 时直接取消）。
    pub fn cancel_pending(&mut self, id: i64) -> Result<bool, DatabaseError> {
        let updated = self.tx.execute(
            "UPDATE tasks SET status = 'cancelled',
             completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
             WHERE id = ?1 AND status = 'pending'",
            params![id],
        )?;
        Ok(updated > 0)
    }

    /// 按状态统计。
    pub fn count_by_status(&self, status: TaskStatus) -> Result<i64, DatabaseError> {
        let n: i64 = self.tx.query_row(
            "SELECT COUNT(*) FROM tasks WHERE status = ?1",
            params![status.as_str()],
            |r| r.get(0),
        )?;
        Ok(n)
    }

    /// 列出任务（按 created_at DESC，可选 limit）。
    pub fn list(&self, limit: Option<usize>) -> Result<Vec<TaskRow>, DatabaseError> {
        let sql = if limit.is_some() {
            "SELECT id, kind_json, priority, status, retry_count, max_retries, error,
                    created_at, started_at, completed_at
             FROM tasks ORDER BY created_at DESC LIMIT ?1"
        } else {
            "SELECT id, kind_json, priority, status, retry_count, max_retries, error,
                    created_at, started_at, completed_at
             FROM tasks ORDER BY created_at DESC"
        };
        let mut stmt = self.tx.prepare(sql)?;
        let mut rows = if let Some(l) = limit {
            stmt.query(params![l as i64])?
        } else {
            stmt.query([])?
        };
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_task(row)?);
        }
        Ok(out)
    }

    /// 按状态过滤列出。
    pub fn list_by_status(
        &self,
        status: TaskStatus,
        limit: Option<usize>,
    ) -> Result<Vec<TaskRow>, DatabaseError> {
        let sql = if limit.is_some() {
            "SELECT id, kind_json, priority, status, retry_count, max_retries, error,
                    created_at, started_at, completed_at
             FROM tasks WHERE status = ?1 ORDER BY created_at DESC LIMIT ?2"
        } else {
            "SELECT id, kind_json, priority, status, retry_count, max_retries, error,
                    created_at, started_at, completed_at
             FROM tasks WHERE status = ?1 ORDER BY created_at DESC"
        };
        let mut stmt = self.tx.prepare(sql)?;
        let mut rows = if let Some(l) = limit {
            stmt.query(params![status.as_str(), l as i64])?
        } else {
            stmt.query(params![status.as_str()])?
        };
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_task(row)?);
        }
        Ok(out)
    }
}

fn row_to_task(row: &rusqlite::Row<'_>) -> Result<TaskRow, DatabaseError> {
    let parse_dt = |s: String| -> Result<DateTime<Utc>, DatabaseError> {
        DateTime::parse_from_rfc3339(&s)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| DatabaseError::Conversion(format!("datetime parse: {e}")))
    };
    let status_str: String = row.get(3)?;
    Ok(TaskRow {
        id: row.get(0)?,
        kind_json: row.get(1)?,
        priority: row.get(2)?,
        status: TaskStatus::parse(&status_str)?,
        retry_count: row.get(4)?,
        max_retries: row.get(5)?,
        error: row.get(6)?,
        created_at: parse_dt(row.get(7)?)?,
        started_at: row
            .get::<_, Option<String>>(8)?
            .map(parse_dt)
            .transpose()?,
        completed_at: row
            .get::<_, Option<String>>(9)?
            .map(parse_dt)
            .transpose()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_roundtrip() {
        assert_eq!(TaskStatus::Pending.as_str(), "pending");
        assert_eq!(TaskStatus::Running.as_str(), "running");
        assert_eq!(TaskStatus::Completed.as_str(), "completed");
        assert_eq!(TaskStatus::Failed.as_str(), "failed");
        assert_eq!(TaskStatus::Cancelled.as_str(), "cancelled");
        assert_eq!(TaskStatus::parse("completed").unwrap(), TaskStatus::Completed);
        assert!(TaskStatus::parse("nope").is_err());
    }
}
