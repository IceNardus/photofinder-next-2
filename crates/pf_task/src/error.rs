//! `pf_task` 错误类型。

use thiserror::Error;

/// `pf_task` 的统一错误。
#[derive(Debug, Error)]
pub enum TaskError {
    /// 任务未找到
    #[error("task not found: {0}")]
    NotFound(i64),

    /// 已经在跑（重复 enqueue 等）
    #[error("task already running")]
    AlreadyRunning,

    /// 已被取消
    #[error("task cancelled")]
    Cancelled,

    /// 数据库错误（来自 `pf_database`）
    #[error("database: {0}")]
    Database(String),

    /// 执行器错误
    #[error("executor: {0}")]
    Executor(String),

    /// 状态非法（如对未 Running 的任务 cancel）
    #[error("invalid state: {0}")]
    InvalidState(String),
}