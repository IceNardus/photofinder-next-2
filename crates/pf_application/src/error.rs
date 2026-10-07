//! `pf_application` 错误类型。
//!
//! 所有下层错误通过 `#[from]` 自动转换。

use thiserror::Error;

use pf_ai::AIError;
use pf_config::ConfigError;
use pf_database::DatabaseError;
use pf_platform::PlatformError;
use pf_task::TaskError;
use pf_vector::VectorError;

/// `pf_application` 的统一错误（也是对外暴露的唯一错误）。
#[derive(Debug, Error)]
pub enum ApplicationError {
    /// 平台错误
    #[error("platform: {0}")]
    Platform(#[from] PlatformError),

    /// 数据库错误
    #[error("database: {0}")]
    Database(#[from] DatabaseError),

    /// AI 错误
    #[error("ai: {0}")]
    AI(#[from] AIError),

    /// 向量错误
    #[error("vector: {0}")]
    Vector(#[from] VectorError),

    /// 任务错误
    #[error("task: {0}")]
    Task(#[from] TaskError),

    /// 配置错误
    #[error("config: {0}")]
    Config(#[from] ConfigError),

    /// 资源未找到
    #[error("not found: {0}")]
    NotFound(String),

    /// 状态非法
    #[error("invalid state: {0}")]
    InvalidState(String),

    /// 内部错误（应当是 invariant violation）
    #[error("internal: {0}")]
    Internal(String),
}