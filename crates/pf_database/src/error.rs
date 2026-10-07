//! `pf_database` 错误类型。

use thiserror::Error;

/// `pf_database` 的统一错误。
#[derive(Debug, Error)]
pub enum DatabaseError {
    /// rusqlite 错误
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// 连接池错误
    #[error("pool: {0}")]
    Pool(String),

    /// Migration 错误
    #[error("migration: {0}")]
    Migration(String),

    /// 未找到
    #[error("not found: {0}")]
    NotFound(String),

    /// 约束冲突
    #[error("constraint: {0}")]
    Constraint(String),

    /// IO 错误
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// 类型转换错误
    #[error("conversion: {0}")]
    Conversion(String),
}

/// r2d2 错误转换。
impl From<r2d2::Error> for DatabaseError {
    fn from(e: r2d2::Error) -> Self {
        DatabaseError::Pool(e.to_string())
    }
}