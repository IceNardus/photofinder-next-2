//! `pf_vector` 错误类型。

use thiserror::Error;

/// `pf_vector` 的统一错误。
#[derive(Debug, Error)]
pub enum VectorError {
    /// IO 错误
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// 维度不匹配
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    Dimension {
        /// 期望维度
        expected: usize,
        /// 实际维度
        actual: usize,
    },

    /// 索引本身错误（底层 HNSW 库返回）
    #[error("index: {0}")]
    Index(String),

    /// 存储层错误
    #[error("storage: {0}")]
    Storage(String),

    /// 序列化 / 反序列化错误
    #[error("serialization: {0}")]
    Serialization(String),
}