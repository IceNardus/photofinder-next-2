//! `pf_config` 错误类型。

use thiserror::Error;

use pf_platform::PlatformError;

/// `pf_config` 的统一错误。
#[derive(Debug, Error)]
pub enum ConfigError {
    /// 模型文件缺失
    #[error("model file not found: {0}")]
    ModelNotFound(String),

    /// 模型 SHA256 校验失败
    #[error("model checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch {
        /// 期望值
        expected: String,
        /// 实际值
        actual: String,
    },

    /// 模型加载失败
    #[error("model load failed: {0}")]
    ModelLoad(String),

    /// 配置解析失败
    #[error("config parse: {0}")]
    Parse(String),

    /// IO 错误
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// 模型已经在内存中
    #[error("model already loaded: {0}")]
    AlreadyLoaded(String),

    /// 模型未加载
    #[error("model not loaded: {0}")]
    NotLoaded(String),

    /// 未知 ModelId
    #[error("unknown model id: {0}")]
    UnknownModel(String),

    /// 平台错误
    #[error("platform: {0}")]
    Platform(#[from] PlatformError),

    /// 下载失败
    #[error("download failed: {0}")]
    Download(String),
}