//! `pf_platform` 错误类型。

use thiserror::Error;

/// `pf_platform` 的统一错误。
#[derive(Debug, Error)]
pub enum PlatformError {
    /// IO 错误
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// 权限被拒
    #[error("permission denied: {0}")]
    PermissionDenied(String),

    /// 照片未找到
    #[error("photo not found: {0}")]
    NotFound(String),

    /// JNI / FFI 桥错误（Android / iOS）
    #[error("bridge: {0}")]
    Bridge(String),

    /// 平台不支持的操作
    #[error("unsupported on this platform: {0}")]
    Unsupported(String),

    /// Walkdir / 遍历错误
    #[error("walk: {0}")]
    Walk(String),

    /// 图片解码错误
    #[error("image decode: {0}")]
    ImageDecode(String),

    /// 模型文件未找到
    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// 路径解析失败
    #[error("path resolution failed: {0}")]
    Path(String),
}