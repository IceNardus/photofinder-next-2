//! 路径解析器 trait。
//!
//! 平台不同实现不同：
//! - Desktop：基于 `dirs` crate（`~/Library/Application Support/PhotoFinderNext/` 等）
//! - Android：`/data/data/<pkg>/files/...`
//! - iOS：`Library/Application Support/PhotoFinderNext/`
//!
//! 注：trait 接受 `&str` 文件名而不是 `ModelId`，以避免依赖 pf_config 的具体类型。
//! 上层（pf_config::ModelManager）负责 ModelId → file_name 的映射。

use std::path::PathBuf;

use crate::error::PlatformError;

/// 路径解析器。
pub trait PathResolver: Send + Sync {
    /// 模型文件完整路径。
    ///
    /// `file_name` 是模型文件本身（如 `scrfd_500m_bnkps.onnx`），不含父目录。
    /// 实现负责按平台优先级（exe 旁 / bundle / 数据目录 / CWD）搜索。
    fn model_path(&self, file_name: &str) -> Result<PathBuf, PlatformError>;

    /// 数据目录（SQLite、缩略图、HNSW 索引）。
    fn data_dir(&self) -> Result<PathBuf, PlatformError>;

    /// 缓存目录（下载模型、临时文件）。
    fn cache_dir(&self) -> Result<PathBuf, PlatformError>;

    /// 数据库文件完整路径。
    fn db_path(&self) -> Result<PathBuf, PlatformError> {
        Ok(self.data_dir()?.join("photofinder.db"))
    }

    /// 日志目录。
    fn log_dir(&self) -> Result<PathBuf, PlatformError> {
        Ok(self.data_dir()?.join("logs"))
    }

    /// 模型根目录（不含文件名）。
    ///
    /// 默认：从 `model_path` 任一模型推断；桌面端有更复杂的搜索逻辑，单独 override。
    fn models_dir(&self) -> Option<PathBuf> {
        // 兜底：取 model_path 的父目录（适用于移动端 — 所有模型都在 PF_MODELS_DIR）
        self.model_path("__sentinel__").ok().and_then(|p| p.parent().map(|x| x.to_path_buf()))
    }
}