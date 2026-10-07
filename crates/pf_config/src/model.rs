//! 模型管理。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use pf_core::ModelVersion;
use pf_platform::PathResolver;

use crate::error::ConfigError;

/// 模型 ID 类型。
///
/// 使用 newtype 包装 `String`。
/// 已知 ID 通过 `KnownModelIds` 提供常量和辅助构造。
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(pub String);

impl ModelId {
    /// 从字符串构造。
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// 字符串引用。
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// ArcFace w600k_r50 (512-d)
    pub fn arcface_w600k_r50() -> Self {
        Self("arcface-w600k-r50".into())
    }
    /// SCRFD 500M (人脸检测)
    pub fn scrfd_500m_bnkps() -> Self {
        Self("scrfd-500m-bnkps".into())
    }
    /// YOLOv8n (对象检测)
    pub fn yolov8n() -> Self {
        Self("yolov8n".into())
    }
    /// MobileCLIP-s2 (图像 embedding)
    pub fn mobileclip_s2() -> Self {
        Self("mobileclip-s2".into())
    }
    /// SuperPoint (Phase 3 关键点)
    pub fn superpoint() -> Self {
        Self("superpoint".into())
    }
    /// LightGlue (Phase 3 匹配)
    pub fn lightglue() -> Self {
        Self("lightglue".into())
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// 单个模型条目（来自 manifest.toml）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// 模型 ID
    pub id: ModelId,
    /// 版本
    pub version: ModelVersion,
    /// 文件名
    pub file_name: String,
    /// SHA256
    pub sha256: String,
    /// 下载 URL（可选）
    pub download_url: Option<String>,
    /// Embedding 维度（仅 embedder 有）
    pub dimension: Option<usize>,
    /// 输入 shape（如 [1, 3, 112, 112]）
    pub input_shape: Option<Vec<usize>>,
    /// 文件大小（bytes）
    pub size_bytes: u64,
}

/// 模型注册表。
#[derive(Debug, Clone, Default)]
pub struct ModelRegistry {
    entries: HashMap<ModelId, ModelEntry>,
}

impl ModelRegistry {
    /// 空注册表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 注册一个模型。
    pub fn register(&mut self, entry: ModelEntry) {
        let id = entry.id.clone();
        self.entries.insert(id, entry);
    }

    /// 查找。
    pub fn get(&self, id: ModelId) -> Option<&ModelEntry> {
        self.entries.get(&id)
    }

    /// 列出所有。
    pub fn list(&self) -> Vec<&ModelEntry> {
        self.entries.values().collect()
    }
}

/// 模型状态。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatus {
    /// 模型 ID
    pub id: ModelId,
    /// 是否已加载
    pub loaded: bool,
    /// 文件路径
    pub path: PathBuf,
    /// 文件大小（bytes）
    pub size_bytes: u64,
    /// SHA256 是否校验通过
    pub verified: bool,
}

/// 模型管理器。
///
/// 唯一允许加载 / 释放 / 校验模型的入口。
/// 任何模块都不能 `Session::builder().commit_from_file(...)` 散落。
pub struct ModelManager {
    registry: Arc<ModelRegistry>,
    path_resolver: Arc<dyn PathResolver>,
    loaded: Mutex<HashMap<ModelId, Arc<dyn std::any::Any + Send + Sync>>>,
}

impl ModelManager {
    /// 构造（不加载任何模型）。
    pub fn new(registry: Arc<ModelRegistry>, path_resolver: Arc<dyn PathResolver>) -> Self {
        Self {
            registry,
            path_resolver,
            loaded: Mutex::new(HashMap::new()),
        }
    }

    /// 模型文件路径（解析 ModelId → manifest 中的 file_name 后委托 PathResolver）。
    pub fn path(&self, id: ModelId) -> Result<PathBuf, ConfigError> {
        let entry = self
            .registry
            .get(id.clone())
            .ok_or_else(|| ConfigError::UnknownModel(id.to_string()))?;
        self.path_resolver.model_path(&entry.file_name).map_err(Into::into)
    }

    /// 是否已加载。
    pub fn is_loaded(&self, id: ModelId) -> bool {
        self.loaded.lock().contains_key(&id)
    }

    /// 列出所有状态。
    pub fn status(&self) -> Vec<ModelStatus> {
        let loaded = self.loaded.lock();
        self.registry
            .list()
            .iter()
            .map(|entry| {
                let id = entry.id.clone();
                let path = self
                    .path_resolver
                    .model_path(&entry.file_name)
                    .unwrap_or_else(|_| PathBuf::from("<unresolved>"));
                let is_loaded = loaded.contains_key(&id);
                ModelStatus {
                    id,
                    loaded: is_loaded,
                    path,
                    size_bytes: entry.size_bytes,
                    verified: false, // Phase 1+ 实现
                }
            })
            .collect()
    }

    /// 注册表引用。
    pub fn registry(&self) -> &ModelRegistry {
        &self.registry
    }

    /// 释放所有模型（用于退出 / 内存压力）。
    pub async fn unload_all(&self) -> Result<(), ConfigError> {
        self.loaded.lock().clear();
        Ok(())
    }

    /// 取出一个已加载的模型（由调用方 downcast）。
    ///
    /// Phase 0 仅占位；Phase 1 由 `ModelLoader<T>` trait 提供具体加载。
    pub fn get<T: Send + Sync + 'static>(&self, id: ModelId) -> Option<Arc<T>> {
        let loaded = self.loaded.lock();
        loaded
            .get(&id)
            .and_then(|any| any.clone().downcast::<T>().ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_model_ids() {
        assert_eq!(ModelId::arcface_w600k_r50().as_str(), "arcface-w600k-r50");
        assert_eq!(ModelId::scrfd_500m_bnkps().as_str(), "scrfd-500m-bnkps");
    }
}