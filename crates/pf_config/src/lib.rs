//! # pf_config — 统一配置 + ModelManager
//!
//! 集中所有 magic number / threshold / 模型版本。
//!
//! 规则：
//! - 默认值硬编码在 Rust struct
//! - 可被 TOML 覆盖
//! - 可被环境变量覆盖（`PF_FACE_MIN_QUALITY=0.5`）
//! - 可被命令行参数覆盖
//!
//! 不知道 ONNX runtime API；不知道具体业务。

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::Deserialize;

pub mod config;
pub mod error;
pub mod model;

pub use config::{Config, FaceConfig, IdentityExecutionMode, ObjectConfig, PatchConfig, RoiConfig, ScannerConfig, ShadowModeConfig, ShadowPromotionPolicy, VectorConfig};
pub use error::ConfigError;
pub use model::{ModelEntry, ModelId, ModelManager, ModelRegistry, ModelStatus};

// re-export 平台层的 PathResolver，方便调用方 use pf_config::PathResolver
pub use pf_platform::PathResolver;

/// 从 TOML value 解析 Config。
pub fn deserialize_config(value: &toml::Value) -> Result<Config, ConfigError> {
    Config::deserialize(value.clone()).map_err(|e| ConfigError::Parse(e.to_string()))
}

/// 从 manifest.toml 加载 ModelRegistry。
pub fn load_manifest_toml(path: &std::path::Path) -> Result<ModelRegistry, ConfigError> {
    let s = std::fs::read_to_string(path)?;
    let manifest: ManifestToml =
        toml::from_str(&s).map_err(|e| ConfigError::Parse(format!("manifest: {e}")))?;
    let mut reg = ModelRegistry::new();
    for entry in manifest.models {
        reg.register(entry);
    }
    Ok(reg)
}

/// 便捷函数：从 ModelManager 取模型路径，缺失时 warn 而不是 panic。
pub fn load_model_or_warn(
    mgr: &ModelManager,
    id: ModelId,
) -> Result<std::path::PathBuf, ConfigError> {
    match mgr.path(id.clone()) {
        Ok(p) => Ok(p),
        Err(e) => {
            tracing::warn!("model {} not available: {}", id, e);
            Err(e)
        }
    }
}

#[derive(Debug, serde::Deserialize)]
struct ManifestToml {
    #[serde(default)]
    models: Vec<ModelEntry>,
}