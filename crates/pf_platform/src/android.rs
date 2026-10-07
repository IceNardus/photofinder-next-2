//! Android 平台实现。
//!
//! ## PathResolver
//!
//! Android app 的数据目录是 `/data/data/<pkg>/files/...`。
//! Tauri 启动时通过 JNI 注入到 `PF_DATA_DIR` / `PF_CACHE_DIR` / `PF_MODELS_DIR` 环境变量；
//! 此处直接读 env。如果变量未设置（开发期裸跑），fallback 到 `std::env::temp_dir()` 子目录。
//!
//! ## MediaStorePhotoProvider
//!
//! 通过 JNI 调 `MediaStore.Images` API（Kotlin 端 stub）。
//! 当前为占位：所有方法返回 `PlatformError::NotImplemented`。
//! 真实 impl 需要在 Android 端用 Kotlin 写 `MediaStoreBridge` JNI 类，由 Tauri 注入。
//!
//! ## 与桌面端的差异
//!
//! - `list_folders`：返回单个虚拟 "Device Library" 文件夹
//! - `list_photos`：通过 MediaStore URI 列表分页
//! - `local_path`：永远返回 `None`（MediaStore 走 content:// URI，无 file path）
//! - `request_permission`：触发 Android 13+ 的 `READ_MEDIA_IMAGES` 权限请求
//! - `delete_photo`：通过 MediaStore 标记删除

use std::path::PathBuf;

use async_trait::async_trait;
use tracing::warn;

use crate::error::PlatformError;
use crate::path_resolver::PathResolver;
use crate::types::{
    PageRequest, PermissionStatus, PhotoEntry, PhotoFolder, PhotoId, PhotoMetadata, PhotoProvider,
};

// ============================================================================
// PathResolver
// ============================================================================

/// Android 路径解析器。
///
/// 路径通过环境变量注入（`PF_DATA_DIR` / `PF_CACHE_DIR` / `PF_MODELS_DIR`），
/// 由 Android 端 `MainActivity.onCreate` 在 `Tauri.create` 之前设置。
/// 变量未设置时 fallback 到临时目录（开发期可走通编译）。
#[derive(Debug, Default, Clone)]
pub struct AndroidPathResolver;

impl AndroidPathResolver {
    /// 构造。
    pub fn new() -> Self {
        Self
    }

    fn env_or_tmp(name: &str) -> PathBuf {
        match std::env::var(name) {
            Ok(s) => PathBuf::from(s),
            Err(_) => {
                let mut p = std::env::temp_dir();
                p.push("photofinder-android");
                p.push(name.trim_start_matches("PF_").to_lowercase());
                warn!(
                    "{} not set, fallback to {}",
                    name,
                    p.display()
                );
                p
            }
        }
    }
}

impl PathResolver for AndroidPathResolver {
    fn model_path(&self, file_name: &str) -> Result<PathBuf, PlatformError> {
        let dir = Self::env_or_tmp("PF_MODELS_DIR");
        Ok(dir.join(file_name))
    }

    fn data_dir(&self) -> Result<PathBuf, PlatformError> {
        let dir = Self::env_or_tmp("PF_DATA_DIR");
        std::fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    fn cache_dir(&self) -> Result<PathBuf, PlatformError> {
        let dir = Self::env_or_tmp("PF_CACHE_DIR");
        std::fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    fn db_path(&self) -> Result<PathBuf, PlatformError> {
        Ok(self.data_dir()?.join("photofinder.db"))
    }

    fn log_dir(&self) -> Result<PathBuf, PlatformError> {
        Ok(self.data_dir()?.join("logs"))
    }
}

// ============================================================================
// MediaStorePhotoProvider (stub)
// ============================================================================

/// MediaStore PhotoProvider — 通过 JNI 调 Android `MediaStore.Images`。
///
/// **当前是 stub**：所有方法返回 `NotImplemented`。
/// 完整实现需要：
/// 1. Android 端创建 `MediaStoreBridge.kt`（JNI 类）
/// 2. 在 `MainActivity.onCreate` 里调 `System.loadLibrary("photofinder")`
/// 3. 把 bridge 实例指针通过 env var 传给 Rust
/// 4. 本 impl 接收指针，调用 JNI 方法
///
/// 占位 impl 让 Rust 端代码可以编译并通过移动端 bootstrap。
#[derive(Debug, Default, Clone)]
pub struct MediaStorePhotoProvider;

impl MediaStorePhotoProvider {
    /// 构造。
    pub fn new() -> Self {
        Self
    }
}

fn not_implemented(method: &str) -> PlatformError {
    PlatformError::Unsupported(format!(
        "MediaStorePhotoProvider::{method} not yet implemented (requires JNI bridge)"
    ))
}

#[async_trait]
impl PhotoProvider for MediaStorePhotoProvider {
    async fn list_folders(&self) -> Result<Vec<PhotoFolder>, PlatformError> {
        // 占位：返回单虚拟 "Device Library"
        Ok(vec![PhotoFolder {
            id: "device-library".to_string(),
            display_name: "Device Library".to_string(),
            count_estimate: None,
        }])
    }

    async fn list_photos(
        &self,
        _folder: &PhotoFolder,
        _page: PageRequest,
    ) -> Result<Vec<PhotoEntry>, PlatformError> {
        Err(not_implemented("list_photos"))
    }

    async fn get_metadata(&self, _photo_id: &PhotoId) -> Result<PhotoMetadata, PlatformError> {
        Err(not_implemented("get_metadata"))
    }

    async fn get_thumbnail(
        &self,
        _photo_id: &PhotoId,
        _max_size: u32,
    ) -> Result<Vec<u8>, PlatformError> {
        Err(not_implemented("get_thumbnail"))
    }

    async fn get_image(&self, _photo_id: &PhotoId) -> Result<Vec<u8>, PlatformError> {
        Err(not_implemented("get_image"))
    }

    async fn request_permission(&self) -> Result<PermissionStatus, PlatformError> {
        // 占位：假设已授权（真实 impl 通过 JNI 触发权限请求）
        Ok(PermissionStatus::Granted)
    }

    async fn delete_photo(&self, _photo_id: &PhotoId) -> Result<(), PlatformError> {
        Err(not_implemented("delete_photo"))
    }

    async fn local_path(&self, _photo_id: &PhotoId) -> Result<Option<PathBuf>, PlatformError> {
        // MediaStore URI 没有 file path
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn android_path_resolver_falls_back_to_tmp() {
        // 清掉 env 强制 fallback
        std::env::remove_var("PF_DATA_DIR");
        std::env::remove_var("PF_CACHE_DIR");
        std::env::remove_var("PF_MODELS_DIR");

        let r = AndroidPathResolver::new();
        let dd = r.data_dir().unwrap();
        let cd = r.cache_dir().unwrap();
        let mp = r.model_path("test.onnx").unwrap();

        assert!(dd.exists(), "data_dir should be created");
        assert!(cd.exists(), "cache_dir should be created");
        assert!(mp.to_string_lossy().contains("test.onnx"));
    }
}
