//! iOS 平台实现。
//!
//! ## PathResolver
//!
//! iOS app 的数据目录是 `<sandbox>/Library/Application Support/PhotoFinderNext/`。
//! 沙箱根目录通过 `NSTemporaryDirectory` 父级或 `Bundle.main.bundlePath` 同级 `Library/` 推断。
//! Tauri 启动时通过 FFI 注入到 `PF_DATA_DIR` / `PF_CACHE_DIR` / `PF_MODELS_DIR` 环境变量。
//!
//! ## PhotoKitPhotoProvider
//!
//! 通过 FFI 调 `PHAsset` / `PHImageManager` API（Swift 端 stub）。
//! 当前为占位：所有方法返回 `PlatformError::NotImplemented`。
//!
//! ## 与桌面端的差异
//!
//! - `list_folders`：返回单个虚拟 "Photo Library" 文件夹
//! - `list_photos`：通过 PHFetchOptions 分页
//! - `local_path`：永远返回 `None`（PHAsset 走 `localIdentifier`）
//! - `request_permission`：触发 `PHPhotoLibrary.requestAuthorization`（异步）
//! - `delete_photo`：通过 PHAssetChangeRequest. deleteAssets

use std::path::PathBuf;

use async_trait::async_trait;
use serde::Deserialize;
use tracing::warn;

use crate::error::PlatformError;
use crate::path_resolver::PathResolver;
use crate::types::{
    PageRequest, PermissionStatus, PhotoEntry, PhotoFolder, PhotoId, PhotoMetadata, PhotoProvider,
};

// ============================================================================
// PathResolver
// ============================================================================

/// iOS 路径解析器。
///
/// 路径通过环境变量注入（`PF_DATA_DIR` / `PF_CACHE_DIR` / `PF_MODELS_DIR`），
/// 由 iOS 端 `AppDelegate.application(didFinishLaunchingWithOptions:)` 在 Tauri 启动前设置。
/// 变量未设置时 fallback 到 `<sandbox>/Library/...`（开发期裸跑）。
#[derive(Debug, Default, Clone)]
pub struct IosPathResolver;

impl IosPathResolver {
    /// 构造。
    pub fn new() -> Self {
        Self
    }

    fn env_or_sandbox(name: &str) -> PathBuf {
        match std::env::var(name) {
            Ok(s) => PathBuf::from(s),
            Err(_) => {
                // 尝试通过 CWD 推断沙箱根（Cargo ios runner 会设 CWD 到 app 根）
                let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
                let candidate = cwd.join("Library").join("Application Support").join("PhotoFinderNext");
                warn!(
                    "{} not set, fallback to {}",
                    name,
                    candidate.display()
                );
                candidate
            }
        }
    }
}

impl PathResolver for IosPathResolver {
    fn model_path(&self, file_name: &str) -> Result<PathBuf, PlatformError> {
        let dir = Self::env_or_sandbox("PF_MODELS_DIR");
        Ok(dir.join(file_name))
    }

    fn data_dir(&self) -> Result<PathBuf, PlatformError> {
        let dir = Self::env_or_sandbox("PF_DATA_DIR");
        std::fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    fn cache_dir(&self) -> Result<PathBuf, PlatformError> {
        let dir = Self::env_or_sandbox("PF_CACHE_DIR");
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
// NativeMediaBridge — PhotoKit 反向桥抽象（Rust → Swift）
// ============================================================================

/// iOS PhotoKit 反向桥：Rust 通过该 trait 调 Swift `PhotoKitBridge.swift`
/// 的 `@_cdecl` 实现。trait 方法返回 owned Rust 类型，unsafe FFI 细节
/// 隐藏在 `IosNativeMediaBridge` 实现里。
///
/// JSON 协议：所有跨边界结构化数据用 JSON 字符串。
/// 字节 blob 用 base64 字符串。
/// 所有权：Swift 端 `strdup` 分配的字符串由 Rust 端 `libc::free` 释放
///（同一个 libc allocator）。
///
/// Phase 2 step 1 — trait 骨架，impl 在 step 2/3 填。
pub trait NativeMediaBridge: Send + Sync {
    /// 当前照片库权限状态（0 = 未授权，1 = 已授权/limited）。
    fn photo_permission_status(&self) -> i32;

    /// 请求权限。返回 1 表示授权成功，0 拒绝。
    fn request_photo_permission(&self) -> i32;

    /// 列出所有照片（JSON 数组）。
    /// 每项：`{"id": String, "uri": String, "size": Number, "modified_time": Number}`。
    /// `None` = 调用失败 / Swift 端未连接。
    fn list_photo_items(&self) -> Option<String>;

    /// 按 URI 查单个照片元数据（JSON 对象，同上 schema）。
    fn photo_item_by_uri(&self, uri: &str) -> Option<String>;

    /// 把全分辨率图片导出到临时文件，返回绝对路径。
    /// Swift 端负责清理。
    fn export_to_temp_file(&self, uri: &str) -> Option<String>;

    /// 导出缩略图到临时文件，返回绝对路径。
    fn export_thumbnail_to_temp_file(&self, uri: &str, max_size: u32) -> Option<String>;

    /// 读取原始字节（base64 字符串）。
    fn read_bytes_base64(&self, uri: &str) -> Option<String>;

    /// 注册后台处理任务。
    fn schedule_bg_task(&self, id: &str, wait_secs: u64) -> i32;
}

/// 默认 no-op bridge — 用于开发环境无 Swift 端时。
/// 所有方法返回"未授权/失败"语义，不 panic。
pub struct NoopNativeMediaBridge;

impl NativeMediaBridge for NoopNativeMediaBridge {
    fn photo_permission_status(&self) -> i32 { 0 }
    fn request_photo_permission(&self) -> i32 { 0 }
    fn list_photo_items(&self) -> Option<String> { None }
    fn photo_item_by_uri(&self, _uri: &str) -> Option<String> { None }
    fn export_to_temp_file(&self, _uri: &str) -> Option<String> { None }
    fn export_thumbnail_to_temp_file(&self, _uri: &str, _max_size: u32) -> Option<String> { None }
    fn read_bytes_base64(&self, _uri: &str) -> Option<String> { None }
    fn schedule_bg_task(&self, _id: &str, _wait_secs: u64) -> i32 { -1 }
}

/// iOS 端 `PhotoKitBridge.swift` 的真实 bridge（Phase 2 step 2 实现）。
/// Phase 2 step 1 仅有骨架 struct，impl 留待 step 2 填 unsafe extern "C" 调用。
#[allow(dead_code)]
pub struct IosNativeMediaBridge {
    // Phase 2 step 2: 加 unsafe extern "C" 声明 + unsafe impl NativeMediaBridge
}

impl IosNativeMediaBridge {
    /// 构造（Phase 2 step 2 会链接 Swift @_cdecl 符号）。
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {}
    }
}

// ============================================================================
// iOS target: unsafe extern "C" 声明 + impl NativeMediaBridge
// ============================================================================
//
// C ABI 契约（与 `apps/ios/Sources/photofinder_desktop/PhotoKitBridge.swift`
// 的 `@_cdecl` 实现对齐）：
// - 字符串都是 NUL-terminated
// - 返回字符串 Swift 端用 `strdup` 分配，Rust 端用 `libc::free` 释放
// - NULL pointer = 调用失败（Swift 端控制台 log 详情）
//
// unsafe extern "C" 块只在 iOS target 编译。非 iOS target 时
// IosNativeMediaBridge 走下面的 no-op fallback impl，让 host 测试能跑。

#[cfg(target_os = "ios")]
mod imp {
    use super::*;
    use std::ffi::{c_char, CStr, CString};
    use std::os::raw::c_longlong;

    #[allow(unsafe_code)]
    unsafe extern "C" {
        fn pf_ios_photo_permission_status() -> i32;
        fn pf_ios_request_photo_permission() -> i32;
        fn pf_ios_list_photo_items() -> *mut c_char;
        fn pf_ios_photo_item_by_uri(uri: *const c_char) -> *mut c_char;
        fn pf_ios_export_to_temp_file(uri: *const c_char) -> *mut c_char;
        fn pf_ios_export_thumbnail_to_temp_file(uri: *const c_char, max_size: u32) -> *mut c_char;
        fn pf_ios_read_bytes(uri: *const c_char) -> *mut c_char;
        fn pf_ios_schedule_bg_task(id: *const c_char, wait_secs: c_longlong) -> i32;
    }

    /// Take ownership of a Swift-side `strdup`'d C string.
    /// `None` for NULL pointer.
    #[allow(unsafe_code)]
    unsafe fn take_c_string(ptr: *mut c_char) -> Option<String> {
        if ptr.is_null() {
            return None;
        }
        let s = CStr::from_ptr(ptr).to_string_lossy().into_owned();
        libc::free(ptr.cast());
        Some(s)
    }

    impl NativeMediaBridge for IosNativeMediaBridge {
        #[allow(unsafe_code)]
        fn photo_permission_status(&self) -> i32 {
            unsafe { pf_ios_photo_permission_status() }
        }

        #[allow(unsafe_code)]
        fn request_photo_permission(&self) -> i32 {
            unsafe { pf_ios_request_photo_permission() }
        }

        #[allow(unsafe_code)]
        fn list_photo_items(&self) -> Option<String> {
            unsafe { take_c_string(pf_ios_list_photo_items()) }
        }

        #[allow(unsafe_code)]
        fn photo_item_by_uri(&self, uri: &str) -> Option<String> {
            let uri_c = CString::new(uri).ok()?;
            unsafe { take_c_string(pf_ios_photo_item_by_uri(uri_c.as_ptr())) }
        }

        #[allow(unsafe_code)]
        fn export_to_temp_file(&self, uri: &str) -> Option<String> {
            let uri_c = CString::new(uri).ok()?;
            unsafe { take_c_string(pf_ios_export_to_temp_file(uri_c.as_ptr())) }
        }

        #[allow(unsafe_code)]
        fn export_thumbnail_to_temp_file(&self, uri: &str, max_size: u32) -> Option<String> {
            let uri_c = CString::new(uri).ok()?;
            unsafe {
                take_c_string(pf_ios_export_thumbnail_to_temp_file(uri_c.as_ptr(), max_size))
            }
        }

        #[allow(unsafe_code)]
        fn read_bytes_base64(&self, uri: &str) -> Option<String> {
            let uri_c = CString::new(uri).ok()?;
            unsafe { take_c_string(pf_ios_read_bytes(uri_c.as_ptr())) }
        }

        #[allow(unsafe_code)]
        fn schedule_bg_task(&self, id: &str, wait_secs: u64) -> i32 {
            // schedule 返回 i32（错误码），CString::new 失败时返回 -1 哨兵
            let id_c = match CString::new(id) {
                Ok(s) => s,
                Err(_) => return -1,
            };
            unsafe { pf_ios_schedule_bg_task(id_c.as_ptr(), wait_secs as c_longlong) }
        }
    }
}

// 非 iOS target 的 no-op fallback — 让 `IosNativeMediaBridge` 在 host 上也能
// 当作 `Box<dyn NativeMediaBridge>` 使用（desktop 测试 / 静态分析）。
#[cfg(not(target_os = "ios"))]
impl NativeMediaBridge for IosNativeMediaBridge {
    fn photo_permission_status(&self) -> i32 { 0 }
    fn request_photo_permission(&self) -> i32 { 0 }
    fn list_photo_items(&self) -> Option<String> { None }
    fn photo_item_by_uri(&self, _uri: &str) -> Option<String> { None }
    fn export_to_temp_file(&self, _uri: &str) -> Option<String> { None }
    fn export_thumbnail_to_temp_file(&self, _uri: &str, _max_size: u32) -> Option<String> { None }
    fn read_bytes_base64(&self, _uri: &str) -> Option<String> { None }
    fn schedule_bg_task(&self, _id: &str, _wait_secs: u64) -> i32 { -1 }
}

/// Wire format for a native photo item（Swift → Rust JSON schema）。
#[derive(Debug, Clone, Deserialize)]
pub struct NativePhotoItemDto {
    pub id: String,
    pub uri: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub modified_time: i64,
}

/// 把 JSON 字符串解析成 `NativePhotoItemDto` 列表。
///
/// 暴露成 pub(crate) 让 `PhotoKitPhotoProvider` 调（step 2.3 接入）。
#[allow(dead_code)]
pub(crate) fn parse_photo_items(json: &str) -> Result<Vec<NativePhotoItemDto>, PlatformError> {
    serde_json::from_str(json)
        .map_err(|e| PlatformError::Bridge(format!("invalid photo items JSON: {e}")))
}

// ============================================================================
// PhotoKitPhotoProvider — 持有 Optional<NativeMediaBridge>
// ============================================================================

/// PhotoKit PhotoProvider — 通过 FFI 调 iOS `PHPhotoLibrary` / `PHAsset`。
///
/// 两种构造：
/// - `new()` — 无 bridge，用于开发/桌面。所有方法返回 `Unsupported`。
/// - `with_bridge(b)` — iOS 启动时注入；list_photos / get_image 等真正调 Swift。
pub struct PhotoKitPhotoProvider {
    bridge: Option<Box<dyn NativeMediaBridge>>,
}

impl std::fmt::Debug for PhotoKitPhotoProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhotoKitPhotoProvider")
            .field("bridge", &self.bridge.as_ref().map(|_| "<NativeMediaBridge>"))
            .finish()
    }
}

impl Default for PhotoKitPhotoProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for PhotoKitPhotoProvider {
    fn clone(&self) -> Self {
        // Box<dyn Trait> 不可 Clone — 重建无 bridge 版本。
        // iOS 端一般单实例,不需要 Clone。
        Self::new()
    }
}

impl PhotoKitPhotoProvider {
    /// 无 bridge 构造（开发/桌面 fallback）。
    pub fn new() -> Self {
        Self { bridge: None }
    }

    /// 带 bridge 构造（iOS 启动时由 bootstrap 调用）。
    pub fn with_bridge(bridge: Box<dyn NativeMediaBridge>) -> Self {
        Self { bridge: Some(bridge) }
    }
}

fn not_implemented(method: &str) -> PlatformError {
    PlatformError::Unsupported(format!(
        "PhotoKitPhotoProvider::{method} requires NativeMediaBridge injection"
    ))
}

/// NativePhotoItemDto → PhotoMetadata (用 placeholder 填 width/height/mime)。
fn dto_to_metadata(dto: &NativePhotoItemDto) -> PhotoMetadata {
    PhotoMetadata {
        width: 0,
        height: 0,
        size: dto.size,
        captured_at: None,
        mime_type: "image/jpeg".to_string(),
        hash: None,
    }
}

/// 把 dto list 拆成 (PhotoEntry) list。
fn dtos_to_entries(dtos: Vec<NativePhotoItemDto>) -> Vec<PhotoEntry> {
    dtos.into_iter()
        .map(|dto| {
            let meta = dto_to_metadata(&dto);
            PhotoEntry {
                id: PhotoId::from(dto.id),
                metadata: meta,
            }
        })
        .collect()
}

#[async_trait]
impl PhotoProvider for PhotoKitPhotoProvider {
    async fn list_folders(&self) -> Result<Vec<PhotoFolder>, PlatformError> {
        // 始终返回单虚拟 "Photo Library"
        Ok(vec![PhotoFolder {
            id: "photo-library".to_string(),
            display_name: "Photo Library".to_string(),
            count_estimate: None,
        }])
    }

    async fn list_photos(
        &self,
        _folder: &PhotoFolder,
        page: PageRequest,
    ) -> Result<Vec<PhotoEntry>, PlatformError> {
        let bridge = self.bridge.as_ref().ok_or_else(|| not_implemented("list_photos"))?;
        let json = bridge
            .list_photo_items()
            .ok_or_else(|| PlatformError::Bridge("list_photo_items returned None".into()))?;
        let all = parse_photo_items(&json)?;
        // 应用分页
        let start = page.offset.min(all.len());
        let end = (start + page.limit).min(all.len());
        Ok(dtos_to_entries(all[start..end].to_vec()))
    }

    async fn get_metadata(&self, photo_id: &PhotoId) -> Result<PhotoMetadata, PlatformError> {
        let bridge = self.bridge.as_ref().ok_or_else(|| not_implemented("get_metadata"))?;
        let uri = photo_id.as_str();
        let json = bridge
            .photo_item_by_uri(uri)
            .ok_or_else(|| PlatformError::Bridge(format!("photo_item_by_uri({uri}) returned None")))?;
        let dto: NativePhotoItemDto = serde_json::from_str(&json)
            .map_err(|e| PlatformError::Bridge(format!("invalid photo_item JSON: {e}")))?;
        Ok(dto_to_metadata(&dto))
    }

    async fn get_thumbnail(
        &self,
        photo_id: &PhotoId,
        max_size: u32,
    ) -> Result<Vec<u8>, PlatformError> {
        let bridge = self.bridge.as_ref().ok_or_else(|| not_implemented("get_thumbnail"))?;
        let path = bridge
            .export_thumbnail_to_temp_file(photo_id.as_str(), max_size)
            .ok_or_else(|| {
                PlatformError::Bridge("export_thumbnail_to_temp_file returned None".into())
            })?;
        std::fs::read(&path).map_err(|e| {
            PlatformError::Bridge(format!("read thumbnail {}: {e}", path))
        })
    }

    async fn get_image(&self, photo_id: &PhotoId) -> Result<Vec<u8>, PlatformError> {
        let bridge = self.bridge.as_ref().ok_or_else(|| not_implemented("get_image"))?;
        let path = bridge
            .export_to_temp_file(photo_id.as_str())
            .ok_or_else(|| PlatformError::Bridge("export_to_temp_file returned None".into()))?;
        std::fs::read(&path)
            .map_err(|e| PlatformError::Bridge(format!("read image {}: {e}", path)))
    }

    async fn request_permission(&self) -> Result<PermissionStatus, PlatformError> {
        match self.bridge.as_ref() {
            Some(b) => {
                let status = b.photo_permission_status();
                Ok(match status {
                    1 => PermissionStatus::Granted,
                    0 => PermissionStatus::Denied,
                    _ => PermissionStatus::NotDetermined,
                })
            }
            None => {
                // 无 bridge 时模拟已授权（开发环境 / 桌面）
                Ok(PermissionStatus::Granted)
            }
        }
    }

    async fn delete_photo(&self, _photo_id: &PhotoId) -> Result<(), PlatformError> {
        // PHAsset delete 走 PHAssetChangeRequest.deleteAssets，需 batch context。
        // iOS bridge 暂未暴露，留 NotImplemented。
        Err(not_implemented("delete_photo"))
    }

    async fn local_path(&self, _photo_id: &PhotoId) -> Result<Option<PathBuf>, PlatformError> {
        // PHAsset 没有 file path
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ios_path_resolver_falls_back_to_sandbox() {
        // 清掉 env 强制 fallback
        std::env::remove_var("PF_DATA_DIR");
        std::env::remove_var("PF_CACHE_DIR");
        std::env::remove_var("PF_MODELS_DIR");

        let r = IosPathResolver::new();
        let dd = r.data_dir().unwrap();
        let cd = r.cache_dir().unwrap();
        let mp = r.model_path("test.onnx").unwrap();

        assert!(dd.exists(), "data_dir should be created");
        assert!(cd.exists(), "cache_dir should be created");
        assert!(mp.to_string_lossy().contains("test.onnx"));
    }

    #[tokio::test]
    async fn provider_without_bridge_returns_unsupported() {
        let p = PhotoKitPhotoProvider::new();
        let folder = PhotoFolder {
            id: "x".into(),
            display_name: "x".into(),
            count_estimate: None,
        };
        let err = p
            .list_photos(&folder, PageRequest { offset: 0, limit: 10 })
            .await
            .unwrap_err();
        assert!(matches!(err, PlatformError::Unsupported(_)));
    }

    #[tokio::test]
    async fn provider_with_bridge_calls_swift() {
        // Noop bridge returns None → list_photos returns Bridge error (not Unsupported).
        let bridge: Box<dyn NativeMediaBridge> = Box::new(NoopNativeMediaBridge);
        let p = PhotoKitPhotoProvider::with_bridge(bridge);
        let folder = PhotoFolder {
            id: "x".into(),
            display_name: "x".into(),
            count_estimate: None,
        };
        let err = p
            .list_photos(&folder, PageRequest { offset: 0, limit: 10 })
            .await
            .unwrap_err();
        assert!(matches!(err, PlatformError::Bridge(_)));
    }

    #[tokio::test]
    async fn noop_bridge_returns_none_for_list() {
        let b = NoopNativeMediaBridge;
        assert_eq!(b.photo_permission_status(), 0);
        assert_eq!(b.request_photo_permission(), 0);
        assert!(b.list_photo_items().is_none());
        assert!(b.photo_item_by_uri("x").is_none());
        assert!(b.export_to_temp_file("x").is_none());
        assert!(b.export_thumbnail_to_temp_file("x", 256).is_none());
        assert!(b.read_bytes_base64("x").is_none());
        assert_eq!(b.schedule_bg_task("x", 0), -1);
    }

    #[test]
    fn parse_photo_items_handles_valid_json() {
        let json = r#"[
            {"id": "asset/1", "uri": "phasset://1", "size": 1024, "modified_time": 1234},
            {"id": "asset/2", "uri": "phasset://2", "size": 2048}
        ]"#;
        let items = parse_photo_items(json).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].id, "asset/1");
        assert_eq!(items[1].size, 2048);
        assert_eq!(items[1].modified_time, 0); // default
    }
}
