//! 平台抽象的核心类型。

use std::path::PathBuf;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::PlatformError;

/// 照片 ID。
///
/// 桌面端 = 文件路径的字符串
/// Android = MediaStore URI 的字符串
/// iOS = PHAsset localIdentifier 的字符串
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PhotoId(pub String);

impl PhotoId {
    /// 字符串引用。
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for PhotoId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for PhotoId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for PhotoId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// 照片来源文件夹。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotoFolder {
    /// 文件夹 ID（平台相关）
    pub id: String,
    /// 显示名
    pub display_name: String,
    /// 照片数估算（移动端可能为 None）
    pub count_estimate: Option<u64>,
}

/// 分页请求。
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PageRequest {
    /// offset
    pub offset: usize,
    /// limit
    pub limit: usize,
}

impl PageRequest {
    /// 第一页，limit=100。
    pub fn first(limit: usize) -> Self {
        Self { offset: 0, limit }
    }

    /// 下一页（offset += limit）。
    pub fn next(&self) -> Self {
        Self {
            offset: self.offset + self.limit,
            limit: self.limit,
        }
    }
}

/// 一张照片的入口信息。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotoEntry {
    /// 照片 ID
    pub id: PhotoId,
    /// 元数据
    pub metadata: PhotoMetadata,
}

/// 照片元数据。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotoMetadata {
    /// 宽（像素）
    pub width: u32,
    /// 高（像素）
    pub height: u32,
    /// 大小（bytes）
    pub size: u64,
    /// 拍摄时间（来自 EXIF，可能为 None）
    pub captured_at: Option<DateTime<Utc>>,
    /// MIME type
    pub mime_type: String,
    /// BLAKE3 hash（如果已计算）
    pub hash: Option<String>,
}

/// 权限状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermissionStatus {
    /// 已授权
    Granted,
    /// 被拒
    Denied,
    /// 受限（家长控制等）
    Restricted,
    /// 尚未决定（移动端首次请求）
    NotDetermined,
}

/// 照片提供器。
///
/// 每个平台独立 impl：
/// - Desktop：`FileSystemPhotoProvider`（walkdir + image crate）
/// - Android：`MediaStorePhotoProvider`（Kotlin → JNI → Rust）
/// - iOS：`PhotoKitPhotoProvider`（Swift → FFI → Rust）
#[async_trait]
pub trait PhotoProvider: Send + Sync {
    /// 列出所有照片来源文件夹（Desktop only - android/ios 返回设备 library）。
    async fn list_folders(&self) -> Result<Vec<PhotoFolder>, PlatformError>;

    /// 分页列出某文件夹下所有照片。
    async fn list_photos(
        &self,
        folder: &PhotoFolder,
        page: PageRequest,
    ) -> Result<Vec<PhotoEntry>, PlatformError>;

    /// 获取元数据。
    async fn get_metadata(&self, photo_id: &PhotoId) -> Result<PhotoMetadata, PlatformError>;

    /// 获取缩略图 bytes（max_size 为最大边长像素）。
    async fn get_thumbnail(
        &self,
        photo_id: &PhotoId,
        max_size: u32,
    ) -> Result<Vec<u8>, PlatformError>;

    /// 获取完整图片 bytes。
    async fn get_image(&self, photo_id: &PhotoId) -> Result<Vec<u8>, PlatformError>;

    /// 请求权限（Mobile only）。
    async fn request_permission(&self) -> Result<PermissionStatus, PlatformError>;

    /// 删除照片（Mobile only）。
    async fn delete_photo(&self, photo_id: &PhotoId) -> Result<(), PlatformError>;

    /// 获取本地文件系统路径（Desktop only - 移动端返回 None）。
    async fn local_path(&self, photo_id: &PhotoId) -> Result<Option<PathBuf>, PlatformError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_request_paging() {
        let p1 = PageRequest::first(50);
        assert_eq!(p1.offset, 0);
        assert_eq!(p1.limit, 50);
        let p2 = p1.next();
        assert_eq!(p2.offset, 50);
        assert_eq!(p2.limit, 50);
    }
}