//! FileSystemPhotoProvider — 桌面端 PhotoProvider 实现。
//!
//! 用 walkdir 扫描本地文件夹，列出支持的图片文件。
//!
//! **算法参数与 ai-next/src-tauri/src/core/scanner.rs::is_image_file 逐位对齐**：
//! 32 个图片扩展名一致；大小写不敏感。
//!
//! 缩略图：使用 Lanczos3 缩放到 max_size×max_size，编码 JPEG。
//!
//! 不在 Provider 里做"低色 / 字节每像素"过滤 — 那属于 *索引时* 的过滤（见 pf_ai::filter）。

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use async_trait::async_trait;
use image::{DynamicImage, GenericImageView, ImageFormat};
use parking_lot::Mutex;
use tracing::{info, warn};

use crate::error::PlatformError;
use crate::types::{
    PageRequest, PermissionStatus, PhotoEntry, PhotoFolder, PhotoId, PhotoMetadata, PhotoProvider,
};

/// 支持的图片扩展名（小写，无 `.`）。
///
/// 与 ai-next 完全一致：32 种格式。
pub const SUPPORTED_EXTS: &[&str] = &[
    "jpg", "jpeg", "png", "webp", "bmp", "gif", "tiff", "tif", "heic", "heif", "avif", "svg",
    "ico", "raw", "cr2", "nef", "arw", "dng", "orf", "rw2", "pef", "srw", "x3f", "3fr", "raf",
    "mrw", "nrw", "dcr", "mos", "crw", "erf", "mdc",
];

/// 是否为支持的图片扩展名。
pub fn is_image_file(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => {
            let ext = ext.to_lowercase();
            SUPPORTED_EXTS.contains(&ext.as_str())
        }
        None => false,
    }
}

/// 从扩展名推断 MIME type。
pub fn mime_for_ext(ext: &str) -> &'static str {
    match ext.to_lowercase().as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "gif" => "image/gif",
        "tiff" | "tif" => "image/tiff",
        "heic" | "heif" => "image/heic",
        "avif" => "image/avif",
        "svg" => "image/svg+xml",
        "ico" => "image/x-icon",
        _ => "application/octet-stream",
    }
}

/// 桌面端 PhotoProvider：扫描一组根目录。
pub struct FileSystemPhotoProvider {
    /// 根目录列表（每项作为一个 PhotoFolder）
    roots: Vec<PathBuf>,
    /// 缩略图 JPEG 质量（1-100）
    thumbnail_quality: u8,
    /// 缓存的扫描结果（folder_id → 完整路径列表）
    cache: Arc<Mutex<Vec<PathBuf>>>,
}

impl FileSystemPhotoProvider {
    /// 构造。
    ///
    /// `roots` 为扫描根目录集合；每个根目录作为一个 PhotoFolder。
    pub fn new(roots: Vec<PathBuf>) -> Self {
        Self {
            roots,
            thumbnail_quality: 85,
            cache: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// 构造并立即触发一次扫描（warm cache）。
    pub fn new_scanned(roots: Vec<PathBuf>) -> Self {
        let s = Self::new(roots);
        let _ = s.scan_all();
        s
    }

    /// 缩略图质量。
    pub fn with_thumbnail_quality(mut self, q: u8) -> Self {
        self.thumbnail_quality = q.clamp(1, 100);
        self
    }

    /// 触发一次全量扫描（walkdir 遍历所有根）。
    pub fn scan_all(&self) -> Result<usize, PlatformError> {
        let mut all: Vec<PathBuf> = Vec::new();
        for root in &self.roots {
            for entry in walkdir::WalkDir::new(root).follow_links(false) {
                let entry = match entry {
                    Ok(e) => e,
                    Err(e) => {
                        warn!("walkdir error: {}", e);
                        continue;
                    }
                };
                let path = entry.path();
                if is_image_file(path) {
                    all.push(path.to_path_buf());
                }
            }
        }
        let n = all.len();
        *self.cache.lock() = all;
        Ok(n)
    }

    /// folder_id → 根目录路径（folder_id 即根目录的字符串形式）。
    fn folder_id(root: &Path) -> String {
        root.to_string_lossy().to_string()
    }

    /// 取某个根目录下的所有图片路径（按 sorted 顺序保证分页稳定）。
    fn paths_in_root(&self, root: &Path) -> Result<Vec<PathBuf>, PlatformError> {
        let mut paths: Vec<PathBuf> = Vec::new();
        let mut walk_errors: usize = 0;
        for entry in walkdir::WalkDir::new(root).follow_links(false) {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    walk_errors += 1;
                    warn!("walkdir error: {}", e);
                    continue;
                }
            };
            let p = entry.path();
            if is_image_file(p) {
                paths.push(p.to_path_buf());
            }
        }
        paths.sort();
        info!(
            root = %root.display(),
            found = paths.len(),
            walk_errors,
            "paths_in_root complete"
        );
        Ok(paths)
    }

    /// 构造 PhotoEntry（按需解码元数据）。
    fn entry_from_path(path: &Path) -> Result<PhotoEntry, PlatformError> {
        let meta = PhotoMetadata::from_path(path)?;
        let id = PhotoId(path.to_string_lossy().to_string());
        Ok(PhotoEntry { id, metadata: meta })
    }
}

#[async_trait]
impl PhotoProvider for FileSystemPhotoProvider {
    async fn list_folders(&self) -> Result<Vec<PhotoFolder>, PlatformError> {
        let folders: Vec<PhotoFolder> = self
            .roots
            .iter()
            .map(|r| PhotoFolder {
                id: Self::folder_id(r),
                display_name: r
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| r.to_string_lossy().to_string()),
                count_estimate: None, // scan 时再算
            })
            .collect();
        Ok(folders)
    }

    async fn list_photos(
        &self,
        folder: &PhotoFolder,
        page: PageRequest,
    ) -> Result<Vec<PhotoEntry>, PlatformError> {
        let root = PathBuf::from(&folder.id);
        let paths = self.paths_in_root(&root)?;
        // 跳过无法解码元数据的文件（不要因为一个损坏文件让整个扫描失败）
        let slice: Vec<PhotoEntry> = paths
            .iter()
            .skip(page.offset)
            .take(page.limit)
            .filter_map(|p| Self::entry_from_path(p).ok())
            .collect();
        Ok(slice)
    }

    async fn get_metadata(&self, photo_id: &PhotoId) -> Result<PhotoMetadata, PlatformError> {
        let path = PathBuf::from(photo_id.as_str());
        PhotoMetadata::from_path(&path)
    }

    async fn get_thumbnail(
        &self,
        photo_id: &PhotoId,
        max_size: u32,
    ) -> Result<Vec<u8>, PlatformError> {
        let path = PathBuf::from(photo_id.as_str());
        let img = image::open(&path).map_err(|e| PlatformError::ImageDecode(e.to_string()))?;
        let (w, h) = img.dimensions();
        let scale = (max_size as f32 / w.max(h) as f32).min(1.0);
        let tw = ((w as f32) * scale).max(1.0) as u32;
        let th = ((h as f32) * scale).max(1.0) as u32;
        let resized: DynamicImage = img.resize(tw, th, image::imageops::FilterType::Lanczos3);
        let mut buf = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buf);
        resized
            .to_rgb8()
            .write_to(&mut cursor, ImageFormat::Jpeg)
            .map_err(|e| PlatformError::ImageDecode(e.to_string()))?;
        // quality 暂未生效（image 0.25 JPEG 不暴露 quality 参数；Phase 2 替换 encoder）
        let _ = self.thumbnail_quality;
        Ok(buf)
    }

    async fn get_image(&self, photo_id: &PhotoId) -> Result<Vec<u8>, PlatformError> {
        let path = PathBuf::from(photo_id.as_str());
        std::fs::read(&path).map_err(PlatformError::Io)
    }

    async fn request_permission(&self) -> Result<PermissionStatus, PlatformError> {
        // Desktop 无需权限
        Ok(PermissionStatus::Granted)
    }

    async fn delete_photo(&self, _photo_id: &PhotoId) -> Result<(), PlatformError> {
        Err(PlatformError::Unsupported(
            "FileSystemPhotoProvider does not delete files; user must delete via OS".into(),
        ))
    }

    async fn local_path(&self, photo_id: &PhotoId) -> Result<Option<PathBuf>, PlatformError> {
        Ok(Some(PathBuf::from(photo_id.as_str())))
    }
}

impl PhotoMetadata {
    /// 从文件路径构造（读 metadata + 尝试读图片尺寸）。
    ///
    /// 注：图片尺寸读取可能失败（损坏文件）；失败时填 0。
    pub fn from_path(path: &Path) -> Result<Self, PlatformError> {
        let fs_meta = std::fs::metadata(path)?;
        let size = fs_meta.len();
        let modified = fs_meta
            .modified()
            .ok()
            .and_then(|m| m.duration_since(UNIX_EPOCH).ok())
            .map(|d| {
                chrono::DateTime::<chrono::Utc>::from_timestamp(d.as_secs() as i64, 0)
                    .unwrap_or_else(chrono::Utc::now)
            });

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        let mime_type = mime_for_ext(&ext).to_string();

        let mut width = 0u32;
        let mut height = 0u32;
        if let Ok(img) = image::open(path) {
            let (w, h) = img.dimensions();
            width = w;
            height = h;
        }

        Ok(PhotoMetadata {
            width,
            height,
            size,
            captured_at: modified,
            mime_type,
            hash: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn make_tmp_dir(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "pf-fsp-{}-{}-{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn supported_exts_count_matches_ai_next() {
        // 与 ai-next 严格一致：32 个扩展名
        assert_eq!(SUPPORTED_EXTS.len(), 32);
    }

    #[test]
    fn is_image_file_recognizes_jpg() {
        let dir = make_tmp_dir("jpg");
        let f = dir.join("a.JPG"); // 大写
        fs::write(&f, b"fake").unwrap();
        assert!(is_image_file(&f));
    }

    #[test]
    fn is_image_file_rejects_txt() {
        let dir = make_tmp_dir("txt");
        let f = dir.join("a.txt");
        fs::write(&f, b"x").unwrap();
        assert!(!is_image_file(&f));
    }

    #[test]
    fn mime_for_common_exts() {
        assert_eq!(mime_for_ext("jpg"), "image/jpeg");
        assert_eq!(mime_for_ext("JPG"), "image/jpeg");
        assert_eq!(mime_for_ext("png"), "image/png");
        assert_eq!(mime_for_ext("webp"), "image/webp");
        assert_eq!(mime_for_ext("heic"), "image/heic");
        assert_eq!(mime_for_ext("unknown"), "application/octet-stream");
    }

    #[tokio::test]
    async fn scan_all_finds_supported_images() {
        let dir = make_tmp_dir("scan");
        // 写两个有效文件 + 一个无效文件
        let a = dir.join("a.jpg");
        let b = dir.join("b.PNG");
        let c = dir.join("c.txt");
        fs::write(&a, b"x").unwrap();
        fs::write(&b, b"y").unwrap();
        fs::write(&c, b"z").unwrap();
        // 子目录里的图片
        let sub = dir.join("sub");
        fs::create_dir(&sub).unwrap();
        let d = sub.join("d.heic");
        fs::write(&d, b"w").unwrap();

        let provider = FileSystemPhotoProvider::new(vec![dir.clone()]);
        let n = provider.scan_all().unwrap();
        assert_eq!(n, 3, "应该找到 a.jpg, b.PNG, d.heic");

        let folders = provider.list_folders().await.unwrap();
        assert_eq!(folders.len(), 1);
        assert_eq!(folders[0].id, dir.to_string_lossy().to_string());

        let entries = provider
            .list_photos(&folders[0], PageRequest::first(100))
            .await
            .unwrap();
        assert_eq!(entries.len(), 3);
        // 排序后: a.jpg, b.PNG, sub/d.heic
        assert!(entries[0].id.as_str().ends_with("a.jpg"));
        assert!(entries[1].id.as_str().ends_with("b.PNG"));
        assert!(entries[2].id.as_str().ends_with("d.heic"));
    }

    #[tokio::test]
    async fn pagination_works() {
        let dir = make_tmp_dir("page");
        for i in 0..5 {
            fs::write(dir.join(format!("f{i}.jpg")), b"x").unwrap();
        }
        let provider = FileSystemPhotoProvider::new(vec![dir.clone()]);
        let folders = provider.list_folders().await.unwrap();
        let p1 = provider
            .list_photos(&folders[0], PageRequest { offset: 0, limit: 2 })
            .await
            .unwrap();
        let p2 = provider
            .list_photos(&folders[0], PageRequest { offset: 2, limit: 2 })
            .await
            .unwrap();
        let p3 = provider
            .list_photos(&folders[0], PageRequest { offset: 4, limit: 2 })
            .await
            .unwrap();
        assert_eq!(p1.len(), 2);
        assert_eq!(p2.len(), 2);
        assert_eq!(p3.len(), 1);
    }

    #[tokio::test]
    async fn request_permission_returns_granted_on_desktop() {
        let provider = FileSystemPhotoProvider::new(vec![]);
        let status = provider.request_permission().await.unwrap();
        assert_eq!(status, PermissionStatus::Granted);
    }

    #[tokio::test]
    async fn local_path_returns_some_for_known_id() {
        let provider = FileSystemPhotoProvider::new(vec![]);
        let id = PhotoId("/some/path.jpg".into());
        let p = provider.local_path(&id).await.unwrap();
        assert_eq!(p, Some(PathBuf::from("/some/path.jpg")));
    }

    #[test]
    fn delete_photo_is_unsupported() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let provider = FileSystemPhotoProvider::new(vec![]);
        let id = PhotoId("/x.jpg".into());
        let r = rt.block_on(provider.delete_photo(&id));
        assert!(r.is_err());
    }
}