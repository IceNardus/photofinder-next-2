//! ScanService — 扫描文件夹 → 入库 → enqueue 索引任务。
//!
//! 设计（对齐 ai-next `core/scanner.rs`）：
//! - 分页列出照片 + walkdir 枚举
//! - 文件签名查重（path + size），对齐 ai-next（无 modified_time 故用 size）
//! - 扫描时计算 BLAKE3 hash（对齐 ai-next）
//! - ImageTypeClassifier 过滤（非 Photo 则跳过）
//! - bytes_per_pixel 过滤（icon/vector 等）
//! - 批量写入 DB（500条/批）
//! - 每个新 image enqueue `IndexFace` + `IndexObject` + `IndexPatch` 任务
//! - 返回 ScanSummary（总数 / 新增 / 跳过）
//! - 通过 `on_image` callback 实时通知每张已入库的图片

use std::path::Path;
use std::sync::Arc;

use pf_ai::ImageTypeClassifier;
use pf_database::{Database, NewImage, ScanStatus};
use pf_platform::{PageRequest, PhotoFolder, PhotoProvider};
use pf_task::{Priority, TaskKind, TaskScheduler};
use tracing::{debug, info, warn};

use crate::error::ApplicationError;

/// 对齐 ai-next BATCH_SIZE = 500。
const BATCH_SIZE: usize = 500;

/// 每张图片入库后的回调签名 (image_id, path)。
/// 扫描过程中每入库一张图片就调用一次，用于实时通知。
pub type OnImageCallback = Arc<dyn Fn(i64, &str) + Send + Sync>;

/// 扫描结果摘要。
#[derive(Debug, Default, Clone)]
pub struct ScanSummary {
    /// 扫描路径（字符串）
    pub path: String,
    /// 候选文件数（已过滤扩展名）
    pub candidate_count: usize,
    /// 新插入 image 数
    pub inserted: usize,
    /// 已存在跳过数
    pub skipped: usize,
    /// 被 ImageTypeClassifier 过滤的图数
    pub filtered: usize,
    /// bytes_per_pixel 过滤的图数
    pub filtered_bpp: usize,
    /// 入队的 IndexFace 任务数
    pub queued_face_tasks: usize,
    /// 入队的 IndexObject 任务数
    pub queued_object_tasks: usize,
    /// 已发现的图片 (image_id, path)
    pub discovered_images: Vec<(i64, String)>,
}

/// ScanService。
pub struct ScanService {
    db: Arc<Database>,
    scheduler: Arc<dyn TaskScheduler>,
    photo_provider: Arc<dyn PhotoProvider>,
    image_classifier: ImageTypeClassifier,
    /// 是否入队 IndexPatch（依赖 patch_extractor）。patch_extractor 未配置时为 false。
    enable_patches: bool,
}

impl ScanService {
    /// 构造。
    pub fn new(
        db: Arc<Database>,
        scheduler: Arc<dyn TaskScheduler>,
        photo_provider: Arc<dyn PhotoProvider>,
        enable_patches: bool,
    ) -> Self {
        Self {
            db,
            scheduler,
            photo_provider,
            image_classifier: ImageTypeClassifier::new(),
            enable_patches,
        }
    }

    /// 扫描一个文件夹（异步）。
    pub async fn scan_folder(&self, folder_path: &Path) -> Result<ScanSummary, ApplicationError> {
        self.scan_folder_with_callback(folder_path, None).await
    }

    /// 扫描一个文件夹，带实时图片回调（异步）。
    /// `on_image` 每入库一张图片调用一次 (image_id, path)。
    pub async fn scan_folder_with_callback(
        &self,
        folder_path: &Path,
        on_image: Option<OnImageCallback>,
    ) -> Result<ScanSummary, ApplicationError> {
        let folder = PhotoFolder {
            id: folder_path.to_string_lossy().into_owned(),
            display_name: folder_path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| folder_path.to_string_lossy().into_owned()),
            count_estimate: None,
        };
        self.scan_folder_entry(&folder, on_image).await
    }

    /// 扫描 `PhotoFolder`（异步 — 走 `PhotoProvider`）。
    /// `on_image` 每入库一张图片调用一次 (image_id, path)，用于实时通知。
    pub async fn scan_folder_entry(
        &self,
        folder: &PhotoFolder,
        on_image: Option<OnImageCallback>,
    ) -> Result<ScanSummary, ApplicationError> {
        let mut summary = ScanSummary {
            path: folder.id.clone(),
            ..Default::default()
        };

        // 1) 分页列出所有照片
        let mut page = PageRequest::first(500);
        let mut all = Vec::new();
        loop {
            let entries = self
                .photo_provider
                .list_photos(folder, page)
                .await
                .map_err(ApplicationError::Platform)?;
            let count = entries.len();
            all.extend(entries);
            if count < page.limit {
                break;
            }
            page = page.next();
        }
        summary.candidate_count = all.len();
        info!(folder = %folder.id, candidate = all.len(), "scan folder complete");

        // 2) 批量处理（对齐 ai-next：500条/批）
        let mut batch_paths: Vec<String> = Vec::with_capacity(BATCH_SIZE);
        let mut batch_hashes: Vec<String> = Vec::with_capacity(BATCH_SIZE);
        let mut batch_sizes: Vec<u64> = Vec::with_capacity(BATCH_SIZE);
        let mut batch_mtimes: Vec<i64> = Vec::with_capacity(BATCH_SIZE);

        for entry in all {
            let Some(local_path) = self
                .photo_provider
                .local_path(&entry.id)
                .await
                .map_err(ApplicationError::Platform)?
            else {
                debug!(photo = %entry.id, "no local path, skip");
                summary.skipped += 1;
                continue;
            };

            let path_str = local_path.to_string_lossy().into_owned();

            // 2a) 文件签名查重（path + size），对齐 ai-next（无 modified_time 故用 size）
            let metadata = match std::fs::metadata(&local_path) {
                Ok(m) => m,
                Err(e) => {
                    warn!(path = %path_str, error = %e, "metadata failed, skip");
                    summary.skipped += 1;
                    continue;
                }
            };
            let size = metadata.len();
            let mtime = metadata
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);

            let (is_new, hash, skipped) =
                self.check_file_signature(&path_str, size);
            if skipped {
                summary.skipped += 1;
                continue;
            }
            if !is_new {
                // 文件已在 DB（path+size 匹配，或 hash 已存在）
                summary.skipped += 1;
                continue;
            }

            // 2b) bytes_per_pixel 过滤（对齐 ai-next should_skip_for_face_pipeline）
            if should_skip_for_face_pipeline(&local_path, metadata.len()) {
                summary.filtered_bpp += 1;
                continue;
            }

            // 2c) ImageTypeClassifier 过滤（非 Photo 则跳过）
            let (acceptable, image_type, type_reason) = match image::open(&local_path) {
                Ok(img) => {
                    let (t, reason) = self.image_classifier.classify_image(&img);
                    (t == pf_ai::ImageType::Photo, t, reason)
                }
                Err(e) => {
                    warn!(path = %path_str, error = %e, "decode failed, skipping");
                    (false, pf_ai::ImageType::Unknown, format!("decode: {e}"))
                }
            };
            if !acceptable {
                debug!(
                    path = %path_str,
                    image_type = %image_type,
                    reason = %type_reason,
                    "filtered by ImageTypeClassifier"
                );
                summary.filtered += 1;
                continue;
            }

            // 3) 收集到批次
            batch_paths.push(path_str);
            batch_hashes.push(hash);
            batch_sizes.push(size);
            batch_mtimes.push(mtime);

            // 达到批量大小时 flush
            if batch_paths.len() >= BATCH_SIZE {
                let inserted = self.flush_batch(&mut batch_paths, &mut batch_hashes, &mut batch_sizes, &mut batch_mtimes, &mut summary)
                    .await?;
                // 实时通知每张入库的图片
                if let Some(ref cb) = on_image {
                    for (id, path) in &inserted {
                        cb(*id, path);
                    }
                }
                summary.discovered_images.extend(inserted);
                tokio::task::yield_now().await;
            }
        }

        // 4) 处理剩余批次
        info!(remaining = batch_paths.len(), "processing remaining batch");
        if !batch_paths.is_empty() {
            let inserted = self.flush_batch(&mut batch_paths, &mut batch_hashes, &mut batch_sizes, &mut batch_mtimes, &mut summary)
                .await?;
            // 实时通知每张入库的图片
            if let Some(ref cb) = on_image {
                for (id, path) in &inserted {
                    cb(*id, path);
                }
            }
            summary.discovered_images.extend(inserted);
        }

        // 5) 入队文件夹 Scan 任务（进度追踪用）
        let folder_id: i64 = folder
            .id
            .parse()
            .unwrap_or_else(|_| chrono::Utc::now().timestamp());
        let _ = self
            .scheduler
            .enqueue(TaskKind::Scan { folder_id }, Priority::Low)
            .await;

        Ok(summary)
    }

    /// 对齐 ai-next check_file_signature：
    /// - 返回 (true, hash, false) = 新文件，需计算 BLAKE3 hash
    /// - 返回 (false, _, true) = 已存在/重复，跳过
    fn check_file_signature(
        &self,
        path: &str,
        size: u64,
    ) -> (bool, String, bool) {
        self.db
            .transaction(|tx| {
                let repo = tx.images();
                // 先按 path 查找
                if let Some(existing) = repo.get_by_path(path).unwrap_or(None) {
                    if existing.size == size {
                        // path 和 size 都匹配 → 文件未变，跳过
                        return Ok((false, existing.hash.clone(), true));
                    }
                    // size 变了，计算 hash 查重
                    let hash = compute_blake3(path);
                    if let Ok(Some(_)) = repo.get_by_hash(&hash) {
                        // hash 已存在 → 同一文件（内容相同，大小不同，如缩略图）
                        return Ok((false, hash, true));
                    }
                    // hash 不存在 → 新文件（内容变了）
                    return Ok((true, hash, false));
                }
                // path 不存在，hash 查重
                let hash = compute_blake3(path);
                if let Ok(Some(_)) = repo.get_by_hash(&hash) {
                    // hash 已存在 → 同一文件不同路径
                    return Ok((false, hash, true));
                }
                // 全新文件
                Ok((true, hash, false))
            })
            .unwrap_or((true, String::new(), false))
    }

    /// 批量 flush 到 DB（对齐 ai-next flush_batch）。
    /// 返回插入的图片信息 (image_id, path)。
    async fn flush_batch(
        &self,
        paths: &mut Vec<String>,
        hashes: &mut Vec<String>,
        sizes: &mut Vec<u64>,
        mtimes: &mut Vec<i64>,
        summary: &mut ScanSummary,
    ) -> Result<Vec<(i64, String)>, ApplicationError> {
        if paths.is_empty() {
            return Ok(Vec::new());
        }

        info!(count = paths.len(), "flush_batch called");

        // 保留 paths 副本用于返回（因为后面会 clear）
        let paths_out = paths.clone();

        let image_ids: Vec<i64> = self
            .db
            .transaction(|tx| {
                let mut ids = Vec::with_capacity(paths.len());
                for i in 0..paths.len() {
                    let new = NewImage {
                        path: paths[i].clone(),
                        hash: hashes[i].clone(),
                        size: sizes[i],
                        modified_time: mtimes[i],
                        width: 0,
                        height: 0,
                        captured_at: None,
                    };
                    let mut repo = tx.images();
                    match repo.insert(&new) {
                        Ok(id) => {
                            ids.push(id);
                            drop(repo); // release mutable borrow before add_to_duplicate_group
                            if !hashes[i].is_empty() {
                                let _ = pf_database::repositories::image::add_to_duplicate_group(tx, &hashes[i]);
                            }
                        }
                        Err(e) => {
                            warn!(path = %paths[i], error = %e, "insert failed");
                        }
                    }
                }
                Ok(ids)
            })
            .map_err(|e| ApplicationError::Internal(format!("flush batch: {e}")))?;

        // 组装返回 (id, path) 对
        let inserted: Vec<(i64, String)> = image_ids
            .iter()
            .zip(paths_out.iter())
            .map(|(id, path)| (*id, path.clone()))
            .collect();

        // 入队索引任务（每个 image 2 个 task：IndexImage 合并 face+object，避免并发写竞争）
        for image_id in &image_ids {
            match self
                .scheduler
                .enqueue(TaskKind::IndexImage { image_id: *image_id }, Priority::Normal)
                .await
            {
                Ok(id) => {
                    info!(image_id, task_id = id, "IndexImage task enqueued (face+object batched)");
                    summary.queued_face_tasks += 1;
                    summary.queued_object_tasks += 1;
                }
                Err(e) => warn!(image_id, error = %e, "enqueue IndexImage failed"),
            }
            if self.enable_patches {
                match self
                    .scheduler
                    .enqueue(TaskKind::IndexPatch { image_id: *image_id }, Priority::Normal)
                    .await
                {
                    Ok(_) => {}
                    Err(e) => warn!(image_id, error = %e, "enqueue IndexPatch failed"),
                }
            }
            summary.inserted += 1;
        }

        paths.clear();
        hashes.clear();
        sizes.clear();

        Ok(inserted)
    }

    /// 把待扫描 image 列表拿出来（不实际扫描，只查 DB）。
    /// 用于 bootstrap 时重建索引。
    pub async fn list_pending(&self, limit: usize) -> Result<Vec<i64>, ApplicationError> {
        let rows = self
            .db
            .transaction(|tx| tx.images().list_pending(limit))
            .map_err(|e| ApplicationError::Internal(format!("list_pending: {e}")))?;
        Ok(rows.into_iter().map(|r| r.id).collect())
    }

    /// 标记 image scan 完成（不动索引状态）。
    pub async fn mark_indexed(&self, image_id: i64) -> Result<(), ApplicationError> {
        self.db
            .transaction(|tx| tx.images().update_scan_status(image_id, ScanStatus::Indexed))
            .map_err(|e| ApplicationError::Internal(format!("update_scan_status: {e}")))?;
        Ok(())
    }
}

/// 计算文件的 BLAKE3 hash（对齐 ai-next）。
fn compute_blake3(path: &str) -> String {
    use std::fs;
    match fs::read(path) {
        Ok(content) => {
            let hash = blake3::hash(&content);
            hash.to_hex().to_string()
        }
        Err(_) => String::new(),
    }
}

/// 对齐 ai-next should_skip_for_face_pipeline：
/// bytes_per_pixel < 0.15 && unique_colors < 100 → skip（low-color image）
///
/// 历史上有 `bpp < 0.05 → skip（vector/icon）` 的硬规则，已移除：
/// 现代高分辨率 JPEG（pexels / 手机出图）压缩后 bpp 经常落在 0.03-0.05，
/// 会误杀真实照片。bpp<0.15 + colors<100 这一档对矢量 / icon 仍有效。
fn should_skip_for_face_pipeline(path: &Path, file_size: u64) -> bool {
    if let Ok(img) = image::open(path) {
        let w = img.width();
        let h = img.height();
        let pixels = (w * h) as f32;
        if pixels < 1.0 {
            return false;
        }
        let bpp = file_size as f32 / pixels;

        if bpp < 0.15 {
            if let Some(colors) = count_unique_colors(&img) {
                if colors < 100 {
                    info!(path = %path.display(), colors = colors, "skipping low-color image");
                    return true;
                }
            }
        }
    }
    false
}

/// 对齐 ai-next count_unique_colors：缩放到 50x50 采样。
fn count_unique_colors(img: &image::DynamicImage) -> Option<usize> {
    let img = img.resize(50, 50, image::imageops::FilterType::Nearest);
    let rgba = img.to_rgba8();
    let mut unique: std::collections::HashSet<u32> = std::collections::HashSet::new();

    for pixel in rgba.pixels() {
        let rgb = ((pixel[0] as u32) << 16) | ((pixel[1] as u32) << 8) | (pixel[2] as u32);
        unique.insert(rgb);
        if unique.len() > 10_000 {
            return Some(unique.len());
        }
    }
    Some(unique.len())
}
