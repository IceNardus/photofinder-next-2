//! IPC 数据传输对象（DTO）。
//!
//! 设计：
//! - 全部字段 snake_case（serde default）
//! - 全部 derive Serialize + Deserialize（前端用 TS 类型对接）
//! - 不暴露内部 Rust 类型（`BBox` / `Embedding` / `Task`），都展开成 f32 / i64 / string
//! - 命名与 ai-next legacy 命令保持一致（前端迁移成本最低）

use serde::{Deserialize, Serialize};

// ============================================================================
// Scan
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct ScanResultDto {
    /// 候选文件数（已过滤扩展名 / 大小）
    pub total_found: usize,
    /// 新插入 image 数
    pub new_images: usize,
    /// 已存在跳过数
    pub skipped: usize,
    /// 被 ImageTypeClassifier 拒掉的数（非 Photo 类型）
    pub filtered: usize,
    /// 被 bytes_per_pixel 过滤的数（vector / icon / 低色图）
    pub filtered_bpp: usize,
    /// 入队的人脸索引任务数
    pub queued_face_tasks: usize,
    /// 入队的对象索引任务数
    pub queued_object_tasks: usize,
    /// 扫描路径
    pub path: String,
}

// ============================================================================
// Search
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct SearchHitDto {
    /// 命中的图片 id
    pub image_id: i64,
    /// 命中的具体目标 id（face_id / object_id）
    pub target_id: i64,
    /// 相似度分数（cosine，0..1，越大越相似）
    pub score: f32,
    /// 排名（1-based）
    pub rank: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResultDto {
    pub image_id: i64,
    pub target_id: i64,
    pub score: f32,
    pub rank: usize,
    /// 缩略图（data URL，前端 `<img>` 直接用）
    pub thumbnail: Option<String>,
    /// 原始图片路径
    pub image_path: Option<String>,
}

// ============================================================================
// Person / Faces
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct PersonDto {
    pub id: i64,
    pub name: Option<String>,
    pub face_count: i64,
    pub body_count: i64,
    pub status: String,
    pub identity_status: String,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FaceDto {
    pub id: i64,
    pub image_id: i64,
    pub bbox: [f32; 4],
    pub quality: f32,
    pub detector_score: f32,
    pub image_path: Option<String>,
    pub thumbnail: Option<String>,
}

// ============================================================================
// Library / Image
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct LibraryImageDto {
    pub id: i64,
    pub path: String,
    pub thumbnail: Option<String>,
    pub scan_status: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    /// 检测到的人脸数
    pub face_count: i64,
    /// 检测到的物品数
    pub object_count: i64,
}

// ============================================================================
// Statistics
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct StatisticsDto {
    pub image_count: i64,
    pub face_count: i64,
    pub object_count: i64,
    pub person_count: i64,
    pub pending_task_count: i64,
    pub completed_task_count: i64,
    pub failed_task_count: i64,
    pub face_index_size: usize,
    pub object_index_size: Option<usize>,
    pub db_path: String,
    pub models_dir: Option<String>,
    // Phase 4 新增（与 ai-next legacy 对齐）
    /// patch 总数（patches 表行数）
    pub patch_count: i64,
    /// 索引目录总大小（bytes；HNSW .data + .graph）
    pub index_size_bytes: u64,
    /// 向量库大小（bytes；face/object/patch 三个 HNSW instance 的 .data 部分）
    pub vector_store_size_bytes: u64,
    /// 数据库大小（bytes；photofinder.db）
    pub database_size_bytes: u64,
    /// 缩略图张数（暂作 stub）
    pub thumbnail_count: u64,
    /// 缩略图总大小（bytes；暂作 stub）
    pub thumbnail_size_bytes: u64,
}

// ============================================================================
// Tasks
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct TaskDto {
    pub id: i64,
    pub kind: String,
    pub status: String,
    pub priority: i32,
    pub retry_count: u32,
    pub max_retries: u32,
    pub error: Option<String>,
    pub created_at: String,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
}

// ============================================================================
// Models
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct ModelStatusDto {
    pub id: String,
    pub loaded: bool,
    pub path: Option<String>,
    pub size_bytes: u64,
    pub verified: bool,
}

// ============================================================================
// App info
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct AppInfoDto {
    pub name: &'static str,
    pub version: &'static str,
    pub data_dir: String,
    pub db_path: String,
    pub models_dir: Option<String>,
    pub platform: &'static str,
}

// ============================================================================
// Clear database result
// ============================================================================

#[derive(Debug, Default, Clone, Serialize)]
pub struct ClearDatabaseResultDto {
    pub success: bool,
    pub cleared_images: bool,
    pub cleared_faces: bool,
    pub cleared_objects: bool,
    pub cleared_persons: bool,
    pub cleared_tasks: bool,
    pub cleared_vectors: bool,
    pub errors: Vec<String>,
}

// ============================================================================
// Commands inputs (deserialized from frontend)
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct SearchByFaceImageArgs {
    pub bytes: Vec<u8>,
    pub top_k: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchByFaceEmbeddingArgs {
    pub embedding: Vec<f32>,
    pub top_k: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ListByClassArgs {
    pub class_id: i32,
    pub limit: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RenamePersonArgs {
    pub person_id: i64,
    pub name: Option<String>,
}

/// Phase 5: 给定 person_id 列出该 person 出现过的所有 image。
///
/// 后端走 prototype 聚合(frontal + left_profile + right_profile +
/// high_quality + general),用每个 prototype 做 HNSW 粗筛,聚合
/// per-image max cosine,阈值过滤后按 score 排序,返回 top_k。
#[derive(Debug, Clone, Deserialize)]
pub struct SearchByPersonArgs {
    pub person_id: i64,
    pub top_k: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ListLibraryArgs {
    pub limit: usize,
    pub offset: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GetThumbnailArgs {
    pub image_id: i64,
    pub max_size: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScanFolderArgs {
    pub path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ListTasksArgs {
    pub status: Option<String>,
    pub limit: usize,
}

// ============================================================================
// Phase 4 新增 DTOs
// ============================================================================

/// 实时扫描进度（前端 1s 轮询）。
#[derive(Debug, Clone, Serialize)]
pub struct ScanStatusDto {
    pub is_scanning: bool,
    pub total_images: usize,
    pub processed_images: usize,
    pub pending_tasks: usize,
    pub current_file: String,
    pub current_faces: usize,
    pub current_objects: usize,
    pub current_patches: usize,
}

/// 实时处理进度（更细的字段，与 ai-next 对齐）。
#[derive(Debug, Clone, Serialize)]
pub struct ProcessingStatusDto {
    pub current_image: String,
    pub current_faces: usize,
    pub current_patches: usize,
    pub current_objects: usize,
    pub log_message: String,
    pub last_completion_message: String,
}

/// 对象搜索命中（带融合分数）。
#[derive(Debug, Clone, Serialize)]
pub struct ObjectSearchHitDto {
    pub image_id: i64,
    pub image_path: String,
    pub image_name: String,
    pub thumbnail_path: Option<String>,
    pub confidence: f32,
    pub similarity: f32,
    pub embedding_score: f32,
    pub inlier_ratio: f32,
    pub inlier_count: usize,
    pub match_count: usize,
    pub bbox_overlap: f32,
}

/// 重建缩略图进度。
#[derive(Debug, Clone, Serialize)]
pub struct RebuildProgress {
    pub total: usize,
    pub generated: usize,
    pub failed: usize,
    pub speed: f32,
}

/// 写查询/裁剪图片 args。
#[derive(Debug, Clone, Deserialize)]
pub struct WriteImageArgs {
    pub data: Vec<u8>,
    pub mime: String,
    /// 裁剪 bbox（仅 cropping 用）
    pub crop: Option<CropArgs>,
}

/// 裁剪参数。
#[derive(Debug, Clone, Deserialize)]
pub struct CropArgs {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// Objects search args.
#[derive(Debug, Clone, Deserialize)]
pub struct SearchObjectsArgs {
    pub query_image: String,
    pub top_k: usize,
}

/// Copy files args.
///
/// 两种用法（互斥）：
/// - `source_paths` 非空 → 直接复制
/// - `image_ids` 非空 → 在 DB 查 path 再复制（前端导出选中的搜索结果时用）
#[derive(Debug, Clone, Deserialize)]
pub struct CopyFilesArgs {
    #[serde(default)]
    pub source_paths: Vec<String>,
    #[serde(default)]
    pub image_ids: Vec<i64>,
    pub dest_dir: String,
}

/// 单 file 拷贝结果。
#[derive(Debug, Clone, Serialize)]
pub struct CopyFileResult {
    pub src: String,
    pub dst: String,
    pub success: bool,
    pub error: Option<String>,
}

/// Get image thumbnail args (legacy path-based).
#[derive(Debug, Clone, Deserialize)]
pub struct GetImageThumbnailArgs {
    pub image_path: String,
    pub max_size: u32,
}


#[cfg(test)]
mod tests {
    use super::*;

    /// DTO 序列化必须包含 `filtered` + `filtered_bpp`（前端要靠这俩字段诊断
    /// 「扫了 0 张脸」是不是 image_classifier 把图库拒光了）。
    #[test]
    fn scan_result_dto_exposes_filtered_counts() {
        let dto = ScanResultDto {
            total_found: 100,
            new_images: 10,
            skipped: 5,
            filtered: 80,        // 80 张被 ImageTypeClassifier 拒
            filtered_bpp: 5,     // 5 张被 bpp 过滤
            queued_face_tasks: 10,
            queued_object_tasks: 10,
            path: "/tmp/x".into(),
        };
        let json = serde_json::to_value(&dto).expect("serialize");
        assert_eq!(json["total_found"], 100);
        assert_eq!(json["new_images"], 10);
        assert_eq!(json["skipped"], 5);
        assert_eq!(json["filtered"], 80, "filtered field must be present");
        assert_eq!(json["filtered_bpp"], 5, "filtered_bpp field must be present");
        assert_eq!(json["queued_face_tasks"], 10);
        assert_eq!(json["queued_object_tasks"], 10);
        assert_eq!(json["path"], "/tmp/x");
    }

    /// 即使 filtered=0 也要存在（不能因为 0 而被 serde 跳过 — usize 不会被 skip，
    /// 但保留这个 test 防止有人误加 `skip_serializing_if`）。
    #[test]
    fn scan_result_dto_includes_zero_filtered() {
        let dto = ScanResultDto {
            total_found: 0,
            new_images: 0,
            skipped: 0,
            filtered: 0,
            filtered_bpp: 0,
            queued_face_tasks: 0,
            queued_object_tasks: 0,
            path: "".into(),
        };
        let json = serde_json::to_value(&dto).expect("serialize");
        assert!(json.get("filtered").is_some());
        assert!(json.get("filtered_bpp").is_some());
    }
}
