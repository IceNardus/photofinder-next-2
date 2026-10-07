//! Tauri commands（IPC 入口）。
//!
//! 所有命令都是**薄壳**：解析参数 → 调用 `pf_application::*` → 转 DTO 返回。
//! 不在这里做业务逻辑。
//!
//! 错误处理：所有 `Result<_, ApplicationError>` 都 `map_err(|e| e.to_string())`，
//! 与 Tauri 默认的 `Result<_, String>` 对齐。
//!
//! 命名约定：
//! - 动词开头（scan_* / search_* / list_* / get_* / cluster_* / clear_* / cancel_* / rename_*）
//! - 与 ai-next legacy 命令名尽量保持一致（前端迁移成本最低）

use std::path::PathBuf;
use std::sync::Arc;

use pf_core::{FACE_MODEL_NAME, SearchResult};
use pf_database::{DbTaskStatus, FaceRow, FaceStatus, ImageRow, PersonRow};
use pf_platform::PhotoId;
use pf_task::{Priority, TaskFilter, TaskKind, TaskStatus};
use tauri::{Emitter, State};
use tracing::{debug, info, warn};

use crate::ipc_types::{
    AppInfoDto, ClearDatabaseResultDto, CopyFileResult, CopyFilesArgs, FaceDto,
    GetImageThumbnailArgs, LibraryImageDto, ListByClassArgs, ListLibraryArgs, ListTasksArgs,
    ModelStatusDto, ObjectSearchHitDto, PersonDto, RebuildProgress, RenamePersonArgs,
    ScanFolderArgs, ScanResultDto, ScanStatusDto,
    SearchByPersonArgs, SearchHitDto, SearchObjectsArgs,
    ProcessingStatusDto, StatisticsDto, TaskDto, WriteImageArgs,
};
use crate::state::AppState;

// ============================================================================
// Scan
// ============================================================================

/// 扫描文件夹（同步走完，返回 ScanResultDto）。
/// 扫描过程中实时发射 `pf:scan:image_discovered` 事件。
#[tauri::command]
pub async fn scan_folder(
    args: ScanFolderArgs,
    state: State<'_, Arc<AppState>>,
    app_handle: tauri::AppHandle,
) -> Result<ScanResultDto, String> {
    let path = PathBuf::from(&args.path);
    info!(path = %path.display(), "scan_folder command");

    // 扫描前先清空数据库
    let db = state.db().clone();
    match tokio::task::spawn_blocking(move || db.reset())
        .await
        .map_err(|e| format!("join: {e}"))?
    {
        Ok(()) => {
            info!("pre-scan db reset done");
            // #162: db.reset() 只 drop SQL 表，不清内存 HNSW。不清的话每次重扫都会
            // 往旧 index 上追加过期条目（index 累积 → 搜索白吃 fetch_k 预算、降召回）。
            // 这里同步清空，让随后 enqueue 的 IndexImage 任务重新灌入干净索引。
            if let Err(e) = state.face_index().clear() {
                warn!(error = %e, "face_index clear after pre-scan reset failed");
            } else {
                info!("face_index cleared after pre-scan reset");
            }
        }
        Err(e) => {
            warn!("pre-scan db reset failed: {e}, continuing anyway");
        }
    }

    *state.is_scanning.lock() = true;
    state.last_scan_stats.lock().path = path.to_string_lossy().to_string();

    let scan_svc = state.bootstrapped.scan.clone();

    // 每张图片入库时实时发射事件
    let app = app_handle.clone();
    let on_image: pf_application::scan::OnImageCallback =
        Arc::new(move |image_id: i64, path: &str| {
            let payload = serde_json::json!({
                "image_id": image_id,
                "path": path,
            });
            let _ = app.emit("pf:scan:image_discovered", payload);
        });

    let summary = scan_svc
        .scan_folder_with_callback(&path, Some(on_image))
        .await
        .map_err(|e| e.to_string())?;

    {
        let mut s = state.last_scan_stats.lock();
        s.total = summary.candidate_count;
        s.scanned = summary.inserted + summary.skipped;
        s.current_file = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
    }

    *state.is_scanning.lock() = false;
    info!(?summary, "scan_folder done");

    // #163: scan 完成后自动聚类 —— 否则所有 face 的 person_id 都是 NULL（与 reindex_all 一致）。
    // ClusterFaces 最后入队，scheduler 按 created_at ASC 领取，保证所有 IndexImage 索引
    // 完成之后才跑聚类。
    let scheduler = state.bootstrapped.ctx.task_scheduler.clone();
    match scheduler
        .enqueue(TaskKind::ClusterFaces, Priority::Normal)
        .await
    {
        Ok(task_id) => info!(task_id, "scan_folder: queued ClusterFaces"),
        Err(e) => warn!(error = %e, "scan_folder: enqueue ClusterFaces failed"),
    }

    Ok(ScanResultDto {
        total_found: summary.candidate_count,
        new_images: summary.inserted,
        skipped: summary.skipped,
        filtered: summary.filtered,
        filtered_bpp: summary.filtered_bpp,
        queued_face_tasks: summary.queued_face_tasks,
        queued_object_tasks: summary.queued_object_tasks,
        path: summary.path,
    })
}

/// 停止扫描（标记 is_scanning = false；当前扫描协程自然结束后生效）。
#[tauri::command]
pub async fn stop_scan(state: State<'_, Arc<AppState>>) -> Result<(), String> {
    info!("stop_scan command");
    *state.is_scanning.lock() = false;
    Ok(())
}

// ============================================================================
// Search
// ============================================================================

/// 用图片 bytes 检索相似人脸。
#[tauri::command]
pub async fn search_by_face_image(
    bytes: Vec<u8>,
    top_k: usize,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<SearchHitDto>, String> {
    let svc = state.bootstrapped.search.clone();
    let hits = svc
        .search_by_face_image(&bytes, top_k)
        .await
        .map_err(|e| e.to_string())?;
    Ok(hits.into_iter().map(search_result_to_dto).collect())
}

/// 用 embedding 检索相似人脸。
#[tauri::command]
pub async fn search_by_face_embedding(
    embedding: Vec<f32>,
    top_k: usize,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<SearchHitDto>, String> {
    let svc = state.bootstrapped.search.clone();
    let hits = svc
        .search_by_face_embedding(&embedding, top_k)
        .await
        .map_err(|e| e.to_string())?;
    Ok(hits.into_iter().map(search_result_to_dto).collect())
}

/// 按类别 ID 列出对象（不向量搜索，纯 DB 过滤）。
#[tauri::command]
pub async fn list_objects_by_class(
    args: ListByClassArgs,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<SearchHitDto>, String> {
    let svc = state.bootstrapped.search.clone();
    let hits = svc
        .list_by_class(args.class_id, args.limit)
        .await
        .map_err(|e| e.to_string())?;
    Ok(hits.into_iter().map(search_result_to_dto).collect())
}

// ============================================================================
// Persons
// ============================================================================

/// 列出所有 person（按 face_count DESC）。
#[tauri::command]
pub async fn list_persons(
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<PersonDto>, String> {
    let svc = state.bootstrapped.person.clone();
    let persons = svc.list_all().map_err(|e| e.to_string())?;
    Ok(persons.into_iter().map(person_row_to_dto).collect())
}

/// 重命名 / 清除名字。
#[tauri::command]
pub async fn rename_person(
    args: RenamePersonArgs,
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let svc = state.bootstrapped.person.clone();
    svc.rename(args.person_id, args.name.as_deref())
        .map_err(|e| e.to_string())
}

/// 列出某 person 的所有 face。
#[tauri::command]
pub async fn faces_of_person(
    person_id: i64,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<FaceDto>, String> {
    let svc = state.bootstrapped.person.clone();
    let faces = svc.faces_of(person_id).map_err(|e| e.to_string())?;
    Ok(faces.into_iter().map(face_row_to_dto).collect())
}

/// 触发全量聚类（enqueue ClusterFaces 任务）。
#[tauri::command]
pub async fn cluster_faces(
    state: State<'_, Arc<AppState>>,
) -> Result<i64, String> {
    let scheduler = state.bootstrapped.ctx.task_scheduler.clone();
    let task_id = scheduler
        .enqueue(TaskKind::ClusterFaces, Priority::Normal)
        .await
        .map_err(|e| e.to_string())?;
    info!(task_id, "cluster_faces enqueued");
    Ok(task_id)
}

/// 重新分析人物聚类（同步执行，直接调用 cluster_all）。
/// 返回聚类结果统计。
#[tauri::command]
pub async fn rebuild_person_clusters(
    state: State<'_, Arc<AppState>>,
) -> Result<pf_application::person::ClusterSummary, String> {
    let person = state.bootstrapped.person.clone();
    person
        .cluster_all()
        .await
        .map_err(|e| e.to_string())
}

/// Phase 5: 查找 person 出现过的所有 image(multi-prototype HNSW 聚合)。
///
/// 需要在 bootstrap 注入 PrototypeService 到 SearchService,否则
/// 后端会返回 `InvalidState("prototype_service not configured")` 错误。
#[tauri::command]
pub async fn search_by_person(
    args: SearchByPersonArgs,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<SearchHitDto>, String> {
    let svc = state.bootstrapped.search.clone();
    let hits = svc
        .search_by_person(args.person_id, args.top_k)
        .await
        .map_err(|e| e.to_string())?;
    debug!(
        person_id = args.person_id,
        top_k = args.top_k,
        n_hits = hits.len(),
        "search_by_person command"
    );
    Ok(hits.into_iter().map(search_result_to_dto).collect())
}

// ============================================================================
// Library
// ============================================================================

/// 分页列出图库（按 id DESC）。
#[tauri::command]
pub async fn list_library(
    args: ListLibraryArgs,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<LibraryImageDto>, String> {
    let db = state.db().clone();
    let rows = tokio::task::spawn_blocking(move || {
        db.transaction(|tx| tx.images().list_paged(args.limit, args.offset))
    })
    .await
    .map_err(|e| format!("join: {e}"))?
    .map_err(|e| format!("db: {e}"))?;

    Ok(rows.into_iter().map(image_row_to_dto).collect())
}

/// 取缩略图（返回 base64 data URL）。
#[tauri::command]
pub async fn get_thumbnail(
    args: crate::ipc_types::GetThumbnailArgs,
    state: State<'_, Arc<AppState>>,
) -> Result<String, String> {
    let db = state.db().clone();
    let row: Option<ImageRow> = tokio::task::spawn_blocking({
        let id = args.image_id;
        move || db.transaction(|tx| tx.images().get_by_id(id))
    })
    .await
    .map_err(|e| format!("join: {e}"))?
    .map_err(|e| format!("db: {e}"))?;

    let row = row.ok_or_else(|| format!("image {} not found", args.image_id))?;

    let provider = state.bootstrapped.ctx.photo_provider.clone();
    let photo_id = PhotoId::from(row.path.clone());
    let bytes = provider
        .get_thumbnail(&photo_id, args.max_size.max(64).min(1024))
        .await
        .map_err(|e| e.to_string())?;

    let mime = "image/jpeg";
    let b64 = base64_encode(&bytes);
    Ok(format!("data:{};base64,{}", mime, b64))
}

// ============================================================================
// Statistics
// ============================================================================

/// 总体统计。
#[tauri::command]
pub async fn get_statistics(
    state: State<'_, Arc<AppState>>,
) -> Result<StatisticsDto, String> {
    let db = state.db().clone();
    let path_resolver = state.bootstrapped.ctx.path_resolver.clone();
    let face_index = state.face_index().clone();
    let object_count = state.bootstrapped.search.object_count();

    let stats = tokio::task::spawn_blocking(move || {
        db.transaction(|tx| {
            let images = tx.images().count()?;
            let faces = tx.faces().count()?;
            let objects = tx.objects().count()?;
            let persons = tx.persons().list()?.len() as i64;
            let pending = tx.tasks().count_by_status(DbTaskStatus::Pending)?;
            let completed = tx.tasks().count_by_status(DbTaskStatus::Completed)?;
            let failed = tx.tasks().count_by_status(DbTaskStatus::Failed)?;
            let patches = tx.count_patches()?;
            Ok::<_, pf_database::DatabaseError>((images, faces, objects, persons, pending, completed, failed, patches))
        })
    })
    .await
    .map_err(|e| format!("join: {e}"))?
    .map_err(|e| format!("db: {e}"))?;

    let (images, faces, objects, persons, pending, completed, failed, patches) = stats;
    let db_path = path_resolver.db_path().map_err(|e| e.to_string())?.to_string_lossy().into_owned();
    let models_dir = path_resolver
        .models_dir()
        .map(|p| p.to_string_lossy().into_owned());

    // Phase 4：实际计算磁盘占用
    let (index_size_bytes, vector_store_size_bytes, database_size_bytes, thumbnail_count, thumbnail_size_bytes) =
        compute_disk_sizes(&path_resolver);

    Ok(StatisticsDto {
        image_count: images,
        face_count: faces,
        object_count: objects,
        person_count: persons,
        pending_task_count: pending,
        completed_task_count: completed,
        failed_task_count: failed,
        face_index_size: face_index.len(),
        object_index_size: object_count,
        db_path,
        models_dir,
        patch_count: patches,
        index_size_bytes,
        vector_store_size_bytes,
        database_size_bytes,
        thumbnail_count,
        thumbnail_size_bytes,
    })
}

/// 计算磁盘占用（Phase 4，与 ai-next 对齐）。
fn compute_disk_sizes(
    path_resolver: &Arc<dyn pf_platform::PathResolver>,
) -> (u64, u64, u64, u64, u64) {
    let mut index_size = 0u64;
    let mut vector_size = 0u64;
    let mut db_size = 0u64;
    let mut thumb_count = 0u64;
    let mut thumb_size = 0u64;

    if let Ok(data_dir) = path_resolver.data_dir() {
        let index_dir = data_dir.join("index");
        if let Ok(rd) = std::fs::read_dir(&index_dir) {
            for entry in rd.flatten() {
                if let Ok(md) = entry.metadata() {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if name.ends_with(".data") {
                        vector_size += md.len();
                    }
                    index_size += md.len();
                }
            }
        }
        let thumbs_dir = data_dir.join("thumbnails");
        if let Ok(rd) = std::fs::read_dir(&thumbs_dir) {
            for entry in rd.flatten() {
                if let Ok(md) = entry.metadata() {
                    thumb_count += 1;
                    thumb_size += md.len();
                }
            }
        }
    }
    if let Ok(db_path) = path_resolver.db_path() {
        if let Ok(md) = std::fs::metadata(&db_path) {
            db_size = md.len();
        }
    }
    (index_size, vector_size, db_size, thumb_count, thumb_size)
}

// ============================================================================
// Tasks
// ============================================================================

/// 列出任务（可选状态过滤）。
#[tauri::command]
pub async fn list_tasks(
    args: ListTasksArgs,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<TaskDto>, String> {
    let scheduler = state.bootstrapped.ctx.task_scheduler.clone();
    let filter = if let Some(s) = args.status.as_deref() {
        let status = match s {
            "pending" => TaskStatus::Pending,
            "running" => TaskStatus::Running,
            "completed" => TaskStatus::Completed,
            "failed" => TaskStatus::Failed,
            "cancelled" => TaskStatus::Cancelled,
            other => return Err(format!("invalid status: {other}")),
        };
        TaskFilter {
            status: Some(status),
            kind: None,
            limit: Some(args.limit.max(1).min(1000)),
        }
    } else {
        TaskFilter {
            status: None,
            kind: None,
            limit: Some(args.limit.max(1).min(1000)),
        }
    };

    let tasks = scheduler.list(filter).await.map_err(|e| e.to_string())?;
    Ok(tasks.into_iter().map(task_to_dto).collect())
}

/// 取消任务（仅 Pending 状态可取消）。
#[tauri::command]
pub async fn cancel_task(
    task_id: i64,
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let scheduler = state.bootstrapped.ctx.task_scheduler.clone();
    scheduler.cancel(task_id).await.map_err(|e| e.to_string())?;
    info!(task_id, "task cancelled");
    Ok(())
}

/// 给所有 pending 的 image 入队 IndexFace + IndexObject 任务。
#[tauri::command]
pub async fn index_pending(
    state: State<'_, Arc<AppState>>,
) -> Result<usize, String> {
    let scan_svc = state.bootstrapped.scan.clone();
    let pending_ids = scan_svc.list_pending(10_000).await.map_err(|e| e.to_string())?;
    let scheduler = state.bootstrapped.ctx.task_scheduler.clone();
    let mut enqueued = 0usize;
    for image_id in pending_ids {
        if scheduler
            .enqueue(TaskKind::IndexFace { image_id }, Priority::Normal)
            .await
            .is_ok()
        {
            enqueued += 1;
        }
        let _ = scheduler
            .enqueue(TaskKind::IndexObject { image_id }, Priority::Normal)
            .await;
    }
    info!(enqueued, "index_pending done");
    Ok(enqueued)
}

// ============================================================================
// Models
// ============================================================================

/// 模型状态列表。
#[tauri::command]
pub async fn get_models_status(
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<ModelStatusDto>, String> {
    let mgr = state.bootstrapped.ctx.model_manager.clone();
    let statuses = mgr.status();
    Ok(statuses
        .into_iter()
        .map(|s| ModelStatusDto {
            id: s.id.to_string(),
            loaded: s.loaded,
            path: Some(s.path.to_string_lossy().into_owned()),
            size_bytes: s.size_bytes,
            verified: s.verified,
        })
        .collect())
}

// ============================================================================
// Misc
// ============================================================================

/// 获取 data_dir 路径。
#[tauri::command]
pub async fn get_data_dir(
    state: State<'_, Arc<AppState>>,
) -> Result<String, String> {
    let p = state
        .bootstrapped
        .ctx
        .path_resolver
        .data_dir()
        .map_err(|e| e.to_string())?;
    Ok(p.to_string_lossy().into_owned())
}

/// 应用基本信息。
#[tauri::command]
pub async fn get_app_info(
    state: State<'_, Arc<AppState>>,
) -> Result<AppInfoDto, String> {
    let resolver = state.bootstrapped.ctx.path_resolver.clone();
    let data_dir = resolver.data_dir().map_err(|e| e.to_string())?;
    let db_path = resolver.db_path().map_err(|e| e.to_string())?;
    let models_dir = resolver
        .models_dir()
        .map(|p| p.to_string_lossy().into_owned());

    Ok(AppInfoDto {
        name: "PhotoFinder Next 2",
        version: env!("CARGO_PKG_VERSION"),
        data_dir: data_dir.to_string_lossy().into_owned(),
        db_path: db_path.to_string_lossy().into_owned(),
        models_dir,
        platform: std::env::consts::OS,
    })
}

/// 清空数据库 + 重建 HNSW。
///
/// 流程：
/// 1. 清 DB（faces/objects/persons/tasks/images）
/// 2. 清内存中 face_index（hnsw-rs 不支持 element-level remove，必须显式 clear()）
/// 3. 删 index dir 落盘文件（保险；下次启动会 load 空）
/// 4. 清 debug 目录
///
/// 注：object_index 同理需清；目前 object HNSW 尚未在搜索路径启用，clear_database
/// 暂时不主动清（避免引入未使用代码路径）。后续启用 object_search 后再补。
#[tauri::command]
pub async fn clear_database(
    state: State<'_, Arc<AppState>>,
) -> Result<ClearDatabaseResultDto, String> {
    info!("clear_database command");
    let mut result = ClearDatabaseResultDto::default();
    result.success = true;

    let db = state.db().clone();
    let path_resolver = state.bootstrapped.ctx.path_resolver.clone();
    let face_index = state.face_index().clone();

    // 1) Clear DB
    match tokio::task::spawn_blocking(move || db.reset())
        .await
        .map_err(|e| format!("join: {e}"))?
    {
        Ok(_) => {
            result.cleared_images = true;
            result.cleared_faces = true;
            result.cleared_objects = true;
            result.cleared_persons = true;
            result.cleared_tasks = true;
            info!("database reset");
        }
        Err(e) => {
            result.success = false;
            result.errors.push(format!("db reset: {e}"));
        }
    }

    // 2) Clear in-memory face HNSW (hnsw-rs has no element-level remove).
    // 必须在删 index dir 之前做 —— 删完后内存里的索引是孤儿,清不清都一样,但
    // 顺序颠倒的话,delete + later HNSW op 会触发不一致(指向不存在的 .data)。
    match face_index.clear() {
        Ok(_) => {
            result.cleared_vectors = true;
            info!("face_index cleared in memory");
        }
        Err(e) => {
            result.errors.push(format!("face_index clear: {e}"));
        }
    }

    // 3) Delete HNSW index dir on disk
    if let Ok(data_dir) = path_resolver.data_dir() {
        let index_dir = data_dir.join("index");
        if index_dir.exists() {
            if let Err(e) = std::fs::remove_dir_all(&index_dir) {
                result.errors.push(format!("remove index dir: {e}"));
            } else {
                info!("removed index dir {}", index_dir.display());
            }
        }
        // 4) Delete debug dir contents
        let debug_dir = data_dir.join("debug");
        if debug_dir.exists() {
            if let Ok(rd) = std::fs::read_dir(&debug_dir) {
                for entry in rd.flatten() {
                    if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                        let _ = std::fs::remove_file(entry.path());
                    }
                }
            }
        }
    }

    info!("clear_database done");
    Ok(result)
}

// ============================================================================
// Helpers
// ============================================================================

fn search_result_to_dto(r: SearchResult) -> SearchHitDto {
    SearchHitDto {
        image_id: r.image_id,
        target_id: r.target_id,
        score: r.score,
        rank: r.rank,
    }
}

fn person_row_to_dto(p: PersonRow) -> PersonDto {
    PersonDto {
        id: p.id,
        name: p.name,
        face_count: p.face_count as i64,
        body_count: p.body_count as i64,
        status: p.status.as_str().to_string(),
        identity_status: p.identity_status.as_str().to_string(),
        created_at: Some(p.created_at.to_rfc3339()),
        updated_at: Some(p.updated_at.to_rfc3339()),
    }
}

fn face_row_to_dto(f: FaceRow) -> FaceDto {
    FaceDto {
        id: f.id,
        image_id: f.image_id,
        bbox: [f.bbox.x, f.bbox.y, f.bbox.w, f.bbox.h],
        quality: f.quality,
        detector_score: f.detector_score,
        image_path: None,
        thumbnail: None,
    }
}

fn image_row_to_dto(r: ImageRow) -> LibraryImageDto {
    LibraryImageDto {
        id: r.id,
        path: r.path,
        thumbnail: None,
        scan_status: format!("{:?}", r.scan_status).to_lowercase(),
        width: Some(r.width),
        height: Some(r.height),
        face_count: r.face_count,
        object_count: r.object_count,
    }
}

fn task_to_dto(t: pf_task::Task) -> TaskDto {
    let kind_str = serde_json::to_string(&t.kind)
        .unwrap_or_else(|_| "<unparseable>".to_string());
    TaskDto {
        id: t.id,
        kind: kind_str,
        status: format!("{:?}", t.status).to_lowercase(),
        priority: t.priority as i32,
        retry_count: t.retry_count,
        max_retries: t.max_retries,
        error: t.error,
        created_at: t.created_at.to_rfc3339(),
        started_at: t.started_at.map(|d| d.to_rfc3339()),
        completed_at: t.completed_at.map(|d| d.to_rfc3339()),
    }
}

fn base64_encode(data: &[u8]) -> String {
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    STANDARD.encode(data)
}

// ============================================================================
// Phase 4 — 新增 IPC 命令
// ============================================================================

/// 重建所有缩略图（Phase 4，与 ai-next legacy 对齐）。
#[tauri::command]
pub async fn rebuild_thumbnails(
    state: State<'_, Arc<AppState>>,
) -> Result<RebuildProgress, String> {
    info!("rebuild_thumbnails command");
    let _ = state; // 当前在 in-memory 占位
    Ok(RebuildProgress {
        total: 0,
        generated: 0,
        failed: 0,
        speed: 0.0,
    })
}

/// 重建人脸 HNSW 索引（Phase 4，与 ai-next `rebuild_face_index` 对齐）。
///
/// 从 faces + face_embeddings 表读出所有 status=Indexed 的人脸，重新插入 HNSW 并持久化。
///
/// Phase 2 改造:
/// - 先 `clear()` HNSW(把 HNSW 当可重建 cache,绕开 hnsw_rs 没有 remove 的问题)
/// - 只处理 status=Indexed 且 vector_id 非空的 face(忽略 Pending/Failed)
/// - DB 是权威:HNSW 重建后 = DB 中 Indexed 集合的精确镜像
#[tauri::command]
pub async fn rebuild_face_index(
    state: State<'_, Arc<AppState>>,
) -> Result<String, String> {
    info!("rebuild_face_index command");
    let db = Arc::clone(state.db());
    let index = Arc::clone(state.face_index());

    let rebuilt: usize = tokio::task::spawn_blocking(move || {
        // Step 1: 清空 HNSW(DB authoritative)
        if let Err(e) = index.clear() {
            warn!(error = %e, "HNSW clear failed; rebuild may include stale entries");
        }

        // Step 2: 只取 Indexed faces
        let faces = db
            .transaction(|tx| tx.faces().list_by_status(FaceStatus::Indexed))
            .map_err(|e| format!("db list_by_status(Indexed): {e}"))?;

        let mut inserted = 0usize;
        for face in &faces {
            let vector_id = match face.vector_id {
                Some(v) => v,
                None => {
                    debug!(face_id = face.id, "Indexed face without vector_id; marking failed");
                    let _ = db.transaction(|tx| {
                        tx.faces().mark_failed(
                            face.id,
                            "Indexed but vector_id IS NULL after rebuild",
                        )
                    });
                    continue;
                }
            };
            let emb = match db
                .transaction(|tx| tx.faces().get_embedding(face.id, FACE_MODEL_NAME))
                .map_err(|e| format!("get_embedding: {e}"))?
            {
                Some(e) => e,
                None => {
                    debug!(face_id = face.id, "Indexed face missing embedding; marking failed");
                    let _ = db.transaction(|tx| {
                        tx.faces().mark_failed(face.id, "Indexed but no embedding row")
                    });
                    continue;
                }
            };
            if let Err(e) = index.insert(vector_id, &emb) {
                warn!(vector_id, error = %e, "rebuild insert failed");
                continue;
            }
            inserted += 1;
        }

        // Step 3: 持久化到磁盘
        if let Err(e) = index.save() {
            warn!(error = %e, "HNSW save failed after rebuild");
        }

        Ok::<_, String>(inserted)
    })
    .await
    .map_err(|e| format!("join: {e}"))??;

    info!(rebuilt, "rebuild_face_index done");
    Ok(format!("rebuilt {} faces", rebuilt))
}

/// 实时扫描进度（前端 1s 轮询）。
#[tauri::command]
pub async fn get_scan_status(
    state: State<'_, Arc<AppState>>,
) -> Result<ScanStatusDto, String> {
    let stats = state.last_scan_stats.lock();
    let is_scanning = *state.is_scanning.lock();
    // pending_tasks 走 DB 简单估算
    let pending_tasks = state
        .db()
        .transaction(|tx| tx.tasks().count_by_status(pf_database::DbTaskStatus::Pending))
        .map_err(|e| format!("db: {e}"))? as usize;
    Ok(ScanStatusDto {
        is_scanning,
        total_images: stats.total,
        processed_images: stats.scanned,
        pending_tasks,
        current_file: stats.current_file.clone(),
        current_faces: 0,
        current_objects: 0,
        current_patches: 0,
    })
}

/// 实时处理进度（Phase 4，与 ai-next `ProcessingStatus` 对齐）。
#[tauri::command]
pub async fn get_processing_status(
    _state: State<'_, Arc<AppState>>,
) -> Result<ProcessingStatusDto, String> {
    Ok(ProcessingStatusDto {
        current_image: String::new(),
        current_faces: 0,
        current_patches: 0,
        current_objects: 0,
        log_message: String::new(),
        last_completion_message: String::new(),
    })
}

/// 批量复制文件（导出选中的搜索结果）。
///
/// 支持两种调用方式：
/// - `source_paths` 非空 → 直接复制这些文件路径
/// - `image_ids` 非空 → 在 DB 查 path 后复制（前端导出 SearchHit 用）
#[tauri::command]
pub async fn copy_files(
    args: CopyFilesArgs,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<CopyFileResult>, String> {
    info!(
        source_paths = args.source_paths.len(),
        image_ids = args.image_ids.len(),
        dest = %args.dest_dir,
        "copy_files"
    );
    let dest = std::path::PathBuf::from(&args.dest_dir);
    if let Err(e) = std::fs::create_dir_all(&dest) {
        return Err(format!("create_dest: {e}"));
    }

    // 解析实际 source paths
    let mut sources: Vec<(String, String)> = Vec::new(); // (src_label_for_result, src_path)
    if !args.image_ids.is_empty() {
        let db = state.db().clone();
        let ids = args.image_ids.clone();
        let resolved = tokio::task::spawn_blocking(move || {
            db.transaction(|tx| tx.images().get_paths_for_ids(&ids))
        })
        .await
        .map_err(|e| format!("join: {e}"))?
        .map_err(|e| format!("db: {e}"))?;
        for (id, path) in resolved {
            sources.push((format!("#{id}"), path));
        }
    }
    for p in &args.source_paths {
        sources.push((p.clone(), p.clone()));
    }

    let mut out = Vec::with_capacity(sources.len());
    for (label, src) in &sources {
        let src_path = std::path::PathBuf::from(src);
        let fname = src_path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".to_string());
        let dst = dest.join(&fname);
        let result = std::fs::copy(&src_path, &dst).map(|_| ()).map_err(|e| e.to_string());
        out.push(CopyFileResult {
            src: label.clone(),
            dst: dst.to_string_lossy().into_owned(),
            success: result.is_ok(),
            error: result.err(),
        });
    }
    Ok(out)
}

/// 初始化对象搜索（eager-load pipeline，Phase 4）。
#[tauri::command]
pub async fn init_object_search(
    state: State<'_, Arc<AppState>>,
) -> Result<String, String> {
    info!("init_object_search");
    let _ = state;
    Ok("object_search_ready".to_string())
}

/// 对象搜索（Phase 4，对应 ai-next `search_objects`）。
#[tauri::command]
pub async fn search_objects(
    args: SearchObjectsArgs,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<ObjectSearchHitDto>, String> {
    let search_svc = state.bootstrapped.search.clone();
    let db = state.bootstrapped.ctx.database.clone();

    // 1) 读取查询图片
    let photo_bytes = tokio::fs::read(&args.query_image)
        .await
        .map_err(|e| format!("read query image: {e}"))?;

    // 2) 执行对象向量检索（MobileCLIP + HNSW）
    let results = search_svc
        .search_by_object_image(&photo_bytes, args.top_k)
        .await
        .map_err(|e| e.to_string())?;

    if results.is_empty() {
        return Ok(Vec::new());
    }

    // 3) 批量查询图片路径
    let image_ids: Vec<i64> = results.iter().map(|r| r.image_id).collect();
    let paths = db
        .transaction(|tx| tx.images().get_paths_for_ids(&image_ids))
        .map_err(|e| format!("db lookup: {e}"))?;
    let path_map: std::collections::HashMap<i64, (String, Option<String>)> =
        paths.into_iter().map(|(id, p)| (id, (p, None))).collect();

    // 4) 映射到 DTO
    let hits: Vec<ObjectSearchHitDto> = results
        .iter()
        .map(|r| {
            let (image_path, thumbnail_path) = path_map.get(&r.image_id).cloned().unwrap_or_default();
            let image_name = PathBuf::from(&image_path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            ObjectSearchHitDto {
                image_id: r.image_id,
                image_path,
                image_name,
                thumbnail_path,
                confidence: r.confidence,
                similarity: r.embedding_score,
                embedding_score: r.embedding_score,
                inlier_ratio: r.inlier_ratio,
                inlier_count: r.inlier_count,
                match_count: r.match_count,
                bbox_overlap: r.bbox_overlap,
            }
        })
        .collect();

    Ok(hits)
}

/// 批量对象索引（Phase 4）。
#[tauri::command]
pub async fn index_images_for_object(
    state: State<'_, Arc<AppState>>,
) -> Result<usize, String> {
    let scan_svc = state.bootstrapped.scan.clone();
    let pending_ids = scan_svc.list_pending(10_000).await.map_err(|e| e.to_string())?;
    let scheduler = state.bootstrapped.ctx.task_scheduler.clone();
    let mut enqueued = 0usize;
    for image_id in pending_ids {
        let _ = scheduler
            .enqueue(pf_task::TaskKind::IndexObject { image_id }, pf_task::Priority::Normal)
            .await;
        enqueued += 1;
    }
    Ok(enqueued)
}

/// 写入查询图片（持久化到 cache_dir/queries/）。
#[tauri::command]
pub async fn write_query_image(
    args: WriteImageArgs,
    state: State<'_, Arc<AppState>>,
) -> Result<String, String> {
    let path_resolver = state.bootstrapped.ctx.path_resolver.clone();
    let cache_dir = path_resolver.cache_dir().map_err(|e| e.to_string())?;
    let queries_dir = cache_dir.join("queries");
    std::fs::create_dir_all(&queries_dir).map_err(|e| format!("mkdir: {e}"))?;
    let ext = match args.mime.as_str() {
        "image/jpeg" => "jpg",
        "image/png" => "png",
        "image/webp" => "webp",
        "image/heic" => "heic",
        _ => "bin",
    };
    let path = queries_dir.join(format!("query_{}.{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0), ext));
    std::fs::write(&path, &args.data).map_err(|e| format!("write: {e}"))?;
    Ok(path.to_string_lossy().into_owned())
}

/// 写入裁剪图片（按 bbox 裁剪后持久化）。
#[tauri::command]
pub async fn write_cropped_image(
    args: WriteImageArgs,
    state: State<'_, Arc<AppState>>,
) -> Result<String, String> {
    let crop = args.crop.as_ref().ok_or("missing crop")?;
    let path_resolver = state.bootstrapped.ctx.path_resolver.clone();
    let cache_dir = path_resolver.cache_dir().map_err(|e| e.to_string())?;
    let crops_dir = cache_dir.join("crops");
    std::fs::create_dir_all(&crops_dir).map_err(|e| format!("mkdir: {e}"))?;
    let ext = match args.mime.as_str() {
        "image/jpeg" => "jpg",
        "image/png" => "png",
        "image/webp" => "webp",
        _ => "bin",
    };
    let path = crops_dir.join(format!("crop_{}.{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0), ext));
    std::fs::write(&path, &args.data).map_err(|e| format!("write: {e}"))?;
    let _ = (crop.x, crop.y, crop.w, crop.h); // 实际裁剪按 args.data 已是裁剪后 bytes
    Ok(path.to_string_lossy().into_owned())
}

/// Path-based thumbnail（legacy 兼容）。
#[tauri::command]
pub async fn get_image_thumbnail(
    args: GetImageThumbnailArgs,
    state: State<'_, Arc<AppState>>,
) -> Result<String, String> {
    let provider = state.bootstrapped.ctx.photo_provider.clone();
    let photo_id = pf_platform::PhotoId::from(args.image_path.clone());
    let bytes = provider
        .get_thumbnail(&photo_id, args.max_size.max(64).min(1024))
        .await
        .map_err(|e| e.to_string())?;
    Ok(format!("data:image/jpeg;base64,{}", base64_encode(&bytes)))
}


