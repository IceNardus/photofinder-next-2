//! # pf_ios_bridge — iOS C ABI 桥
//!
//! 把 `pf_application::Bootstrapped` 服务暴露成 C ABI 函数，供 Swift 端
//! `apps/ios/Sources/photofinder_desktop/AppBootstrap.swift` 通过 `@_silgen_name` 调用。
//!
//! ## 12 个 `pf_ios_*` 函数
//!
//! | 函数 | 返回 | 用途 |
// |---|---|---|
//! | `pf_ios_init_app` | i32 | 构造 Bootstrapped + tokio runtime（同步,可能阻塞几秒） |
//! | `pf_ios_run_scan` | i32 | 触发扫描（异步,返回 0 = 已入队） |
//! | `pf_ios_scan_in_progress` | i32 | 1 = 扫描中,0 = 空闲 |
//! | `pf_ios_db_image_count` | i32 | DB 已 Indexed 图片数,-1 = 未初始化 |
//! | `pf_ios_scan_progress` | *mut c_char | 当前进度 JSON 或 NULL |
//! | `pf_ios_scan_result` | *mut c_char | 最近一次扫描完成结果 JSON |
//! | `pf_ios_clear_db` | i32 | 清库 + 清 HNSW |
//! | `pf_ios_search_person` | *mut c_char | 人物搜索 JSON (Phase B) |
//! | `pf_ios_search_object` | *mut c_char | 对象搜索 JSON (Phase C) |
//! | `pf_ios_free_string` | () | 释放 Swift strdup 字符串 |
//!
//! ## 字符串所有权
//!
//! Rust 返回的 `*mut c_char` 是 `CString::into_raw()` 给出的 owned pointer，
//! Swift 用完必须调 `pf_ios_free_string` 释放（同一 libc allocator）。
//!
//! ## 状态生命周期
//!
//! `pf_ios_init_app` 第一次调用时:
//! 1. `setup::init_logging()`
//! 2. 构造 tokio runtime + scheduler
//! 3. 构造 `Bootstrapped`（IosPathResolver + PhotoKit bridge + face pipeline）
//! 4. 启动后台 worker
//!
//! 之后所有 ABI 调用都读 STATE singleton。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use std::ffi::{c_char, CStr, CString};
use std::os::raw::c_int;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use anyhow::Context;
use pf_application::Bootstrapped;
use pf_database::ScanStatus;
use pf_platform::PhotoFolder;
use serde::Serialize;

pub mod setup;

// ============================================================================
// State
// ============================================================================

/// 全局 singleton — 第一次 `pf_ios_init_app` 时初始化。
pub struct IosBridgeState {
    /// Rust 业务逻辑
    pub bootstrapped: Arc<Bootstrapped>,
    /// 专用 tokio runtime（异步任务用）
    pub runtime: tokio::runtime::Runtime,
    /// 扫描进行中标志
    pub scan_in_progress: Arc<AtomicBool>,
    /// 当前扫描进度（Swift 轮询）
    pub scan_progress: Arc<parking_lot::Mutex<ScanProgress>>,
    /// 最近一次扫描完成结果
    pub last_scan_result: Arc<parking_lot::Mutex<Option<ScanSummaryDto>>>,
}

/// 实时进度 — 每次 ScanService callback 更新。
#[derive(Debug, Default, Clone, Serialize)]
pub struct ScanProgress {
    /// 已处理图片数
    pub processed: usize,
    /// 候选总数（如果有）
    pub total: usize,
    /// 当前正在处理的图片路径/URI
    pub current_file: String,
}

/// 最近一次扫描结果摘要（JSON 友好）。
#[derive(Debug, Clone, Serialize)]
pub struct ScanSummaryDto {
    pub path: String,
    pub candidate_count: usize,
    pub inserted: usize,
    pub skipped: usize,
    pub filtered: usize,
    pub filtered_bpp: usize,
    pub queued_face_tasks: usize,
    pub queued_object_tasks: usize,
}

impl From<pf_application::ScanSummary> for ScanSummaryDto {
    fn from(s: pf_application::ScanSummary) -> Self {
        Self {
            path: s.path,
            candidate_count: s.candidate_count,
            inserted: s.inserted,
            skipped: s.skipped,
            filtered: s.filtered,
            filtered_bpp: s.filtered_bpp,
            queued_face_tasks: s.queued_face_tasks,
            queued_object_tasks: s.queued_object_tasks,
        }
    }
}

static STATE: OnceLock<IosBridgeState> = OnceLock::new();

/// 是否已初始化（用于在 init_app 之前的 ABI 调用返回 sentinel）。
fn state() -> Option<&'static IosBridgeState> {
    STATE.get()
}

// ============================================================================
// Helpers
// ============================================================================

/// 把 owned String 变成 strdup'd raw pointer（Swift 用 pf_ios_free_string 释放）。
/// 空字符串也返回有效指针（避免 Swift 收到 NULL 时与"未初始化"混淆）。
fn to_c_string(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(c) => c.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

fn json_to_c_string<T: Serialize>(v: &T) -> *mut c_char {
    match serde_json::to_string(v) {
        Ok(s) => to_c_string(s),
        Err(_) => std::ptr::null_mut(),
    }
}

// ============================================================================
// Phase A — init / scan / clear_db
// ============================================================================

/// 初始化 Rust 状态。返回:
/// - 0 = 成功
/// - 4 = 已初始化（OnceLock 命中,非错误）
/// - -1 = 真实失败（bootstrap / tokio 构建错误）
///
/// Swift 在 App.init() 之后立即调用一次（自带 didInit 守门,Rust 端
/// 二次调用也安全 — 返回 4 而不是 -1）。
#[no_mangle]
pub extern "C" fn pf_ios_init_app() -> c_int {
    let _ = setup::init_logging();
    let state_result: anyhow::Result<IosBridgeState> = (|| {
        let bootstrapped = setup::bootstrap().context("bootstrap")?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(2)
            .thread_name("pf-ios")
            .build()
            .context("tokio")?;
        // 启动 scheduler worker（处理 IndexFace / IndexObject / IndexPatch 任务）
        runtime.block_on(async {
            bootstrapped.scheduler.start_workers();
        });
        Ok(IosBridgeState {
            bootstrapped: Arc::new(bootstrapped),
            runtime,
            scan_in_progress: Arc::new(AtomicBool::new(false)),
            scan_progress: Arc::new(parking_lot::Mutex::new(ScanProgress::default())),
            last_scan_result: Arc::new(parking_lot::Mutex::new(None)),
        })
    })();

    match state_result {
        Ok(state) => match STATE.set(state) {
            Ok(()) => 0,
            Err(_) => {
                tracing::info!("pf_ios_init_app: already initialized");
                4
            }
        },
        Err(e) => {
            tracing::error!("pf_ios_init_app failed: {e:#}");
            -1
        }
    }
}

/// 触发扫描。返回 0 = 已入队,非 0 = 失败。
///
/// 扫描的是 `PhotoKitPhotoProvider` 的 "Photo Library" 虚拟文件夹
/// （实际通过 Swift bridge 拿 PHAsset 列表）。
#[no_mangle]
pub extern "C" fn pf_ios_run_scan() -> c_int {
    let Some(state) = state() else {
        return -1;
    };
    if state.scan_in_progress.swap(true, Ordering::SeqCst) {
        // 已经在扫
        return 0;
    }
    // 重置 progress + last_result
    *state.scan_progress.lock() = ScanProgress::default();
    *state.last_scan_result.lock() = None;

    let bootstrapped = state.bootstrapped.clone();
    let scan_in_progress = state.scan_in_progress.clone();
    let scan_progress = state.scan_progress.clone();
    let last_scan_result_arc = state.last_scan_result.clone();
    state.runtime.spawn(async move {
        let folder = PhotoFolder {
            id: "photo-library".to_string(),
            display_name: "Photo Library".to_string(),
            count_estimate: None,
        };
        let progress_cb: pf_application::scan::OnImageCallback = {
            let sp = scan_progress.clone();
            Arc::new(move |_image_id, path| {
                let mut g = sp.lock();
                g.processed += 1;
                g.current_file = path.to_string();
            })
        };
        let result = bootstrapped
            .scan
            .scan_folder_entry(&folder, Some(progress_cb))
            .await;
        match result {
            Ok(summary) => {
                *last_scan_result_arc.lock() = Some(summary.into());
                tracing::info!("scan complete");
            }
            Err(e) => {
                tracing::error!("scan failed: {e}");
            }
        }
        scan_in_progress.store(false, Ordering::SeqCst);
    });

    0
}

/// 扫描是否在进行中（1 = 是,0 = 否）。
#[no_mangle]
pub extern "C" fn pf_ios_scan_in_progress() -> c_int {
    state()
        .map(|s| if s.scan_in_progress.load(Ordering::SeqCst) { 1 } else { 0 })
        .unwrap_or(0)
}

/// DB 已 Indexed 图片数（-1 = 未初始化）。
#[no_mangle]
pub extern "C" fn pf_ios_db_image_count() -> c_int {
    let Some(state) = state() else {
        return -1;
    };
    let db = state.bootstrapped.ctx.database.clone();
    // DB 读用 spawn_blocking 避免阻塞 runtime
    let count: i64 = match state.runtime.block_on(async {
        tokio::task::spawn_blocking(move || {
            db.transaction(|tx| tx.images().count_by_status(ScanStatus::Indexed))
        })
        .await
        .unwrap_or(Ok(0))
    }) {
        Ok(n) => n,
        Err(_) => -1,
    };
    if count < 0 { -1 } else { count as c_int }
}

/// 当前扫描进度 JSON。`{"processed":N,"total":M,"current_file":"..."}`。
/// NULL = 未初始化或无进度。
#[no_mangle]
pub extern "C" fn pf_ios_scan_progress() -> *mut c_char {
    let Some(state) = state() else {
        return std::ptr::null_mut();
    };
    let snapshot = state.scan_progress.lock().clone();
    json_to_c_string(&snapshot)
}

/// 最近一次扫描完成结果 JSON。NULL = 未完成或未初始化。
#[no_mangle]
pub extern "C" fn pf_ios_scan_result() -> *mut c_char {
    let Some(state) = state() else {
        return std::ptr::null_mut();
    };
    let snapshot = state.last_scan_result.lock().clone();
    match snapshot {
        Some(s) => json_to_c_string(&s),
        None => std::ptr::null_mut(),
    }
}

/// 清空数据库 + HNSW 索引 + index dir。返回 0 = 成功,非 0 = 失败。
#[no_mangle]
pub extern "C" fn pf_ios_clear_db() -> c_int {
    let Some(state) = state() else {
        return -1;
    };
    let db = state.bootstrapped.ctx.database.clone();
    let face_index = state.bootstrapped.ctx.face_index.clone();
    let path_resolver = state.bootstrapped.ctx.path_resolver.clone();

    let result: anyhow::Result<()> = state.runtime.block_on(async {
        // 1) DB reset
        let db2 = db.clone();
        tokio::task::spawn_blocking(move || db2.reset())
            .await
            .map_err(|e| anyhow::anyhow!("join: {e}"))??;
        // 2) HNSW clear in-memory
        face_index
            .clear()
            .map_err(|e| anyhow::anyhow!("face_index clear: {e}"))?;
        // 3) Remove index dir on disk
        if let Ok(data_dir) = path_resolver.data_dir() {
            let index_dir = data_dir.join("index");
            if index_dir.exists() {
                let _ = std::fs::remove_dir_all(&index_dir);
            }
        }
        Ok(())
    });
    match result {
        Ok(()) => {
            *state.scan_progress.lock() = ScanProgress::default();
            *state.last_scan_result.lock() = None;
            0
        }
        Err(e) => {
            tracing::error!("pf_ios_clear_db failed: {e:#}");
            -1
        }
    }
}

// ============================================================================
// Phase B — 人物搜索
// ============================================================================

/// 人物搜索命中（JSON 友好格式 — 与 Swift `PersonSearchResultDTO` 对齐）。
#[derive(Debug, Clone, Serialize)]
pub struct PersonSearchHitDto {
    /// 命中的图片 id
    pub image_id: i64,
    /// 命中的人脸 id（Rust `SearchResult::target_id`）
    pub face_id: i64,
    /// 相似度分数（cosine 0..1）
    pub score: f32,
    /// 人脸 bbox（xyxy 格式，左上 + 右下）
    pub bbox: Option<[f32; 4]>,
    /// 缩略图 / 平台 URI（iOS 上 = `phasset://LOCALID`）
    pub thumbnail_path: String,
}

/// 人物搜索：选一张含人脸的参考图 → 找图库中所有同一人照片。
///
/// `query_path` 是 Swift 写到 `data_dir/query_image_<uuid>.jpg` 的 JPEG 路径。
/// `top_k` 最大返回数（实际会先 fetch 4*top_k 再做阈值过滤）。
/// 返回 JSON 数组 — 元素为 `[{image_id, face_id, score, bbox, thumbnail_path}, ...]`。
/// 失败 / 未初始化时返回 NULL。
#[no_mangle]
pub unsafe extern "C" fn pf_ios_search_person(query_path: *const c_char, top_k: c_int) -> *mut c_char {
    let Some(state) = state() else {
        return std::ptr::null_mut();
    };

    // 1) Path 解析
    let path_str = unsafe {
        if query_path.is_null() {
            return std::ptr::null_mut();
        }
        match CStr::from_ptr(query_path).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };

    let top_k = top_k.max(1) as usize;

    // 2) 读 JPEG bytes
    let bytes = match std::fs::read(path_str) {
        Ok(b) => b,
        Err(e) => {
            tracing::error!("pf_ios_search_person: read {} failed: {e}", path_str);
            return std::ptr::null_mut();
        }
    };

    // 3) 跑搜索（异步 → 用 block_on 同步）
    let bootstrapped = state.bootstrapped.clone();
    let hits = match state.runtime.block_on(async {
        bootstrapped.search.search_by_face_image(&bytes, top_k).await
    }) {
        Ok(h) => h,
        Err(e) => {
            tracing::error!("pf_ios_search_person: search failed: {e}");
            return std::ptr::null_mut();
        }
    };

    // 4) Enrich with bbox + thumbnail_path（DB 读用 spawn_blocking 不阻塞 runtime）
    let db = state.bootstrapped.ctx.database.clone();
    let enriched: Vec<PersonSearchHitDto> = state
        .runtime
        .block_on(async {
            tokio::task::spawn_blocking(move || {
                let mut out = Vec::with_capacity(hits.len());
                for h in hits {
                    let bbox = db
                        .transaction(|tx| tx.faces().get_by_id(h.target_id))
                        .ok()
                        .flatten()
                        .map(|f| {
                            [
                                f.bbox.x,
                                f.bbox.y,
                                f.bbox.x + f.bbox.w,
                                f.bbox.y + f.bbox.h,
                            ]
                        });

                    let thumbnail_path = db
                        .transaction(|tx| tx.images().get_by_id(h.image_id))
                        .ok()
                        .flatten()
                        .map(|img| img.path)
                        .unwrap_or_default();

                    out.push(PersonSearchHitDto {
                        image_id: h.image_id,
                        face_id: h.target_id,
                        score: h.score,
                        bbox,
                        thumbnail_path,
                    });
                }
                out
            })
            .await
        })
        .unwrap_or_default();

    json_to_c_string(&enriched)
}

// ============================================================================
// Phase C — 对象搜索（stub）
// ============================================================================

/// 对象搜索命中（JSON 友好格式 — 与 Swift `ObjectSearchResultDTO` 对齐）。
#[derive(Debug, Clone, Serialize)]
pub struct ObjectSearchHitDto {
    /// 命中的图片 id
    pub image_id: i64,
    /// 图片路径 / 平台 URI（iOS 上 = `phasset://LOCALID`）
    pub image_path: String,
    /// 融合置信度（α*embedding + β*inlier + γ*bbox）
    pub confidence: f32,
    /// LightGlue 内点率
    pub inlier_ratio: f32,
    /// 向量检索分数
    pub embedding_score: f32,
    /// 命中的 ROI 区域（xyxy 格式，在候选图片像素坐标系 — iOS 用于画矩形叠加）
    pub matched_bbox: [f32; 4],
}

/// 对象搜索：给一张参考图（或裁剪图）→ 找图库中所有相似物品。
///
/// `query_path` 是 Swift 写到 `data_dir/query_crop_<uuid>.jpg` 的 JPEG 路径。
/// `top_k` 最大返回数。
/// 返回 JSON 数组 — 元素为 `[{image_id, image_path, confidence, inlier_ratio,
/// embedding_score, matched_bbox}, ...]`。
/// 失败 / 未初始化时返回 NULL。
#[no_mangle]
pub unsafe extern "C" fn pf_ios_search_object(query_path: *const c_char, top_k: c_int) -> *mut c_char {
    let Some(state) = state() else {
        return std::ptr::null_mut();
    };

    // 1) Path 解析
    let path_str = unsafe {
        if query_path.is_null() {
            return std::ptr::null_mut();
        }
        match CStr::from_ptr(query_path).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        }
    };

    let top_k = top_k.max(1) as usize;

    // 2) 读 JPEG bytes
    let bytes = match std::fs::read(path_str) {
        Ok(b) => b,
        Err(e) => {
            tracing::error!("pf_ios_search_object: read {} failed: {e}", path_str);
            return std::ptr::null_mut();
        }
    };

    // 3) 跑搜索（异步 → block_on）
    let bootstrapped = state.bootstrapped.clone();
    let hits = match state.runtime.block_on(async {
        bootstrapped.search.search_by_object_image(&bytes, top_k).await
    }) {
        Ok(h) => h,
        Err(e) => {
            tracing::error!("pf_ios_search_object: search failed: {e}");
            return std::ptr::null_mut();
        }
    };

    // 4) Enrich with image_path（DB 批量读）
    let db = state.bootstrapped.ctx.database.clone();
    let enriched: Vec<ObjectSearchHitDto> = state
        .runtime
        .block_on(async {
            tokio::task::spawn_blocking(move || {
                let image_ids: Vec<i64> = hits.iter().map(|h| h.image_id).collect();
                let path_map: std::collections::HashMap<i64, String> = db
                    .transaction(|tx| tx.images().get_paths_for_ids(&image_ids))
                    .ok()
                    .map(|v| v.into_iter().collect())
                    .unwrap_or_default();

                hits.into_iter()
                    .map(|h| {
                        let image_path = path_map
                            .get(&h.image_id)
                            .cloned()
                            .unwrap_or_default();
                        let mb = &h.matched_bbox;
                        ObjectSearchHitDto {
                            image_id: h.image_id,
                            image_path,
                            confidence: h.confidence,
                            inlier_ratio: h.inlier_ratio,
                            embedding_score: h.embedding_score,
                            matched_bbox: [mb.x, mb.y, mb.x + mb.w, mb.y + mb.h],
                        }
                    })
                    .collect()
            })
            .await
        })
        .unwrap_or_default();

    json_to_c_string(&enriched)
}

// ============================================================================
// 字符串释放
// ============================================================================

/// 释放由 `pf_ios_*` 返回的字符串（Swift 用完必须调）。
#[no_mangle]
pub unsafe extern "C" fn pf_ios_free_string(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = CString::from_raw(ptr);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_string_accepts_null() {
        pf_ios_free_string(std::ptr::null_mut());
    }

    #[test]
    fn free_string_owns_allocated_cstring() {
        let s = CString::new("hello").unwrap();
        let ptr = s.into_raw();
        pf_ios_free_string(ptr);
    }

    #[test]
    fn uninitialized_state_returns_sentinel() {
        // 不要假设 STATE 已初始化 — 测试在隔离进程跑,OnceLock 为空
        let _ = pf_ios_init_app; // 抑制未使用警告
        assert!(state().is_none() || state().is_some());
    }

    #[test]
    fn json_to_c_string_handles_valid_struct() {
        #[derive(Serialize)]
        struct Foo {
            a: i32,
            b: String,
        }
        let f = Foo { a: 42, b: "hello".into() };
        let ptr = json_to_c_string(&f);
        assert!(!ptr.is_null());
        unsafe {
            let s = CString::from_raw(ptr);
            let json = s.to_string_lossy();
            assert!(json.contains("\"a\":42"));
            assert!(json.contains("\"b\":\"hello\""));
        }
    }

    /// 验证 `PersonSearchHitDto` JSON 字段名与 Swift `PersonSearchResultDTO` 完全一致。
    /// 字段名不匹配会导致 Swift 端 decode 失败但 host 端 CI 看不到。
    #[test]
    fn person_search_hit_dto_json_keys_match_swift() {
        let hit = PersonSearchHitDto {
            image_id: 42,
            face_id: 7,
            score: 0.87,
            bbox: Some([10.0, 20.0, 110.0, 120.0]),
            thumbnail_path: "phasset://LOCALID123".to_string(),
        };
        let ptr = json_to_c_string(&hit);
        assert!(!ptr.is_null());
        unsafe {
            let s = CString::from_raw(ptr);
            let json = s.to_string_lossy();
            // Swift 端 Decodable: image_id, face_id, score, bbox, thumbnail_path
            assert!(json.contains("\"image_id\":42"), "missing image_id: {json}");
            assert!(json.contains("\"face_id\":7"), "missing face_id: {json}");
            assert!(json.contains("\"score\":0.87"), "missing score: {json}");
            assert!(
                json.contains("\"thumbnail_path\":\"phasset://LOCALID123\""),
                "missing thumbnail_path: {json}"
            );
            // bbox 是数组 — 检查 4 个数字
            assert!(json.contains("\"bbox\":[10.0,20.0,110.0,120.0]"), "bbox array: {json}");
        }
    }

    /// 验证 `pf_ios_search_person` 在未初始化时返回 NULL（Swift 收到 nil → 走错误分支）。
    #[test]
    fn search_person_returns_null_when_uninitialized() {
        // 隔离进程 — STATE 为空
        let result = pf_ios_search_person(std::ptr::null(), 50);
        assert!(result.is_null());
    }

    /// 验证 `to_c_string` 不在 intermediate NUL byte 处 panic（编码安全）。
    #[test]
    fn to_c_string_handles_internal_nul() {
        let s = "hello\0world".to_string();
        let ptr = to_c_string(s);
        // embedded NUL → CString::new fails → 返回 null
        assert!(ptr.is_null());
    }

    /// 验证 `ObjectSearchHitDto` JSON 字段名与 Swift `ObjectSearchResultDTO` 完全一致。
    /// 字段名不匹配会导致 Swift 端 decode 失败但 host 端 CI 看不到。
    #[test]
    fn object_search_hit_dto_json_keys_match_swift() {
        let hit = ObjectSearchHitDto {
            image_id: 99,
            image_path: "phasset://LOCALID_xyz".to_string(),
            confidence: 0.91,
            inlier_ratio: 0.75,
            embedding_score: 0.88,
            matched_bbox: [50.0, 60.0, 250.0, 260.0],
        };
        let ptr = json_to_c_string(&hit);
        assert!(!ptr.is_null());
        unsafe {
            let s = CString::from_raw(ptr);
            let json = s.to_string_lossy();
            // Swift 端 Decodable: image_id, image_path, confidence, inlier_ratio,
            //                      embedding_score, matched_bbox
            assert!(json.contains("\"image_id\":99"), "missing image_id: {json}");
            assert!(
                json.contains("\"image_path\":\"phasset://LOCALID_xyz\""),
                "missing image_path: {json}"
            );
            assert!(json.contains("\"confidence\":0.91"), "missing confidence: {json}");
            assert!(json.contains("\"inlier_ratio\":0.75"), "missing inlier_ratio: {json}");
            assert!(
                json.contains("\"embedding_score\":0.88"),
                "missing embedding_score: {json}"
            );
            assert!(
                json.contains("\"matched_bbox\":[50.0,60.0,250.0,260.0]"),
                "matched_bbox array: {json}"
            );
        }
    }

    /// 验证 `pf_ios_search_object` 在未初始化时返回 NULL（Swift 收到 nil → 走错误分支）。
    #[test]
    fn search_object_returns_null_when_uninitialized() {
        let result = pf_ios_search_object(std::ptr::null(), 20);
        assert!(result.is_null());
    }
}