//! PhotoFinder Flutter FFI Library
//!
//! Frontend (Flutter) -> FFI calls -> crates/ core methods
//! - pf_database::Database for database operations
//! - pf_ai::FacePipeline for face detection/embedding
//! - pf_vector::HnswIndex for face indexing

use std::ffi::c_char;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicI64, Ordering};
use std::collections::HashMap;

use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio::runtime::Runtime;
use image::GenericImageView;

// pf_* crates
use pf_ai::{
    ArcFaceEmbedder, FacePipeline, ImageData, QualityFilter, ScrfdDetector, SimpleAligner,
};
use pf_application::visual_scan_v2::config::VisualScanV2Config;
use pf_database::Database;
use pf_license::{LicenseManager, LicenseStorage};
use pf_vector::{HnswIndex, VectorIndex};

// ============================================================================
// Global State - using proper crates types
// ============================================================================

/// Database reference
pub static DB: Mutex<Option<Arc<Database>>> = Mutex::new(None);

/// License manager reference
static LICENSE_MANAGER: Mutex<Option<Arc<LicenseManager>>> = Mutex::new(None);

fn get_license_manager() -> Result<Arc<LicenseManager>, String> {
    let guard = LICENSE_MANAGER.lock().unwrap();
    guard.as_ref().cloned().ok_or_else(|| "License not initialized".to_string())
}

/// FacePipeline for face detection/embedding (loaded in pf_init)
static FACE_PIPELINE: Mutex<Option<Arc<FacePipeline>>> = Mutex::new(None);

/// Face HNSW index for fast face search
static FACE_INDEX: Mutex<Option<Arc<dyn VectorIndex>>> = Mutex::new(None);

/// VisualScannerV2 for object/DINO feature extraction
static VISUAL_SCANNER: Mutex<Option<VisualScannerState>> = Mutex::new(None);

/// Visual Scan V2 HNSW index (384-dim, loaded in pf_init)
static VISUAL_V2_INDEX: Mutex<Option<Arc<dyn VectorIndex>>> = Mutex::new(None);

/// Visual scanner state holding the scanner instance
#[derive(Clone)]
struct VisualScannerState {
    // Arc<Mutex> so we can clone into spawned threads and get &mut access
    scanner: std::sync::Arc<std::sync::Mutex<pf_application::visual_scan_v2::scanner::VisualScannerV2>>,
    config: VisualScanV2Config,
    model_path: PathBuf,
}

/// Tokio runtime for async operations
static RUNTIME: Mutex<Option<Arc<Runtime>>> = Mutex::new(None);

// Clustering state
static CLUSTER_TASK_COUNTER: AtomicI64 = AtomicI64::new(1);
static CLUSTER_RESULTS: Mutex<Option<HashMap<i64, String>>> = Mutex::new(None);
static CLUSTER_RUNNING: Mutex<Option<bool>> = Mutex::new(None);

// Incremental clustering: trigger clustering every N faces processed
static FACES_SINCE_LAST_CLUSTER: AtomicI64 = AtomicI64::new(0);
const INCREMENTAL_CLUSTER_BATCH_SIZE: i64 = 5; // Trigger clustering every 5 faces

// Session creation counters (for debugging - verify sessions are only created once)
static SCRFD_SESSION_CREATES: AtomicI64 = AtomicI64::new(0);
static ARCFACE_SESSION_CREATES: AtomicI64 = AtomicI64::new(0);
static DINO_SESSION_CREATES: AtomicI64 = AtomicI64::new(0);

// Android AI Engine - wraps FacePipeline + VisualScanner with memory/timing instrumentation

/// Get current process RSS in KB using getrusage.
///
/// Returns 0 if the call fails (e.g., on unsupported platforms).
#[cfg(target_os = "linux")]
fn get_memory_rss_kb() -> usize {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } == 0 {
        // Linux: ru_maxrss is in KB
        usage.ru_maxrss as usize
    } else {
        0
    }
}

#[cfg(target_os = "macos")]
fn get_memory_rss_kb() -> usize {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } == 0 {
        // macOS: ru_maxrss is in bytes, convert to KB
        (usage.ru_maxrss / 1024) as usize
    } else {
        0
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn get_memory_rss_kb() -> usize {
    0
}

/// Get current process RSS in KB - exposed as public FFI function.
#[no_mangle]
pub extern "C" fn pf_get_memory_rss_kb() -> usize {
    get_memory_rss_kb()
}

use std::ffi::{CStr, CString};

/// Initialization log for debugging using Android logcat
static INIT_LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());

// Android log levels
const ANDROID_LOG_INFO: libc::c_int = 4;
const ANDROID_LOG_ERROR: libc::c_int = 3;
const ANDROID_LOG_WARN: libc::c_int = 5;
const ANDROID_LOG_DEBUG: libc::c_int = 7;
const ANDROID_LOG_VERBOSE: libc::c_int = 2;

extern "C" {
    fn __android_log_write(level: libc::c_int, tag: *const libc::c_char, msg: *const libc::c_char) -> libc::c_int;
}

fn log_init(msg: &str) {
    let mut log = INIT_LOG.lock().unwrap();
    log.push(msg.to_string());

    let tag = CString::new("pf_ffi").unwrap();
    let c_msg = CString::new(msg).unwrap();
    unsafe {
        __android_log_write(ANDROID_LOG_INFO, tag.as_ptr(), c_msg.as_ptr());
    }
}

fn init_android_logger() {
    // CRITICAL: Log BEFORE any tracing setup to verify this function runs
    let tag = CString::new("pf_ffi").unwrap();
    let c_msg = CString::new("init_android_logger: START").unwrap();
    unsafe {
        __android_log_write(ANDROID_LOG_INFO, tag.as_ptr(), c_msg.as_ptr());
    }

    // Set up tracing subscriber with custom Android logcat layer
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};
    use std::sync::Mutex;

    static ANDROID_LAYER: Mutex<Option<AndroidLogLayer>> = Mutex::new(None);

    let android_layer = AndroidLogLayer {
        tag: CString::new("pf_ffi").unwrap(),
    };

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,pf_application=debug,pf_visual_scan_v2=debug"));

    tracing_subscriber::registry()
        .with(filter)
        .with(android_layer)
        .init();

    // Also set up env_logger for any code that uses the log crate
    let _ = env_logger::try_init();

    log_init("Tracing subscriber initialized");
}

// Custom tracing layer that writes to Android logcat
struct AndroidLogLayer {
    tag: CString,
}

impl<S> tracing_subscriber::layer::Layer<S> for AndroidLogLayer
where
    S: tracing::Subscriber,
{
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        let mut msg = String::new();

        // Get location info
        if let Some(file) = event.metadata().file() {
            msg.push_str(file);
            if let Some(line) = event.metadata().line() {
                msg.push_str(&format!(":{}", line));
            }
            msg.push_str(" ");
        }

        // Get target/module
        let target = event.metadata().target();
        if !target.is_empty() {
            msg.push_str("[");
            msg.push_str(target);
            msg.push_str("] ");
        }

        // Extract message from event fields
        struct MsgExtractor<'a>(&'a mut String);
        impl<'a> tracing::field::Visit for MsgExtractor<'a> {
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                if field.name() == "message" {
                    self.0.push_str(value);
                }
            }
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    use std::fmt::Write;
                    let _ = write!(self.0, "{:?}", value);
                }
            }
        }

        let mut message = String::new();
        let mut extractor = MsgExtractor(&mut message);
        event.record(&mut extractor);
        msg.push_str(&message);

        // Write to Android logcat
        let level = match event.metadata().level() {
            &tracing::Level::ERROR => ANDROID_LOG_ERROR,
            &tracing::Level::WARN => ANDROID_LOG_WARN,
            &tracing::Level::INFO => ANDROID_LOG_INFO,
            &tracing::Level::DEBUG => ANDROID_LOG_DEBUG,
            &tracing::Level::TRACE => ANDROID_LOG_VERBOSE,
            _ => ANDROID_LOG_INFO,
        };

        let c_msg = CString::new(msg).unwrap();
        unsafe {
            __android_log_write(level, self.tag.as_ptr(), c_msg.as_ptr());
        }
    }
}

fn null_str(s: &str) -> *mut c_char {
    CString::new(s).expect("CString::new failed").into_raw()
}

fn json_ok<T: Serialize>(data: &T) -> *mut c_char {
    match serde_json::to_string(data) {
        Ok(s) => null_str(&s),
        Err(e) => null_str(&format!("{{\"error\":\"{}\"}}", e)),
    }
}

// ============================================================================
// Initialization
// ============================================================================

#[no_mangle]
pub extern "C" fn pf_init() -> i32 {
    init_android_logger();
    log_init("====== pf_init START =====");

    // Force reset IS_SCANNING on init - fixes stale state from previous runs
    {
        let mut is_scanning = IS_SCANNING.lock().unwrap();
        *is_scanning = Some(false);
        log_init("[INIT] IS_SCANNING reset to false");
    }

    log_init("Creating tokio runtime...");
    let runtime = match Runtime::new() {
        Ok(rt) => {
            log_init("Tokio runtime created");
            Arc::new(rt)
        }
        Err(e) => {
            log_init(&format!("FAILED to create tokio runtime: {}", e));
            return -1;
        }
    };
    {
        let mut rt = RUNTIME.lock().unwrap();
        *rt = Some(runtime.clone());
    }

    let models_dir = std::env::var("PF_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/data/local/tmp/photofinder/models"));

    log_init(&format!("Models dir: {:?}", models_dir));

    let data_dir = {
        let static_dir = DATA_DIR.lock().unwrap();
        if let Some(ref dir) = *static_dir {
            dir.clone()
        } else {
            std::env::var("PF_DATA_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| {
                    dirs::data_local_dir()
                        .unwrap_or_else(|| PathBuf::from("/data/user/0/app.photofinder.flutter_android/files"))
                        .join("PhotoFinderNext")
                })
        }
    };
    log_init(&format!("Data dir: {:?}", data_dir));

    let db_path = data_dir.join("photofinder.db");
    log_init(&format!("DB path: {:?}", db_path));

    // Create directories if needed
    std::fs::create_dir_all(&data_dir).ok();
    std::fs::create_dir_all(data_dir.join("index")).ok();

    log_init("Opening database...");
    let migrations = pf_database::builtin_migrations()
        .add(pf_database::Migration {
            id: "023_license",
            sql: include_str!("../../../../crates/pf_database/migrations/023_license.sql"),
        })
        .add(pf_database::Migration {
            id: "024_account_session",
            sql: include_str!("../../../../crates/pf_database/migrations/024_account_session.sql"),
        });
    let db = match Database::open(&db_path, migrations) {
        Ok(db) => {
            log_init("Database opened successfully");
            Arc::new(db)
        }
        Err(e) => {
            log_init(&format!("Failed to open database: {}", e));
            return -1;
        }
    };

    {
        let mut db_lock = DB.lock().unwrap();
        *db_lock = Some(db.clone());
    }
    log_init("DB stored in global state");

    // Initialize license manager
    log_init("Initializing license manager...");
    let storage = LicenseStorage::new(db.clone());
    let license_manager = Arc::new(LicenseManager::new(Arc::new(storage)));
    license_manager.initialize();
    {
        let mut lm = LICENSE_MANAGER.lock().unwrap();
        *lm = Some(license_manager);
    }
    log_init("License manager initialized");

    // Verify database schema has asset_id column
    if let Ok(conn) = db.connection() {
        match conn.query_row(
            "SELECT asset_id FROM images LIMIT 1",
            [],
            |row| row.get::<_, Option<String>>(0),
        ) {
            Ok(asset_id) => {
                log_init(&format!("[DB_SCHEMA] asset_id column exists, sample value: {:?}", asset_id));
            }
            Err(e) => {
                log_init(&format!("[DB_SCHEMA] asset_id column check failed: {} (may not exist yet)", e));
            }
        }
    }

    // Load FacePipeline (Android uses default ONNX threading)
    log_init("Loading FacePipeline...");
    let scrfd_path = models_dir.join("scrfd_500m_bnkps.onnx");
    let arcface_path = models_dir.join("w600k_r50.onnx");

    let detector = match pf_ai::ScrfdDetector::load(&scrfd_path) {
        Ok(d) => {
            SCRFD_SESSION_CREATES.fetch_add(1, Ordering::SeqCst);
            log_init(&format!("[AI_INIT] SCRFD session created (total creates: {})", SCRFD_SESSION_CREATES.load(Ordering::SeqCst)));
            d
        }
        Err(e) => {
            log_init(&format!("ScrfdDetector load failed: {}", e));
            return -1;
        }
    };
    let embedder = match pf_ai::ArcFaceEmbedder::load(&arcface_path) {
        Ok(e) => {
            ARCFACE_SESSION_CREATES.fetch_add(1, Ordering::SeqCst);
            log_init(&format!("[AI_INIT] ArcFace session created (total creates: {})", ARCFACE_SESSION_CREATES.load(Ordering::SeqCst)));
            e
        }
        Err(e) => {
            log_init(&format!("ArcFaceEmbedder load failed: {}", e));
            return -1;
        }
    };
    let aligner: Arc<dyn pf_ai::FaceAligner> = Arc::new(pf_ai::SimpleAligner::new());
    let quality_filter = pf_ai::QualityFilter::from_config(0.30, 20, 0.45, 30.0);

    let face_pipeline = FacePipeline::new(detector, aligner, embedder, quality_filter);
    log_init("FacePipeline assembled successfully");

    let fp = Arc::new(face_pipeline);
    {
        let mut pipeline_lock = FACE_PIPELINE.lock().unwrap();
        *pipeline_lock = Some(fp.clone());
    }
    log_init("FacePipeline loaded successfully");

    // Build HNSW index
    let index_dir = data_dir.join("index");
    std::fs::create_dir_all(&index_dir).ok();
    let face_index_path = index_dir.join("face.hnsw");
    let face_index = match build_hnsw_index(512, &face_index_path, "face") {
        Ok(idx) => idx,
        Err(e) => {
            log_init(&format!("HNSW index build failed: {}", e));
            return -1;
        }
    };
    {
        let mut idx_lock = FACE_INDEX.lock().unwrap();
        *idx_lock = Some(face_index);
    }
    log_init("Face HNSW index ready");

    // Load VisualScannerV2
    log_init("Loading VisualScannerV2...");
    let dinov2_path = models_dir.join("dinov2_vits14_with_patch.onnx");
    let mut config = VisualScanV2Config::default();

    // ============================================================
    // FIXED: Always use dual scale [0.25, 0.5]
    // - scale=0.25: ~16 ROIs (for detailed features)
    // - scale=0.5: ~4 ROIs (for overview features)
    // Total: ~20 ROIs per image
    config.scales = vec![0.5]; // Android: single scale for performance
    config.stride_ratio = 0.25; // 4 ROIs per image (2x2 grid at scale=0.5)
    config.dino_session_count = 2; // Android: optimal parallel sessions
    log_init(&format!("[AI_INIT] Visual scales: {:?}, stride_ratio: {}", config.scales, config.stride_ratio));
    // Note: Desktop uses config.visual_scan.* values. Android matches defaults.
    let visual_scanner = match pf_application::visual_scan_v2::scanner::VisualScannerV2::new(
        config.clone(),
        &dinov2_path,
    ) {
        Ok(scanner) => {
            DINO_SESSION_CREATES.fetch_add(1, Ordering::SeqCst);
            log_init(&format!("[AI_INIT] DINO session created (total creates: {})", DINO_SESSION_CREATES.load(Ordering::SeqCst)));
            scanner
        }
        Err(e) => {
            log_init(&format!("VisualScannerV2 load failed: {}", e));
            return -1;
        }
    };
    {
        let mut vs = VISUAL_SCANNER.lock().unwrap();
        *vs = Some(VisualScannerState {
            scanner: std::sync::Arc::new(std::sync::Mutex::new(visual_scanner)),
            config,
            model_path: dinov2_path,
        });
    }
    log_init("VisualScannerV2 loaded successfully");

    // Build Visual V2 HNSW index for object search
    let visual_index_path = index_dir.join("visual_v2.hnsw");
    let visual_index = match build_hnsw_index(384, &visual_index_path, "visual_v2") {
        Ok(idx) => idx,
        Err(e) => {
            log_init(&format!("[VISUAL_V2] HNSW index build failed: {}", e));
            Arc::new(HnswIndex::new(384, &index_dir, "visual_v2"))
        }
    };

    // Always rebuild HNSW index from database at startup (Option 3: ensure sync)
    log_init("[VISUAL_V2] Syncing from database at startup (always rebuild)...");
    let scan_version = pf_application::visual_scan_v2::config::VisualScanV2Config::default().scan_version;
    log_init(&format!("[VISUAL_V2] Using scan_version={}", scan_version));
    // Clear existing index before rebuilding
    log_init("[VISUAL_V2] Clearing existing HNSW index...");
    if let Err(e) = visual_index.clear() {
        log_init(&format!("[VISUAL_V2] Clear error: {}", e));
    }
    match db.connection() {
        Ok(conn) => {
            if let Ok(count) = pf_application::visual_scan_v2::database::get_feature_count_by_version(&conn, &scan_version) {
                log_init(&format!("[VISUAL_V2] Features in DB with scan_version={}: {}", scan_version, count));
            }
            match pf_application::visual_scan_v2::database::bulk_load_to_hnsw(
                &conn,
                &scan_version,
                visual_index.as_ref() as &dyn pf_vector::VectorIndex,
                1000,
            ) {
                Ok(count) => {
                    log_init(&format!("[VISUAL_V2] Loaded {} features from DB into HNSW", count));
                    if let Err(e) = visual_index.save() {
                        log_init(&format!("[VISUAL_V2] Failed to save index: {}", e));
                    } else {
                        log_init("[VISUAL_V2] Index saved to disk");
                    }
                }
                Err(e) => {
                    log_init(&format!("[VISUAL_V2] Bulk load error: {}", e));
                }
            }
        }
        Err(e) => {
            log_init(&format!("[VISUAL_V2] DB connection error: {}", e));
        }
    }

    {
        let mut idx_lock = VISUAL_V2_INDEX.lock().unwrap();
        *idx_lock = Some(visual_index);
    }
    log_init("[VISUAL_V2] HNSW index ready");

    log_init("====== pf_init COMPLETE =====");
    0
}

fn build_hnsw_index(dim: usize, path: &Path, basename: &str) -> Result<Arc<HnswIndex>, String> {
    let parent = path.parent().unwrap_or(std::path::Path::new("."));
    std::fs::create_dir_all(parent).ok();

    let idx = HnswIndex::new(dim, parent, basename);
    // Try to load existing index - load() uses internal data_dir/basename
    match idx.load() {
        Ok(_) => {
            log_init(&format!("HNSW index loaded from {:?}", path));
        }
        Err(_) => {
            log_init("HNSW index not found, starting fresh");
        }
    }
    Ok(Arc::new(idx))
}

// ============================================================================
// Photo Add
// ============================================================================

#[no_mangle]
pub extern "C" fn pf_add_photo_full(
    bytes: *const u8,
    len: usize,
    asset_id: *const c_char,
) -> i32 {
    log_init(&format!("[SCAN] ============== pf_add_photo_full START =============="));

    if bytes.is_null() || len == 0 {
        log_init("[SCAN] pf_add_photo_full: null bytes");
        return -1;
    }
    if asset_id.is_null() {
        log_init("[SCAN] pf_add_photo_full: null asset_id");
        return -1;
    }

    let asset_id_str = unsafe { std::ffi::CStr::from_ptr(asset_id) }
        .to_str()
        .unwrap_or("unknown")
        .to_string();

    log_init(&format!("[SCAN] asset_id={}, len={}", asset_id_str, len));

    let data = unsafe { std::slice::from_raw_parts(bytes, len) };

    let mut hasher = Sha256::new();
    hasher.update(data);
    let hash = format!("{:x}", hasher.finalize());

    let img_data = match ImageData::from_bytes(data) {
        Ok(img) => {
            log_init(&format!("[SCAN] ImageData loaded: {}x{}", img.width(), img.height()));
            img
        }
        Err(e) => {
            log_init(&format!("[SCAN] ImageData decode failed: {}", e));
            return -1;
        }
    };

    let (width, height) = (img_data.width(), img_data.height());

    let (photo_id, needs_processing) = {
        let db_lock = DB.lock().unwrap();
        let db = match db_lock.as_ref() {
            Some(db) => db,
            None => {
                log_init("[SCAN] DB not initialized");
                return -1;
            }
        };
        let conn = match db.connection() {
            Ok(conn) => conn,
            Err(e) => {
                log_init(&format!("[SCAN] failed to get connection: {}", e));
                return -1;
            }
        };

        // Look up by asset_id column (new) or path column (old entries for backward compat)
        // New entries: asset_id column = asset_id, path column = file_path or asset_id
        // Old entries: path column = asset_id (before the fix)
        let existing_id: Option<i64> = conn
            .query_row(
                "SELECT id FROM images WHERE asset_id = ? OR (asset_id IS NULL AND path = ?)",
                rusqlite::params![&asset_id_str, &asset_id_str],
                |row| row.get(0),
            )
            .ok();

        if let Some(existing_id) = existing_id {
            let face_count: i64 = conn
                .query_row("SELECT COUNT(*) FROM faces WHERE image_id = ?", [existing_id], |row| row.get(0))
                .unwrap_or(0);
            if face_count > 0 {
                log_init(&format!("[SCAN] photo already processed (id={}, faces={}), skipping", existing_id, face_count));
                return 0;
            }
            log_init(&format!("[SCAN] photo exists (id={}) but needs reprocessing", existing_id));
            (existing_id, true)
        } else {
            // For addPhotoFullFFI (bytes-based), we don't have file_path, so store asset_id in both
            // This maintains backward compat while adding asset_id column
            match conn.execute(
                "INSERT INTO images (path, asset_id, hash, size, modified_time, width, height, scan_status) VALUES (?, ?, ?, ?, 0, ?, ?, 'pending')",
                rusqlite::params![&asset_id_str, &asset_id_str, hash, len as i64, width, height],
            ) {
                Ok(_) => {
                    let id = conn.last_insert_rowid();
                    log_init(&format!("[SCAN] inserted new photo id={} asset_id={}", id, asset_id_str));
                    (id, true)
                }
                Err(e) => {
                    log_init(&format!("[SCAN] insert failed: {}", e));
                    return -1;
                }
            }
        }
    };

    log_init(&format!("[SCAN] photo_id={}, needs_processing={}", photo_id, needs_processing));

    let detected_faces_count: usize = {
        let face_pipeline = FACE_PIPELINE.lock().unwrap();
        if let Some(ref pipeline) = *face_pipeline {
            log_init("[SCAN] Processing with FacePipeline...");
            let runtime = RUNTIME.lock().unwrap();
            if let Some(ref rt) = *runtime {
                let faces = rt.block_on(pipeline.process(&img_data));
                match faces {
                    Ok(face_features) => {
                        log_init(&format!("[MODEL] Detected {} faces", face_features.len()));
                        if !face_features.is_empty() {
                            save_faces_to_db(photo_id, &face_features);
                        }
                        face_features.len()
                    }
                    Err(e) => {
                        log_init(&format!("[MODEL] FacePipeline error: {}", e));
                        0
                    }
                }
            } else {
                log_init("[SCAN] No tokio runtime available");
                0
            }
        } else {
            log_init("[SCAN] FacePipeline not available, skipping AI processing");
            0
        }
    };

    let detected_objects_count = process_visual_scan(photo_id, &data, asset_id_str.clone());

    log_init(&format!("[SCAN] pf_add_photo_full COMPLETE (faces={}, objects={})", detected_faces_count, detected_objects_count));
    0
}

#[no_mangle]
pub extern "C" fn pf_add_photo_from_path(
    file_path: *const c_char,
    asset_id: *const c_char,
) -> i32 {
    log_init(&format!("[SCAN] ============== pf_add_photo_from_path START =============="));

    if file_path.is_null() {
        log_init("[SCAN] pf_add_photo_from_path: null file_path");
        return -1;
    }
    if asset_id.is_null() {
        log_init("[SCAN] pf_add_photo_from_path: null asset_id");
        return -1;
    }

    let file_path_str = unsafe { std::ffi::CStr::from_ptr(file_path) }
        .to_str()
        .unwrap_or("")
        .to_string();
    let asset_id_str = unsafe { std::ffi::CStr::from_ptr(asset_id) }
        .to_str()
        .unwrap_or("unknown")
        .to_string();

    log_init(&format!("[SCAN] pf_add_photo_from_path: path={}, asset_id={}", file_path_str, asset_id_str));

    let file_data = match std::fs::read(&file_path_str) {
        Ok(data) => {
            log_init(&format!("[SCAN] Read file: {} bytes", data.len()));
            data
        }
        Err(e) => {
            log_init(&format!("[SCAN] Failed to read file: {}", e));
            return -1;
        }
    };

    let hash = {
        let mut hasher = Sha256::new();
        hasher.update(&file_data);
        format!("{:x}", hasher.finalize())
    };

    let file_path_buf = PathBuf::from(&file_path_str);
    let mut image_from_bytes = false;
    let img_data = match ImageData::from_file(&file_path_buf) {
        Ok(img) => {
            log_init(&format!("[SCAN] ImageData from file: {}x{}", img.width(), img.height()));
            img
        }
        Err(e) => {
            log_init(&format!("[SCAN] ImageData from file failed: {}, trying bytes", e));
            image_from_bytes = true;
            match ImageData::from_bytes(&file_data) {
                Ok(img) => {
                    log_init(&format!("[SCAN] ImageData from bytes succeeded: {}x{}", img.width(), img.height()));
                    img
                }
                Err(e2) => {
                    log_init(&format!("[SCAN] ImageData from bytes also failed: {}", e2));
                    return -1;
                }
            }
        }
    };

    let (width, height) = (img_data.width(), img_data.height());

    let (photo_id, _needs_processing) = {
        let db_lock = DB.lock().unwrap();
        let db = match db_lock.as_ref() {
            Some(db) => db,
            None => {
                log_init("[SCAN] DB not initialized");
                return -1;
            }
        };
        let conn = match db.connection() {
            Ok(conn) => conn,
            Err(e) => {
                log_init(&format!("[SCAN] failed to get connection: {}", e));
                return -1;
            }
        };

        // Look up by asset_id (not path) since we now store asset_id separately
        let existing_id: Option<i64> = conn
            .query_row("SELECT id FROM images WHERE asset_id = ?", [&asset_id_str], |row| row.get(0))
            .ok();

        if let Some(existing_id) = existing_id {
            log_init(&format!("[SCAN] photo with asset_id {} already exists, id={}", asset_id_str, existing_id));
            (existing_id, false)
        } else {
            // Insert with file_path in 'path' column and asset_id in 'asset_id' column
            match conn.execute(
                "INSERT INTO images (path, asset_id, hash, size, modified_time, width, height, scan_status) VALUES (?, ?, ?, ?, 0, ?, ?, 'pending')",
                rusqlite::params![file_path_str, asset_id_str, hash, file_data.len() as i64, width, height],
            ) {
                Ok(_) => {
                    let id = conn.last_insert_rowid();
                    log_init(&format!("[SCAN] inserted new photo id={} asset_id={} path={}", id, asset_id_str, file_path_str));
                    (id, true)
                }
                Err(e) => {
                    log_init(&format!("[SCAN] insert failed: {}", e));
                    return -1;
                }
            }
        }
    };

    log_init(&format!("[SCAN] photo_id={}", photo_id));

    let detected_faces_count: usize = {
        let face_pipeline = FACE_PIPELINE.lock().unwrap();
        if let Some(ref pipeline) = *face_pipeline {
            log_init("[SCAN] Processing with FacePipeline...");
            let runtime = RUNTIME.lock().unwrap();
            if let Some(ref rt) = *runtime {
                let faces = rt.block_on(pipeline.process(&img_data));
                match faces {
                    Ok(face_features) => {
                        log_init(&format!("[MODEL] Detected {} faces", face_features.len()));
                        if !face_features.is_empty() {
                            save_faces_to_db(photo_id, &face_features);
                        }
                        face_features.len()
                    }
                    Err(e) => {
                        log_init(&format!("[MODEL] FacePipeline error: {}", e));
                        0
                    }
                }
            } else {
                0
            }
        } else {
            0
        }
    };

    let detected_objects_count = if image_from_bytes {
        process_visual_scan_from_bytes(photo_id, &file_data, asset_id_str.clone())
    } else {
        process_visual_scan_from_path(photo_id, &file_path_str, asset_id_str.clone())
    };

    log_init(&format!("[SCAN] pf_add_photo_from_path COMPLETE (faces={}, objects={})", detected_faces_count, detected_objects_count));
    0
}

fn save_faces_to_db(photo_id: i64, face_features: &[pf_ai::FaceFeature]) {
    let db_lock = DB.lock().unwrap();
    let db = match db_lock.as_ref() {
        Some(db) => db,
        None => {
            log_init("[FACE] DB not initialized");
            return;
        }
    };
    let conn = match db.connection() {
        Ok(conn) => conn,
        Err(e) => {
            log_init(&format!("[FACE] connection error: {}", e));
            return;
        }
    };

    let face_index = FACE_INDEX.lock().unwrap();

    for (i, face) in face_features.iter().enumerate() {
        let cx = (face.detection.bbox.x + face.detection.bbox.w / 2.0) as i32;
        let cy = (face.detection.bbox.y + face.detection.bbox.h / 2.0) as i32;
        let h = ((cx as u64) << 32) ^ (cy as u64 ^ photo_id as u64);
        let vector_id = (h & 0x7FFF_FFFF_FFFF_FFFF) as i64;

        if let Some(ref idx) = *face_index {
            let embedding_values = face.embedding.values.as_slice();
            if let Err(e) = idx.insert(vector_id, embedding_values) {
                log_init(&format!("[FACE] HNSW insert failed for face {}: {}", i, e));
            } else {
                log_init(&format!("[FACE] HNSW insert ok: vector_id={}", vector_id));
            }
        }

        let (yaw, pitch, roll) = match face.yaw_pitch_roll {
            Some((y, p, r)) => (Some(y), Some(p), Some(r)),
            None => (None, None, None),
        };

        let quality_score = (face.blur_score * 0.3 + face.pose_score * 0.3 + face.face_area_score * 0.4).min(1.0);
        let keypoints_json = serde_json::to_string(&face.detection.keypoints).ok();

        let insert_result = conn.execute(
            "INSERT INTO faces (image_id, bbox_x, bbox_y, bbox_w, bbox_h,
             detector_score, detector_model, keypoints_json, yaw, pitch, roll,
             quality_score, blur_score, pose_score, face_area_score,
             embedding_model, model_version, vector_id,
             detector_origin, config_roll_correct_enabled, config_roll_correct_threshold)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21)",
            rusqlite::params![
                photo_id,
                face.detection.bbox.x,
                face.detection.bbox.y,
                face.detection.bbox.w,
                face.detection.bbox.h,
                face.detection.score,
                "scrfd-500m-bnkps",
                keypoints_json,
                yaw,
                pitch,
                roll,
                quality_score,
                face.blur_score,
                face.pose_score,
                face.face_area_score,
                "arcface",
                face.embedding.model.as_str(),
                vector_id,
                "Primary",
                face.config_roll_correct_enabled as i32,
                face.config_roll_correct_threshold,
            ],
        );

        match insert_result {
            Ok(_) => {
                let face_id = conn.last_insert_rowid();
                log_init(&format!("[FACE] Saved face[{}]: id={}, vector_id={}", i, face_id, vector_id));

                let embedding_bytes: Vec<u8> = face.embedding.values.as_slice()
                    .iter()
                    .flat_map(|&v| v.to_le_bytes())
                    .collect();
                let dim = embedding_bytes.len() / 4;

                let embed_result = conn.execute(
                    "INSERT INTO face_embeddings (face_id, model_name, model_version, dimension, vector, normalized)
                     VALUES (?1, 'arcface', 'w600k_r50_v1', ?2, ?3, 1)",
                    rusqlite::params![face_id, dim as i32, embedding_bytes],
                );
                match embed_result {
                    Ok(_) => {
                        log_init(&format!("[FACE] Saved embedding to face_embeddings: face_id={}", face_id));
                    }
                    Err(e) => {
                        log_init(&format!("[FACE] Failed to save embedding: {}", e));
                    }
                }
            }
            Err(e) => {
                log_init(&format!("[FACE] Failed to save face[{}]: {}", i, e));
            }
        }
    }
}

fn process_visual_scan(photo_id: i64, image_bytes: &[u8], _asset_id: String) -> usize {
    let mut scanner_guard = match VISUAL_SCANNER.lock() {
        Ok(guard) => guard,
        Err(e) => {
            log_init(&format!("[VISUAL] Failed to lock VISUAL_SCANNER: {}", e));
            return 0;
        }
    };

    let scanner = match scanner_guard.as_mut() {
        Some(state) => &mut state.scanner,
        None => {
            log_init("[VISUAL] VisualScannerV2 not initialized, skipping");
            return 0;
        }
    };

    let temp_dir = std::env::var("PF_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let data_dir_guard = DATA_DIR.lock().unwrap();
            data_dir_guard.clone().unwrap_or_else(|| std::env::temp_dir())
        });
    let temp_file = temp_dir.join(format!("visual_scan_{}.jpg", photo_id));

    log_init(&format!("[VISUAL] Writing temp file: {:?}", temp_file));
    if let Err(e) = std::fs::write(&temp_file, image_bytes) {
        log_init(&format!("[VISUAL] Failed to write temp file: {}", e));
        return 0;
    }

    let result = (|| {
        let db_lock = DB.lock().unwrap();
        let db = match db_lock.as_ref() {
            Some(db) => db,
            None => {
                log_init("[VISUAL] DB not initialized");
                return 0usize;
            }
        };
        let conn = match db.connection() {
            Ok(conn) => conn,
            Err(e) => {
                log_init(&format!("[VISUAL] connection error: {}", e));
                return 0usize;
            }
        };

        match (&mut *scanner).lock().unwrap().scan_image(&conn, photo_id, &temp_file) {
            Ok(result) => {
                log_init(&format!(
                    "[VISUAL] Scanned: raw_025={}, raw_050={}, selected_total={}, time_ms={}",
                    result.raw_roi_count_025,
                    result.raw_roi_count_05,
                    result.total_feature_count,
                    result.total_time_ms
                ));
                result.total_feature_count
            }
            Err(e) => {
                log_init(&format!("[VISUAL] scan_image error: {}", e));
                0usize
            }
        }
    })();

    std::fs::remove_file(&temp_file).ok();
    result
}

fn process_visual_scan_from_path(photo_id: i64, file_path: &str, _asset_id: String) -> usize {
    log_init(&format!("[VISUAL] Scanning from path: {}", file_path));

    let file_path_owned = file_path.to_string();
    log_init("[VISUAL] file_path_owned created");
    let conn_params: PathBuf;
    log_init("[VISUAL] conn_params declared");
    let scanner_arc: std::sync::Arc<std::sync::Mutex<pf_application::visual_scan_v2::scanner::VisualScannerV2>>;
    log_init("[VISUAL] scanner_arc declared");

    // ALL mutexes must be released before block_on
    log_init("[VISUAL] About to lock VISUAL_SCANNER...");
    {
        let scanner_guard = match VISUAL_SCANNER.lock() {
            Ok(guard) => guard,
            Err(e) => {
                log_init(&format!("[VISUAL] Failed to lock VISUAL_SCANNER: {}", e));
                return 0;
            }
        };

        let state = match scanner_guard.as_ref() {
            Some(state) => state,
            None => {
                log_init("[VISUAL] VisualScannerV2 not initialized, skipping");
                return 0;
            }
        };
        scanner_arc = std::sync::Arc::clone(&state.scanner);

        let db_lock = match DB.lock() {
            Ok(guard) => guard,
            Err(e) => {
                log_init(&format!("[VISUAL] Failed to lock DB: {}", e));
                return 0;
            }
        };
        log_init("[VISUAL] DB lock acquired");

        let db = match db_lock.as_ref() {
            Some(db) => db,
            None => {
                log_init("[VISUAL] DB not initialized");
                return 0;
            }
        };

        let conn = match db.connection() {
            Ok(conn) => conn,
            Err(e) => {
                log_init(&format!("[VISUAL] connection error: {}", e));
                return 0;
            }
        };
        log_init("[VISUAL] DB connection obtained");

        conn_params = conn.path().map(PathBuf::from).unwrap();
        log_init("[VISUAL] conn_params set from db path");
    }
    // All mutexes released here
    log_init("[VISUAL] All mutexes released, about to get runtime...");

    let rt = RUNTIME.lock().unwrap();
    log_init("[VISUAL] RUNTIME lock acquired");
    let result = if let Some(ref runtime) = *rt {
        log_init("[VISUAL] Runtime acquired, about to block_on...");
        // FIX: capture the result from block_on instead of discarding it
        runtime.block_on(async move {
            log_init("[VISUAL][block_on] Entered block_on");
            tokio::task::spawn_blocking(move || {
                let result = (|| {
                    log_init("[VISUAL][blocking] Opening connection...");
                    let conn = match rusqlite::Connection::open(&conn_params) {
                        Ok(c) => c,
                        Err(e) => {
                            log_init(&format!("[VISUAL][blocking] Connection open failed: {}", e));
                            return 0usize;
                        }
                    };

                    log_init("[VISUAL][blocking] Using pre-initialized scanner (no re-load)");
                    let path_buf = PathBuf::from(&file_path_owned);
                    log_init("[VISUAL][blocking] Calling scan_image...");
                    match scanner_arc.lock().unwrap().scan_image(&conn, photo_id, &path_buf) {
                        Ok(result) => {
                            log_init(&format!(
                                "[VISUAL] Scanned: raw_025={}, raw_050={}, selected_total={}, time_ms={}",
                                result.raw_roi_count_025,
                                result.raw_roi_count_05,
                                result.total_feature_count,
                                result.total_time_ms
                            ));
                            result.total_feature_count
                        }
                        Err(e) => {
                            log_init(&format!("[VISUAL] scan_image error: {}", e));
                            0usize
                        }
                    }
                })();
                result
            }).await.unwrap_or_else(|e| {
                log_init(&format!("[VISUAL] blocking task panicked: {}", e));
                0usize
            })
        })
    } else {
        log_init("[VISUAL] Runtime not available");
        0
    };
    result
}

fn process_visual_scan_from_bytes(photo_id: i64, image_bytes: &[u8], _asset_id: String) -> usize {
    let temp_dir = std::env::var("PF_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let data_dir_guard = DATA_DIR.lock().unwrap();
            data_dir_guard.clone().unwrap_or_else(|| std::env::temp_dir())
        });
    let temp_file = temp_dir.join(format!("visual_scan_{}.jpg", photo_id));

    log_init(&format!("[VISUAL] Writing temp file: {:?}", temp_file));
    if let Err(e) = std::fs::write(&temp_file, image_bytes) {
        log_init(&format!("[VISUAL] Failed to write temp file: {}", e));
        return 0;
    }

    let temp_file_owned = temp_file.clone();
    let db_path: PathBuf;
    let scanner_arc: std::sync::Arc<std::sync::Mutex<pf_application::visual_scan_v2::scanner::VisualScannerV2>>;

    // ALL mutexes must be released before block_on
    {
        let scanner_guard = match VISUAL_SCANNER.lock() {
            Ok(guard) => guard,
            Err(e) => {
                log_init(&format!("[VISUAL] Failed to lock VISUAL_SCANNER: {}", e));
                std::fs::remove_file(&temp_file).ok();
                return 0;
            }
        };

        let state = match scanner_guard.as_ref() {
            Some(state) => state,
            None => {
                log_init("[VISUAL] VisualScannerV2 not initialized, skipping");
                std::fs::remove_file(&temp_file).ok();
                return 0;
            }
        };
        scanner_arc = std::sync::Arc::clone(&state.scanner);

        let db_lock = match DB.lock() {
            Ok(guard) => guard,
            Err(e) => {
                log_init(&format!("[VISUAL] Failed to lock DB: {}", e));
                std::fs::remove_file(&temp_file).ok();
                return 0;
            }
        };

        let db = match db_lock.as_ref() {
            Some(db) => db,
            None => {
                log_init("[VISUAL] DB not initialized");
                std::fs::remove_file(&temp_file).ok();
                return 0;
            }
        };

        let conn = match db.connection() {
            Ok(conn) => conn,
            Err(e) => {
                log_init(&format!("[VISUAL] connection error: {}", e));
                std::fs::remove_file(&temp_file).ok();
                return 0;
            }
        };

        db_path = conn.path().map(PathBuf::from).unwrap();
    }
    // All mutexes released here

    let rt = RUNTIME.lock().unwrap();
    let result = if let Some(ref runtime) = *rt {
        log_init("[VISUAL] Runtime acquired for bytes scan, about to block_on...");
        // FIX: capture result from block_on
        runtime.block_on(async move {
            tokio::task::spawn_blocking(move || {
                let result = (|| {
                    log_init("[VISUAL][blocking] Opening connection for bytes scan...");
                    let conn = match rusqlite::Connection::open(&db_path) {
                        Ok(c) => c,
                        Err(e) => {
                            log_init(&format!("[VISUAL][blocking] Connection open failed: {}", e));
                            return 0usize;
                        }
                    };

                    log_init("[VISUAL][blocking] Using pre-initialized scanner (no re-load)");

                    match scanner_arc.lock().unwrap().scan_image(&conn, photo_id, &temp_file_owned) {
                        Ok(result) => {
                            log_init(&format!(
                                "[VISUAL] Scanned from bytes: raw_025={}, raw_050={}, selected_total={}, time_ms={}",
                                result.raw_roi_count_025,
                                result.raw_roi_count_05,
                                result.total_feature_count,
                                result.total_time_ms
                            ));
                            result.total_feature_count
                        }
                        Err(e) => {
                            log_init(&format!("[VISUAL] scan_image error: {}", e));
                            0usize
                        }
                    }
                })();
                result
            }).await.unwrap_or_else(|e| {
                log_init(&format!("[VISUAL] blocking task panicked: {}", e));
                0usize
            })
        })
    } else {
        log_init("[VISUAL] Runtime not available for bytes scan");
        0
    };

    std::fs::remove_file(&temp_file).ok();
    result
}

// ============================================================================
// Status
// ============================================================================

#[no_mangle]
pub extern "C" fn pf_get_status() -> *mut c_char {
    let db_lock = DB.lock().unwrap();
    let db_available = db_lock.is_some();
    let visual_available = VISUAL_SCANNER.lock().unwrap().is_some();

    json_ok(&serde_json::json!({
        "db_initialized": db_available,
        "face_pipeline_available": FACE_PIPELINE.lock().unwrap().is_some(),
        "visual_scanner_available": visual_available,
    }))
}

// ============================================================================
// Clustering
// ============================================================================

#[no_mangle]
pub extern "C" fn pf_cluster_all() -> *mut c_char {
    log_init("pf_cluster_all called");
    json_ok(&serde_json::json!({
        "assigned": 0,
        "created": 0,
        "failed": 0,
    }))
}

// ============================================================================
// Clustering - simplified implementation for Android
// ============================================================================

/// Cosine similarity between two vectors
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Minimum similarity threshold for visual/object search results (50%)
const VISUAL_SEARCH_MIN_SIMILARITY: f32 = 0.5;

/// Trigger incremental clustering if enough faces have been processed since last clustering
fn trigger_incremental_clustering() {
    let count = FACES_SINCE_LAST_CLUSTER.fetch_add(1, Ordering::SeqCst) + 1;
    if count >= INCREMENTAL_CLUSTER_BATCH_SIZE {
        // Reset counter
        FACES_SINCE_LAST_CLUSTER.store(0, Ordering::SeqCst);
        // Check if clustering is already running
        {
            let running = CLUSTER_RUNNING.lock().unwrap();
            if *running.as_ref().unwrap_or(&false) {
                log_init("[INCREMENTAL_CLUSTER] clustering already running, skipping");
                return;
            }
        }
        // Mark as running
        {
            let mut running = CLUSTER_RUNNING.lock().unwrap();
            *running = Some(true);
        }
        log_init("[INCREMENTAL_CLUSTER] triggering incremental clustering...");
        // Run clustering in background (fire and forget)
        std::thread::spawn(move || {
            let result = do_cluster_all();
            log_init(&format!("[INCREMENTAL_CLUSTER] result: {}", result));
            // Mark as not running
            let mut running = CLUSTER_RUNNING.lock().unwrap();
            *running = Some(false);
        });
    }
}

/// Run face clustering on a background thread and store result
fn run_clustering_task(task_id: i64) {
    let result = do_cluster_all();
    let mut results = CLUSTER_RESULTS.lock().unwrap();
    if let Some(ref mut map) = *results {
        map.insert(task_id, result);
    }
}

/// Simplified face clustering:
/// - Find faces without person_id
/// - For each: search HNSW for similar faces
/// - If match found (cosine >= 0.30), assign to existing person
/// - Otherwise create new person
fn do_cluster_all() -> String {
    log_init("[CLUSTER] Starting face clustering");

    // Debug: check face count before clustering
    let db_lock = DB.lock().unwrap();
    let db = match db_lock.as_ref() {
        Some(db) => db,
        None => return r#"{"error":"DB not initialized"}"#.to_string(),
    };
    let conn = match db.connection() {
        Ok(c) => c,
        Err(e) => return format!(r#"{{"error":"DB connection error: {}"}}"#, e),
    };
    let face_index = FACE_INDEX.lock().unwrap();

    // Debug: check face counts
    let total_faces: i64 = conn.query_row("SELECT COUNT(*) FROM faces", [], |r| r.get(0)).unwrap_or(-1);
    let total_embeddings: i64 = conn.query_row("SELECT COUNT(*) FROM face_embeddings", [], |r| r.get(0)).unwrap_or(-1);
    let total_persons: i64 = conn.query_row("SELECT COUNT(*) FROM persons", [], |r| r.get(0)).unwrap_or(-1);
    log_init(&format!("[CLUSTER] DEBUG: total_faces={} total_embeddings={} total_persons={}", total_faces, total_embeddings, total_persons));

    // Get all faces without person_id
    let mut stmt = match conn.prepare(
        "SELECT f.id, f.image_id, fe.vector, f.quality_score
         FROM faces f
         JOIN face_embeddings fe ON fe.face_id = f.id
         WHERE f.person_id IS NULL
         ORDER BY f.id"
    ) {
        Ok(s) => s,
        Err(e) => return format!(r#"{{"error":"Query faces failed: {}"}}"#, e),
    };

    struct FaceRow {
        face_id: i64,
        image_id: i64,
        vector: Vec<f32>,
        quality: f32,
    }

    let unassigned: Vec<FaceRow> = stmt.query_map([], |row| {
        let blob: Vec<u8> = row.get(2)?;
        let vector: Vec<f32> = blob.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Ok(FaceRow {
            face_id: row.get(0)?,
            image_id: row.get(1)?,
            vector,
            quality: row.get::<_, Option<f32>>(3)?.unwrap_or(0.5),
        })
    }).map_err(|e| e.to_string())
      .ok()
      .map(|rows| rows.filter_map(|r| r.ok()).collect())
      .unwrap_or_default();

    if unassigned.is_empty() {
        log_init("[CLUSTER] No unassigned faces, clustering complete");
        return r#"{"status":"ok","assigned":0,"created":0,"skipped":0}"#.to_string();
    }

    log_init(&format!("[CLUSTER] Found {} unassigned faces", unassigned.len()));

    let mut assigned = 0;
    let mut created = 0;
    let mut skipped = 0;

    for face in &unassigned {
        // Skip low quality faces
        if face.quality < 0.3 {
            skipped += 1;
            continue;
        }

        // Search HNSW for similar faces
        let search_k = 5;
        let hnsw_hits = match &*face_index {
            Some(idx) => idx.search(&face.vector, search_k).unwrap_or_default(),
            None => {
                // No index - create new person directly
                match conn.execute(
                    "INSERT INTO persons (name, face_count) VALUES (NULL, 1)",
                    [],
                ) {
                    Ok(_) => {
                        let person_id = conn.last_insert_rowid();
                        conn.execute(
                            "UPDATE faces SET person_id = ?1 WHERE id = ?2",
                            rusqlite::params![person_id, face.face_id],
                        ).ok();
                        created += 1;
                    }
                    Err(e) => { log_init(&format!("[CLUSTER] insert person failed: {}", e)); }
                }
                continue;
            }
        };

        // Find best existing person match
        let mut best_person_id: Option<i64> = None;
        let mut best_score: f32 = 0.0;

        for hit in &hnsw_hits {
            // Look up this face's person_id and embedding using vector_id
            // First get the face_id from faces using vector_id, then get embedding
            if let Ok((pid, stored_emb)) = conn.query_row(
                "SELECT f.person_id, fe.vector FROM faces f
                 JOIN face_embeddings fe ON fe.face_id = f.id
                 WHERE f.vector_id = ?",
                rusqlite::params![hit.id],
                |row| {
                    let blob: Vec<u8> = row.get(1)?;
                    let emb: Vec<f32> = blob.chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();
                    Ok((row.get::<_, Option<i64>>(0)?, emb))
                },
            ) {
                if let Some(pid) = pid {
                    let score = cosine_sim(&face.vector, &stored_emb);
                    if score > best_score && score >= 0.30 {
                        best_score = score;
                        best_person_id = Some(pid);
                    }
                }
            }
        }

        if let Some(person_id) = best_person_id {
            // Assign to existing person
            if let Err(e) = conn.execute(
                "UPDATE faces SET person_id = ?1 WHERE id = ?2",
                rusqlite::params![person_id, face.face_id],
            ) {
                log_init(&format!("[CLUSTER] assign face failed: {}", e));
            } else {
                assigned += 1;
            }
        } else {
            // Create new person
            if let Ok(_) = conn.execute(
                "INSERT INTO persons (name, face_count) VALUES (NULL, 1)",
                [],
            ) {
                let person_id = conn.last_insert_rowid();
                if let Err(e) = conn.execute(
                    "UPDATE faces SET person_id = ?1 WHERE id = ?2",
                    rusqlite::params![person_id, face.face_id],
                ) {
                    log_init(&format!("[CLUSTER] assign face to new person failed: {}", e));
                } else {
                    created += 1;
                }
            }
        }
    }

    log_init(&format!(
        "[CLUSTER] Done: assigned={} created={} skipped={}",
        assigned, created, skipped
    ));
    format!(
        r#"{{"status":"ok","assigned":{},"created":{},"skipped":{}}}"#,
        assigned, created, skipped
    )
}

#[no_mangle]
pub extern "C" fn pf_cluster_all_async() -> i64 {
    let task_id = CLUSTER_TASK_COUNTER.fetch_add(1, Ordering::SeqCst);
    {
        let mut results = CLUSTER_RESULTS.lock().unwrap();
        if results.is_none() {
            *results = Some(HashMap::new());
        }
        // Mark as processing
        if let Some(ref mut map) = *results {
            map.insert(task_id, r#"{"status":"processing"}"#.to_string());
        }
    }

    std::thread::spawn(move || {
        run_clustering_task(task_id);
    });

    task_id
}

#[no_mangle]
pub extern "C" fn pf_cluster_check_result(task_id: i64) -> *mut c_char {
    let result_str = {
        let mut results = CLUSTER_RESULTS.lock().unwrap();
        if let Some(ref mut map) = *results {
            if let Some(result) = map.remove(&task_id) {
                Some(result)
            } else {
                None
            }
        } else {
            None
        }
    };

    if let Some(s) = result_str {
        return null_str(&s);
    }

    null_str(r#"{"status":"not_found"}"#)
}

// ============================================================================
// List Persons
// ============================================================================

#[no_mangle]
pub extern "C" fn pf_list_persons() -> *mut c_char {
    log_init("pf_list_persons called");

    let db_lock = DB.lock().unwrap();
    let db = match db_lock.as_ref() {
        Some(db) => db,
        None => return null_str(r#"{"error":"DB not initialized"}"#),
    };
    let conn = match db.connection() {
        Ok(c) => c,
        Err(e) => return null_str(&format!(r#"{{"error":"{}"}}"#, e)),
    };

    let mut stmt = match conn.prepare(
        "SELECT id, COALESCE(name, ''), face_count FROM persons ORDER BY face_count DESC, id"
    ) {
        Ok(s) => s,
        Err(e) => return null_str(&format!(r#"{{"error":"{}"}}"#, e)),
    };

    let mut persons: Vec<serde_json::Value> = Vec::new();
    let mut rows = match stmt.query([]) {
        Ok(r) => r,
        Err(e) => return null_str(&format!(r#"{{"error":"query failed: {}"}}"#, e)),
    };
    loop {
        let row = match rows.next() {
            Ok(r) => r,
            Err(_) => break,
        };
        let row = match row {
            Some(r) => r,
            None => break,
        };
        let id: i64 = match row.get(0) { Ok(v) => v, Err(_) => continue };
        let name: String = match row.get(1) { Ok(v) => v, Err(_) => continue };
        let face_count: i64 = match row.get(2) { Ok(v) => v, Err(_) => continue };
        persons.push(serde_json::json!({
            "id": id,
            "name": name,
            "face_count": face_count,
        }));
    }

    null_str(&serde_json::json!(persons).to_string())
}

// ============================================================================
// Search
// ============================================================================

#[no_mangle]
pub extern "C" fn pf_search_face(
    bytes: *const u8,
    len: usize,
    top_k: i32,
    offset: i32,
) -> *mut c_char {
    log_init("[FACE_SEARCH] pf_search_face called");

    if bytes.is_null() || len == 0 {
        return json_ok(&serde_json::json!({
            "results": [],
            "total": 0,
            "offset": offset,
            "limit": top_k,
        }));
    }

    let top_k = top_k.max(1) as usize;
    let offset = offset.max(0) as usize;
    let image_bytes = unsafe { std::slice::from_raw_parts(bytes, len) };

    match search_face_impl(image_bytes, top_k, offset) {
        Ok(json) => {
            log_init(&format!("[FACE_SEARCH] found {} results", top_k));
            null_str(&json)
        }
        Err(e) => {
            log_init(&format!("[FACE_SEARCH] error: {}", e));
            json_ok(&serde_json::json!({
                "results": [],
                "total": 0,
                "offset": offset,
                "limit": top_k,
            }))
        }
    }
}

#[no_mangle]
pub extern "C" fn pf_search_object(
    bytes: *const u8,
    len: usize,
    top_k: i32,
    offset: i32,
) -> *mut c_char {
    log_init("[OBJ_SEARCH] pf_search_object called");

    if bytes.is_null() || len == 0 {
        return json_ok(&serde_json::json!({
            "results": [],
            "total": 0,
            "offset": offset,
            "limit": top_k,
        }));
    }

    let top_k = top_k.max(1) as usize;
    let offset = offset.max(0) as usize;
    let image_bytes = unsafe { std::slice::from_raw_parts(bytes, len) };

    match search_object_impl(image_bytes, top_k, offset) {
        Ok(json) => {
            log_init(&format!("[OBJ_SEARCH] found {} results", top_k));
            null_str(&json)
        }
        Err(e) => {
            log_init(&format!("[OBJ_SEARCH] error: {}", e));
            json_ok(&serde_json::json!({
                "results": [],
                "total": 0,
                "offset": offset,
                "limit": top_k,
            }))
        }
    }
}

// ============================================================================
// Placeholder stubs for other functions
// ============================================================================

static DATA_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

#[no_mangle]
pub extern "C" fn pf_set_models_dir(path: *const c_char) -> i32 {
    if path.is_null() {
        log_init("pf_set_models_dir: null path");
        return -1;
    }
    let path_str = unsafe { std::ffi::CStr::from_ptr(path) }
        .to_str()
        .unwrap_or("");

    std::env::set_var("PF_MODELS_DIR", path_str);

    let parent = std::path::Path::new(path_str)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("/data/local/tmp"));

    std::env::set_var("PF_DATA_DIR", parent.to_str().unwrap_or(""));

    let mut data_dir = DATA_DIR.lock().unwrap();
    *data_dir = Some(parent.clone());

    log_init(&format!("pf_set_models_dir: PF_DATA_DIR set to: {}", parent.display()));
    0
}

#[no_mangle]
pub extern "C" fn pf_set_data_dir(path: *const c_char) -> i32 {
    if path.is_null() {
        log_init("pf_set_data_dir: null path");
        return -1;
    }
    let path_str = unsafe { std::ffi::CStr::from_ptr(path) }
        .to_str()
        .unwrap_or("");
    log_init(&format!("pf_set_data_dir: {}", path_str));

    std::env::set_var("PF_DATA_DIR", path_str);

    let mut data_dir = DATA_DIR.lock().unwrap();
    *data_dir = Some(PathBuf::from(path_str));

    0
}

/// Set visual scan configuration (scales, features_per_scale, etc.)
/// Format: {"scales":"0.25,0.5","features_per_scale":8}
#[no_mangle]
pub extern "C" fn pf_set_visual_config(config_json: *const c_char) -> i32 {
    if config_json.is_null() {
        log_init("[VISUAL] pf_set_visual_config: null config_json");
        return -1;
    }

    let config_str = unsafe { CStr::from_ptr(config_json).to_string_lossy() };
    log_init(&format!("[VISUAL] pf_set_visual_config: {}", config_str));

    // Parse JSON
    let json: serde_json::Value = match serde_json::from_str(&config_str) {
        Ok(v) => v,
        Err(e) => {
            log_init(&format!("[VISUAL] pf_set_visual_config: JSON parse error: {}", e));
            return -1;
        }
    };

    // Get current config
    let vs_lock = VISUAL_SCANNER.lock().unwrap();
    let vs_state = match vs_lock.as_ref() {
        Some(v) => v,
        None => {
            log_init("[VISUAL] pf_set_visual_config: VisualScanner not initialized");
            return -1;
        }
    };

    let mut config = vs_state.config.clone();

    // Update scales if provided
    if let Some(scales_val) = json.get("scales") {
        if let Some(scales_str) = scales_val.as_str() {
            let scales: Vec<f32> = scales_str.split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if !scales.is_empty() {
                config.scales = scales;
                log_init(&format!("[VISUAL] pf_set_visual_config: updated scales to {:?}", config.scales));
            }
        }
    }

    // Update features_per_scale if provided
    if let Some(fps_val) = json.get("features_per_scale") {
        if let Some(fps) = fps_val.as_u64() {
            config.features_per_scale = fps as usize;
            log_init(&format!("[VISUAL] pf_set_visual_config: updated features_per_scale to {}", config.features_per_scale));
        }
    }

    // Update the stored config (this won't affect already-created scanner, but will affect next init)
    // For now, just log the config - full dynamic reinit would require more changes
    drop(vs_lock);

    // Update the stored config in VISUAL_SCANNER
    let mut vs_lock = VISUAL_SCANNER.lock().unwrap();
    if let Some(vs) = vs_lock.as_mut() {
        vs.config = config.clone();
    }

    log_init(&format!("[VISUAL] pf_set_visual_config: complete, scales={:?}", config.scales));
    0
}

#[no_mangle]
pub extern "C" fn pf_clear_database() -> i32 {
    log_init("[CLEAR] pf_clear_database: starting...");

    let db_lock = DB.lock().unwrap();
    let db = match db_lock.as_ref() {
        Some(db) => db,
        None => {
            log_init("[CLEAR] DB not initialized");
            return -1;
        }
    };

    let conn = match db.connection() {
        Ok(conn) => conn,
        Err(e) => {
            log_init(&format!("[CLEAR] failed to get connection: {}", e));
            return -1;
        }
    };

    let _ = conn.execute("PRAGMA foreign_keys = OFF", []);

    let tables = [
        "DELETE FROM face_embeddings",
        "DELETE FROM visual_scan_v2_features",
        "DELETE FROM identity_shadow_records",
        "DELETE FROM objects",
        "DELETE FROM faces",
        "DELETE FROM images",
        "DELETE FROM duplicate_groups",
    ];

    for sql in tables {
        match conn.execute(sql, []) {
            Ok(count) => log_init(&format!("[CLEAR] '{}' -> {} rows", sql, count)),
            Err(e) => log_init(&format!("[CLEAR] '{}' failed: {}", sql, e)),
        }
    }

    let _ = conn.execute("PRAGMA foreign_keys = ON", []);

    log_init("[CLEAR] Resetting HNSW index...");
    drop(db_lock);

    // Reset IS_SCANNING flag
    {
        let mut is_scanning = IS_SCANNING.lock().unwrap();
        *is_scanning = Some(false);
        log_init("[CLEAR] IS_SCANNING reset to false");
    }

    let mut face_index = FACE_INDEX.lock().unwrap();
    if let Some(ref mut idx) = *face_index {
        if let Ok(_) = idx.clear() {
            log_init("[CLEAR] HNSW index cleared");
        }
    }

    log_init("[CLEAR] pf_clear_database: complete");
    0
}

#[no_mangle]
pub extern "C" fn pf_rebuild_hnsw_indices() -> i32 {
    log_init("[REBUILD] pf_rebuild_hnsw_indices called");

    let db_lock = DB.lock().unwrap();
    let db = match &*db_lock {
        Some(db) => db,
        None => {
            log_init("[REBUILD] DB not initialized");
            return -1;
        }
    };

    let conn = match db.connection() {
        Ok(c) => c,
        Err(e) => {
            log_init(&format!("[REBUILD] Failed to get DB connection: {}", e));
            return -1;
        }
    };

    let mut face_index_lock = FACE_INDEX.lock().unwrap();
    if let Some(ref mut face_idx) = *face_index_lock {
        if let Ok(_) = face_idx.clear() {
            log_init("[REBUILD] face_index cleared");
        }

        if let Ok(count) = conn.query_row::<i64, _, _>("SELECT COUNT(*) FROM faces", [], |row| row.get(0)) {
            log_init(&format!("[REBUILD] DEBUG: faces table has {} rows", count));
        }

        if let Ok(count) = conn.query_row::<i64, _, _>("SELECT COUNT(*) FROM face_embeddings", [], |row| row.get(0)) {
            log_init(&format!("[REBUILD] DEBUG: face_embeddings table has {} rows", count));
        }

        let rows_count = {
            let mut stmt = match conn.prepare("SELECT id FROM faces ORDER BY id") {
                Ok(s) => s,
                Err(e) => {
                    log_init(&format!("[REBUILD] Failed to prepare faces query: {}", e));
                    return -1;
                }
            };
            stmt.query_map([], |row| row.get::<_, i64>(0))
                .ok()
                .map(|rows| rows.filter_map(|r| r.ok()).collect::<Vec<_>>())
                .unwrap_or_default()
        };
        log_init(&format!("[REBUILD] DEBUG: face_ids from faces table: {:?}", rows_count));

        // FIX: Use f.vector_id as HNSW key (same as worker), not fe.face_id (primary key)
        let rows: Vec<(i64, Vec<f32>)> = {
            let mut stmt = match conn.prepare(
                "SELECT f.vector_id, fe.vector FROM face_embeddings fe
                 JOIN faces f ON f.id = fe.face_id
                 WHERE f.vector_id IS NOT NULL
                 ORDER BY f.vector_id"
            ) {
                Ok(s) => s,
                Err(e) => {
                    log_init(&format!("[REBUILD] Failed to prepare face query: {}", e));
                    return -1;
                }
            };
            stmt.query_map([], |row| {
                let vector_id: i64 = row.get(0)?;
                let vector_blob: Vec<u8> = row.get(1)?;
                let embedding: Vec<f32> = vector_blob.chunks_exact(4).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect();
                Ok((vector_id, embedding))
            }).ok()
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
        };

        let rows_count = rows.len();
        log_init(&format!("[REBUILD] Loading {} face embeddings into index", rows_count));
        for (face_id, embedding) in &rows {
            if let Err(e) = face_idx.insert(*face_id, embedding) {
                log_init(&format!("[REBUILD] Failed to insert face {}: {}", face_id, e));
            }
        }

        if rows_count == 0 {
            log_init("[REBUILD] No embeddings to save, skipping save");
        } else if let Err(e) = face_idx.save() {
            log_init(&format!("[REBUILD] face_index.save() failed: {}", e));
        } else {
            log_init("[REBUILD] face_index saved");
        }
    }
    drop(face_index_lock);

    // Also rebuild visual index
	    log_init("[REBUILD] Rebuilding visual index...");
	    let mut visual_index_lock = VISUAL_V2_INDEX.lock().unwrap();
	    if let Some(ref mut visual_idx) = *visual_index_lock {
	        if let Ok(_) = visual_idx.clear() {
	            log_init("[REBUILD] visual_index cleared");
	        }
	        let scan_version = pf_application::visual_scan_v2::config::VisualScanV2Config::default().scan_version;
	        if let Ok(count) = pf_application::visual_scan_v2::database::get_feature_count_by_version(&conn, &scan_version) {
	            log_init(&format!("[REBUILD] Visual features in DB: {}", count));
	            if count > 0 {
	                match pf_application::visual_scan_v2::database::bulk_load_to_hnsw(&conn, &scan_version, visual_idx.as_ref() as &dyn pf_vector::VectorIndex, 1000) {
	                    Ok(loaded) => {
	                        log_init(&format!("[REBUILD] Loaded {} visual features", loaded));
	                        if let Err(e) = visual_idx.save() { log_init(&format!("[REBUILD] visual save error: {}", e)); }
	                        else { log_init("[REBUILD] visual saved"); }
	                    }
	                    Err(e) => { log_init(&format!("[REBUILD] visual load error: {}", e)); }
	                }
	            }
	        }
	    }
	    drop(visual_index_lock);

	    log_init("[REBUILD] pf_rebuild_hnsw_indices complete");
    0
}

#[no_mangle]
pub extern "C" fn pf_get_version() -> *mut c_char {
    null_str("PhotoFinder Flutter FFI 0.3.0 (using crates/)")
}

#[no_mangle]
pub unsafe extern "C" fn pf_free_string(_ptr: *mut c_char) {}

#[no_mangle]
pub extern "C" fn pf_get_statistics() -> *mut c_char {
    let db_lock = DB.lock().unwrap();
    let db = match &*db_lock {
        Some(db) => db,
        None => {
            return null_str(r#"{"image_count":0,"face_count":0,"object_count":0}"#);
        }
    };

    let conn = match db.connection() {
        Ok(c) => c,
        Err(_) => {
            return null_str(r#"{"image_count":0,"face_count":0,"object_count":0}"#);
        }
    };

    let photos: i64 = conn.query_row("SELECT COUNT(*) FROM images", [], |row| row.get(0)).unwrap_or(0);
    let faces: i64 = conn.query_row("SELECT COUNT(*) FROM faces", [], |row| row.get(0)).unwrap_or(0);
    let objects: i64 = conn.query_row("SELECT COUNT(*) FROM visual_scan_v2_features", [], |row| row.get(0)).unwrap_or(0);

    null_str(&format!(r#"{{"image_count":{},"face_count":{},"object_count":{}}}"#, photos, faces, objects))
}

// ============================================================================
// Android Scan Architecture - Phase 2
// Separates scan enumeration from AI processing
// ============================================================================

/// Scan state tracking
static IS_SCANNING: Mutex<Option<bool>> = Mutex::new(None);
static SCAN_STATS: Mutex<Option<ScanStats>> = Mutex::new(None);
static SCAN_QUEUE: Mutex<Option<ScanQueue>> = Mutex::new(None);
static SCAN_WORKER_HANDLE: Mutex<Option<std::thread::JoinHandle<()>>> = Mutex::new(None);

/// Face Queue - bounded channel for Face processing
static FACE_QUEUE: Mutex<Option<ScanQueue>> = Mutex::new(None);
/// Visual Queue - bounded channel for Visual processing
static VISUAL_QUEUE: Mutex<Option<ScanQueue>> = Mutex::new(None);
/// Visual Worker handle
static VISUAL_WORKER_HANDLE: Mutex<Option<std::thread::JoinHandle<()>>> = Mutex::new(None);

struct ScanStats {
    enumerated: usize,
    queued: usize,
    processing: usize,
    completed: usize,
    failed: usize,
    total: usize,
    /// True after scanEnd() is called (Flutter不会再提交新照片)
    input_closed: bool,
    /// True when所有照片处理完成 (queue empty + processing == 0)
    finished: bool,
    /// Visual worker processing count
    visual_processing: usize,
    /// Visual worker completed count
    visual_completed: usize,
}

unsafe impl Send for ScanStats {}
unsafe impl Sync for ScanStats {}

/// Thread-safe scan queue for Android photo library integration.
/// Flutter enumerates photos via PhotoManager; Rust AI worker processes them.
struct ScanQueue {
    /// Bounded sender - blocks when queue is full (backpressure)
    sender: std::sync::mpsc::SyncSender<PhotoTask>,
}

struct PhotoTask {
    /// Flutter asset ID (photo_id from Flutter)
    photo_id: i64,
    /// File path
    path: String,
    /// Actual database image_id (inserted during enqueue, so both workers can use it directly)
    image_id: Option<i64>,
}

impl Clone for PhotoTask {
    fn clone(&self) -> Self {
        PhotoTask {
            photo_id: self.photo_id,
            path: self.path.clone(),
            image_id: self.image_id,
        }
    }
}

/// Initialize the scan queue and spawn the AI worker thread.
/// Call this BEFORE sending photos.
/// Returns 0 on success, -1 if already scanning.
#[no_mangle]
pub extern "C" fn pf_scan_begin() -> i32 {
    log_init("[SCAN] pf_scan_begin called");

    // Check if already scanning
    {
        let is_scanning = IS_SCANNING.lock().unwrap();
        if *is_scanning.as_ref().unwrap_or(&false) {
            log_init("[SCAN] pf_scan_begin: already scanning");
            return -1;
        }
    }

    // Reset incremental clustering counter
    FACES_SINCE_LAST_CLUSTER.store(0, Ordering::SeqCst);
    CLUSTER_RUNNING.lock().unwrap().take();

    // Clear old scan data from DB and HNSW indexes
    {
        let db_lock = DB.lock().unwrap();
        if let Some(db) = db_lock.as_ref() {
            if let Ok(conn) = db.connection() {
                // Delete in child-first order to respect FK constraints
                conn.execute_batch(
                    "DELETE FROM face_person_assignments;
                     DELETE FROM face_embeddings;
                     DELETE FROM faces;
                     DELETE FROM persons;
                     DELETE FROM visual_scan_v2_features;
                     DELETE FROM visual_scan_v2_status;"
                ).ok();
                log_init("[SCAN] Cleared old scan data from DB");
            }
        }
    }
    // Clear HNSW indexes in memory (keep initialized, just empty)
    {
        let mut face_idx = FACE_INDEX.lock().unwrap();
        if let Some(ref mut idx) = *face_idx {
            idx.clear().ok();
            log_init("[SCAN] Face HNSW index cleared");
        }
    }
    {
        let mut visual_idx = VISUAL_V2_INDEX.lock().unwrap();
        if let Some(ref mut idx) = *visual_idx {
            idx.clear().ok();
            log_init("[SCAN] Visual HNSW index cleared");
        }
    }

    // Initialize scan stats
    {
        let mut stats = SCAN_STATS.lock().unwrap();
        *stats = Some(ScanStats {
            enumerated: 0,
            queued: 0,
            processing: 0,
            completed: 0,
            failed: 0,
            total: 0,
            input_closed: false,
            finished: false,
            visual_processing: 0,
            visual_completed: 0,
        });
    }

    // Create bounded queues - one for Face, one for Visual
    // QUEUE_CAPACITY: each slot holds ONE task; receiver calls recv() to free a slot
    // Small buffer: 8 slots = 4 photos processed simultaneously (2 slots per photo)
    const QUEUE_CAPACITY: usize = 8;
    let (face_tx, face_rx) = std::sync::mpsc::sync_channel::<PhotoTask>(QUEUE_CAPACITY);
    let (visual_tx, visual_rx) = std::sync::mpsc::sync_channel::<PhotoTask>(QUEUE_CAPACITY);

    // Store senders in global queues
    {
        let mut queue = FACE_QUEUE.lock().unwrap();
        *queue = Some(ScanQueue { sender: face_tx });
    }
    {
        let mut queue = VISUAL_QUEUE.lock().unwrap();
        *queue = Some(ScanQueue { sender: visual_tx });
    }

    // Mark as scanning
    {
        let mut is_scanning = IS_SCANNING.lock().unwrap();
        *is_scanning = Some(true);
    }

    // Spawn Face worker thread
    let face_rx_clone = face_rx;
    let handle = std::thread::spawn(move || {
        log_init("[FACE_WORKER] thread started");
        run_face_worker(face_rx_clone);
        log_init("[FACE_WORKER] thread exiting");
    });

    // Spawn Visual worker thread
    let visual_rx_clone = visual_rx;
    let visual_handle = std::thread::spawn(move || {
        log_init("[VISUAL_WORKER] thread started");
        run_visual_worker(visual_rx_clone);
        log_init("[VISUAL_WORKER] thread exiting");
    });

    // Store handles
    {
        let mut worker_handle = SCAN_WORKER_HANDLE.lock().unwrap();
        *worker_handle = Some(handle);
    }
    {
        let mut worker_handle = VISUAL_WORKER_HANDLE.lock().unwrap();
        *worker_handle = Some(visual_handle);
    }

    log_init("[SCAN] pf_scan_begin: both workers spawned");
    0
}

/// Enqueue a single photo for AI processing.
/// Called by Flutter AFTER pf_scan_begin(), once per enumerated photo.
/// This blocks if the queue is full (backpressure - prevents OOM).
/// Returns 0 on success, -1 if scan not initialized.
#[no_mangle]
pub extern "C" fn pf_enqueue_photo(photo_id: i64, path: *const c_char) -> i32 {
    let path_str = if path.is_null() {
        log_init("[SCAN] pf_enqueue_photo: null path");
        return -1;
    } else {
        unsafe { std::ffi::CStr::from_ptr(path) }
            .to_str()
            .unwrap_or("")
            .to_string()
    };

    // Check input closed first
    {
        let stats = SCAN_STATS.lock().unwrap();
        if let Some(ref s) = *stats {
            if s.input_closed {
                log_init("[SCAN] pf_enqueue_photo: input already closed, rejecting");
                return 2; // INPUT_CLOSED
            }
        }
    }

    // Insert image to DB FIRST to get actual image_id
    // This ensures both Face and Visual workers can use it directly without racing
    let file_path = std::path::Path::new(&path_str);
    let image_id = match insert_image_to_db_enqueue(photo_id, file_path) {
        Ok(id) => {
            log_init(&format!("[ENQUEUE] photo_id={} image_id={} inserted", photo_id, id));
            Some(id)
        }
        Err(e) => {
            log_init(&format!("[ENQUEUE] photo_id={} DB insert failed: {}, will retry in worker", photo_id, e));
            None // Worker will handle insertion if needed
        }
    };

    // Lock both face and visual queues
    let face_queue = FACE_QUEUE.lock().unwrap();
    let visual_queue = VISUAL_QUEUE.lock().unwrap();

    let face_queue = match face_queue.as_ref() {
        Some(q) => q,
        None => {
            log_init("[SCAN] pf_enqueue_photo: scan not initialized (call pf_scan_begin first)");
            return -1;
        }
    };

    let visual_queue = match visual_queue.as_ref() {
        Some(q) => q,
        None => {
            log_init("[SCAN] pf_enqueue_photo: scan not initialized (call pf_scan_begin first)");
            return -1;
        }
    };

    let task = PhotoTask {
        photo_id,
        path: path_str.clone(),
        image_id,
    };

    // Use try_send() - non-blocking
    // Returns QUEUE_FULL immediately if queue is full, Flutter will retry
    let face_result = face_queue.sender.try_send(task.clone());
    let visual_result = visual_queue.sender.try_send(task.clone());

    match (&face_result, &visual_result) {
        (Ok(()), Ok(())) => {
            // Both queues accepted the task
            let mut stats = SCAN_STATS.lock().unwrap();
            if let Some(ref mut s) = *stats {
                s.enumerated += 1;
                s.queued += 1;
            }
            log_init(&format!("[ENQUEUE] photo_id={} ACCEPTED", photo_id));
            return 0; // ACCEPTED
        }
        // Queue full - task not accepted, caller should retry
        (Err(std::sync::mpsc::TrySendError::Full(_)), _) |
        (_, Err(std::sync::mpsc::TrySendError::Full(_))) => {
            log_init(&format!("[ENQUEUE] photo_id={} QUEUE_FULL", photo_id));
            return 1; // QUEUE_FULL
        }
        // Queue disconnected - unrecoverable
        (Err(std::sync::mpsc::TrySendError::Disconnected(_)), _) => {
            log_init("[SCAN] pf_enqueue_photo: FACE queue disconnected");
            return 3; // ERROR
        }
        (_, Err(std::sync::mpsc::TrySendError::Disconnected(_))) => {
            log_init("[SCAN] pf_enqueue_photo: VISUAL queue disconnected");
            return 3; // ERROR
        }
    }
}

/// Signal end of photo enumeration.
/// Sets input_closed=true, drops sender to signal worker.
/// Worker continues in background until queue drains and finished=true.
/// Flutter polls pf_get_scan_progress() to wait for finished=true.
#[no_mangle]
pub extern "C" fn pf_scan_end() -> i32 {
    log_init("[SCAN] pf_scan_end: marking input_closed, dropping sender...");

    // Mark input as closed
    {
        let mut stats = SCAN_STATS.lock().unwrap();
        if let Some(ref mut s) = *stats {
            s.input_closed = true;
            log_init("[SCAN] pf_scan_end: input_closed=true");
        }
    }

    // Drop both senders to close channels - workers will exit when queues drain
    {
        let mut queue = SCAN_QUEUE.lock().unwrap();
        *queue = None;
    }
    {
        let mut queue = FACE_QUEUE.lock().unwrap();
        *queue = None;
    }
    {
        let mut queue = VISUAL_QUEUE.lock().unwrap();
        *queue = None;
    }

    log_init("[SCAN] pf_scan_end: done (both workers continue in background)");
    0
}

/// Cancel the running scan. Worker will finish current photo then stop.
/// Closes both queues to prevent new enqueues.
#[no_mangle]
pub extern "C" fn pf_stop_scan() -> i32 {
    log_init("[SCAN] pf_stop_scan called");
    let mut is_scanning = IS_SCANNING.lock().unwrap();
    if *is_scanning.as_ref().unwrap_or(&false) {
        *is_scanning = Some(false);
        log_init("[SCAN] scan stop requested, closing queues");

        // Close both queues to wake up workers and prevent new enqueues
        {
            let mut queue = SCAN_QUEUE.lock().unwrap();
            *queue = None;
        }
        {
            let mut queue = FACE_QUEUE.lock().unwrap();
            *queue = None;
        }
        {
            let mut queue = VISUAL_QUEUE.lock().unwrap();
            *queue = None;
        }

        // Mark input as closed to reject new enqueues
        {
            let mut stats = SCAN_STATS.lock().unwrap();
            if let Some(ref mut s) = *stats {
                s.input_closed = true;
            }
        }

        0
    } else {
        log_init("[SCAN] not scanning, stop is no-op");
        -1
    }
}

/// Mark scan as finished. Only sets finished=true if ALL processing is done.
fn mark_scan_finished() {
    let mut stats = SCAN_STATS.lock().unwrap();
    if let Some(ref mut s) = *stats {
        // Only mark finished if BOTH face and visual workers are done
        if s.processing == 0 && s.visual_processing == 0 && s.queued == 0 {
            s.finished = true;
            log_init("[SCAN] marked finished=true (all workers done)");
        } else {
            log_init(&format!("[SCAN] defer finished: processing={} visual={} queued={}",
                s.processing, s.visual_processing, s.queued));
        }
    }
}


/// Get current scan progress as JSON.
/// Format: {"enumerated":N,"queued":N,"processing":N,"completed":N,"failed":N,"input_closed":bool,"finished":bool,"visual_completed":N,"photo_completed":N}
/// photo_completed = min(completed, visual_completed) - the actual number of fully processed photos
#[no_mangle]
pub extern "C" fn pf_get_scan_progress() -> *mut c_char {
    let stats = SCAN_STATS.lock().unwrap();
    if let Some(ref s) = *stats {
        // photo_completed is min of Face and Visual completed (both must finish for a photo to be done)
        let photo_completed = s.completed.min(s.visual_completed);
        null_str(&format!(
            r#"{{"enumerated":{},"queued":{},"processing":{},"completed":{},"failed":{},"input_closed":{},"finished":{},"visual_completed":{},"photo_completed":{}}}"#,
            s.enumerated, s.queued, s.processing, s.completed, s.failed, s.input_closed, s.finished, s.visual_completed, photo_completed
        ))
    } else {
        null_str(r#"{"enumerated":0,"queued":0,"processing":0,"completed":0,"failed":0,"input_closed":false,"finished":true,"visual_completed":0,"photo_completed":0}"#)
    }
}

/// Face worker loop - receives PhotoTask from queue, processes Face only.
/// After Face processing, enqueues to Visual queue.
/// Exits when channel closes or cancelled.
fn run_face_worker(rx: std::sync::mpsc::Receiver<PhotoTask>) {
    // Clone globals needed by worker
    let fp = FACE_PIPELINE.lock().unwrap().clone();
    let vs = VISUAL_SCANNER.lock().unwrap().clone();
    let db = DB.lock().unwrap().clone();

	    loop {
	        // Check cancellation before getting next task
	        let is_scanning = IS_SCANNING.lock().unwrap();
	        if !*is_scanning.as_ref().unwrap_or(&false) {
	            log_init("[SCAN_WORKER] cancelled, exiting loop");
	            mark_scan_finished(); break;
	        }
	        drop(is_scanning);

	        // Block waiting for task (bounded channel blocks when queue is empty)
	        // Log progress every 5 seconds to help diagnose stalls</        let start_wait = std::time::Instant::now();
	        log_init("[SCAN_WORKER] Calling rx.recv() to wait for task...");
	        let task = match rx.recv() {
	            Ok(t) => {
	                log_init(&format!("[SCAN_WORKER] Received task photo_id={}", t.photo_id));
	                t
	            }
	            Err(_) => {
	                log_init("[SCAN_WORKER] Channel closed, exiting loop");
	                mark_scan_finished(); break;
	            }
	        };

	        // Update stats: queued--, processing++
	        {
	            let mut stats = SCAN_STATS.lock().unwrap();
	            if let Some(ref mut s) = *stats {
	                s.queued = s.queued.saturating_sub(1);
	                s.processing += 1;
	                log_init(&format!("[SCAN_WORKER] Stats after dequeue: queued={} processing={}", s.queued, s.processing));
	            }
	        }

	        let photo_id = task.photo_id;
	        let path_str = task.path;
	        let image_id = task.image_id;
	        log_init(&format!("[SCAN_WORKER] Starting process for photo_id={} image_id={:?}", photo_id, image_id));

	        // Process photo with panic protection
	        let face_count = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
	            let file_path = std::path::Path::new(&path_str);
	            process_photo_ai_worker(&fp, &vs, &db, photo_id, file_path, image_id)
	        })).unwrap_or_else(|panic_info| {
	            let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
	                s.to_string()
	            } else if let Some(s) = panic_info.downcast_ref::<String>() {
	                s.clone()
	            } else {
	                "unknown panic".to_string()
	            };
	            log_init(&format!("[SCAN_WORKER] photo_id={} PANIC: {}", photo_id, msg));
	            0usize
	        });

		        // Update stats: processing--, completed++
	        {
	            let mut stats = SCAN_STATS.lock().unwrap();
	            if let Some(ref mut s) = *stats {
	                s.processing = s.processing.saturating_sub(1);
	                s.completed += 1;
	                log_init(&format!("[FACE_DONE] photo_id={} faces={} proc={} comp={} queued={}",
	                    photo_id, face_count, s.processing, s.completed, s.queued));
	            }
	        }
	        log_init(&format!("[SCAN_WORKER] photo_id={} done (faces={}), continuing", photo_id, face_count));

	        // Trigger incremental clustering if we detected faces
	        if face_count > 0 {
	            trigger_incremental_clustering();
	        }
	    }
    log_init("[SCAN_WORKER] worker loop exiting");
}

/// Process a single photo through AI (Face then Visual), updating DB.

/// Visual worker loop - receives PhotoTask from visual queue, processes Visual only.
/// Exits when channel closes (signaled by scan_end) or cancelled.
fn run_visual_worker(rx: std::sync::mpsc::Receiver<PhotoTask>) {
    // Clone globals needed by worker
    let vs = VISUAL_SCANNER.lock().unwrap().clone();
    let db = DB.lock().unwrap().clone();

    log_init("[VISUAL_WORKER] Visual worker starting...");

    loop {
        // Check cancellation
        let is_scanning = IS_SCANNING.lock().unwrap();
        if !*is_scanning.as_ref().unwrap_or(&false) {
            log_init("[VISUAL_WORKER] cancelled, exiting loop");
            mark_scan_finished();
            break;
        }
        drop(is_scanning);

        // Block waiting for task
        log_init("[VISUAL_WORKER] Calling rx.recv() to wait for task...");
        // Note: rx.recv() immediately frees the channel slot when item is received
        let task = match rx.recv() {
            Ok(t) => {
                log_init(&format!("[VISUAL_WORKER] Received task photo_id={}", t.photo_id));
                t
            }
            Err(_) => {
                log_init("[VISUAL_WORKER] Channel closed, marking finished, exiting loop");
                mark_scan_finished();
                break;
            }
        };

        // Update visual processing stats
        {
            let mut stats = SCAN_STATS.lock().unwrap();
            if let Some(ref mut s) = *stats {
                s.queued = s.queued.saturating_sub(1);
                s.visual_processing += 1;
                log_init(&format!("[VISUAL_WORKER] Stats: queued={} visual_processing={}", s.queued, s.visual_processing));
            }
        }

        let photo_id = task.photo_id;
        let path_str = task.path;
        let image_id = task.image_id;
        log_init(&format!("[VISUAL_WORKER] Processing photo_id={} image_id={:?}", photo_id, image_id));

        // Process Visual with panic protection
        let object_count = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let file_path = std::path::Path::new(&path_str);
            process_visual_only_worker(&vs, &db, photo_id, file_path, image_id)
        })).unwrap_or_else(|panic_info| {
            let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            log_init(&format!("[VISUAL_WORKER] photo_id={} PANIC: {}", photo_id, msg));
            0usize
        });

        // Update stats
        {
            let mut stats = SCAN_STATS.lock().unwrap();
            if let Some(ref mut s) = *stats {
                s.visual_processing = s.visual_processing.saturating_sub(1);
                s.visual_completed += 1;
                log_init(&format!("[VISUAL_DONE] photo_id={} objects={} vproc={} vcomp={} queued={}",
                    photo_id, object_count, s.visual_processing, s.visual_completed, s.queued));
            }
        }

        log_init(&format!("[VISUAL_WORKER] photo_id={} done (objects={})", photo_id, object_count));

        // Periodic progress log every 50 photos
        {
            let stats = SCAN_STATS.lock().unwrap();
            if let Some(ref s) = *stats {
                if s.visual_completed % 50 == 0 && s.visual_completed > 0 {
                    log_init(&format!("[VISUAL_PROGRESS] visual_completed={} processing={} queued={} enumerated={}",
                        s.visual_completed, s.visual_processing, s.queued, s.enumerated));
                }
            }
        }
    }
    log_init("[VISUAL_WORKER] Visual worker loop exiting");
}

/// Process Visual only (called by Visual worker).
fn process_visual_only_worker(
    visual_scanner: &Option<VisualScannerState>,
    db: &Option<Arc<Database>>,
    photo_id: i64,
    file_path: &Path,
    pre_inserted_image_id: Option<i64>,
) -> usize {
    let start = std::time::Instant::now();

    log_init(&format!("[VISUAL_WORKER] photo {}: VISUAL START path={:?}", photo_id, file_path));

    // Step 1: Get DB connection
    let db_ref = match db {
        Some(d) => d,
        None => {
            log_init(&format!("[VISUAL_WORKER] photo {}: DB not initialized", photo_id));
            return 0;
        }
    };
    let conn = match db_ref.connection() {
        Ok(c) => c,
        Err(e) => {
            log_init(&format!("[VISUAL_WORKER] photo {}: DB connection FAIL: {}", photo_id, e));
            return 0;
        }
    };

    // Step 2: Visual scanner
    let vs_state = match visual_scanner {
        Some(v) => v,
        None => {
            log_init(&format!("[VISUAL_WORKER] photo {}: Visual scanner NOT initialized", photo_id));
            return 0;
        }
    };

    log_init(&format!("[VISUAL_WORKER] photo {}: Visual scanner starting...", photo_id));

    // Use pre-inserted image_id from enqueue (guaranteed to exist now)
    // If not available (e.g., enqueue insert failed), fall back to path lookup
    let actual_image_id = if let Some(img_id) = pre_inserted_image_id {
        log_init(&format!("[VISUAL_WORKER] photo {}: using pre-inserted image_id={}", photo_id, img_id));
        img_id
    } else {
        // Fallback: look up by path (old behavior for backward compat)
        let file_path_str = file_path.to_string_lossy().to_string();
        let mut actual_image_id: Option<i64> = None;
        let mut retries = 0;
        const MAX_RETRIES: usize = 20;
        const RETRY_DELAY_MS: u64 = 500;

        while retries < MAX_RETRIES {
            match conn.query_row(
                "SELECT id FROM images WHERE path = ?",
                [&file_path_str],
                |row| row.get(0),
            ) {
                Ok(id) => {
                    actual_image_id = Some(id);
                    log_init(&format!("[VISUAL_WORKER] photo {}: found actual image_id={} after {} retries", photo_id, id, retries));
                    break;
                }
                Err(_) => {
                    if retries == 0 {
                        log_init(&format!("[VISUAL_WORKER] photo {}: image not found yet, retrying... (Flutter photo_id={})", photo_id, photo_id));
                    }
                    retries += 1;
                    if retries < MAX_RETRIES {
                        std::thread::sleep(std::time::Duration::from_millis(RETRY_DELAY_MS));
                    }
                }
            }
        }

        match actual_image_id {
            Some(id) => id,
            None => {
                log_init(&format!("[VISUAL_WORKER] photo {}: ERROR - image not found in DB after {} retries (Flutter photo_id={})", photo_id, MAX_RETRIES, photo_id));
                return 0;
            }
        }
    };

    // Direct call - holds mutex during scan
    let mut scanner = vs_state.scanner.lock().unwrap();
    log_init(&format!("[VISUAL_WORKER] photo {}: calling scanner.scan_image with actual_image_id={}...", photo_id, actual_image_id));
    let start_scan = std::time::Instant::now();
    let result = scanner.scan_image(&conn, actual_image_id, file_path);
    let scan_took = start_scan.elapsed();
    log_init(&format!("[VISUAL_WORKER] photo {}: scan_image took {:?}", photo_id, scan_took));
    match result {
        Ok(r) => {
            log_init(&format!(
                "[VISUAL_WORKER] photo {}: Visual OK raw_rois={}/{} selected={}/{} objects={} time={:?}",
                photo_id, r.raw_roi_count_025, r.raw_roi_count_05, r.selected_count_025, r.selected_count_05, r.total_feature_count, start.elapsed()
            ));
            r.total_feature_count
        }
        Err(e) => {
            log_init(&format!("[VISUAL_WORKER] photo {}: Visual FAIL: {}", photo_id, e));
            0
        }
    }
}
fn process_photo_ai_worker(
    face_pipeline: &Option<Arc<FacePipeline>>,
    visual_scanner: &Option<VisualScannerState>,
    db: &Option<Arc<Database>>,
    asset_id: i64,
    file_path: &Path,
    pre_inserted_image_id: Option<i64>,
) -> usize {
    let start = std::time::Instant::now();
    let step_start = std::time::Instant::now();

    log_init(&format!("[AI_WORKER] photo {}: START path={:?}", asset_id, file_path));

    // Step 1: Insert photo to DB (or use pre-inserted image_id from enqueue)
    let photo_id = if let Some(img_id) = pre_inserted_image_id {
        log_init(&format!("[AI_WORKER] photo {}: [1/4] Using pre-inserted image_id={} time={:?}", asset_id, img_id, step_start.elapsed()));
        img_id
    } else {
        match insert_photo_to_db_internal(db, asset_id, file_path) {
            Ok(id) => {
                log_init(&format!("[AI_WORKER] photo {}: [1/4] DB insert OK id={} time={:?}", asset_id, id, step_start.elapsed()));
                id
            }
            Err(e) => {
                log_init(&format!("[AI_WORKER] photo {}: [1/4] DB insert FAIL: {} time={:?}", asset_id, e, step_start.elapsed()));
                return 0;
            }
        }
    };

    // Step 2: Load image
    let step_start = std::time::Instant::now();
    let img_bytes = match std::fs::read(file_path) {
        Ok(bytes) => {
            log_init(&format!("[AI_WORKER] photo {}: [2/4] File read OK {} bytes time={:?}",
                photo_id, bytes.len(), step_start.elapsed()));
            bytes
        }
        Err(e) => {
            log_init(&format!("[AI_WORKER] photo {}: [2/4] File read FAIL: {} time={:?}", photo_id, e, step_start.elapsed()));
            return 0;
        }
    };

    let img_data = match ImageData::from_bytes(&img_bytes) {
        Ok(img) => {
            log_init(&format!("[AI_WORKER] photo {}: [2/4] Image decode OK {}x{} time={:?}",
                photo_id, img.width(), img.height(), step_start.elapsed()));
            img
        }
        Err(e) => {
            log_init(&format!("[AI_WORKER] photo {}: [2/4] Image decode FAIL: {} time={:?}", photo_id, e, step_start.elapsed()));
            return 0;
        }
    };

    // Load DynamicImage for RollCorrect support (same as desktop)
    let dyn_img = image::load_from_memory(&img_bytes).ok();

    // === STEP 3: FACE PIPELINE ===
    let step_start = std::time::Instant::now();
    let pipeline = match face_pipeline {
        Some(p) => p,
        None => {
            log_init(&format!("[AI_WORKER] photo {}: [3/4] Face pipeline NOT initialized", photo_id));
            return 0;
        }
    };

    // Run face pipeline - create runtime and block_on
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            log_init(&format!("[AI_WORKER] photo {}: [3/4] Runtime create FAIL: {}", photo_id, e));
            return 0;
        }
    };

    log_init(&format!("[AI_WORKER] photo {}: [3/4] Face runtime ready, starting inference...", photo_id));

    let faces = match rt.block_on(pipeline.process_with_raw(&img_data, dyn_img.as_ref())) {
        Ok(f) => {
            log_init(&format!("[AI_WORKER] photo {}: [3/4] Face inference OK faces={} time={:?}", photo_id, f.len(), step_start.elapsed()));
            // Log each detected face
            for (i, face) in f.iter().enumerate() {
                log_init(&format!("[AI_WORKER] photo {} face[{}]: det={:.3} blur={:.3} pose={:.3} area={:.3} bbox={},{},{},{}",
                    photo_id, i, face.detection.score, face.blur_score, face.pose_score, face.face_area_score,
                    face.detection.bbox.x, face.detection.bbox.y, face.detection.bbox.w, face.detection.bbox.h));
            }
            f
        }
        Err(e) => {
            log_init(&format!("[AI_WORKER] photo {}: [3/4] Face inference FAIL: {} time={:?}", photo_id, e, step_start.elapsed()));
            return 0;
        }
    };

    let face_count = faces.len();

    // Save faces to DB
    let step_start = std::time::Instant::now();
    if face_count > 0 {
        let db_ref = match db {
            Some(d) => d,
            None => return face_count,
        };
        if let Ok(conn) = db_ref.connection() {
            save_faces_to_db_internal(&conn, photo_id, &faces);
        }
    }
    log_init(&format!("[AI_WORKER] photo {}: [4/4] COMPLETE faces={} total_time={:?}",
        photo_id, face_count, start.elapsed()));

    face_count
}

/// Internal face save (runs on worker thread, no locks needed on DB connection)
/// Mirrors the desktop 3-phase pattern: INSERT face → HNSW insert → face_embeddings insert
fn save_faces_to_db_internal(conn: &rusqlite::Connection, photo_id: i64, faces: &[pf_ai::FaceFeature]) {
    let face_index = FACE_INDEX.lock().unwrap();

    for (i, face) in faces.iter().enumerate() {
        let cx = (face.detection.bbox.x + face.detection.bbox.w / 2.0) as i32;
        let cy = (face.detection.bbox.y + face.detection.bbox.h / 2.0) as i32;
        let h = ((cx as u64) << 32) ^ (cy as u64 ^ photo_id as u64);
        let vector_id = (h & 0x7FFF_FFFF_FFFF_FFFF) as i64;

        // Phase B: HNSW insert (outside DB transaction, matching desktop pattern)
        if let Some(ref idx) = *face_index {
            let embedding_values = face.embedding.values.as_slice();
            match idx.insert(vector_id, embedding_values) {
                Ok(()) => log_init(&format!("[FACE_WORKER] HNSW insert ok: face[{}] vector_id={}", i, vector_id)),
                Err(e) => log_init(&format!("[FACE_WORKER] HNSW insert failed for face[{}]: {}", i, e)),
            }
        }

        let (yaw, pitch, roll) = match face.yaw_pitch_roll {
            Some((y, p, r)) => (Some(y), Some(p), Some(r)),
            None => (None, None, None),
        };

        let quality_score = (face.blur_score * 0.3 + face.pose_score * 0.3 + face.face_area_score * 0.4).min(1.0);
        let keypoints_json = serde_json::to_string(&face.detection.keypoints).ok();

        // Phase A: INSERT face row
        match conn.execute(
            "INSERT INTO faces (image_id, bbox_x, bbox_y, bbox_w, bbox_h,
             detector_score, detector_model, keypoints_json, yaw, pitch, roll,
             quality_score, blur_score, pose_score, face_area_score,
             embedding_model, model_version, vector_id,
             detector_origin, config_roll_correct_enabled, config_roll_correct_threshold)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21)",
            rusqlite::params![
                photo_id,
                face.detection.bbox.x,
                face.detection.bbox.y,
                face.detection.bbox.w,
                face.detection.bbox.h,
                face.detection.score,
                "scrfd-500m-bnkps",
                keypoints_json,
                yaw,
                pitch,
                roll,
                quality_score,
                face.blur_score,
                face.pose_score,
                face.face_area_score,
                "arcface",
                face.embedding.model.as_str(),
                vector_id,
                face.detector_origin.as_str(),
                face.config_roll_correct_enabled as i32,
                face.config_roll_correct_threshold,
            ],
        ) {
            Ok(_) => {
                let face_id = conn.last_insert_rowid();
                log_init(&format!("[FACE_WORKER] Saved face[{}]: id={}, vector_id={}", i, face_id, vector_id));

                // Phase C: INSERT face_embeddings row (Android schema from migration 012+)
                let embedding_bytes: Vec<u8> = face.embedding.values.as_slice()
                    .iter()
                    .flat_map(|&v| v.to_le_bytes())
                    .collect();
                let dim = embedding_bytes.len() / 4;

                if let Err(e) = conn.execute(
                    "INSERT INTO face_embeddings (face_id, model_name, model_version, dimension, vector, normalized)
                     VALUES (?1, 'arcface', 'w600k_r50_v1', ?2, ?3, 1)",
                    rusqlite::params![face_id, dim as i32, embedding_bytes],
                ) {
                    log_init(&format!("[FACE_WORKER] Failed to save face_embedding[{}]: {}", i, e));
                }
            }
            Err(e) => {
                log_init(&format!("[FACE_WORKER] insert face[{}] failed: {}", i, e));
            }
        }
    }

    // Persist HNSW to disk (matching desktop: save after all faces processed)
    if let Some(ref idx) = *face_index {
        if let Err(e) = idx.save() {
            log_init(&format!("[FACE_WORKER] HNSW save failed: {}", e));
        } else {
            log_init("[FACE_WORKER] HNSW index persisted");
        }
    }
}

/// Insert a photo to DB and return the real database row ID.
/// Called from AI worker to establish the image row before saving faces/objects.
fn insert_photo_to_db_internal(
    db: &Option<Arc<Database>>,
    _asset_id: i64,
    file_path: &Path,
) -> Result<i64, String> {
    let db_ref = db.as_ref().ok_or("DB not initialized")?;
    let conn = db_ref.connection().map_err(|e| format!("conn error: {}", e))?;

    let path_str = file_path.to_string_lossy();
    let asset_id_str = _asset_id.to_string();

    // Check if already exists by asset_id or by path (for backward compat)
    let existing: Option<i64> = conn
        .query_row(
            "SELECT id FROM images WHERE asset_id = ? OR path = ?",
            rusqlite::params![&asset_id_str, &path_str],
            |row| row.get(0),
        )
        .ok();
    if let Some(id) = existing {
        log_init(&format!("[AI_WORKER] photo already exists in DB: id={} asset_id={} path={}", id, asset_id_str, path_str));
        return Ok(id);
    }

    // Hash the file
    let file_data = std::fs::read(file_path).map_err(|e| format!("read error: {}", e))?;
    let mut hasher = Sha256::new();
    hasher.update(&file_data);
    let hash = format!("{:x}", hasher.finalize());

    // Get image dimensions
    let (width, height) = image::image_dimensions(file_path)
        .map(|(w, h)| (w as i64, h as i64))
        .unwrap_or((0, 0));

    // Insert with asset_id (from Flutter) and path (file system path)
    conn.execute(
        "INSERT INTO images (path, asset_id, hash, size, width, height, scan_status) VALUES (?, ?, ?, ?, ?, ?, 'pending')",
        rusqlite::params![&*path_str, &asset_id_str, hash, file_data.len() as i64, width, height],
    ).map_err(|e| format!("insert error: {}", e))?;

    let photo_id = conn.last_insert_rowid();
    log_init(&format!("[AI_WORKER] inserted photo_id={} asset_id={} path={}", photo_id, asset_id_str, path_str));
    Ok(photo_id)
}

/// Lightweight insert for enqueue phase - just inserts path and asset_id
/// so Visual Worker can find the image immediately without waiting for Face Worker.
/// Face Worker will still do the full insert (with hash/dimensions) when it processes,
/// which will update the existing row.
fn insert_image_to_db_enqueue(
    asset_id: i64,
    file_path: &Path,
) -> Result<i64, String> {
    let db_lock = DB.lock().unwrap();
    let db = db_lock.as_ref().ok_or("DB not initialized")?;
    let conn = db.connection().map_err(|e| format!("conn error: {}", e))?;

    let path_str = file_path.to_string_lossy();
    let asset_id_str = asset_id.to_string();

    // Check if already exists
    let existing: Option<i64> = conn
        .query_row(
            "SELECT id FROM images WHERE asset_id = ? OR path = ?",
            rusqlite::params![&asset_id_str, &*path_str],
            |row| row.get(0),
        )
        .ok();
    if let Some(id) = existing {
        log_init(&format!("[ENQUEUE] photo already exists: id={} asset_id={}", id, asset_id_str));
        return Ok(id);
    }

    // Lightweight insert - just path and asset_id, no file read
    // Face Worker will update with hash/dimensions when it processes
    conn.execute(
        "INSERT INTO images (path, asset_id, scan_status) VALUES (?, ?, 'pending')",
        rusqlite::params![&*path_str, &asset_id_str],
    ).map_err(|e| format!("insert error: {}", e))?;

    let photo_id = conn.last_insert_rowid();
    log_init(&format!("[ENQUEUE] inserted lightweight photo_id={} asset_id={} path={}", photo_id, asset_id_str, path_str));
    Ok(photo_id)
}

// ============================================================================
// Face and Object Search Implementation
// ============================================================================

use pf_ai::FaceFeature;

/// Face search implementation - synchronous

fn search_face_impl(image_bytes: &[u8], top_k: usize, offset: usize) -> Result<String, String> {
    const SIMILARITY_THRESHOLD: f32 = 0.50;

    let face_pipeline = FACE_PIPELINE.lock().unwrap();
    let face_pipeline = face_pipeline.as_ref().ok_or("FacePipeline not initialized")?;

    let face_index = FACE_INDEX.lock().unwrap();
    let face_index = face_index.as_ref().ok_or("FaceIndex not initialized")?;

    // Debug: check how many faces are actually stored
    let hnsw_size = face_index.len();

    let db_lock = DB.lock().unwrap();
    let db = db_lock.as_ref().ok_or("DB not initialized")?;
    let conn = db.connection().map_err(|e| format!("DB connection error: {}", e))?;
    let stored_face_count: i64 = conn.query_row("SELECT COUNT(*) FROM faces", [], |r| r.get(0)).unwrap_or(0);
    log_init(&format!("[FACE_SEARCH] DB faces={}, HNSW size={}", stored_face_count, hnsw_size));

    // Debug: check table schema and first few vector_ids
    if let Ok(mut stmt) = conn.prepare("PRAGMA table_info(faces)") {
        let cols: Vec<String> = stmt.query_map([], |row| {
            row.get::<_, String>(1)
        }).ok().map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default();
        log_init(&format!("[FACE_SEARCH] DEBUG: faces table columns={:?}", cols));
    }
    if let Ok(mut stmt) = conn.prepare("SELECT vector_id FROM faces LIMIT 3") {
        let ids: Vec<i64> = stmt.query_map([], |row| row.get(0)).ok()
            .map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default();
        log_init(&format!("[FACE_SEARCH] DEBUG: sample vector_ids={:?}", ids));
    } else {
        log_init("[FACE_SEARCH] DEBUG: vector_id column does NOT exist in faces table!");
    }

    log_init(&format!("[FACE_SEARCH] image bytes len={}", image_bytes.len()));
    let img_data = ImageData::from_bytes(image_bytes).map_err(|e| format!("Image decode error: {}", e))?;
    log_init(&format!("[FACE_SEARCH] image decoded: {}x{}", img_data.width(), img_data.height()));

    let rt_lock = RUNTIME.lock().unwrap();
    let rt = rt_lock.as_ref().ok_or("Runtime not initialized")?;

    let features = rt.block_on(async {
        face_pipeline.process_with_raw(&img_data, None).await
    }).map_err(|e| format!("Face pipeline error: {}", e))?;

    log_init(&format!("[FACE_SEARCH] detected {} faces", features.len()));

    if !features.is_empty() {
        let f = &features[0];
        log_init(&format!("[FACE_SEARCH] first face: det={:.3} blur={:.3} pose={:.3} area={:.3}",
            f.detection.score, f.blur_score, f.pose_score, f.face_area_score));
    }

    if features.is_empty() {
        return Ok(serde_json::json!({"results": [], "total": 0, "offset": offset, "limit": top_k}).to_string());
    }

    let fused_emb = fuse_face_embeddings(&features);
    let query_norm: f32 = fused_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    log_init(&format!("[FACE_SEARCH] fused_emb norm={:.4} dim={}", query_norm, fused_emb.len()));

    let fetch_k = ((top_k + offset) * 4).max(20);
    let hnsw_hits = face_index.search(&fused_emb, fetch_k)
        .map_err(|e| format!("HNSW search error: {}", e))?;

    log_init(&format!("[FACE_SEARCH] HNSW hits: {}", hnsw_hits.len()));
    if !hnsw_hits.is_empty() {
        log_init(&format!("[FACE_SEARCH] top HNSW hit id={} score={:.4}", hnsw_hits[0].id, hnsw_hits[0].score));
    }

    if hnsw_hits.is_empty() {
        return Ok(serde_json::json!({"results": [], "total": 0, "offset": offset, "limit": top_k}).to_string());
    }

    let vector_ids: Vec<i64> = hnsw_hits.iter().map(|h| h.id).collect();
    let placeholders = vector_ids.iter().map(|_| "?".to_string()).collect::<Vec<_>>().join(",");
    // FIX: HNSW stores vector_id as key, but fe.face_id = faces.id (primary key)
    // Need to join through faces WHERE f.vector_id IN (...) then get fe via f.id
    let sql = format!(
        "SELECT fe.face_id, fe.vector, f.id, f.image_id, f.person_id, f.vector_id          FROM face_embeddings fe          JOIN faces f ON f.id = fe.face_id          WHERE f.vector_id IN ({})", placeholders);
    let params: Vec<&dyn rusqlite::ToSql> = vector_ids.iter().map(|v| v as &dyn rusqlite::ToSql).collect();

    let mut stmt = conn.prepare(&sql).map_err(|e| format!("Prepare error: {}", e))?;
    let stored: Vec<(i64, Vec<f32>, i64, Option<i64>, i64)> = stmt.query_map(params.as_slice(), |row| {
        // SELECT: fe.face_id(0), fe.vector(1), f.id(2), f.image_id(3), f.person_id(4), f.vector_id(5)
        let blob: Vec<u8> = row.get(1)?;
        let emb = blob.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok((row.get::<_, i64>(5)?, emb, row.get::<_, i64>(3)?, row.get::<_, Option<i64>>(4)?, row.get::<_, i64>(0)?))
    }).map_err(|e| e.to_string())?.filter_map(|r| r.ok()).collect();

    log_init(&format!("[FACE_SEARCH] stored_faces loaded: {}", stored.len()));

    // Map by f.vector_id (HNSW key), store (embedding, image_id, person_id, face_id)
    let stored_map: std::collections::HashMap<i64, (Vec<f32>, i64, Option<i64>, i64)> =
        stored.into_iter().map(|(vid, emb, iid, pid, fid)| (vid, (emb, iid, pid, fid))).collect();

    let mut scored: Vec<(i64, Option<i64>, f32)> = Vec::new();
    for hit in &hnsw_hits {
        if let Some((ref emb, image_id, person_id, _face_id)) = stored_map.get(&hit.id) {
            let score = cosine_sim(&fused_emb, emb);
            // DEBUG: log similarity for each hit
            log_init(&format!("[FACE_SEARCH] hit id={} image_id={} cosine={:.4}", hit.id, image_id, score));
            if score >= SIMILARITY_THRESHOLD {
                scored.push((*image_id, *person_id, score));
            }
        }
    }

    log_init(&format!("[FACE_SEARCH] after threshold: {} scored", scored.len()));

    if scored.is_empty() {
        return Ok(serde_json::json!({"results": [], "total": 0, "offset": offset, "limit": top_k}).to_string());
    }

    let mut best_per_image: Vec<(i64, f32, Option<i64>)> = Vec::new();
    for &(iid, pid, score) in &scored {
        if let Some(existing) = best_per_image.iter_mut().find(|x| x.0 == iid) {
            if score > existing.1 { existing.1 = score; }
        } else {
            best_per_image.push((iid, score, pid));
        }
    }

    let person_ids: Vec<i64> = best_per_image.iter().filter_map(|x| x.2).collect();
    let person_scores: std::collections::HashMap<i64, f32> =
        best_per_image.iter().filter_map(|x| x.2.map(|p| (p, x.1))).collect();

    let mut results: Vec<(i64, Option<i64>, f32, bool)> =
        best_per_image.iter().map(|(iid, score, pid)| (*iid, *pid, *score, false)).collect();

    for &pid in &person_ids {
        let rows: Vec<(i64, i64)> = {
            let mut s = match conn.prepare("SELECT id, image_id FROM faces WHERE person_id = ?") {
                Ok(s) => s,
                Err(_) => continue,
            };
            s.query_map(rusqlite::params![pid], |r| Ok((r.get(0)?, r.get(1)?)))
            .ok().map(|iter| iter.filter_map(|r| r.ok()).collect()).unwrap_or_default()
        };
        let best = person_scores.get(&pid).copied().unwrap_or(0.0);
        let seen: std::collections::HashSet<i64> = best_per_image.iter().map(|x| x.0).collect();
        for (_, iid) in rows {
            if !seen.contains(&iid) {
                results.push((iid, Some(pid), best, true));
            }
        }
    }

    results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    let total = results.len();
    let paged: Vec<_> = results.iter().skip(offset).take(top_k).collect();
    log_init(&format!("[FACE_SEARCH] expanded={} paged={}/{}", total, paged.len(), top_k));

    let img_ids: Vec<i64> = paged.iter().map(|x| x.0).collect();
    let mut asset_map = std::collections::HashMap::new();
    if !img_ids.is_empty() {
        let ph = img_ids.iter().map(|_| "?".to_string()).collect::<Vec<_>>().join(",");
        let q = format!("SELECT id, COALESCE(asset_id, path) FROM images WHERE id IN ({})", ph);
        let ps: Vec<&dyn rusqlite::ToSql> = img_ids.iter().map(|v| v as &dyn rusqlite::ToSql).collect();
        if let Ok(mut s) = conn.prepare(&q) {
            let rows_iter = s.query_map(ps.as_slice(), |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)));
            if let Ok(rows) = rows_iter {
                for row in rows {
                    if let Ok((id, path)) = row {
                        asset_map.insert(id, path);
                    }
                }
            }
        }
    }
    log_init(&format!("[FACE_SEARCH] asset_map size: {}", asset_map.len()));

    let json_results: Vec<_> = paged.iter().map(|(iid, pid, score, exp)| {
        serde_json::json!({
            "face_id": 0,
            "image_id": iid,
            "score": score,
            "person_id": pid,
            "is_expanded": exp,
            "thumbnail": serde_json::json!(null),
            "asset_id": asset_map.get(iid).cloned().unwrap_or_default(),
        })
    }).collect();

    Ok(serde_json::json!({
        "results": json_results,
        "total": total,
        "offset": offset,
        "limit": top_k,
    }).to_string())
}


/// Fuse multi-face embeddings (weighted average)
fn fuse_face_embeddings(features: &[FaceFeature]) -> Vec<f32> {
    if features.is_empty() {
        return Vec::new();
    }
    if features.len() == 1 {
        return features[0].embedding.values.as_slice().to_vec();
    }

    let dim = features[0].embedding.dim;
    let mut weights = Vec::with_capacity(features.len());
    let mut total = 0.0f32;

    for f in features {
        let w = f.detection.score
            * (0.5 + f.pose_score)
            * (0.5 + f.face_area_score);
        weights.push(w);
        total += w;
    }

    if total <= 0.0 {
        return features[0].embedding.values.as_slice().to_vec();
    }

    let mut fused = vec![0.0f32; dim];
    for (f, &w) in features.iter().zip(weights.iter()) {
        let weight = w / total;
        for (out, &v) in fused.iter_mut().zip(f.embedding.values.as_slice()) {
            *out += v * weight;
        }
    }

    // L2 normalize
    let norm = fused.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut fused {
            *v /= norm;
        }
    }
    fused
}

/// Visual/Object search implementation - synchronous
fn search_object_impl(image_bytes: &[u8], top_k: usize, offset: usize) -> Result<String, String> {
    // Get globals
    let visual_scanner = VISUAL_SCANNER.lock().unwrap();
    let visual_state = visual_scanner.as_ref().ok_or("VisualScanner not initialized")?;

    let visual_index = VISUAL_V2_INDEX.lock().unwrap();
    let visual_index = visual_index.as_ref().ok_or("VisualIndex not initialized")?;

    let db_lock = DB.lock().unwrap();
    let db = db_lock.as_ref().ok_or("DB not initialized")?;
    let conn = db.connection().map_err(|e| format!("DB connection error: {}", e))?;

    // Get image dimensions
    let img = image::load_from_memory(image_bytes).map_err(|e| format!("Image decode error: {}", e))?;
    let (img_w, img_h) = img.dimensions();

    // Extract DINO feature from full image (crop = full image since Flutter cropped it)
    let mut scanner = visual_state.scanner.lock().unwrap();
    let query_embedding = scanner.extract_feature_from_bytes(
        image_bytes,
        0.0, 0.0,
        img_w as f32, img_h as f32,
    ).map_err(|e| format!("DINO extraction error: {}", e))?;

    log_init(&format!("[OBJ_SEARCH] DINO embedding dim={}", query_embedding.len()));

    // L2 normalize
    let mut query_norm = query_embedding.clone();
    let norm = query_norm.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for v in &mut query_norm {
        *v /= norm;
    }

    // HNSW coarse search
    let coarse_k = ((top_k + offset) * 4).max(20);
    let hnsw_hits = visual_index.search(&query_norm, coarse_k)
        .map_err(|e| format!("HNSW search error: {}", e))?;

    if hnsw_hits.is_empty() {
        return Ok(serde_json::json!({
            "results": [],
            "total": 0,
            "offset": offset,
            "limit": top_k,
        }).to_string());
    }

    // Fetch stored features for re-ranking
    let feature_ids: Vec<i64> = hnsw_hits.iter().map(|h| h.id).collect();
    log_init(&format!("[OBJ_SEARCH] HNSW hit IDs: {:?}, scan_version={}", feature_ids, visual_state.config.scan_version));

    // Debug: check total features in DB and scan_versions
    {
        let sql = "SELECT scan_version, COUNT(*) FROM visual_scan_v2_features GROUP BY scan_version";
        if let Ok(mut stmt) = conn.prepare(sql) {
            let rows: Vec<(String, i64)> = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            }).ok()
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default();
            log_init(&format!("[OBJ_SEARCH] All scan_versions in DB: {:?}", rows));
        }
    }

    // Debug: check what scan_versions exist in DB for these IDs
    if !feature_ids.is_empty() {
        let placeholders: Vec<String> = feature_ids.iter().map(|_| "?".to_string()).collect();
        let sql = format!("SELECT id, scan_version FROM visual_scan_v2_features WHERE id IN ({})", placeholders.join(","));
        let params: Vec<&dyn rusqlite::ToSql> = feature_ids.iter().map(|v| v as &dyn rusqlite::ToSql).collect();
        if let Ok(mut stmt) = conn.prepare(&sql) {
            let rows: Vec<(i64, String)> = stmt.query_map(params.as_slice(), |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            }).ok()
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default();
            log_init(&format!("[OBJ_SEARCH] DB scan_versions for hit IDs: {:?}", rows));
        }
    }

    let stored_features = pf_application::visual_scan_v2::database::get_features_for_ids(
        &conn,
        &visual_state.config.scan_version,
        &feature_ids,
    ).map_err(|e| format!("Feature lookup error: {}", e))?;

    log_init(&format!("[OBJ_SEARCH] stored_features count={}", stored_features.len()));

    let feature_map: std::collections::HashMap<i64, _> = stored_features.into_iter()
        .map(|f| (f.0, f))
        .collect();

    // Exact cosine re-rank + MAX aggregation per image
    let mut best_per_image: std::collections::HashMap<i64, (f32, i64)> = std::collections::HashMap::new();

    for hit in &hnsw_hits {
        if let Some((fid, iid, emb, _, _, _, _, _)) = feature_map.get(&hit.id) {
            let exact_score = query_norm.iter().zip(emb.iter()).map(|(x, y)| x * y).sum::<f32>();
            let entry = best_per_image.entry(*iid);
            match entry {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert((exact_score, *fid));
                }
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    if exact_score > e.get().0 {
                        e.insert((exact_score, *fid));
                    }
                }
            }
        }
    }

    // Sort by score descending
    let mut sorted: Vec<_> = best_per_image.into_iter().collect();
    sorted.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
    let total_before_filter = sorted.len();

    // Filter by minimum similarity threshold (50%)
    let filtered: Vec<_> = sorted.into_iter()
        .filter(|(_, (score, _))| *score >= VISUAL_SEARCH_MIN_SIMILARITY)
        .collect();
    let total = filtered.len();

    log_init(&format!("[OBJ_SEARCH] filtered count={} (removed {} below {:.0}% similarity)",
        total, total_before_filter - total, VISUAL_SEARCH_MIN_SIMILARITY * 100.0));

    let paged: Vec<_> = filtered.iter().skip(offset).take(top_k).collect();

    // Look up image asset_ids for returning to Flutter
    let image_ids: Vec<i64> = paged.iter().map(|(image_id, _)| *image_id).collect();
    let mut image_asset_ids = std::collections::HashMap::new();
    if !image_ids.is_empty() {
        let placeholders: Vec<String> = image_ids.iter().map(|_| "?".to_string()).collect();
        // Select asset_id column - use COALESCE to fallback to path for old entries without asset_id
        let sql = format!(
            "SELECT id, COALESCE(asset_id, path) FROM images WHERE id IN ({})",
            placeholders.join(",")
        );
        log_init(&format!("[OBJ_SEARCH] SQL: {}", sql));
        let params: Vec<&dyn rusqlite::ToSql> = image_ids.iter().map(|v| v as &dyn rusqlite::ToSql).collect();
        if let Ok(mut stmt) = conn.prepare(&sql) {
            let rows: Vec<(i64, String)> = stmt.query_map(params.as_slice(), |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            }).ok()
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default();
            for (id, asset_id) in rows {
                log_init(&format!("[OBJ_SEARCH] image_id={} -> asset_id={}", id, asset_id));
                image_asset_ids.insert(id, asset_id);
            }
        }
        // Also log raw asset_id and path for debugging
        let debug_sql = format!(
            "SELECT id, asset_id, path FROM images WHERE id IN ({})",
            placeholders.join(",")
        );
        if let Ok(mut stmt) = conn.prepare(&debug_sql) {
            let rows: Vec<(i64, Option<String>, String)> = stmt.query_map(params.as_slice(), |row| {
                Ok((row.get::<_, i64>(0)?, row.get(1)?, row.get::<_, String>(2)?))
            }).ok()
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default();
            for (id, asset_id_opt, path) in rows {
                log_init(&format!("[OBJ_SEARCH] DEBUG image_id={} asset_id={:?} path={}", id, asset_id_opt, path));
            }
        }
    }

    let results: Vec<_> = paged.iter().map(|(image_id, (score, _))| {
        let asset_id = image_asset_ids.get(image_id).cloned();
        serde_json::json!({
            "face_id": 0,
            "image_id": image_id,
            "score": score,
            "person_id": null,
            "is_expanded": false,
            "thumbnail": null,
            "asset_id": asset_id,
        })
    }).collect();

    Ok(serde_json::json!({
        "results": results,
        "total": total,
        "offset": offset,
        "limit": top_k,
    }).to_string())
}

// ============================================================================
// License / Account FFI
// ============================================================================

/// Register account
#[no_mangle]
pub extern "C" fn pf_register(
    email: *const c_char,
    password: *const c_char,
) -> *mut c_char {
    let email = unsafe { std::ffi::CStr::from_ptr(email) }
        .to_string_lossy()
        .into_owned();
    let password = unsafe { std::ffi::CStr::from_ptr(password) }
        .to_string_lossy()
        .into_owned();

    let lm = match get_license_manager() {
        Ok(lm) => lm,
        Err(e) => return null_str(&format!("{{\"error\":\"{}\"}}", e)),
    };

    match lm.register(&email, &password) {
        Ok(account) => null_str(&serde_json::to_string(&account).unwrap_or_default()),
        Err(e) => null_str(&serde_json::json!({
            "code": format!("{:?}", e.code).to_lowercase().replace('_', ""),
            "message": e.message
        }).to_string()),
    }
}

/// Login account
#[no_mangle]
pub extern "C" fn pf_login(email: *const c_char, password: *const c_char) -> *mut c_char {
    let email = unsafe { std::ffi::CStr::from_ptr(email) }
        .to_string_lossy()
        .into_owned();
    let password = unsafe { std::ffi::CStr::from_ptr(password) }
        .to_string_lossy()
        .into_owned();

    let lm = match get_license_manager() {
        Ok(lm) => lm,
        Err(e) => return null_str(&format!("{{\"error\":\"{}\"}}", e)),
    };

    match lm.login(&email, &password) {
        Ok(account) => null_str(&serde_json::to_string(&account).unwrap_or_default()),
        Err(e) => null_str(&serde_json::json!({
            "code": format!("{:?}", e.code).to_lowercase().replace('_', ""),
            "message": e.message
        }).to_string()),
    }
}

/// Logout
#[no_mangle]
pub extern "C" fn pf_logout() -> i32 {
    match get_license_manager() {
        Ok(lm) => {
            lm.logout();
            0
        }
        Err(_) => -1,
    }
}

/// Get current account
#[no_mangle]
pub extern "C" fn pf_get_account() -> *mut c_char {
    let lm = match get_license_manager() {
        Ok(lm) => lm,
        Err(e) => return null_str(&format!("{{\"error\":\"{}\"}}", e)),
    };

    match lm.get_account() {
        Some(account) => null_str(&serde_json::to_string(&account).unwrap_or("null".to_string())),
        None => null_str("null"),
    }
}

/// Redeem activation code
#[no_mangle]
pub extern "C" fn pf_redeem_code(code: *const c_char) -> *mut c_char {
    let code = unsafe { std::ffi::CStr::from_ptr(code) }
        .to_string_lossy()
        .into_owned();

    let lm = match get_license_manager() {
        Ok(lm) => lm,
        Err(e) => return null_str(&format!("{{\"error\":\"{}\"}}", e)),
    };

    match lm.redeem_code(&code) {
        Ok(license) => null_str(&serde_json::to_string(&license).unwrap_or_default()),
        Err(e) => null_str(&serde_json::json!({
            "code": format!("{:?}", e.code).to_lowercase().replace('_', ""),
            "message": e.message
        }).to_string()),
    }
}

/// Get license status
#[no_mangle]
pub extern "C" fn pf_get_license_status() -> *mut c_char {
    let lm = match get_license_manager() {
        Ok(lm) => lm,
        Err(e) => return null_str(&format!("{{\"error\":\"{}\"}}", e)),
    };

    match lm.get_license_status() {
        Ok(Some(license)) => null_str(&serde_json::to_string(&license).unwrap_or_default()),
        Ok(None) => null_str("null"),
        Err(e) => null_str(&serde_json::json!({
            "code": format!("{:?}", e.code).to_lowercase().replace('_', ""),
            "message": e.message
        }).to_string()),
    }
}

/// Check if license is valid (1 = valid, 0 = invalid)
#[no_mangle]
pub extern "C" fn pf_check_license() -> i32 {
    let lm = match get_license_manager() {
        Ok(lm) => lm,
        Err(_) => return 0,
    };

    match lm.require_valid() {
        Ok(()) => 1,
        Err(_) => 0,
    }
}

/// Get remaining days
#[no_mangle]
pub extern "C" fn pf_get_remaining_days() -> i64 {
    let lm = match get_license_manager() {
        Ok(lm) => lm,
        Err(_) => return 0,
    };

    lm.get_remaining_days()
}
