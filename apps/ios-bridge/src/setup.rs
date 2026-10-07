//! `bootstrap_ios` — iOS 端启动装配。
//!
//! 流程与 `apps/desktop/src-tauri/src/desktop_setup.rs::bootstrap()` 类似，
//! 但用 iOS-specific providers：
//! - `IosPathResolver` (PF_DATA_DIR 环境变量注入)
//! - `PhotoKitPhotoProvider::with_bridge(IosNativeMediaBridge)` (PHAsset 通过 Swift bridge)

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use pf_ai::{
    ArcFaceEmbedder, FacePipeline, MobileClipEmbedder, ScrfdDetector, SimpleAligner,
};
use pf_application::Bootstrapped;
use pf_config::{Config, ModelManager, ModelRegistry};
use pf_database::Database;
use pf_platform::ios::{
    IosNativeMediaBridge, IosPathResolver, NativeMediaBridge, PhotoKitPhotoProvider,
};
use pf_platform::{PathResolver, PhotoProvider};
use pf_vector::{HnswIndex, VectorIndex};
use tracing::{info, warn};

// ============================================================================
// Logging
// ============================================================================

/// 初始化 tracing。写到 `data_dir/logs/photofinder.log` + stderr。
pub fn init_logging() -> Result<()> {
    let resolver = IosPathResolver::new();
    let log_dir = resolver.log_dir().unwrap_or_else(|_| PathBuf::from("."));
    std::fs::create_dir_all(&log_dir).ok();

    let file_appender = tracing_appender::rolling::daily(&log_dir, "photofinder.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    use std::sync::OnceLock;
    static GUARD: OnceLock<tracing_appender::non_blocking::WorkerGuard> = OnceLock::new();
    let _ = GUARD.set(_guard);

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,ort=warn,hnsw_rs=warn"));

    use tracing_subscriber::fmt::writer::MakeWriterExt;
    let writer = non_blocking.and(std::io::stderr);

    tracing_subscriber::fmt()
        .with_writer(writer)
        .with_env_filter(env_filter)
        .with_ansi(false)
        .init();

    info!("[iOS] logging initialized at {}", log_dir.display());
    Ok(())
}

// ============================================================================
// Config / Model registry（与 desktop 共享，简化版 — iOS app bundle 里只有一处）
// ============================================================================

fn load_config(app_dir: &Path) -> Arc<Config> {
    let path = app_dir.join("config.toml");
    let mut config = Config::default();
    if path.exists() {
        match std::fs::read_to_string(&path) {
            Ok(s) => {
                if let Ok(v) = toml::from_str::<toml::Value>(&s) {
                    if let Ok(c) = pf_config::deserialize_config(&v) {
                        config = c;
                        info!("[iOS] loaded config from {}", path.display());
                    }
                }
            }
            Err(_) => {
                warn!("[iOS] config read failed, using default");
            }
        }
    } else {
        info!("[iOS] no config.toml, using default");
    }
    Arc::new(config)
}

fn load_model_registry(app_dir: &Path) -> Arc<ModelRegistry> {
    // iOS app bundle 结构: <App>/<bin>  +  <App>/models/manifest.toml
    let candidates = [
        app_dir.join("models").join("manifest.toml"),
        app_dir.join("manifest.toml"),
        PathBuf::from("models").join("manifest.toml"),
    ];
    for path in &candidates {
        if path.exists() {
            if let Ok(reg) = pf_config::load_manifest_toml(path) {
                info!("[iOS] loaded model registry from {}", path.display());
                return Arc::new(reg);
            }
        }
    }
    warn!("[iOS] no model manifest found, using empty registry");
    Arc::new(ModelRegistry::new())
}

// ============================================================================
// Model loaders（精简版，只加载 face pipeline + object embedder）
// ============================================================================

fn try_load_face_pipeline(
    model_manager: Arc<ModelManager>,
    config: Arc<Config>,
) -> Option<Arc<FacePipeline>> {
    let detector = match pf_config::load_model_or_warn(
        &model_manager,
        pf_config::ModelId::scrfd_500m_bnkps(),
    ) {
        Ok(p) => match ScrfdDetector::load(&p) {
            Ok(d) => Some(d),
            Err(e) => {
                warn!("[iOS] SCRFD load failed: {e}");
                None
            }
        },
        Err(_) => None,
    };

    let embedder = match pf_config::load_model_or_warn(
        &model_manager,
        pf_config::ModelId::arcface_w600k_r50(),
    ) {
        Ok(p) => match ArcFaceEmbedder::load(&p) {
            Ok(e) => Some(e),
            Err(e) => {
                warn!("[iOS] ArcFace load failed: {e}");
                None
            }
        },
        Err(_) => None,
    };

    match (detector, embedder) {
        (Some(d), Some(e)) => {
            let detector: Arc<dyn pf_ai::FaceDetector> = d;
            let embedder: Arc<dyn pf_ai::FaceEmbedder> = e;
            let aligner: Arc<dyn pf_ai::FaceAligner> = Arc::new(SimpleAligner::new());
            let qf = pf_ai::QualityFilter::from_config(
                config.face.min_detector_score,
                config.face.min_face_size,
                config.face.min_quality,
                config.face.max_yaw,
            );
            info!("[iOS] face pipeline loaded");
            Some(Arc::new(FacePipeline::new(detector, aligner, embedder, qf)))
        }
        _ => {
            warn!("[iOS] face pipeline NOT loaded (model files missing)");
            None
        }
    }
}

fn try_load_object_embedder(
    model_manager: Arc<ModelManager>,
) -> Option<Arc<dyn pf_ai::ObjectEmbedder>> {
    let path = pf_config::load_model_or_warn(
        &model_manager,
        pf_config::ModelId::mobileclip_s2(),
    )
    .ok()?;
    let embedder = match MobileClipEmbedder::load(&path) {
        Ok(e) => {
            info!("[iOS] MobileCLIP embedder loaded");
            e
        }
        Err(e) => {
            warn!("[iOS] MobileCLIP load failed: {e}");
            return None;
        }
    };
    let embedder: Arc<dyn pf_ai::ObjectEmbedder> = embedder;
    Some(embedder)
}

// ============================================================================
// HNSW
// ============================================================================

fn build_hnsw_index(
    dim: usize,
    path: &Path,
    basename: &str,
) -> Result<Arc<dyn VectorIndex>> {
    let parent = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).ok();

    let data_file = parent.join(format!("{}.hnsw.data", basename));
    let graph_file = parent.join(format!("{}.hnsw.graph", basename));

    let mut index = HnswIndex::new(dim, parent, basename);
    if data_file.exists() && graph_file.exists() {
        match index.load() {
            Ok(_) => info!(
                "[iOS] HNSW {} loaded ({} entries)",
                basename,
                index.len()
            ),
            Err(e) => {
                warn!("[iOS] HNSW {basename} load failed: {e}, starting fresh");
                index = HnswIndex::new(dim, parent, basename);
            }
        }
    } else {
        info!("[iOS] creating fresh HNSW index at {}", data_file.display());
    }
    // Phase 1.1: iOS-bridge 没有 Config,沿用 DEFAULT_EF_SEARCH=64。
    index.set_ef_search(pf_vector::DEFAULT_EF_SEARCH);
    Ok(Arc::new(index))
}

// ============================================================================
// Bootstrap 入口
// ============================================================================

/// iOS 端装配入口。返回完整 `Bootstrapped`,Swift 把它存在静态 STATE 里。
///
/// 失败原因通常：models 缺失 / DB 无法创建 / HNSW 初始化失败。
pub fn bootstrap() -> Result<Bootstrapped> {
    let resolver = Arc::new(IosPathResolver::new());
    let app_dir = resolver.data_dir().context("resolve data_dir")?;
    std::fs::create_dir_all(&app_dir).context("create data_dir")?;
    info!("[iOS] data_dir = {}", app_dir.display());

    let config = load_config(&app_dir);
    let registry = load_model_registry(&app_dir);
    let model_manager = Arc::new(ModelManager::new(registry.clone(), resolver.clone()));

    let db_path = resolver.db_path().context("resolve db_path")?;
    let database = Arc::new(Database::open(&db_path, pf_database::builtin_migrations())?);

    let face_pipeline = try_load_face_pipeline(model_manager.clone(), config.clone())
        .ok_or_else(|| anyhow!("iOS: face pipeline required — install models in app bundle"))?;

    let index_dir = app_dir.join("index");
    std::fs::create_dir_all(&index_dir).ok();
    let face_index = build_hnsw_index(512, &index_dir.join("face.hnsw"), "face")?;

    // 启动时检测：HNSW 为空但 DB 有 indexed face → rebuild
    if face_index.len() == 0 {
        let face_count: i64 = database
            .transaction(|tx| tx.faces().count())
            .unwrap_or(0);
        if face_count > 0 {
            warn!(
                face_count,
                "[iOS] HNSW empty but DB has faces, rebuilding"
            );
            // Phase 2.5+ 调 rebuild_face_index_sync（待 helper function）
        }
    }

    let object_embedder = try_load_object_embedder(model_manager.clone());
    let object_index = build_hnsw_index(512, &index_dir.join("object.hnsw"), "object")?;

    // PhotoKit bridge — 每次 list_photos 都通过 Swift extern "C" 调 PhotoKit
    let bridge: Box<dyn NativeMediaBridge> = Box::new(IosNativeMediaBridge::new());
    let photo_provider: Arc<dyn PhotoProvider> =
        Arc::new(PhotoKitPhotoProvider::with_bridge(bridge));

    let boot = pf_application::assemble(
        config,
        model_manager,
        database,
        face_pipeline,
        face_index,
        object_embedder,
        Some(object_index),
        photo_provider,
        resolver.clone(),
        None, // category_search
        None, // superpoint
        None, // lightglue
    )?;

    info!("[iOS] bootstrap complete");
    Ok(boot)
}