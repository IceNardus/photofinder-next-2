//! Mobile bootstrap：把 pf_application 装配到 iOS / Android。
//!
//! 与 desktop_setup 的差异：
//! - PathResolver 用 `AndroidPathResolver` / `IosPathResolver`
//! - PhotoProvider 用 `MediaStorePhotoProvider` / `PhotoKitPhotoProvider`（占位）
//! - 模型路径在移动端指向 app bundle 内 `models/` 目录
//! - 库文件由 `PF_DATA_DIR` / `PF_CACHE_DIR` / `PF_MODELS_DIR` 环境变量指定
//!
//! 设计原则：
//! - 模型缺失不 panic
//! - HNSW 缺失则自动创建空索引
//! - 任何不可恢复错误立即 panic
//!
//! ## 与桌面端的代码复用
//!
//! `pf_application::assemble` 接受 trait object，所以 `PathResolver` 和
//! `PhotoProvider` 都按 `Arc<dyn ...>` 传入。装配步骤与桌面端完全一致，
//! 只换 trait 实现。

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use pf_ai::{ArcFaceEmbedder, CategorySearch, FacePipeline, ObjectPipeline, ScrfdDetector, SimpleAligner};
use pf_application::Bootstrapped;
use pf_config::{Config, ModelManager, ModelRegistry};
use pf_database::Database;
use pf_platform::PathResolver;
use pf_vector::{HnswIndex, VectorIndex};
use tracing::{info, warn};

// ============================================================================
// Logging
// ============================================================================

/// 初始化 logging（移动端写文件到 data_dir/logs，stderr 也输出）。
pub fn init_logging() -> Result<()> {
    use once_cell::sync::OnceCell;
    static GUARD: OnceCell<tracing_appender::non_blocking::WorkerGuard> = OnceCell::new();

    let log_dir = std::env::var("PF_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("photofinder-mobile"));
    let log_dir = log_dir.join("logs");
    std::fs::create_dir_all(&log_dir).ok();

    let file_appender = tracing_appender::rolling::daily(&log_dir, "photofinder.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
    let _ = GUARD.set(guard);

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,ort=warn,hnsw_rs=warn"));

    tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .with_env_filter(env_filter)
        .with_ansi(false)
        .init();

    info!("logging initialized at {}", log_dir.display());
    Ok(())
}

// ============================================================================
// Models
// ============================================================================

/// 加载人脸 detector + aligner + embedder（移动端，模型缺失返回 None）。
pub fn try_load_face_pipeline(
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
                warn!("SCRFD load failed: {e}");
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
                warn!("ArcFace load failed: {e}");
                None
            }
        },
        Err(_) => None,
    };

    match (detector, embedder) {
        (Some(d), Some(e)) => {
            let aligner: Arc<dyn pf_ai::FaceAligner> = Arc::new(SimpleAligner::new());
            let detector: Arc<dyn pf_ai::FaceDetector> = d;
            let embedder: Arc<dyn pf_ai::FaceEmbedder> = e;
            let qf = pf_ai::QualityFilter::from_config(
                config.face.min_detector_score,
                config.face.min_face_size,
                config.face.min_quality,
                config.face.max_yaw,
            );
            info!("face pipeline loaded");
            Some(Arc::new(FacePipeline::new(detector, aligner, embedder, qf)))
        }
        _ => {
            warn!("face pipeline NOT loaded (model files missing)");
            None
        }
    }
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
    let mut index = HnswIndex::new(dim, parent, basename);
    let data_file = parent.join(format!("{}.hnsw.data", basename));
    let graph_file = parent.join(format!("{}.hnsw.graph", basename));
    if data_file.exists() && graph_file.exists() {
        match index.load() {
            Ok(_) => info!("loaded HNSW index from {}", data_file.display()),
            Err(e) => {
                warn!("HNSW load failed: {e}, starting fresh");
                index = HnswIndex::new(dim, parent, basename);
            }
        }
    } else {
        info!("creating fresh HNSW index at {}", data_file.display());
    }
    // Phase 1.1: 移动端没有 Config 注入,用 DEFAULT_EF_SEARCH=64
    // (与 desktop / 之前硬编码值一致)。
    index.set_ef_search(pf_vector::DEFAULT_EF_SEARCH);
    Ok(Arc::new(index))
}

// ============================================================================
// Bootstrap
// ============================================================================

/// 移动端装配入口。
///
/// 调用方（iOS / Android Tauri shell）传 platform-specific PathResolver / PhotoProvider。
#[allow(clippy::too_many_arguments)]
pub fn bootstrap(
    config: Arc<Config>,
    _registry: Arc<ModelRegistry>,
    database: Arc<Database>,
    face_pipeline: Option<Arc<FacePipeline>>,
    face_index: Arc<dyn VectorIndex>,
    object_pipeline: Option<Arc<ObjectPipeline>>,
    object_embedder: Option<Arc<dyn pf_ai::ObjectEmbedder>>,
    object_index: Option<Arc<dyn VectorIndex>>,
    photo_provider: Arc<dyn pf_platform::PhotoProvider>,
    path_resolver: Arc<dyn PathResolver>,
    model_manager: Arc<ModelManager>,
) -> Result<Bootstrapped> {
    let face_pipeline = face_pipeline
        .ok_or_else(|| anyhow!("face pipeline required — install models (see models/manifest.toml)"))?;

    let category_search = object_pipeline.as_ref().map(|p| {
        Arc::new(CategorySearch::new(p.detector(), p.embedder()))
    });

    let boot = pf_application::assemble(
        config,
        model_manager,
        database,
        face_pipeline,
        face_index,
        None, // body_pipeline
        None, // body_index
        object_embedder,
        object_index,
        photo_provider,
        path_resolver,
        category_search,
        None, // superpoint
        None, // lightglue
    )?;

    info!("mobile bootstrap complete");
    Ok(boot)
}

/// 移动端默认配置（直接读 Config::default）。
pub fn default_config() -> Arc<Config> {
    Arc::new(Config::default())
}

/// 空 ModelRegistry（占位 — 实际从 manifest.toml 读）。
pub fn empty_registry() -> Arc<ModelRegistry> {
    Arc::new(ModelRegistry::new())
}

/// 加载 ModelRegistry（读 env: PF_MODELS_DIR/manifest.toml）。
pub fn load_model_registry() -> Result<Arc<ModelRegistry>> {
    let models_dir = std::env::var("PF_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("photofinder-mobile/models"));
    let manifest_path = models_dir.join("manifest.toml");

    if manifest_path.exists() {
        let reg = pf_config::load_manifest_toml(&manifest_path)
            .context("load manifest.toml")?;
        info!("loaded model registry from {}", manifest_path.display());
        Ok(Arc::new(reg))
    } else {
        warn!(
            "no manifest.toml at {}, using empty registry",
            manifest_path.display()
        );
        Ok(Arc::new(ModelRegistry::new()))
    }
}

/// 移动端 PathResolver 工厂（按 target_os 分发）。
pub fn mobile_path_resolver() -> Arc<dyn PathResolver> {
    #[cfg(target_os = "android")]
    {
        Arc::new(pf_platform::AndroidPathResolver::new())
    }
    #[cfg(target_os = "ios")]
    {
        Arc::new(pf_platform::IosPathResolver::new())
    }
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    {
        // 开发期（非移动 target）fallback 到 desktop
        Arc::new(pf_platform::DesktopPathResolver::new())
    }
}

/// 移动端 PhotoProvider 工厂。
pub fn mobile_photo_provider() -> Arc<dyn pf_platform::PhotoProvider> {
    #[cfg(target_os = "android")]
    {
        Arc::new(pf_platform::MediaStorePhotoProvider::new())
    }
    #[cfg(target_os = "ios")]
    {
        Arc::new(pf_platform::PhotoKitPhotoProvider::new())
    }
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    {
        // 开发期 fallback — 用一个空 roots 列表的 filesystem provider
        Arc::new(pf_platform::FileSystemPhotoProvider::new(vec![]))
    }
}

/// 移动端 model manager（用 platform path resolver）。
pub fn mobile_model_manager(path_resolver: Arc<dyn PathResolver>) -> Arc<ModelManager> {
    let registry = match load_model_registry() {
        Ok(r) => r,
        Err(e) => {
            warn!("failed to load manifest: {e}, using empty");
            Arc::new(ModelRegistry::new())
        }
    };
    Arc::new(ModelManager::new(registry, path_resolver))
}

/// 移动端 DB（创建 data_dir/photofinder.db）。
pub fn mobile_database() -> Result<Arc<Database>> {
    let data_dir = std::env::var("PF_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("photofinder-mobile"));
    std::fs::create_dir_all(&data_dir)?;
    let db_path = data_dir.join("photofinder.db");
    let db = Database::open(&db_path, pf_database::builtin_migrations())?;
    Ok(Arc::new(db))
}

/// 移动端 HNSW 索引（创建 data_dir/index/face.hnsw）。
pub fn mobile_face_index() -> Result<Arc<dyn VectorIndex>> {
    let data_dir = std::env::var("PF_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("photofinder-mobile"));
    let index_dir = data_dir.join("index");
    std::fs::create_dir_all(&index_dir)?;
    let face_index_path = index_dir.join("face.hnsw");
    build_hnsw_index(512, &face_index_path, "face")
}
