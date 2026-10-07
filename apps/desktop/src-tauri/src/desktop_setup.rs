//! Desktop 启动装配：从零到 `pf_application::Bootstrapped`。

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use pf_ai::{
    ArcFaceEmbedder, BodyCropStrategy, BodyPipeline, CategorySearch, FacePipeline,
    LightGlueMatcher, MobileClipEmbedder, ObjectEmbedder, ObjectPipeline, ScrfdDetector,
    SimpleAligner, SuperPointExtractor, YouTuReIdEmbedder, YoloV8Detector,
};
use pf_ai::body::BODY_EMBEDDING_DIM;
use pf_ai::similarity::{FeatureMatcher, KeypointExtractor};
use pf_application::Bootstrapped;
use pf_config::{Config, ModelManager, ModelRegistry};
use pf_core::FACE_MODEL_NAME;
use pf_database::Database;
use pf_platform::{DesktopPathResolver, FileSystemPhotoProvider, PathResolver};
use pf_vector::{HnswIndex, VectorIndex};
use tracing::{debug, info, warn};

// ============================================================================
// Logging
// ============================================================================

/// 初始化 logging：写文件 + 输出 stderr。
pub fn init_logging() -> Result<()> {
    let resolver = DesktopPathResolver::new();
    let log_dir = resolver.log_dir().unwrap_or_else(|_| PathBuf::from("."));
    std::fs::create_dir_all(&log_dir).ok();

    let file_appender = tracing_appender::rolling::daily(&log_dir, "photofinder.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    use once_cell::sync::OnceCell;
    static GUARD: OnceCell<tracing_appender::non_blocking::WorkerGuard> = OnceCell::new();
    let _ = GUARD.set(_guard);

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,ort=warn,hnsw_rs=warn"));

    // 使用 tracing-appender 的 `non_blocking` 写文件；
    // 再包一层直接写 stderr（终端）的 writer
    use tracing_subscriber::fmt::writer::MakeWriterExt;
    let writer = non_blocking.and(std::io::stderr);

    tracing_subscriber::fmt()
        .with_writer(writer)
        .with_env_filter(env_filter)
        .with_ansi(false)
        .init();

    info!("logging initialized at {}", log_dir.display());
    Ok(())
}

// ============================================================================
// Config / Model registry
// ============================================================================

/// 加载 Config（尝试读 `config.toml`，失败用 default）。
pub fn load_config(app_dir: &Path) -> Arc<Config> {
    let path = app_dir.join("config.toml");
    let mut config = Config::default();
    if path.exists() {
        match std::fs::read_to_string(&path) {
            Ok(s) => match toml::from_str::<toml::Value>(&s) {
                Ok(v) => {
                    if let Ok(c) = pf_config::deserialize_config(&v) {
                        config = c;
                        info!("loaded config from {}", path.display());
                    }
                }
                Err(e) => warn!("config parse failed: {e}, using default"),
            },
            Err(e) => warn!("config read failed: {e}, using default"),
        }
    } else {
        info!("no config.toml at {}, using default", path.display());
    }
    Arc::new(config)
}

/// 加载 ModelRegistry（读 `models/manifest.toml`，失败用内置 default）。
pub fn load_model_registry(app_dir: &Path) -> Arc<ModelRegistry> {
    // 搜索路径（优先序）：
    // 1. exe 旁 Resources/models/manifest.toml（macOS bundle）
    // 2. exe 旁 models/manifest.toml（dev 或 bundle 根）
    // 3. exe 上层父目录/models/（dev: 项目根，bundle: app bundle 根）
    // 4. app_dir/models/manifest.toml
    // 5. CWD models/manifest.toml
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()));

    let mut candidates: Vec<PathBuf> = Vec::new();

    // 1, 2: exe 旁
    if let Some(ref d) = exe_dir {
        candidates.push(d.join("resources").join("models").join("manifest.toml"));
        candidates.push(d.join("models").join("manifest.toml"));
    }

    // 3: exe 上层父目录（dev & bundle 分别处理）
    // dev (exe = .../photofinder-next-2/target/release/):
    //   ancestors()[3] = photofinder-next-2 (项目根)
    // bundle (exe = .../PhotoFinder Next.app/Contents/MacOS/):
    //   ancestors()[2] = PhotoFinder Next.app (bundle 根)
    if let Some(ref d) = exe_dir {
        for &idx in &[2, 3] {
            if let Some(anc) = d.ancestors().nth(idx) {
                candidates.push(anc.join("Contents").join("Resources").join("models").join("manifest.toml"));
                candidates.push(anc.join("models").join("manifest.toml"));
            }
        }
    }

    // 4: app_dir
    candidates.push(app_dir.join("models").join("manifest.toml"));
    // 5: CWD
    candidates.push(PathBuf::from("models").join("manifest.toml"));

    for path in &candidates {
        if path.exists() {
            match pf_config::load_manifest_toml(path) {
                Ok(reg) => {
                    info!("loaded model registry from {}", path.display());
                    return Arc::new(reg);
                }
                Err(e) => warn!("manifest parse failed at {}: {e}", path.display()),
            }
        }
    }
    warn!("no model manifest found, using empty registry");
    Arc::new(ModelRegistry::new())
}

// ============================================================================
// Models
// ============================================================================

/// 加载人脸 detector + aligner + embedder（模型缺失返回 None）。
pub fn try_load_face_pipeline(
    model_manager: Arc<ModelManager>,
    config: Arc<Config>,
) -> Option<Arc<FacePipeline>> {
    let detector = match pf_config::load_model_or_warn(&model_manager, pf_config::ModelId::scrfd_500m_bnkps()) {
        Ok(p) => match ScrfdDetector::load(&p) {
            Ok(d) => Some(d),
            Err(e) => {
                warn!("SCRFD load failed: {e}");
                None
            }
        },
        Err(_) => None,
    };

    let embedder = match pf_config::load_model_or_warn(&model_manager, pf_config::ModelId::arcface_w600k_r50()) {
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

/// 单独加载 MobileCLIP embedder（用于 ROI 索引，即使没有 YOLOv8n 也能工作）。
pub fn try_load_object_embedder(model_manager: Arc<ModelManager>) -> Option<Arc<dyn ObjectEmbedder>> {
    let embedder = match pf_config::load_model_or_warn(&model_manager, pf_config::ModelId::mobileclip_s2()) {
        Ok(p) => match MobileClipEmbedder::load(&p) {
            Ok(e) => Some(e as Arc<dyn ObjectEmbedder>),
            Err(e) => {
                warn!("MobileCLIP load failed: {e}");
                None
            }
        },
        Err(_) => None,
    };
    if embedder.is_some() {
        info!("object embedder (MobileCLIP) loaded");
    } else {
        warn!("object embedder NOT loaded");
    }
    embedder
}

/// 加载对象 pipeline（YOLOv8n detector + MobileCLIP embedder）。
pub fn try_load_object_pipeline(model_manager: Arc<ModelManager>) -> Option<Arc<ObjectPipeline>> {
    let detector = match pf_config::load_model_or_warn(&model_manager, pf_config::ModelId::yolov8n()) {
        Ok(p) => match YoloV8Detector::load(&p) {
            Ok(d) => Some(d as Arc<dyn pf_ai::ObjectDetector>),
            Err(e) => {
                warn!("YOLOv8n load failed: {e}");
                None
            }
        },
        Err(_) => None,
    };
    let embedder = try_load_object_embedder(model_manager);

    match (detector, embedder) {
        (Some(d), Some(e)) => {
            info!("object pipeline (detector+embedder) loaded");
            Some(Arc::new(ObjectPipeline::new(d, e)))
        }
        _ => {
            // 即使没有 detector，只要有 embedder 就能做 ROI 索引
            None
        }
    }
}

/// 加载 SuperPoint 关键点提取器（用于对象检索的几何校验）。
pub fn try_load_superpoint(model_manager: Arc<ModelManager>) -> Option<Arc<dyn KeypointExtractor>> {
    let extractor = match pf_config::load_model_or_warn(&model_manager, pf_config::ModelId::superpoint()) {
        Ok(p) => match SuperPointExtractor::load(&p) {
            Ok(e) => Some(e as Arc<dyn KeypointExtractor>),
            Err(e) => {
                warn!("SuperPoint load failed: {e}");
                None
            }
        },
        Err(_) => None,
    };
    if extractor.is_some() {
        info!("superpoint loaded");
    } else {
        warn!("superpoint NOT loaded (geometric verification disabled)");
    }
    extractor
}

/// 创建 LightGlue 特征匹配器（无模型文件，直接构造）。
///
/// Phase 5：min_matches 从 `config.object.lightglue_min_matches` 取（默认 3）。
pub fn try_load_lightglue(min_matches: usize) -> Option<Arc<dyn FeatureMatcher>> {
    Some(Arc::new(LightGlueMatcher::new(
        pf_ai::object::lightglue::RANSAC_THRESHOLD,
        pf_ai::object::lightglue::RATIO_THRESHOLD,
        min_matches.max(1),
    )))
}

/// 加载 Body pipeline（YouTu Re-ID embedder + BodyCropStrategy）。
///
/// V2: Body 用于 candidate retrieval，不做 verification。
pub fn try_load_body_pipeline(model_manager: Arc<ModelManager>) -> Option<Arc<BodyPipeline>> {
    // YouTu 模型文件路径（从 ModelManager 获取）
    let model_path = model_manager
        .path(pf_config::ModelId("person_reid_youtu_2021nov".into()))
        .ok()?;

    match YouTuReIdEmbedder::load(&model_path) {
        Ok(embedder) => {
            let pipeline = BodyPipeline::new(embedder, BodyCropStrategy::HeadBody);
            info!("body pipeline (YouTu Re-ID) loaded");
            Some(Arc::new(pipeline))
        }
        Err(e) => {
            warn!("YouTu Re-ID load failed: {e}");
            None
        }
    }
}

// ============================================================================
// Bootstrap
// ============================================================================

/// 应用主装配入口。
pub fn bootstrap() -> Result<Bootstrapped> {
    let resolver = Arc::new(DesktopPathResolver::new());
    let app_dir = resolver
        .data_dir()
        .context("resolve data_dir")?;
    std::fs::create_dir_all(&app_dir).context("create data_dir")?;

    info!("data_dir = {}", app_dir.display());

    // 1) Config + Model registry
    let config = load_config(&app_dir);
    let registry = load_model_registry(&app_dir);
    let model_manager = Arc::new(ModelManager::new(registry.clone(), resolver.clone()));

    // 2) Database
    let db_path = resolver.db_path().context("resolve db_path")?;
    let database = Arc::new(Database::open(&db_path, pf_database::builtin_migrations())?);

    // 3) Face pipeline
    let face_pipeline = try_load_face_pipeline(model_manager.clone(), config.clone())
        .ok_or_else(|| anyhow!("face pipeline required — install models (see models/manifest.toml)"))?;

    // 4) HNSW face index (load or create)
    let index_dir = app_dir.join("index");
    std::fs::create_dir_all(&index_dir).ok();
    let face_index_path = index_dir.join("face.hnsw");
    let face_index = build_hnsw_index(512, &face_index_path, "face", config.vector.ef_search)?;

    // 启动时检测：HNSW index 与 DB 已索引 face 不一致 → 立即 rebuild。
    // 旧逻辑只在 len < db 时重建;但 hnsw_rs 随机后缀 dump + stale 规范文件会让
    // load() 拿到"非空但过时"的索引(线上 bug:图库内查询图搜不到自己)。
    // #162 起也处理 index.len > db 的方向:db.reset() 只 drop SQL 表不清 HNSW,
    // 重扫会往旧 index 累积过期条目 → 磁盘索引点数是 DB 的超集,必须同样重建。
    let db_indexed: i64 = database
        .transaction(|tx| {
            let by_status = tx.faces().count_by_status()?;
            Ok::<_, pf_database::DatabaseError>(
                by_status
                    .iter()
                    .find(|(s, _)| *s == pf_database::FaceStatus::Indexed)
                    .map(|(_, n)| *n)
                    .unwrap_or(0),
            )
        })
        .unwrap_or(0);
    if face_index.len() != db_indexed as usize {
        warn!(
            db_indexed,
            index_len = face_index.len(),
            "HNSW index diverges from DB indexed faces, rebuilding..."
        );
        rebuild_face_index_sync(&database, &*face_index);
    }

    // 5) Object pipeline + embedder + HNSW index + CategorySearch
    let object_pipeline = try_load_object_pipeline(model_manager.clone());
    // 单独加载 embedder（即使没有 YOLOv8n detector，ROI 索引也需要它）
    let object_embedder = try_load_object_embedder(model_manager.clone());
    let category_search = object_pipeline.as_ref().map(|p| {
        Arc::new(CategorySearch::new(p.detector(), p.embedder()))
    });
    let object_index_path = index_dir.join("object.hnsw");
    let object_index = build_hnsw_index(512, &object_index_path, "object", config.vector.ef_search)?;

    // 5b) Body pipeline + HNSW index (V2: YouTu Re-ID for candidate retrieval)
    // Controlled by config.identity.enable_body_reid
    let body_pipeline = if config.identity.enable_body_reid {
        try_load_body_pipeline(model_manager.clone())
    } else {
        info!("body_reid disabled via config, skipping body pipeline load");
        None
    };
    let body_index_path = index_dir.join("body.hnsw");
    let body_index = body_pipeline.as_ref().map(|_| {
        build_hnsw_index(BODY_EMBEDDING_DIM, &body_index_path, "body", config.vector.ef_search)
            .expect("body index creation failed")
    });

    // 6) SuperPoint + LightGlue（几何校验）
    let superpoint = try_load_superpoint(model_manager.clone());
    let lightglue = try_load_lightglue(config.object.lightglue_min_matches);

    // 7) Photo provider
    let library_roots = resolve_library_roots(&app_dir);
    let photo_provider: Arc<dyn pf_platform::PhotoProvider> =
        Arc::new(FileSystemPhotoProvider::new(library_roots));

    // 8) Assemble
    let boot = pf_application::assemble(
        config,
        model_manager,
        database,
        face_pipeline,
        face_index,
        body_pipeline,
        body_index,
        object_embedder,
        Some(object_index),
        photo_provider,
        resolver.clone(),
        category_search,
        superpoint,
        lightglue,
    )?;

    info!("bootstrap complete");
    Ok(boot)
}

/// 创建 / 加载 HNSW 索引。
///
/// Phase 1.1:`ef_search` 在创建后立即写入,所有后续 search 都用此值
/// (hnsw_rs 0.3.4 的 `search(data, knbn, ef_arg)` 每调用取 ef_arg)。
fn build_hnsw_index(
    dim: usize,
    path: &Path,
    basename: &str,
    ef_search: usize,
) -> Result<Arc<dyn pf_vector::VectorIndex>> {
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
    index.set_ef_search(ef_search);
    Ok(Arc::new(index))
}

/// 从 app_dir/library.toml 或 default 读 library roots。
fn resolve_library_roots(app_dir: &Path) -> Vec<PathBuf> {
    let config_path = app_dir.join("library.toml");
    if config_path.exists() {
        if let Ok(s) = std::fs::read_to_string(&config_path) {
            if let Ok(v) = toml::from_str::<toml::Value>(&s) {
                if let Some(arr) = v.get("roots").and_then(|r| r.as_array()) {
                    let roots: Vec<PathBuf> = arr
                        .iter()
                        .filter_map(|x| x.as_str().map(PathBuf::from))
                        .collect();
                    if !roots.is_empty() {
                        info!("loaded {} library roots", roots.len());
                        return roots;
                    }
                }
            }
        }
    }
    if let Some(p) = dirs::picture_dir() {
        info!("using default library root: {}", p.display());
        vec![p]
    } else {
        info!("no picture_dir, using CWD");
        vec![PathBuf::from(".")]
    }
}

/// 从 DB 恢复人脸 HNSW 索引（同步版本，用于启动时空索引检测）。
///
/// Phase 2 改造:
/// - 先 `clear()` HNSW
/// - 只处理 status=Indexed 且 vector_id 非空的 face
/// - 异常 case 自动 mark_failed:vector_id 缺失 / embedding 缺失
fn rebuild_face_index_sync(database: &Database, face_index: &dyn VectorIndex) {
    // Step 1: 清空 HNSW
    if let Err(e) = face_index.clear() {
        warn!(error = %e, "rebuild_face_index: HNSW clear failed; stale entries may persist");
    }

    // Step 2: 只取 Indexed faces
    let faces = match database
        .transaction(|tx| tx.faces().list_by_status(pf_database::FaceStatus::Indexed))
    {
        Ok(f) => f,
        Err(e) => {
            warn!(error = %e, "rebuild_face_index: list_by_status(Indexed) failed");
            return;
        }
    };

    let mut inserted = 0usize;
    for face in &faces {
        let vector_id = match face.vector_id {
            Some(v) => v,
            None => {
                debug!(face_id = face.id, "Indexed face without vector_id; marking failed");
                let _ = database.transaction(|tx| {
                    tx.faces()
                        .mark_failed(face.id, "Indexed but vector_id IS NULL after rebuild")
                });
                continue;
            }
        };
        // 注意:face_embeddings 里实际存的 model_name 是嵌入器的模型名
        // (FACE_MODEL_NAME,见 arcface.rs ModelVersion::new),get_embedding 是
        // exact match,传旧值 "arcface" 会取不到 → mark_failed 打坏整个库。
        let emb = match database.transaction(|tx| {
            tx.faces().get_embedding(face.id, FACE_MODEL_NAME)
        }) {
            Ok(Some(e)) => e,
            Ok(None) => {
                debug!(face_id = face.id, "Indexed face missing embedding; marking failed");
                let _ = database.transaction(|tx| {
                    tx.faces()
                        .mark_failed(face.id, "Indexed but no embedding row")
                });
                continue;
            }
            Err(e) => {
                warn!(error = %e, "rebuild_face_index: get_embedding failed");
                continue;
            }
        };
        if let Err(e) = face_index.insert(vector_id, &emb) {
            warn!(vector_id, error = %e, "rebuild insert failed");
            continue;
        }
        inserted += 1;
    }

    if let Err(e) = face_index.save() {
        warn!(error = %e, "rebuild_face_index: save failed");
    }
    info!(inserted, "rebuild_face_index: done");
}
