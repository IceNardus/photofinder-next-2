//! `bootstrap` — 把所有依赖装配起来，返回 `AppContext`。
//!
//! 调用方（desktop / mobile shell）只需：
//! ```no_run
//! # use pf_application::bootstrap;
//! # // 实际调用见 `assemble`，这里只是说明性 pseudo-code
//! # let _ = bootstrap::default_scheduler_config();
//! ```

use std::sync::Arc;

use pf_ai::{CategorySearch, FacePipeline, ObjectEmbedder};
use pf_ai::similarity::{FeatureMatcher, KeypointExtractor};
use pf_config::{Config, ModelManager, PathResolver};
use pf_database::Database;
use pf_platform::PhotoProvider;
use pf_task::{ExecutorRegistry, SchedulerConfig, SqliteTaskScheduler};
use pf_vector::VectorIndex;
use tracing::info;

use crate::body_prototype_service::BodyPrototypeService;
use crate::context::AppContext;
use crate::error::ApplicationError;
use crate::executors;
use crate::identity_evidence::IdentityPipeline;
use crate::index::IndexService;
use crate::person::PersonService;
use crate::scan::ScanService;
use crate::search::SearchService;

/// 默认装配入口。
pub struct Bootstrapped {
    /// 应用上下文（所有 service / pipeline / index 都挂在 ctx 上）。
    pub ctx: AppContext,
    /// 任务调度器（带 SqliteTaskScheduler 具体类型，方便 start_workers）。
    pub scheduler: Arc<SqliteTaskScheduler>,
    /// executor 注册表（如果需要额外注册）。
    pub registry: Arc<ExecutorRegistry>,
    /// 扫描服务
    pub scan: Arc<ScanService>,
    /// 索引服务
    pub index: Arc<IndexService>,
    /// 搜索服务
    pub search: Arc<SearchService>,
    /// 人脸服务
    pub person: Arc<PersonService>,
    /// V2: Body prototype 服务
    pub body_prototype_service: Option<Arc<BodyPrototypeService>>,
}

/// 装配（带全部已知依赖）。
#[allow(clippy::too_many_arguments)]
pub fn assemble(
    config: Arc<Config>,
    model_manager: Arc<ModelManager>,
    database: Arc<Database>,
    face_pipeline: Arc<FacePipeline>,
    face_index: Arc<dyn VectorIndex>,
    body_pipeline: Option<Arc<pf_ai::BodyPipeline>>,
    body_index: Option<Arc<dyn VectorIndex>>,
    object_embedder: Option<Arc<dyn ObjectEmbedder>>,
    object_index: Option<Arc<dyn VectorIndex>>,
    photo_provider: Arc<dyn PhotoProvider>,
    path_resolver: Arc<dyn PathResolver>,
    category_search: Option<Arc<CategorySearch>>,
    superpoint: Option<Arc<dyn KeypointExtractor>>,
    lightglue: Option<Arc<dyn FeatureMatcher>>,
) -> Result<Bootstrapped, ApplicationError> {
    // 1) Scheduler + Registry
    let registry = Arc::new(ExecutorRegistry::new());
    let scheduler = Arc::new(SqliteTaskScheduler::new(
        (*database).clone(),
        registry.clone(),
    ));

    // 2) Services
    // scan 构造早于 index,因此 enable_patches 在此处固定为 false（patch_extractor 始终 None 直到未来 SuperPoint 加载）。
    let scan = Arc::new(ScanService::new(
        database.clone(),
        scheduler.clone(),
        photo_provider.clone(),
        false, // enable_patches
    ));
    let index = Arc::new(IndexService::new(
        database.clone(),
        face_pipeline.clone(),
        face_index.clone(),
        body_pipeline,
        body_index.clone(),
        object_embedder.clone(),
        object_index.clone(),
        photo_provider.clone(),
        None, // patch_extractor (wired when SuperPoint model loaded)
        None, // patch_index
    ));
    // V2: Body prototype service (created early so it's available for SearchService)
    let body_prototype_service = Arc::new(BodyPrototypeService::new(database.clone()));

    let search = Arc::new(SearchService::new(
        database.clone(),
        config.clone(),
        face_pipeline.clone(),
        face_index.clone(),
        None, // body_index (wired when YouTu model loaded)
        object_embedder,
        object_index.clone(),
        photo_provider.clone(),
        None, // patch_index
        None, // patch_search
        category_search,
        superpoint,
        lightglue,
    )
    .with_prototype_service(Arc::new(
        crate::prototype_service::PrototypeService::new(database.clone()),
    ))
    .with_body_prototype_service(body_prototype_service.clone()));

    // Phase 30/32: Create IdentityPipeline for shadow mode
    // Phase 32: Now with db and prototype_service for proper scoring
    let identity_pipeline = IdentityPipeline::new(
        face_index.clone(),
        body_index.as_ref().map(|b| b.clone()),
        database.clone(),
        Some(Arc::new(crate::prototype_service::PrototypeService::new(database.clone()))),
    );

    // Convert from pf_config::IdentityExecutionMode to identity_evidence::IdentityExecutionMode
    let execution_mode = match config.identity.execution_mode {
        pf_config::IdentityExecutionMode::Legacy => {
            crate::identity_evidence::IdentityExecutionMode::Legacy
        }
        pf_config::IdentityExecutionMode::Shadow => {
            crate::identity_evidence::IdentityExecutionMode::Shadow
        }
        pf_config::IdentityExecutionMode::NewPipeline => {
            crate::identity_evidence::IdentityExecutionMode::NewPipeline
        }
    };

    let person = Arc::new(
        PersonService::new(
            database.clone(),
            face_index.clone(),
            Default::default(),
        )
        .with_prototype_service(Arc::new(
            crate::prototype_service::PrototypeService::new(database.clone()),
        ))
        .with_identity_pipeline(Arc::new(identity_pipeline))
        .with_execution_mode(execution_mode),
    );

    // 3) Register executors
    executors::register_executors(&registry, database.clone(), index.clone(), person.clone());

    // 4) AppContext
    let ctx = AppContext::new(
        config,
        model_manager,
        database,
        face_pipeline,
        face_index,
        scheduler.clone(),
        photo_provider,
        path_resolver,
    );

    info!("bootstrap complete");
    Ok(Bootstrapped {
        ctx,
        scheduler,
        registry,
        scan,
        index,
        search,
        person,
        body_prototype_service: Some(body_prototype_service),
    })
}

/// 启动后台 worker（懒加载，但可以手动触发）。
pub fn start_workers(scheduler: &SqliteTaskScheduler) {
    scheduler.start_workers();
}

/// 便捷 config 工厂。
pub fn default_scheduler_config() -> SchedulerConfig {
    SchedulerConfig::default()
}