//! `AppContext` — 装配所有依赖的根容器。
//!
//! Phase 0 脚手架：仅字段声明 + 构造方法签名，不实际初始化。
//! Phase 1 真实实现在 `pf_application::bootstrap`。

use std::sync::Arc;

use pf_ai::FacePipeline;
use pf_config::{Config, ModelManager, PathResolver};
use pf_database::Database;
use pf_platform::PhotoProvider;
use pf_task::TaskScheduler;
use pf_vector::VectorIndex;

/// 应用上下文（持有所有共享依赖）。
///
/// 设计：
/// - 全部字段都是 `Arc<...>`，clone 整个 ctx 即可在 Tauri command / service 间共享
/// - 不持有任何独占状态（每个 service 自己管理 mutable state）
#[derive(Clone)]
pub struct AppContext {
    /// 配置
    pub config: Arc<Config>,
    /// 模型管理
    pub model_manager: Arc<ModelManager>,
    /// 数据库
    pub database: Arc<Database>,
    /// 人脸 pipeline
    pub face_pipeline: Arc<FacePipeline>,
    /// 人脸向量索引
    pub face_index: Arc<dyn VectorIndex>,
    /// 任务调度器
    pub task_scheduler: Arc<dyn TaskScheduler>,
    /// 照片提供器
    pub photo_provider: Arc<dyn PhotoProvider>,
    /// 路径解析器（方便 service 用）
    pub path_resolver: Arc<dyn PathResolver>,
}

impl AppContext {
    /// 构造（Phase 1 由 `bootstrap::run_desktop` 调用）。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: Arc<Config>,
        model_manager: Arc<ModelManager>,
        database: Arc<Database>,
        face_pipeline: Arc<FacePipeline>,
        face_index: Arc<dyn VectorIndex>,
        task_scheduler: Arc<dyn TaskScheduler>,
        photo_provider: Arc<dyn PhotoProvider>,
        path_resolver: Arc<dyn PathResolver>,
    ) -> Self {
        Self {
            config,
            model_manager,
            database,
            face_pipeline,
            face_index,
            task_scheduler,
            photo_provider,
            path_resolver,
        }
    }
}