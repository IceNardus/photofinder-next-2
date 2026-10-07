//! `TaskExecutor` 实现：把 `TaskKind` 映射到具体业务。
//!
//! 通过 `ctx.task_id` 查 DB 拿到 `kind_json`，反序列化出 image_id 后调用 service。

use std::sync::Arc;

use async_trait::async_trait;
use pf_database::Database;
use pf_task::{TaskContext, TaskError, TaskExecutor, TaskKindDiscriminant, TaskOutcome};
use serde_json::Value;
use tracing::{error, info, warn};

use crate::index::IndexService;

/// 从 task_id 反查 image_id（解析 `kind_json`）。
async fn image_id_from_task(db: &Database, task_id: i64) -> Result<i64, TaskError> {
    let db = db.clone();
    let kind_json = tokio::task::spawn_blocking(move || {
        db.transaction(|tx| tx.tasks().get_by_id(task_id).map(|opt| opt.map(|r| r.kind_json)))
    })
    .await
    .map_err(|e| TaskError::Executor(format!("join: {e}")))?
    .map_err(|e| TaskError::Database(e.to_string()))?
    .ok_or(TaskError::NotFound(task_id))?;

    let val: Value = serde_json::from_str(&kind_json)
        .map_err(|e| TaskError::Executor(format!("invalid kind_json: {e}")))?;

    // TaskKind 序列化格式：{"IndexFace":{"image_id":1}} 或 {"IndexObject":{"image_id":1}} 等
    // 需要先获取 variant 名，再从嵌套对象中取 image_id
    for variant in ["IndexFace", "IndexObject", "IndexImage", "IndexPatch"] {
        if let Some(inner) = val.get(variant).and_then(|v| v.get("image_id")) {
            if let Some(id) = inner.as_i64() {
                return Ok(id);
            }
        }
    }
    Err(TaskError::Executor("missing image_id in kind_json".into()))
}

/// Scan 任务执行器（仅作为 progress marker）。
pub struct ScanExecutor;

#[async_trait]
impl TaskExecutor for ScanExecutor {
    fn kind(&self) -> TaskKindDiscriminant {
        TaskKindDiscriminant::Scan
    }
    fn name(&self) -> &'static str {
        "scan"
    }
    async fn execute(&self, ctx: &TaskContext) -> Result<TaskOutcome, TaskError> {
        if ctx.is_cancelled() {
            return Ok(TaskOutcome::Cancelled);
        }
        info!(task_id = ctx.task_id, "scan task marker processed");
        Ok(TaskOutcome::Success(None))
    }
}

/// IndexFace 任务执行器。
pub struct IndexFaceExecutor {
    pub db: Arc<Database>,
    pub index_service: Arc<IndexService>,
}

#[async_trait]
impl TaskExecutor for IndexFaceExecutor {
    fn kind(&self) -> TaskKindDiscriminant {
        TaskKindDiscriminant::IndexFace
    }
    fn name(&self) -> &'static str {
        "index_face"
    }
    async fn execute(&self, ctx: &TaskContext) -> Result<TaskOutcome, TaskError> {
        if ctx.is_cancelled() {
            return Ok(TaskOutcome::Cancelled);
        }
        let image_id = image_id_from_task(&self.db, ctx.task_id).await?;
        let photo_id = self
            .index_service
            .photo_id_for(image_id)
            .await
            .map_err(|e| TaskError::Executor(format!("photo_id_for: {e}")))?;
        match self
            .index_service
            .index_faces_for_image(image_id, &photo_id)
            .await
        {
            Ok(summary) => {
                ctx.progress.report(1, 1, &format!("indexed image {}", image_id));
                Ok(TaskOutcome::Success(Some(serde_json::to_value(&summary).unwrap_or(Value::Null))))
            }
            Err(e) => {
                error!(image_id, error = %e, "index_image failed");
                Ok(TaskOutcome::Failed(e.to_string()))
            }
        }
    }
}

/// IndexObject 任务执行器。
pub struct IndexObjectExecutor {
    pub db: Arc<Database>,
    pub index_service: Arc<IndexService>,
}

#[async_trait]
impl TaskExecutor for IndexObjectExecutor {
    fn kind(&self) -> TaskKindDiscriminant {
        TaskKindDiscriminant::IndexObject
    }
    fn name(&self) -> &'static str {
        "index_object"
    }
    async fn execute(&self, ctx: &TaskContext) -> Result<TaskOutcome, TaskError> {
        if ctx.is_cancelled() {
            return Ok(TaskOutcome::Cancelled);
        }
        let image_id = match image_id_from_task(&self.db, ctx.task_id).await {
            Ok(id) => id,
            Err(e) => {
                error!(task_id = ctx.task_id, error = %e, "image_id_from_task failed");
                return Err(e);
            }
        };
        let photo_id = match self
            .index_service
            .photo_id_for(image_id)
            .await
        {
            Ok(id) => id,
            Err(e) => {
                error!(image_id, error = %e, "photo_id_for failed");
                return Err(TaskError::Executor(format!("photo_id_for: {e}")));
            }
        };
        info!(image_id, task_id = ctx.task_id, "IndexObject executor starting");
        match self
            .index_service
            .index_objects_for_image(image_id, &photo_id)
            .await
        {
            Ok(summary) => {
                info!(image_id, object_count = summary.object_count, indexed = summary.indexed_object_count, "index_objects_for_image completed");
                Ok(TaskOutcome::Success(
                    Some(serde_json::to_value(&summary).unwrap_or(Value::Null)),
                ))
            }
            Err(e) => {
                error!(image_id, error = %e, "index_objects failed");
                Ok(TaskOutcome::Failed(e.to_string()))
            }
        }
    }
}

/// IndexImage 任务执行器 — 一张图片的完整索引（face + object 顺序执行 + 批量写 DB）。
///
/// 替代 `IndexFace + IndexObject` 配对：从根本上消除同一图片的并发写竞争。
pub struct IndexImageExecutor {
    pub db: Arc<Database>,
    pub index_service: Arc<IndexService>,
}

#[async_trait]
impl TaskExecutor for IndexImageExecutor {
    fn kind(&self) -> TaskKindDiscriminant {
        TaskKindDiscriminant::IndexImage
    }
    fn name(&self) -> &'static str {
        "index_image"
    }
    async fn execute(&self, ctx: &TaskContext) -> Result<TaskOutcome, TaskError> {
        if ctx.is_cancelled() {
            return Ok(TaskOutcome::Cancelled);
        }
        let image_id = image_id_from_task(&self.db, ctx.task_id).await?;
        let photo_id = self
            .index_service
            .photo_id_for(image_id)
            .await
            .map_err(|e| TaskError::Executor(format!("photo_id_for: {e}")))?;
        info!(image_id, task_id = ctx.task_id, "IndexImage executor starting (face→object batched)");
        match self
            .index_service
            .index_image(image_id, &photo_id)
            .await
        {
            Ok(summary) => {
                ctx.progress.report(
                    1,
                    1,
                    &format!(
                        "indexed image {} ({} faces, {} objects)",
                        image_id, summary.indexed_face_count, summary.indexed_object_count
                    ),
                );
                Ok(TaskOutcome::Success(Some(serde_json::to_value(&summary).unwrap_or(Value::Null))))
            }
            Err(e) => {
                error!(image_id, error = %e, "index_image failed");
                Ok(TaskOutcome::Failed(e.to_string()))
            }
        }
    }
}

/// ClusterFaces 任务执行器（触发全量聚类）。
pub struct ClusterFacesExecutor {
    pub person_service: Arc<crate::person::PersonService>,
}

#[async_trait]
impl TaskExecutor for ClusterFacesExecutor {
    fn kind(&self) -> TaskKindDiscriminant {
        TaskKindDiscriminant::ClusterFaces
    }
    fn name(&self) -> &'static str {
        "cluster_faces"
    }
    async fn execute(&self, ctx: &TaskContext) -> Result<TaskOutcome, TaskError> {
        if ctx.is_cancelled() {
            return Ok(TaskOutcome::Cancelled);
        }
        info!(task_id = ctx.task_id, "cluster faces dispatched");
        match self.person_service.cluster_all().await {
            Ok(summary) => Ok(TaskOutcome::Success(
                Some(serde_json::to_value(&summary).unwrap_or(Value::Null)),
            )),
            Err(e) => {
                error!(error = %e, "cluster_all failed");
                Ok(TaskOutcome::Failed(e.to_string()))
            }
        }
    }
}

/// IndexPatch 任务执行器（Phase 3 — SuperPoint + LightGlue patch 索引）。
pub struct IndexPatchExecutor {
    pub db: Arc<Database>,
    pub index_service: Arc<IndexService>,
}

#[async_trait]
impl TaskExecutor for IndexPatchExecutor {
    fn kind(&self) -> TaskKindDiscriminant {
        TaskKindDiscriminant::IndexPatch
    }
    fn name(&self) -> &'static str {
        "index_patch"
    }
    async fn execute(&self, ctx: &TaskContext) -> Result<TaskOutcome, TaskError> {
        if ctx.is_cancelled() {
            return Ok(TaskOutcome::Cancelled);
        }
        let image_id = image_id_from_task(&self.db, ctx.task_id).await?;
        let photo_id = self
            .index_service
            .photo_id_for(image_id)
            .await
            .map_err(|e| TaskError::Executor(format!("photo_id_for: {e}")))?;
        info!(image_id, task_id = ctx.task_id, "index_patch started");
        // Phase E: 真正的 SuperPoint + VLAD patch 索引
        match self.index_service.index_patch_image(image_id, &photo_id).await {
            Ok(count) => {
                ctx.progress.report(1, 1, &format!("patched image {} ({} patches)", image_id, count));
                Ok(TaskOutcome::Success(Some(serde_json::json!({
                    "image_id": image_id,
                    "patched": true,
                    "patch_count": count,
                }))))
            }
            Err(e) => {
                warn!(image_id, error = %e, "index_patch failed");
                // 即使失败也标记完成（避免卡住队列）
                Ok(TaskOutcome::Failed(e.to_string()))
            }
        }
    }
}

/// 把所有 executor 注册到 registry。
pub fn register_executors(
    registry: &pf_task::ExecutorRegistry,
    db: Arc<Database>,
    index_service: Arc<IndexService>,
    person_service: Arc<crate::person::PersonService>,
) {
    registry.register(Arc::new(ScanExecutor));
    registry.register(Arc::new(IndexImageExecutor {
        db: db.clone(),
        index_service: index_service.clone(),
    }));
    registry.register(Arc::new(IndexFaceExecutor {
        db: db.clone(),
        index_service: index_service.clone(),
    }));
    registry.register(Arc::new(IndexObjectExecutor {
        db: db.clone(),
        index_service: index_service.clone(),
    }));
    registry.register(Arc::new(IndexPatchExecutor {
        db,
        index_service,
    }));
    registry.register(Arc::new(ClusterFacesExecutor { person_service }));
}