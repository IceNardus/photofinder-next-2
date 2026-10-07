//! 任务调度器 trait。

use async_trait::async_trait;

use crate::progress::TaskStats;
use crate::task::{Priority, Task, TaskFilter, TaskKind};
use crate::TaskError;

/// 任务调度器抽象。
///
/// 实现方负责：
/// - 持久化（SQLite / 内存）
/// - 优先级排序
/// - 取出一条任务并 dispatch 给对应的 `TaskExecutor`
#[async_trait]
pub trait TaskScheduler: Send + Sync {
    /// 入队一条新任务，返回任务 id。
    async fn enqueue(&self, kind: TaskKind, priority: Priority) -> Result<i64, TaskError>;

    /// 取消任务（仅 Pending 可取消；Running 需通过 `should_cancel` 协作）。
    async fn cancel(&self, task_id: i64) -> Result<(), TaskError>;

    /// 暂停整个调度器。
    async fn pause(&self) -> Result<(), TaskError>;

    /// 恢复调度器。
    async fn resume(&self) -> Result<(), TaskError>;

    /// 整体统计。
    async fn status(&self) -> Result<TaskStats, TaskError>;

    /// 待处理任务数（用于背压）。
    async fn pending_count(&self) -> Result<u64, TaskError>;

    /// 列出任务（按过滤器）。
    async fn list(&self, filter: TaskFilter) -> Result<Vec<Task>, TaskError>;
}

/// `TaskKind` 的判别式（用于过滤 / HashMap 索引）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskKindDiscriminant {
    /// Scan
    Scan,
    /// IndexFace
    IndexFace,
    /// IndexObject
    IndexObject,
    /// IndexImage
    IndexImage,
    /// IndexPatch
    IndexPatch,
    /// BuildIndex
    BuildIndex,
    /// ClusterFaces
    ClusterFaces,
}

impl TaskKindDiscriminant {
    /// 从 `TaskKind` 提取判别式。
    pub fn from_kind(kind: &crate::task::TaskKind) -> Self {
        use crate::task::TaskKind;
        match kind {
            TaskKind::Scan { .. } => Self::Scan,
            TaskKind::IndexFace { .. } => Self::IndexFace,
            TaskKind::IndexObject { .. } => Self::IndexObject,
            TaskKind::IndexImage { .. } => Self::IndexImage,
            TaskKind::IndexPatch { .. } => Self::IndexPatch,
            TaskKind::BuildIndex { .. } => Self::BuildIndex,
            TaskKind::ClusterFaces => Self::ClusterFaces,
        }
    }
}