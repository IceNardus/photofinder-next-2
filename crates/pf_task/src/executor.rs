//! 任务执行器 trait。

use async_trait::async_trait;

use crate::progress::TaskContext;
use crate::TaskError;

/// 任务执行器。
///
/// `pf_task` 不知道具体业务（不 import `pf_ai` / `pf_application`）。
/// `pf_application` 实现本 trait 并注册到 `TaskScheduler`。
#[async_trait]
pub trait TaskExecutor: Send + Sync {
    /// 支持的 `TaskKind` 判别式。
    fn kind(&self) -> crate::scheduler::TaskKindDiscriminant;

    /// 人类可读名字（用于日志 / UI）。
    fn name(&self) -> &'static str;

    /// 执行任务。可通过 `ctx.should_cancel` 协作取消，通过 `ctx.progress` 上报进度。
    async fn execute(&self, ctx: &TaskContext) -> Result<TaskOutcome, TaskError>;
}

/// 执行结果。
#[derive(Debug, Clone)]
pub enum TaskOutcome {
    /// 成功，可携带任意元数据（JSON-friendly）
    Success(Option<serde_json::Value>),
    /// 失败，应附带原因
    Failed(String),
    /// 取消
    Cancelled,
}