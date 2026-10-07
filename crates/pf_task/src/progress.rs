//! 进度上报 + 任务上下文。

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

/// 进度上报 trait。
pub trait ProgressReporter: Send + Sync {
    /// 报告当前进度。
    ///
    /// `completed` / `total` 单位由调用方定义（图片张数 / bytes / patches 等）。
    /// `message` 是人类可读描述（"indexing face 5/10"）。
    fn report(&self, completed: u64, total: u64, message: &str);
}

/// 任务执行上下文。
///
/// 通过 `Arc` 共享给 executor。
#[derive(Clone)]
pub struct TaskContext {
    /// 任务 ID
    pub task_id: i64,
    /// 进度上报器
    pub progress: Arc<dyn ProgressReporter>,
    /// 协作式取消标记
    pub should_cancel: Arc<AtomicBool>,
}

impl TaskContext {
    /// 构造一个新上下文。
    pub fn new(
        task_id: i64,
        progress: Arc<dyn ProgressReporter>,
        should_cancel: Arc<AtomicBool>,
    ) -> Self {
        Self {
            task_id,
            progress,
            should_cancel,
        }
    }

    /// 检查并返回取消状态。
    pub fn is_cancelled(&self) -> bool {
        self.should_cancel.load(Ordering::SeqCst)
    }

    /// 请求取消。
    pub fn request_cancel(&self) {
        self.should_cancel.store(true, Ordering::SeqCst);
    }
}

/// 任务统计快照。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStats {
    /// 待处理数
    pub pending: u64,
    /// 正在执行数
    pub running: u64,
    /// 已完成数
    pub completed: u64,
    /// 失败数
    pub failed: u64,
    /// 已取消数
    pub cancelled: u64,
    /// 当前是否暂停
    pub paused: bool,
}

impl TaskStats {
    /// 全零构造（测试 / 默认值）。
    pub fn zero() -> Self {
        Self {
            pending: 0,
            running: 0,
            completed: 0,
            failed: 0,
            cancelled: 0,
            paused: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_works() {
        let cancel = Arc::new(AtomicBool::new(false));
        let ctx = TaskContext::new(1, Arc::new(NoopProgress), cancel.clone());
        assert!(!ctx.is_cancelled());
        ctx.request_cancel();
        assert!(ctx.is_cancelled());
    }

    struct NoopProgress;
    impl ProgressReporter for NoopProgress {
        fn report(&self, _c: u64, _t: u64, _m: &str) {}
    }
}