//! SQLite-backed `TaskScheduler`。
//!
//! - 任务持久化在 SQLite (`tasks` 表)
//! - worker 线程在独立的 `tokio` task 中轮询
//! - 取消通过 `Arc<AtomicBool>` 写到 `TaskContext` 上的协作式 flag
//! - 默认 2 个 worker（桌面端够用；移动端按需调整）

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use pf_database::{Database, DatabaseError, NewTask, DbTaskStatus};
use serde_json;
use tokio::sync::Notify;
use tracing::{debug, error, info, warn};

use crate::executor::TaskOutcome;
use crate::progress::{ProgressReporter, TaskContext, TaskStats};
use crate::registry::ExecutorRegistry;
use crate::scheduler::{TaskKindDiscriminant, TaskScheduler};
use crate::task::{Priority, Task, TaskFilter, TaskKind, TaskStatus};
use crate::TaskError;

/// 调度器配置。
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// worker 数量
    pub workers: usize,
    /// 任务排队时轮询间隔（空闲时）
    pub poll_interval: std::time::Duration,
    /// 任务默认最大重试次数
    pub default_max_retries: u32,
    /// 首次 `enqueue` 时是否懒启动 worker。
    /// 生产保持 `true`；测试关闭后 `enqueue` 不会触发 worker，
    /// 使 "任务仍处于 Pending" 的断言具有确定性（消除与 `claim_next` 的竞态）。
    pub auto_start_workers: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            workers: 2,
            poll_interval: std::time::Duration::from_millis(500),
            default_max_retries: 3,
            auto_start_workers: true,
        }
    }
}

/// `TaskKind` 判别式 → 进度 channel 端点（用于 UI 订阅）。
pub type ProgressSink = Arc<dyn ProgressReporter>;

/// 默认进度上报器（仅写日志，不上报外部）。
pub struct LoggingProgressReporter;

impl ProgressReporter for LoggingProgressReporter {
    fn report(&self, completed: u64, total: u64, message: &str) {
        debug!(completed, total, message, "task progress");
    }
}

/// SQLite-backed 任务调度器。
pub struct SqliteTaskScheduler {
    db: Database,
    registry: Arc<ExecutorRegistry>,
    /// 进度回调（外部替换；默认仅 logging）
    progress: parking_lot::RwLock<ProgressSink>,
    config: SchedulerConfig,
    paused: Arc<AtomicBool>,
    /// 唤醒 worker 的通知器（enqueue / resume 时触发）
    wake: Arc<Notify>,
    /// running task id → 取消 flag
    ///
    /// `Arc` 让 `start_workers` 出来的 worker 与 `cancel()` 看同一份 map。
    /// 修复:之前 `start_workers` 自己 new 一份 inner map,导致 cancel() 永远拿不到 running flag。
    cancel_flags: Arc<parking_lot::Mutex<std::collections::HashMap<i64, Arc<AtomicBool>>>>,
    worker_handles: parking_lot::Mutex<Vec<std::thread::JoinHandle<()>>>,
    started: parking_lot::Mutex<bool>,
    /// BUG #11 修复:Drop 时通知 worker 退出,而不是让线程在 3600s sleep 里空转。
    stop_signal: Arc<AtomicBool>,
}

impl SqliteTaskScheduler {
    /// 构造（不启动 worker）。
    pub fn new(db: Database, registry: Arc<ExecutorRegistry>) -> Self {
        Self::with_config(db, registry, SchedulerConfig::default())
    }

    /// 自定义 config。
    pub fn with_config(
        db: Database,
        registry: Arc<ExecutorRegistry>,
        config: SchedulerConfig,
    ) -> Self {
        let default_progress: ProgressSink = Arc::new(LoggingProgressReporter);
        Self {
            db,
            registry,
            progress: parking_lot::RwLock::new(default_progress),
            config,
            paused: Arc::new(AtomicBool::new(false)),
            wake: Arc::new(Notify::new()),
            cancel_flags: Arc::new(parking_lot::Mutex::new(Default::default())),
            worker_handles: parking_lot::Mutex::new(Vec::new()),
            started: parking_lot::Mutex::new(false),
            stop_signal: Arc::new(AtomicBool::new(false)),
        }
    }

    /// 替换进度回调（外部用于 Tauri event 推送）。
    pub fn set_progress_sink(&self, sink: ProgressSink) {
        *self.progress.write() = sink;
    }

    /// 启动 worker 线程（懒加载：首次 enqueue 或 start_workers 触发）。
    ///
    /// 与 ai-next 一致：在独立系统线程中创建独立的 Tokio runtime，
    /// 不依赖调用方的 async context。
    pub fn start_workers(&self) {
        let mut started = self.started.lock();
        if *started {
            return;
        }
        *started = true;
        drop(started);

        info!("[Scheduler] start_workers: spawning worker thread");

        let db = self.db.clone();
        let registry = self.registry.clone();
        let progress = self.current_progress();
        let paused = self.paused.clone();
        let wake = self.wake.clone();
        // 共享同一份 cancel_flags map:worker 写、cancel() 读
        let cancel_flags = self.cancel_flags.clone();
        let stop_signal = self.stop_signal.clone();
        let poll = self.config.poll_interval;
        let workers = self.config.workers;
        let mut handles = self.worker_handles.lock();

        let handle = std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                for i in 0..workers {
                    let db = db.clone();
                    let registry = registry.clone();
                    let progress = progress.clone();
                    let paused = paused.clone();
                    let wake = wake.clone();
                    let cancel_flags = cancel_flags.clone();
                    let stop_signal = stop_signal.clone();
                    tokio::spawn(worker_loop(
                        i,
                        db,
                        registry,
                        progress,
                        paused,
                        wake,
                        cancel_flags,
                        stop_signal,
                        poll,
                    ));
                }
                info!(workers, "task scheduler started");
                // 保持 runtime 活跃,直到 stop_signal 置位
                while !stop_signal.load(Ordering::SeqCst) {
                    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                }
                info!("task scheduler runtime exiting (stop_signal set)");
            });
        });

        handles.push(handle);
    }

    /// 取出当前进度回调（克隆 `Arc`）。
    fn current_progress(&self) -> ProgressSink {
        self.progress.read().clone()
    }

    /// 唤醒 worker。
    fn wake(&self) {
        self.wake.notify_waiters();
    }

    /// 取出当前运行的取消 flag。
    pub fn cancel_flag_for(&self, task_id: i64) -> Option<Arc<AtomicBool>> {
        self.cancel_flags.lock().get(&task_id).cloned()
    }
}

#[async_trait::async_trait]
impl TaskScheduler for SqliteTaskScheduler {
    async fn enqueue(&self, kind: TaskKind, priority: Priority) -> Result<i64, TaskError> {
        // 懒启动 worker（测试可关闭，见 SchedulerConfig::auto_start_workers）
        if self.config.auto_start_workers {
            self.start_workers();
        }

        let kind_json = serde_json::to_string(&kind)
            .map_err(|e| TaskError::Executor(format!("serialize TaskKind: {e}")))?;
        let max_retries = self.config.default_max_retries as i32;

        let db = self.db.clone();
        let id = tokio::task::spawn_blocking(move || {
            db.transaction(|tx| {
                let mut repo = tx.tasks();
                let new = NewTask {
                    kind_json,
                    priority: priority as i32,
                    max_retries,
                };
                repo.insert(&new)
            })
        })
        .await
        .map_err(|e| TaskError::Executor(format!("join: {e}")))?
        .map_err(|e| TaskError::Database(e.to_string()))?;

        self.wake();
        Ok(id)
    }

    async fn cancel(&self, task_id: i64) -> Result<(), TaskError> {
        // 1) 如果正在运行 → 设置 cancel flag
        let flag = {
            let flags = self.cancel_flags.lock();
            flags.get(&task_id).cloned()
        };
        if let Some(f) = flag {
            f.store(true, Ordering::SeqCst);
            return Ok(());
        }
        // 2) 否则尝试在 DB 中直接取消 pending
        let db = self.db.clone();
        let updated = tokio::task::spawn_blocking(move || {
            db.transaction(|tx| {
                let mut repo = tx.tasks();
                repo.cancel_pending(task_id)
            })
        })
        .await
        .map_err(|e| TaskError::Executor(format!("join: {e}")))?
        .map_err(|e| TaskError::Database(e.to_string()))?;
        if !updated {
            // 任务不在 pending，也可能不存在 / 已完成
            let db = self.db.clone();
            let exists: Result<bool, DatabaseError> = tokio::task::spawn_blocking(move || {
                db.transaction(|tx| {
                    let repo = tx.tasks();
                    repo.get_by_id(task_id).map(|opt| opt.is_some())
                })
            })
            .await
            .map_err(|e| TaskError::Executor(format!("join: {e}")))?;
            if !exists.unwrap_or(false) {
                return Err(TaskError::NotFound(task_id));
            }
        }
        Ok(())
    }

    async fn pause(&self) -> Result<(), TaskError> {
        self.paused.store(true, Ordering::SeqCst);
        info!("task scheduler paused");
        Ok(())
    }

    async fn resume(&self) -> Result<(), TaskError> {
        self.paused.store(false, Ordering::SeqCst);
        self.wake();
        info!("task scheduler resumed");
        Ok(())
    }

    async fn status(&self) -> Result<TaskStats, TaskError> {
        let db = self.db.clone();
        let counts = tokio::task::spawn_blocking(move || {
            db.transaction(|tx| {
                let repo = tx.tasks();
                Ok::<_, DatabaseError>((
                    repo.count_by_status(DbTaskStatus::Pending)?,
                    repo.count_by_status(DbTaskStatus::Running)?,
                    repo.count_by_status(DbTaskStatus::Completed)?,
                    repo.count_by_status(DbTaskStatus::Failed)?,
                    repo.count_by_status(DbTaskStatus::Cancelled)?,
                ))
            })
        })
        .await
        .map_err(|e| TaskError::Executor(format!("join: {e}")))?
        .map_err(|e| TaskError::Database(e.to_string()))?;
        Ok(TaskStats {
            pending: counts.0 as u64,
            running: counts.1 as u64,
            completed: counts.2 as u64,
            failed: counts.3 as u64,
            cancelled: counts.4 as u64,
            paused: self.paused.load(Ordering::SeqCst),
        })
    }

    async fn pending_count(&self) -> Result<u64, TaskError> {
        let db = self.db.clone();
        let n = tokio::task::spawn_blocking(move || {
            db.transaction(|tx| {
                let repo = tx.tasks();
                repo.count_by_status(DbTaskStatus::Pending)
            })
        })
        .await
        .map_err(|e| TaskError::Executor(format!("join: {e}")))?
        .map_err(|e| TaskError::Database(e.to_string()))?;
        Ok(n as u64)
    }

    async fn list(&self, filter: TaskFilter) -> Result<Vec<Task>, TaskError> {
        let db = self.db.clone();
        let rows = tokio::task::spawn_blocking(move || {
            db.transaction(|tx| {
                let repo = tx.tasks();
                let rows = if let Some(status) = filter.status {
                    let target = match status {
                        TaskStatus::Pending => DbTaskStatus::Pending,
                        TaskStatus::Running => DbTaskStatus::Running,
                        TaskStatus::Completed => DbTaskStatus::Completed,
                        TaskStatus::Failed => DbTaskStatus::Failed,
                        TaskStatus::Cancelled => DbTaskStatus::Cancelled,
                    };
                    repo.list_by_status(target, filter.limit)?
                } else {
                    repo.list(filter.limit)?
                };
                Ok::<_, DatabaseError>(rows)
            })
        })
        .await
        .map_err(|e| TaskError::Executor(format!("join: {e}")))?
        .map_err(|e| TaskError::Database(e.to_string()))?;

        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let kind: TaskKind = serde_json::from_str(&r.kind_json).map_err(|e| {
                TaskError::Executor(format!("deserialize TaskKind id={}: {e}", r.id))
            })?;
            // 过滤 kind（按判别式）
            if let Some(d) = filter.kind {
                if TaskKindDiscriminant::from_kind(&kind) != d {
                    continue;
                }
            }
            out.push(Task {
                id: r.id,
                kind,
                priority: match r.priority {
                    0 => Priority::Low,
                    1 => Priority::Normal,
                    2 => Priority::High,
                    3..=i32::MAX => Priority::Urgent,
                    _ => Priority::Low,
                },
                status: row_to_status(r.status),
                retry_count: r.retry_count.max(0) as u32,
                max_retries: r.max_retries.max(0) as u32,
                error: r.error,
                created_at: r.created_at,
                started_at: r.started_at,
                completed_at: r.completed_at,
            });
        }
        Ok(out)
    }
}

fn row_to_status(s: DbTaskStatus) -> TaskStatus {
    match s {
        DbTaskStatus::Pending => TaskStatus::Pending,
        DbTaskStatus::Running => TaskStatus::Running,
        DbTaskStatus::Completed => TaskStatus::Completed,
        DbTaskStatus::Failed => TaskStatus::Failed,
        DbTaskStatus::Cancelled => TaskStatus::Cancelled,
    }
}

impl Drop for SqliteTaskScheduler {
    fn drop(&mut self) {
        // 取消所有 running 任务
        let flags = self.cancel_flags.lock();
        for (_, f) in flags.iter() {
            f.store(true, Ordering::SeqCst);
        }
        drop(flags);

        // BUG #11 修复:置 stop_signal 并唤醒 worker,等线程退出后再 drop handles。
        self.stop_signal.store(true, Ordering::SeqCst);
        self.wake.notify_waiters();

        let mut handles = self.worker_handles.lock();
        for h in handles.drain(..) {
            let _ = h.join();
        }
    }
}

/// Worker 循环。
async fn worker_loop(
    worker_id: usize,
    db: Database,
    registry: Arc<ExecutorRegistry>,
    progress: ProgressSink,
    paused: Arc<AtomicBool>,
    wake: Arc<Notify>,
    cancel_flags: Arc<parking_lot::Mutex<std::collections::HashMap<i64, Arc<AtomicBool>>>>,
    stop_signal: Arc<AtomicBool>,
    poll_interval: std::time::Duration,
) {
    info!(worker_id, "worker started");
    loop {
        // BUG #11 修复:stop_signal 触发后立即退出循环
        if stop_signal.load(Ordering::SeqCst) {
            info!(worker_id, "worker exiting (stop_signal set)");
            return;
        }
        if paused.load(Ordering::SeqCst) {
            // 等待 wake 或 stop_signal
            tokio::select! {
                _ = wake.notified() => {}
                _ = async {
                    while !stop_signal.load(Ordering::SeqCst) {
                        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                    }
                } => {
                    info!(worker_id, "worker exiting during pause (stop_signal set)");
                    return;
                }
            }
            continue;
        }

        // 取下一条任务
        let db2 = db.clone();
        let next = tokio::task::spawn_blocking(move || {
            db2.transaction(|tx| {
                let mut repo = tx.tasks();
                repo.claim_next()
            })
        })
        .await;

        let claimed = match next {
            Ok(Ok(Some(row))) => row,
            Ok(Ok(None)) => {
                // 队列空，等待 wake 或短 polling
                tokio::select! {
                    _ = wake.notified() => {}
                    _ = tokio::time::sleep(poll_interval) => {}
                }
                continue;
            }
            Ok(Err(e)) => {
                warn!(worker_id, error = %e, "claim_next returned error");
                tokio::time::sleep(poll_interval).await;
                continue;
            }
            Err(e) => {
                error!(worker_id, error = %e, "join error");
                tokio::time::sleep(poll_interval).await;
                continue;
            }
        };

        // 解析 TaskKind
        let kind: TaskKind = match serde_json::from_str(&claimed.kind_json) {
            Ok(k) => {
                info!(task_id = claimed.id, kind = ?TaskKindDiscriminant::from_kind(&k), "worker claimed task");
                k
            }
            Err(e) => {
                error!(task_id = claimed.id, error = %e, "invalid TaskKind JSON");
                let _ = mark_failed_or_retry(&db, claimed.id, &format!("invalid JSON: {e}")).await;
                continue;
            }
        };

        let kind_disc = TaskKindDiscriminant::from_kind(&kind);
        let exec = match registry.get(kind_disc) {
            Some(e) => e,
            None => {
                warn!(task_id = claimed.id, kind = ?kind_disc, "no executor registered");
                let _ = mark_failed_or_retry(
                    &db,
                    claimed.id,
                    &format!("no executor for {:?}", kind_disc),
                )
                .await;
                continue;
            }
        };

        // 准备 context
        let cancel_flag = Arc::new(AtomicBool::new(false));
        {
            let mut flags = cancel_flags.lock();
            flags.insert(claimed.id, cancel_flag.clone());
        }
        let ctx = TaskContext::new(claimed.id, progress.clone(), cancel_flag.clone());

        // 执行
        let start = std::time::Instant::now();
        let exec_result = exec.execute(&ctx).await;
        let elapsed = start.elapsed();

        // 清理 cancel flag
        {
            let mut flags = cancel_flags.lock();
            flags.remove(&claimed.id);
        }

        match exec_result {
            Ok(TaskOutcome::Success(meta)) => {
                debug!(task_id = claimed.id, ?elapsed, meta = ?meta, "task success");
                mark_completed(&db, claimed.id);
            }
            Ok(TaskOutcome::Cancelled) => {
                info!(task_id = claimed.id, "task cancelled");
                mark_cancelled(&db, claimed.id);
            }
            Ok(TaskOutcome::Failed(reason)) => {
                warn!(task_id = claimed.id, reason = %reason, "task failed");
                let _ = mark_failed_or_retry(&db, claimed.id, &reason).await;
            }
            Err(e) => {
                warn!(task_id = claimed.id, error = %e, "task errored");
                let _ = mark_failed_or_retry(&db, claimed.id, &e.to_string()).await;
            }
        }
    }
}

fn mark_completed(db: &Database, id: i64) {
    let db = db.clone();
    let _ = tokio::task::spawn_blocking(move || {
        let _ = db.transaction(|tx| {
            let mut repo = tx.tasks();
            repo.mark_completed(id)
        });
    });
}

fn mark_cancelled(db: &Database, id: i64) {
    let db = db.clone();
    let _ = tokio::task::spawn_blocking(move || {
        let _ = db.transaction(|tx| {
            let mut repo = tx.tasks();
            repo.mark_cancelled(id)
        });
    });
}

async fn mark_failed_or_retry(db: &Database, id: i64, reason: &str) -> Result<bool, TaskError> {
    let db = db.clone();
    let reason = reason.to_string();
    let result = tokio::task::spawn_blocking(move || {
        db.transaction(|tx| {
            let mut repo = tx.tasks();
            repo.mark_failed_or_retry(id, &reason)
        })
    })
    .await
    .map_err(|e| TaskError::Executor(format!("join: {e}")))?;
    result.map_err(|e| TaskError::Database(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::{TaskExecutor, TaskOutcome};
    use crate::progress::TaskContext;
    use crate::TaskError;
    use pf_database::builtin_migrations;
    use tempfile::tempdir;

    struct EchoExec;

    #[async_trait::async_trait]
    impl TaskExecutor for EchoExec {
        fn kind(&self) -> TaskKindDiscriminant {
            TaskKindDiscriminant::IndexFace
        }
        fn name(&self) -> &'static str {
            "echo"
        }
        async fn execute(&self, _ctx: &TaskContext) -> Result<TaskOutcome, TaskError> {
            Ok(TaskOutcome::Success(Some(serde_json::json!({"ok": true}))))
        }
    }

    struct FailingExec;

    #[async_trait::async_trait]
    impl TaskExecutor for FailingExec {
        fn kind(&self) -> TaskKindDiscriminant {
            TaskKindDiscriminant::Scan
        }
        fn name(&self) -> &'static str {
            "failing"
        }
        async fn execute(&self, _ctx: &TaskContext) -> Result<TaskOutcome, TaskError> {
            Ok(TaskOutcome::Failed("simulated failure".into()))
        }
    }

    fn make_db() -> (tempfile::TempDir, Database) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");
        let db = Database::open(&path, builtin_migrations()).unwrap();
        (dir, db)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn enqueue_list_status() {
        let (_dir, db) = make_db();
        let registry = Arc::new(ExecutorRegistry::new());
        registry.register(Arc::new(EchoExec));
        let sched = SqliteTaskScheduler::new(db, registry);
        sched.start_workers();

        let id = sched
            .enqueue(TaskKind::IndexFace { image_id: 42 }, Priority::Normal)
            .await
            .unwrap();
        let list = sched
            .list(TaskFilter {
                status: None,
                kind: Some(TaskKindDiscriminant::IndexFace),
                limit: Some(10),
            })
            .await
            .unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, id);

        // 等 worker 跑
        for _ in 0..50 {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            let stats = sched.status().await.unwrap();
            if stats.completed >= 1 {
                break;
            }
        }
        let stats = sched.status().await.unwrap();
        assert_eq!(stats.completed, 1, "stats = {:?}", stats);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn enqueue_without_workers_does_not_dispatch() {
        // 不启动 workers（auto_start_workers=false），只测试 enqueue + list
        let (_dir, db) = make_db();
        let registry = Arc::new(ExecutorRegistry::new());
        let sched = SqliteTaskScheduler::with_config(
            db,
            registry,
            SchedulerConfig {
                auto_start_workers: false,
                ..Default::default()
            },
        );

        let id = sched
            .enqueue(TaskKind::IndexFace { image_id: 7 }, Priority::High)
            .await
            .unwrap();
        let list = sched
            .list(TaskFilter {
                status: Some(TaskStatus::Pending),
                kind: None,
                limit: Some(10),
            })
            .await
            .unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, id);
        assert_eq!(list[0].status, TaskStatus::Pending);
        assert_eq!(list[0].priority, Priority::High);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_pending_succeeds() {
        // auto_start_workers=false：保证任务在 cancel 前仍是 Pending，
        // 避免 worker 抢先 claim_next 把任务改成 Running（竞态导致 flaky）。
        let (_dir, db) = make_db();
        let registry = Arc::new(ExecutorRegistry::new());
        let sched = SqliteTaskScheduler::with_config(
            db,
            registry,
            SchedulerConfig {
                auto_start_workers: false,
                ..Default::default()
            },
        );

        let id = sched
            .enqueue(TaskKind::IndexFace { image_id: 99 }, Priority::Normal)
            .await
            .unwrap();
        sched.cancel(id).await.unwrap();
        let list = sched
            .list(TaskFilter {
                status: Some(TaskStatus::Cancelled),
                kind: None,
                limit: Some(10),
            })
            .await
            .unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, id);
        assert_eq!(list[0].status, TaskStatus::Cancelled);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_unknown_task_returns_not_found() {
        let (_dir, db) = make_db();
        let registry = Arc::new(ExecutorRegistry::new());
        let sched = SqliteTaskScheduler::new(db, registry);
        let r = sched.cancel(999_999).await;
        assert!(matches!(r, Err(TaskError::NotFound(999_999))));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pause_and_resume() {
        let (_dir, db) = make_db();
        let registry = Arc::new(ExecutorRegistry::new());
        let sched = SqliteTaskScheduler::new(db, registry);
        sched.pause().await.unwrap();
        let stats = sched.status().await.unwrap();
        assert!(stats.paused);
        sched.resume().await.unwrap();
        let stats = sched.status().await.unwrap();
        assert!(!stats.paused);
    }

    /// 回归测试:`cancel()` 必须能找到 running 任务的 cancel flag。
    /// 修复前 cancel_flags 在 `self` 和 worker 内部是两份独立的 map,
    /// cancel() 永远命中不了 running 任务,只能走 DB cancel_pending 分支,
    /// 看似返回 Ok 实际 worker 继续在跑。
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_running_task_sets_shared_flag() {
        use std::sync::atomic::Ordering;
        let (_dir, db) = make_db();
        let registry = Arc::new(ExecutorRegistry::new());

        // 自定义 executor:执行时把自己 task_id 对应的 cancel flag 写到外面能查的 channel
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(i64, Arc<AtomicBool>)>();
        struct CapturingExec(tokio::sync::mpsc::UnboundedSender<(i64, Arc<AtomicBool>)>);
        #[async_trait::async_trait]
        impl TaskExecutor for CapturingExec {
            fn kind(&self) -> TaskKindDiscriminant {
                TaskKindDiscriminant::IndexFace
            }
            fn name(&self) -> &'static str {
                "capturing"
            }
            async fn execute(
                &self,
                ctx: &TaskContext,
            ) -> Result<TaskOutcome, TaskError> {
                let _ = self.0.send((ctx.task_id, ctx.should_cancel.clone()));
                // 阻塞直到被 cancel
                for _ in 0..200 {
                    if ctx.is_cancelled() {
                        return Ok(TaskOutcome::Cancelled);
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                }
                Ok(TaskOutcome::Success(None))
            }
        }
        registry.register(Arc::new(CapturingExec(tx)));

        let sched = SqliteTaskScheduler::new(db, registry);
        sched.start_workers();

        let id = sched
            .enqueue(TaskKind::IndexFace { image_id: 1 }, Priority::Normal)
            .await
            .unwrap();

        // 等 worker claim 并把 flag 发出来
        let (_task_id, flag) = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            rx.recv(),
        )
        .await
        .expect("worker should claim and send flag")
        .expect("channel open");

        assert!(!flag.load(Ordering::SeqCst), "flag 初始应为 false");

        // cancel 后 flag 应立即变 true(共享 map 修复的核心断言)
        sched.cancel(id).await.unwrap();
        assert!(
            flag.load(Ordering::SeqCst),
            "BUG #1 回归:cancel() 必须能 set 到 running 任务的共享 flag"
        );

        // 任务应在 10 秒内收尾(返回 Cancelled)
        for _ in 0..100 {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            let rows = sched
                .list(TaskFilter {
                    status: Some(TaskStatus::Cancelled),
                    kind: None,
                    limit: Some(10),
                })
                .await
                .unwrap();
            if rows.iter().any(|r| r.id == id) {
                return;
            }
        }
        panic!("cancel 后任务未进入 Cancelled 状态");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn failing_task_retries_then_fails() {
        let (_dir, db) = make_db();
        let registry = Arc::new(ExecutorRegistry::new());
        registry.register(Arc::new(FailingExec));
        let sched = SqliteTaskScheduler::new(db, registry);
        sched.start_workers();

        let id = sched
            .enqueue(TaskKind::Scan { folder_id: 1 }, Priority::Normal)
            .await
            .unwrap();

        // 等待重试耗尽（最多 ~3 次，间隔很小）
        for _ in 0..100 {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let stats = sched.status().await.unwrap();
            if stats.failed >= 1 {
                break;
            }
        }
        let rows = sched
            .list(TaskFilter {
                status: Some(TaskStatus::Failed),
                kind: None,
                limit: Some(10),
            })
            .await
            .unwrap();
        assert!(rows.iter().any(|r| r.id == id));
    }
}
