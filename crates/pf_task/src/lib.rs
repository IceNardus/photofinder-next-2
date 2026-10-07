//! # pf_task — 后台任务系统
//!
//! 不依赖具体业务。`TaskExecutor` 由 `pf_application` 提供具体实现并注册到 `pf_task`。
//!
//! 设计：
//! - `Task` 数据持久化在 SQLite（`pf_database`）
//! - `TaskScheduler` trait 抽象队列
//! - `TaskExecutor` trait 抽象执行
//! - 进度通过 `ProgressReporter` 推送（最终由 `pf_application` 转为 Tauri event）

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod executor;
pub mod progress;
pub mod registry;
pub mod scheduler;
pub mod sqlite_scheduler;
pub mod task;

pub use error::TaskError;
pub use executor::{TaskExecutor, TaskOutcome};
pub use progress::{ProgressReporter, TaskContext, TaskStats};
pub use registry::ExecutorRegistry;
pub use scheduler::{TaskKindDiscriminant, TaskScheduler};
pub use sqlite_scheduler::{LoggingProgressReporter, SchedulerConfig, SqliteTaskScheduler};
pub use task::{Priority, Task, TaskFilter, TaskKind, TaskStatus};