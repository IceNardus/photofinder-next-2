//! Tauri 事件 channel 名称。
//!
//! 前端用 `listen<T>(EVENT_SCAN_PROGRESS, cb)` 订阅。
//! 命名规则：`pf:<domain>:<event>`，避免和 Tauri 内置事件冲突。

/// 扫描进度（folder scan 阶段）。
pub const EVENT_SCAN_PROGRESS: &str = "pf:scan:progress";

/// 扫描完成（一次 folder scan 结束，无论成功失败）。
pub const EVENT_SCAN_COMPLETED: &str = "pf:scan:completed";

/// 后台 task 进度（index / cluster 等）。
pub const EVENT_TASK_PROGRESS: &str = "pf:task:progress";

/// 后台 task 完成。
pub const EVENT_TASK_COMPLETED: &str = "pf:task:completed";

/// 后台 task 失败。
pub const EVENT_TASK_FAILED: &str = "pf:task:failed";

/// 应用就绪（bootstrap 完成）。
pub const EVENT_APP_READY: &str = "pf:app:ready";
