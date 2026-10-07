//! Index generation 跟踪 — 用于在 reindex 时区分新旧 face,避免 async rebuild
//! 覆盖正在写入的新 face。
//!
//! 用法:
//! - 启动时 `init(db)` 从 DB 拉当前 MAX(index_generation)
//! - `current()`:读当前 generation(供 NewFace 默认值)
//! - `bump()`:reindex 时调用,返回值给 IndexImageExecutor 写
//!
//! 注:当前是 in-process `AtomicU64`。跨进程 reindex 必须由 caller 显式同步,
//! 否则两个进程可能同时 bump → generation 撞车。
//! Phase 3+:可换 DB-backed `SELECT MAX(...)` 加 advisory lock。

use std::sync::atomic::{AtomicU64, Ordering};

use pf_database::Database;

/// 当前 generation(进程内)。
static CURRENT: AtomicU64 = AtomicU64::new(0);

/// 从 DB 读 max(index_generation) 并初始化 in-process 计数器。
///
/// 必须启动时调用一次;之后所有 `bump()` 都基于这个起点。
pub fn init(db: &Database) {
    let max = db
        .transaction(|tx| {
            let n: i64 = tx
                .query_row(
                    "SELECT COALESCE(MAX(index_generation), 0) FROM faces",
                    [],
                    |r| r.get(0),
                )
                .unwrap_or(0);
            Ok(n)
        })
        .unwrap_or(0);
    CURRENT.store(max.max(0) as u64, Ordering::SeqCst);
}

/// 当前 generation(供 NewFace 默认值)。
pub fn current() -> i64 {
    CURRENT.load(Ordering::SeqCst) as i64
}

/// 原子 +1,返回新值。reindex_all 流程使用。
pub fn bump() -> i64 {
    let new = CURRENT.fetch_add(1, Ordering::SeqCst) + 1;
    new as i64
}

/// 显式设到指定值(测试用)。
pub fn set(g: i64) {
    CURRENT.store(g.max(0) as u64, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bump_increments() {
        let prev = current();
        let next = bump();
        assert_eq!(next, prev + 1);
        assert_eq!(current(), next);
        // 还原
        set(prev);
    }
}