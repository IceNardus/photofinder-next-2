//! 并发写测试 — 验证 BEGIN IMMEDIATE 修复防止 SQLITE_BUSY。
//!
//! 回归场景：在 BEGIN DEFERRED 默认下，2 个并发写事务各以共享锁启动，
//! 首次写入时升级到 reserved 时相互冲突，立即返回 SQLITE_BUSY，
//! busy_timeout 在 lock-upgrade 路径上不生效。
//!
//! 修复：用 BEGIN IMMEDIATE 直接获取 reserved 锁，配合 busy_timeout=5000
//! 让第二个事务等到第一个完成。

use std::sync::Arc;
use std::thread;

use pf_database::repositories::image::ImageRepository;
use pf_database::{builtin_migrations, Database, NewImage};

/// 用 temp file 而非 :memory: — 后者每个 pooled connection 都有独立 DB，
/// 无法模拟真实的多连接并发场景。
fn temp_db() -> Database {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pf_concurrent_test_{}_{}.db",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    // 清理 — 测试前不留尾巴
    let _ = std::fs::remove_file(&p);
    Database::open(&p, builtin_migrations()).unwrap()
}

#[test]
fn two_concurrent_write_transactions_serialize_without_busy() {
    let db = Arc::new(temp_db());

    let mut handles = Vec::new();
    for i in 0..2 {
        let db = db.clone();
        handles.push(thread::spawn(move || {
            db.transaction(|tx| {
                let new = NewImage {
                    path: format!("/tmp/concurrent_test_{i}.jpg"),
                    hash: format!("hash_{i}"),
                    size: 100 + i as u64,
                    modified_time: 0,
                    width: 0,
                    height: 0,
                    captured_at: None,
                };
                let _id = tx.images().insert(&new)?;
                // 模拟稍长的写事务（让另一个线程有时间争抢）
                thread::sleep(std::time::Duration::from_millis(50));
                Ok::<_, pf_database::DatabaseError>(())
            })
        }));
    }

    for h in handles {
        let result = h.join().unwrap();
        assert!(
            result.is_ok(),
            "concurrent write must succeed with BEGIN IMMEDIATE: {result:?}"
        );
    }

    // 验证两条都写入了
    let conn = db.connection().unwrap();
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM images WHERE path LIKE '/tmp/concurrent_test_%'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(count, 2, "both rows should be committed");
}

#[test]
fn transaction_with_retry_handles_busy_under_contention() {
    let db = Arc::new(temp_db());

    // 预先插一条数据，让另一个事务能以写冲突开始
    db.transaction(|tx| {
        let new = NewImage {
            path: "/tmp/preseed.jpg".into(),
            hash: "p".into(),
            size: 1,
            modified_time: 0,
            width: 0,
            height: 0,
            captured_at: None,
        };
        let _ = tx.images().insert(&new)?;
        Ok::<_, pf_database::DatabaseError>(())
    })
    .unwrap();

    // 两个并发写 — retry 路径应至少其中一个成功（实际：begin immediate 让它们都成功）
    let mut handles = Vec::new();
    for i in 0..2 {
        let db = db.clone();
        handles.push(thread::spawn(move || {
            db.transaction_with_retry(|tx| {
                let new = NewImage {
                    path: format!("/tmp/retry_test_{i}.jpg"),
                    hash: format!("r_{i}"),
                    size: 100,
                    modified_time: 0,
                    width: 0,
                    height: 0,
                    captured_at: None,
                };
                let _ = tx.images().insert(&new)?;
                thread::sleep(std::time::Duration::from_millis(30));
                Ok::<_, pf_database::DatabaseError>(())
            })
        }));
    }

    for h in handles {
        let result = h.join().unwrap();
        assert!(result.is_ok(), "retry path must succeed: {result:?}");
    }
}