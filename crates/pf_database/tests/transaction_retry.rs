//! Transaction retry helper 集成测试。

use pf_database::{builtin_migrations, Database};

#[test]
fn transaction_with_retry_succeeds_on_clean_run() {
    let db = Database::open_in_memory(builtin_migrations()).unwrap();

    // 简单 INSERT — 应该一次成功，不触发重试
    let row_id = db
        .transaction_with_retry(|tx| {
            use pf_database::repositories::image::ImageRepository;
            use pf_database::NewImage;
            let new = NewImage {
                path: "/tmp/test.jpg".into(),
                hash: "abc123".into(),
                size: 1024,
                modified_time: 0,
                width: 0,
                height: 0,
                captured_at: None,
            };
            let id = tx.images().insert(&new)?;
            Ok::<_, pf_database::DatabaseError>(id)
        })
        .unwrap();
    assert!(row_id > 0);
}

#[test]
fn transaction_with_retry_propagates_non_busy_errors() {
    let db = Database::open_in_memory(builtin_migrations()).unwrap();

    // 唯一约束冲突 — NOT a SQLITE_BUSY, retry should NOT help, should propagate
    use pf_database::repositories::image::ImageRepository;
    use pf_database::NewImage;
    let new = NewImage {
        path: "/tmp/dup.jpg".into(),
        hash: "hash1".into(),
        size: 100,
        modified_time: 0,
        width: 0,
        height: 0,
        captured_at: None,
    };
    db.transaction(|tx| {
        let _ = tx.images().insert(&new)?;
        Ok::<_, pf_database::DatabaseError>(())
    })
    .unwrap();

    // 第二次插入相同 path — 应该返回 Constraint 错误（不是 BUSY），不会重试
    let result = db.transaction_with_retry(|tx| {
        let _ = tx.images().insert(&new)?;
        Ok::<_, pf_database::DatabaseError>(())
    });
    assert!(result.is_err(), "duplicate path should error");
}

#[test]
fn transaction_uses_immediate_behavior() {
    // 间接验证：`open_in_memory` 用 BEGIN IMMEDIATE 启动后，第二个并发
    // 写事务应在 busy_timeout（5s）内被排队/拒绝（不会 deadlock）。
    // 此处仅验证 transaction() 自身能在 immediate 模式下正常工作。
    let db = Database::open_in_memory(builtin_migrations()).unwrap();

    let result = db.transaction(|tx| {
        use pf_database::repositories::image::ImageRepository;
        use pf_database::NewImage;
        let new = NewImage {
            path: "/tmp/immediate_test.jpg".into(),
            hash: "imm".into(),
            size: 1,
            modified_time: 0,
            width: 0,
            height: 0,
            captured_at: None,
        };
        let _id = tx.images().insert(&new)?;
        Ok::<_, pf_database::DatabaseError>(42)
    });
    assert_eq!(result.unwrap(), 42);
}