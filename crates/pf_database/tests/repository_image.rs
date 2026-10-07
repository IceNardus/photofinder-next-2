//! ImageRepository 集成测试。

use chrono::Utc;
use pf_database::{
    builtin_migrations, Database, ImageRow, NewImage, ScanStatus, ThumbnailStatus,
};

#[test]
fn insert_and_get_by_id() {
    let db = Database::open_in_memory(builtin_migrations()).unwrap();

    let id = db
        .transaction(|tx| {
            let new = NewImage {
                path: "/tmp/test.jpg".into(),
                hash: "abc123".into(),
                size: 1024,
                modified_time: 0,
                width: 800,
                height: 600,
                captured_at: Some(Utc::now()),
            };
            tx.images().insert(&new)
        })
        .unwrap();

    let got = db
        .transaction(|tx| tx.images().get_by_id(id))
        .unwrap()
        .unwrap();

    assert_eq!(got.path, "/tmp/test.jpg");
    assert_eq!(got.hash, "abc123");
    assert_eq!(got.size, 1024);
    assert_eq!(got.width, 800);
    assert_eq!(got.height, 600);
    assert_eq!(got.scan_status, ScanStatus::Pending);
    assert_eq!(got.thumbnail_status, ThumbnailStatus::Pending);
}

#[test]
fn get_by_path_and_hash() {
    let db = Database::open_in_memory(builtin_migrations()).unwrap();

    db.transaction(|tx| {
        tx.images().insert(&NewImage {
            path: "/a/b.jpg".into(),
            hash: "hash-a".into(),
            size: 100,
            modified_time: 0,
            width: 50,
            height: 50,
            captured_at: None,
        })
    })
    .unwrap();

    let by_path: Option<ImageRow> = db
        .transaction(|tx| tx.images().get_by_path("/a/b.jpg"))
        .unwrap();
    assert!(by_path.is_some());
    assert_eq!(by_path.unwrap().hash, "hash-a");

    let by_hash: Option<ImageRow> = db
        .transaction(|tx| tx.images().get_by_hash("hash-a"))
        .unwrap();
    assert!(by_hash.is_some());
}

#[test]
fn update_scan_status() {
    let db = Database::open_in_memory(builtin_migrations()).unwrap();

    let id = db
        .transaction(|tx| {
            tx.images().insert(&NewImage {
                path: "/x.jpg".into(),
                hash: "h".into(),
                size: 1,
                modified_time: 0,
                width: 1,
                height: 1,
                captured_at: None,
            })
        })
        .unwrap();

    db.transaction(|tx| tx.images().update_scan_status(id, ScanStatus::Indexed))
        .unwrap();

    let img = db
        .transaction(|tx| tx.images().get_by_id(id))
        .unwrap()
        .unwrap();
    assert_eq!(img.scan_status, ScanStatus::Indexed);
}

#[test]
fn list_pending_returns_only_pending() {
    let db = Database::open_in_memory(builtin_migrations()).unwrap();

    db.transaction(|tx| {
        for i in 0..5 {
            tx.images().insert(&NewImage {
                path: format!("/p{i}.jpg"),
                hash: format!("h{i}"),
                size: 0,
                modified_time: 0,
                width: 0,
                height: 0,
                captured_at: None,
            })
            .unwrap();
        }
        Ok(())
    })
    .unwrap();

    let pending: Vec<ImageRow> = db
        .transaction(|tx| tx.images().list_pending(10))
        .unwrap();
    assert_eq!(pending.len(), 5);

    // 把第一个改成 indexed
    let first_id = pending[0].id;
    db.transaction(|tx| tx.images().update_scan_status(first_id, ScanStatus::Indexed))
        .unwrap();

    let pending: Vec<ImageRow> = db
        .transaction(|tx| tx.images().list_pending(10))
        .unwrap();
    assert_eq!(pending.len(), 4);
}

#[test]
fn count_returns_correct_total() {
    let db = Database::open_in_memory(builtin_migrations()).unwrap();

    assert_eq!(
        db.transaction(|tx| tx.images().count()).unwrap(),
        0
    );

    db.transaction(|tx| {
        for i in 0..3 {
            tx.images().insert(&NewImage {
                path: format!("/p{i}.jpg"),
                hash: format!("h{i}"),
                size: 0,
                modified_time: 0,
                width: 0,
                height: 0,
                captured_at: None,
            })
            .unwrap();
        }
        Ok(())
    })
    .unwrap();

    assert_eq!(
        db.transaction(|tx| tx.images().count()).unwrap(),
        3
    );
}

#[test]
fn duplicate_path_rejected() {
    let db = Database::open_in_memory(builtin_migrations()).unwrap();

    db.transaction(|tx| {
        tx.images().insert(&NewImage {
            path: "/dup.jpg".into(),
            hash: "h1".into(),
            size: 0,
            modified_time: 0,
            width: 0,
            height: 0,
            captured_at: None,
        })
    })
    .unwrap();

    // 第二次插入同 path 应失败
    let result = db.transaction(|tx| {
        tx.images().insert(&NewImage {
            path: "/dup.jpg".into(),
            hash: "h2".into(),
            size: 0,
            modified_time: 0,
            width: 0,
            height: 0,
            captured_at: None,
        })
    });
    assert!(result.is_err());
}