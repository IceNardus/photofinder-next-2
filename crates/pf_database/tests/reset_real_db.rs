//! Bug #12 回归测试:Database::reset() 在真实磁盘 DB 上能完整 drop 所有迁移对象,
//! 然后干净重跑 migrations。
//!
//! 用 `tempfile::tempdir()` 隔离,不污染 production DB。
//!
//! Bug #12 之前 reset() 硬编码 DROP 列表漏了:
//! - 014 face_person_assignments (post-rename of face_person_matches)
//! - 014 _person_prototypes_legacy
//! - 014 trg_persons_face_count_insert / update / delete
//! - 014 idx_fpa_* / idx_person_prototypes_* 等新索引
//!
//! 这些漏 drop 后,rerun migrations 会因 FK 目标残留或表已存在而失败。

use pf_database::{builtin_migrations, Database};

#[test]
fn reset_on_real_disk_db_works() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("test.db");

    // 第一次:open + reset
    let db = Database::open(&db_path, builtin_migrations()).expect("open");
    db.reset().expect("reset on fresh real DB");

    let conn = db.connection().expect("conn");
    let n_applied: i64 = conn
        .query_row("SELECT COUNT(*) FROM schema_migrations", [], |r| r.get(0))
        .expect("count");
    assert_eq!(n_applied, 14, "all 14 migrations should be re-applied after reset");

    // 第二次:open + insert data + reset + verify re-run OK
    drop(conn);
    db.transaction(|tx| {
        use pf_database::repositories::person::NewPerson;
        tx.persons()
            .insert(&NewPerson {
                name: Some("test_person".into()),
            })
            .map(|_| ())
    })
    .expect("insert before second reset");

    db.reset().expect("reset with data");
    let conn = db.connection().expect("conn");

    // reset 后 person 表应存在但为空
    let n_persons: i64 = conn
        .query_row("SELECT COUNT(*) FROM persons", [], |r| r.get(0))
        .expect("count persons");
    assert_eq!(n_persons, 0, "persons must be empty after reset");

    // reset 后 014 的 face_person_assignments + _person_prototypes_legacy 都应存在
    let has_fpa: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='face_person_assignments'",
            [],
            |r| r.get(0),
        )
        .expect("count fpa");
    assert_eq!(has_fpa, 1, "face_person_assignments must be re-created");

    let has_legacy: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='_person_prototypes_legacy'",
            [],
            |r| r.get(0),
        )
        .expect("count legacy");
    assert_eq!(has_legacy, 1, "_person_prototypes_legacy must be re-created");

    // reset 后 014 的 3 个 trigger 必须重建
    let n_trg: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type='trigger' AND name LIKE 'trg_persons_face_count_%'",
            [],
            |r| r.get(0),
        )
        .expect("count triggers");
    assert_eq!(n_trg, 3, "3 face_count triggers must be re-created");
}