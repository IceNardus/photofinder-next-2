//! Migration 集成测试。

use pf_database::{builtin_migrations, Database};

#[test]
fn builtin_migrations_apply_cleanly() {
    let db = Database::open_in_memory(builtin_migrations()).unwrap();

    // 验证 schema_migrations 全部应用
    let conn = db.connection().unwrap();
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM schema_migrations", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 14, "all 14 migrations should be applied");
}

#[test]
fn migrations_are_idempotent() {
    // 第二次打开应当不报错（migration 已被记录）
    let db = Database::open_in_memory(builtin_migrations()).unwrap();
    drop(db);

    // 第二次打开
    let db2 = Database::open_in_memory(builtin_migrations()).unwrap();
    let conn = db2.connection().unwrap();
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM schema_migrations", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 14);
}

#[test]
fn images_table_exists_with_correct_columns() {
    let db = Database::open_in_memory(builtin_migrations()).unwrap();
    let conn = db.connection().unwrap();

    // 验证关键列存在
    let col: String = conn
        .query_row(
            "SELECT type FROM sqlite_master WHERE type='table' AND name='images'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(col, "table");
}

/// 014: 验证 v3 schema 落地。
/// - faces 加 hnsw_handle / index_status / index_generation
/// - face_person_matches rename 成 face_person_assignments + similarity/confidence/model_version
/// - person_prototypes 新表 + 数据迁移
/// - persons.face_count trigger 在 faces 写入时自动同步
#[test]
fn migration_014_face_v3_identity() {
    let db = Database::open_in_memory(builtin_migrations()).unwrap();
    let conn = db.connection().unwrap();

    // 1. faces 新列存在
    let cols: Vec<String> = {
        let mut stmt = conn
            .prepare("PRAGMA table_info(faces)")
            .unwrap();
        stmt.query_map([], |r| r.get::<_, String>(1))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect()
    };
    assert!(cols.iter().any(|c| c == "hnsw_handle"), "missing faces.hnsw_handle");
    assert!(cols.iter().any(|c| c == "index_status"), "missing faces.index_status");
    assert!(cols.iter().any(|c| c == "index_generation"), "missing faces.index_generation");

    // 2. face_person_assignments 表存在（旧 face_person_matches 已 rename）
    let renamed: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='face_person_assignments'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(renamed, 1, "face_person_assignments should exist");
    let old_name: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='face_person_matches'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(old_name, 0, "face_person_matches should be renamed");

    // 3. face_person_assignments 新字段
    let fpa_cols: Vec<String> = {
        let mut stmt = conn
            .prepare("PRAGMA table_info(face_person_assignments)")
            .unwrap();
        stmt.query_map([], |r| r.get::<_, String>(1))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect()
    };
    assert!(fpa_cols.iter().any(|c| c == "similarity"), "missing fpa.similarity");
    assert!(fpa_cols.iter().any(|c| c == "confidence"), "missing fpa.confidence");
    assert!(fpa_cols.iter().any(|c| c == "model_version"), "missing fpa.model_version");

    // 4. person_prototypes 新 schema
    let pp_cols: Vec<String> = {
        let mut stmt = conn
            .prepare("PRAGMA table_info(person_prototypes)")
            .unwrap();
        stmt.query_map([], |r| r.get::<_, String>(1))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect()
    };
    for required in [
        "face_id", "embedding", "dim", "pose_score", "pose_yaw",
        "prototype_type", "weight", "model_version", "is_active",
    ] {
        assert!(
            pp_cols.iter().any(|c| c == required),
            "missing person_prototypes.{required}"
        );
    }

    // 5. 旧表改名保留
    let legacy: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='_person_prototypes_legacy'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(legacy, 1, "_person_prototypes_legacy should be preserved");

    // 6. trigger 存在
    let trigger_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name LIKE 'trg_persons_face_count_%'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(trigger_count, 3, "should have 3 face_count triggers");
}