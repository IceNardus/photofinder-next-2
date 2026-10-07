//! Migration runner。

use rusqlite::{Connection, Transaction};
use tracing::{debug, info};

use crate::error::DatabaseError;

/// 单条 migration。
#[derive(Debug, Clone)]
pub struct Migration {
    /// 唯一 ID（如 "001_init"）
    pub id: &'static str,
    /// SQL 内容
    pub sql: &'static str,
}

/// Migration 集合。
#[derive(Debug, Clone, Default)]
pub struct MigrationSet {
    migrations: Vec<Migration>,
}

impl MigrationSet {
    /// 空集合。
    pub fn new() -> Self {
        Self::default()
    }

    /// 添加 migration（按 id 升序排列）。
    pub fn add(mut self, m: Migration) -> Self {
        self.migrations.push(m);
        self.migrations.sort_by_key(|m| m.id);
        self
    }

    /// 条目数。
    pub fn len(&self) -> usize {
        self.migrations.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.migrations.is_empty()
    }

    /// 迭代。
    pub fn iter(&self) -> impl Iterator<Item = &Migration> {
        self.migrations.iter()
    }
}

/// 加载 v2 内置 migrations（编译期 include_str!）。
pub fn builtin_migrations() -> MigrationSet {
    MigrationSet::new()
        .add(Migration {
            id: "001_init",
            sql: include_str!("../migrations/001_init.sql"),
        })
        .add(Migration {
            id: "002_images",
            sql: include_str!("../migrations/002_images.sql"),
        })
        .add(Migration {
            id: "003_tasks",
            sql: include_str!("../migrations/003_tasks.sql"),
        })
        .add(Migration {
            id: "004_faces",
            sql: include_str!("../migrations/004_faces.sql"),
        })
        .add(Migration {
            id: "005_persons",
            sql: include_str!("../migrations/005_persons.sql"),
        })
        .add(Migration {
            id: "006_objects",
            sql: include_str!("../migrations/006_objects.sql"),
        })
        .add(Migration {
            id: "007_patches",
            sql: include_str!("../migrations/007_patches.sql"),
        })
        .add(Migration {
            id: "008_face_embeddings",
            sql: include_str!("../migrations/008_face_embeddings.sql"),
        })
        .add(Migration {
            id: "009_faces_extended",
            sql: include_str!("../migrations/009_faces_extended.sql"),
        })
        .add(Migration {
            id: "010_duplicate_groups",
            sql: include_str!("../migrations/010_duplicate_groups.sql"),
        })
        .add(Migration {
            id: "011_object_roi_metadata",
            sql: include_str!("../migrations/011_object_roi_metadata.sql"),
        })
        .add(Migration {
            id: "012_face_identity_v2",
            sql: include_str!("../migrations/012_face_identity_v2.sql"),
        })
        .add(Migration {
            id: "013_face_indexing_status",
            sql: include_str!("../migrations/013_face_indexing_status.sql"),
        })
        .add(Migration {
            id: "014_face_v3_identity",
            sql: include_str!("../migrations/014_face_v3_identity.sql"),
        })
        .add(Migration {
            id: "015_bodies",
            sql: include_str!("../migrations/015_bodies.sql"),
        })
        .add(Migration {
            id: "016_person_body_prototypes",
            sql: include_str!("../migrations/016_person_body_prototypes.sql"),
        })
        .add(Migration {
            id: "017_identity_shadow_records",
            sql: include_str!("../migrations/017_identity_shadow_records.sql"),
        })
        .add(Migration {
            id: "018_shadow_enhanced_fields",
            sql: include_str!("../migrations/018_shadow_enhanced_fields.sql"),
        })
        .add(Migration {
            id: "019_face_cascade_roll_correct",
            sql: include_str!("../migrations/019_face_cascade_roll_correct.sql"),
        })
}

/// 判断某 SQLite 错误消息是否是 migration rerun 时可容忍的兼容错误。
///
/// 合法可忽略(只这两类):
/// - `"duplicate column name"`:ALTER TABLE ADD COLUMN 重跑(老 DB 已有此列)
/// - `"already exists"`:个别不用 IF NOT EXISTS 的 CREATE 二次执行防御
///
/// 不再吞 `"no such table"` / `"no such index"` / `"no such column"` —
/// migrations 里所有 CREATE/DROP 都用 IF [NOT] EXISTS,触发这类错误意味着
/// SQL 有 typo 或 schema 不一致,应该 abort 让用户知道。
fn is_ignorable_error(err: &str) -> bool {
    err.contains("duplicate column name") || err.contains("already exists")
}

/// 运行 migration（逐语句执行，跳过可选 schema 对象的缺失错误）。
pub(crate) fn run(conn: &mut Connection, set: &MigrationSet) -> Result<(), DatabaseError> {

    // 确保 meta 表存在
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_migrations (
            id TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );",
    )?;

    let tx = conn.transaction()?;

    for m in set.iter() {
        if is_applied(&tx, m.id)? {
            debug!(migration = m.id, "already applied, skip");
            continue;
        }

        debug!(migration = m.id, "applying");

        // 逐条 SQL 语句执行，兼容旧 DB 列/索引缺失
        for stmt in split_sql_statements(m.sql) {
            let trimmed = stmt.trim();
            if trimmed.is_empty() {
                continue;
            }
            let result = tx.execute_batch(trimmed);
            if let Err(ref e) = result {
                let err_str = e.to_string();
                if is_ignorable_error(&err_str) {
                    debug!(migration = m.id, "ignored: {}", err_str);
                } else {
                    return Err(DatabaseError::Migration(format!("[{}]: {}", m.id, e)));
                }
            }
        }

        tx.execute(
            "INSERT INTO schema_migrations (id) VALUES (?1)",
            rusqlite::params![m.id],
        )?;
    }

    tx.commit()?;
    info!(count = set.len(), "migrations complete");
    Ok(())
}

/// 按分号拆分成独立 SQL 语句（保留 CREATE TABLE / INDEX / TRIGGER 等完整语义）。
/// 移除 `-- comment` 行注释，避免注释内容干扰 SQL 解析。
///
/// 跟踪三层嵌套深度：
/// - `paren_depth` ( )：用于函数/列约束
/// - `begin_depth` BEGIN/END：用于 CREATE TRIGGER 体内的 `;`
/// - `string_depth` 'string literal'：避免字符串内的 `;` 被当成语句结束
fn split_sql_statements(sql: &str) -> Vec<String> {
    // 1. 移除行注释（-- ... \n）
    let no_comments = strip_line_comments(sql);

    let mut stmts = Vec::new();
    let mut paren_depth = 0usize;
    let mut begin_depth = 0usize;
    let mut string_char: Option<char> = None;
    let mut start = 0usize;
    let bytes = no_comments.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let ch = bytes[i] as char;
        if let Some(s) = string_char {
            if ch == s {
                string_char = None;
            } else if ch == '\\' && i + 1 < bytes.len() {
                i += 1; // skip escaped
            }
        } else {
            match ch {
                '\'' | '"' => string_char = Some(ch),
                '(' => paren_depth += 1,
                ')' => paren_depth = paren_depth.saturating_sub(1),
                // BEGIN (case-insensitive prefix match against "BEGIN")
                'B' | 'b' => {
                    if paren_depth == 0
                        && no_comments[i..].to_ascii_uppercase().starts_with("BEGIN")
                        && (i + 5 == bytes.len()
                            || bytes[i + 5].is_ascii_whitespace()
                            || bytes[i + 5] == b';')
                    {
                        begin_depth += 1;
                        i += 4; // skip "BEGIN"
                    }
                }
                'E' | 'e' => {
                    if paren_depth == 0
                        && begin_depth > 0
                        && no_comments[i..].to_ascii_uppercase().starts_with("END")
                        && (i + 3 == bytes.len()
                            || bytes[i + 3].is_ascii_whitespace()
                            || bytes[i + 3] == b';')
                    {
                        begin_depth -= 1;
                        i += 2; // skip "END"
                    }
                }
                ';' if paren_depth == 0 && begin_depth == 0 => {
                    let stmt = no_comments[start..i].trim().to_string();
                    if !stmt.is_empty() {
                        stmts.push(stmt);
                    }
                    start = i + 1;
                }
                _ => {}
            }
        }
        i += 1;
    }
    let tail = no_comments[start..].trim().to_string();
    if !tail.is_empty() {
        stmts.push(tail);
    }
    stmts
}

/// 移除 SQL 行注释（-- 到行尾）。
fn strip_line_comments(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for line in s.lines() {
        if let Some(pos) = line.find("--") {
            result.push_str(&line[..pos]);
            result.push('\n');
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}

fn is_applied(tx: &Transaction<'_>, id: &str) -> Result<bool, DatabaseError> {
    let count: i64 = tx
        .query_row(
            "SELECT COUNT(*) FROM schema_migrations WHERE id = ?1",
            rusqlite::params![id],
            |r| r.get(0),
        )
        .map_err(DatabaseError::Sqlite)?;
    Ok(count > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_set_sorts_by_id() {
        let s = MigrationSet::new()
            .add(Migration {
                id: "003_third",
                sql: "",
            })
            .add(Migration {
                id: "001_first",
                sql: "",
            })
            .add(Migration {
                id: "002_second",
                sql: "",
            });

        let ids: Vec<&str> = s.iter().map(|m| m.id).collect();
        assert_eq!(ids, vec!["001_first", "002_second", "003_third"]);
    }

    /// Bug #14 回归测试:确认所有 builtin migrations 在 fresh in-memory DB 上
    /// 都能成功跑完(包括 012/014 等含 schema 重命名/重建的复杂 migration)。
    /// 之前的 is_ignorable_error 太宽松,会吞掉 012 line 101 的
    /// `CREATE INDEX idx_face_embeddings_image_id ON face_embeddings(image_id)`
    /// 引用已删列导致的 "no such column",看起来"成功",实际 schema 不完整。
    #[test]
    fn builtin_migrations_run_on_fresh_db() {
        let mut conn = rusqlite::Connection::open_in_memory().expect("open in-memory");
        let set = builtin_migrations();
        assert!(set.len() >= 14, "expected >=14 migrations, got {}", set.len());

        // 必须成功,不应 abort
        run(&mut conn, &set).expect("builtin migrations should run cleanly on fresh DB");

        // 验证 schema_migrations 全部标为已应用
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM schema_migrations", [], |r| r.get(0))
            .expect("query schema_migrations");
        assert_eq!(count, set.len() as i64, "all migrations must be marked applied");

        // 验证 face_embeddings 表没有 image_id 列(Bug #14 防御)
        // 跑 SQLite pragma 表信息,如果含 image_id 列就是回归
        let has_image_id: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('face_embeddings') WHERE name = 'image_id'",
                [],
                |r| r.get(0),
            )
            .expect("pragma_table_info");
        assert_eq!(
            has_image_id, 0,
            "face_embeddings must NOT have image_id column (Bug #14 regression)"
        );

        // 验证 face_embeddings 应该有 face_id 列
        let has_face_id: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('face_embeddings') WHERE name = 'face_id'",
                [],
                |r| r.get(0),
            )
            .expect("pragma_table_info");
        assert_eq!(has_face_id, 1, "face_embeddings must have face_id column");
    }

    /// Bug #13 回归测试:确认 is_ignorable_error 只吞合法的兼容错误
    /// (duplicate column name / already exists),不再吞 "no such column/table/index"。
    #[test]
    fn ignorable_error_predicate_only_known_safe() {
        // 合法
        assert!(is_ignorable_error("duplicate column name: foo"));
        assert!(is_ignorable_error("table foo already exists"));

        // 非法 — 必须不再被吞,否则 schema mismatch 会被静默掩盖
        assert!(!is_ignorable_error("no such column: bar"));
        assert!(!is_ignorable_error("no such table: baz"));
        assert!(!is_ignorable_error("no such index: idx_x"));
        assert!(!is_ignorable_error("syntax error near..."));
        assert!(!is_ignorable_error("constraint failed: UNIQUE"));
    }
}