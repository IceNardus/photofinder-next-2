//! SQLite 连接 + 连接池。

use std::path::Path;

use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::Connection;
use rusqlite::OptionalExtension;
use rusqlite::Transaction as SqliteTransaction;
use tracing::{debug, info};

use crate::error::DatabaseError;
use crate::migration::MigrationSet;

/// PRAGMA 应用函数。
///
/// 返回 `rusqlite::Error` 因为这是 `SqliteConnectionManager::with_init` 的约束。
fn apply_pragmas(conn: &mut Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA foreign_keys = ON;
         PRAGMA busy_timeout = 5000;
         PRAGMA temp_store = MEMORY;",
    )?;
    Ok(())
}

/// `Database` 顶层句柄。
#[derive(Clone)]
pub struct Database {
    pool: Pool<SqliteConnectionManager>,
    migrations: MigrationSet,
}

impl Database {
    /// 打开数据库（自动跑 migration）。
    pub fn open(path: &Path, migrations: MigrationSet) -> Result<Self, DatabaseError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let manager = SqliteConnectionManager::file(path).with_init(apply_pragmas);
        let pool = Pool::builder()
            .max_size(8)
            .build(manager)
            .map_err(|e| DatabaseError::Pool(e.to_string()))?;

        let db = Self { pool, migrations };

        // 跑 migration
        db.run_migrations()?;

        // Phase 1.3: 启动时检查 face_embeddings.model_version 一致性
        // (m014 §5: 多版本混合 HNSW 需 reindex)。
        db.check_model_versions()?;

        info!(path = %path.display(), "database opened");
        Ok(db)
    }

    /// 内存数据库（仅测试）。
    pub fn open_in_memory(migrations: MigrationSet) -> Result<Self, DatabaseError> {
        let manager = SqliteConnectionManager::memory().with_init(apply_pragmas);
        let pool = Pool::builder()
            .max_size(4)
            .build(manager)
            .map_err(|e| DatabaseError::Pool(e.to_string()))?;

        let db = Self { pool, migrations };
        db.run_migrations()?;
        db.check_model_versions()?;
        Ok(db)
    }

    /// Phase 1.3: 启动时校验 face_embeddings 是否有多种 model_version。
    ///
    /// 多版本同时存在意味着 HNSW index 与 DB 里的 embedding 不一致
    /// (向量维度/语义可能不同),需要 reindex_all。当前只 warn,不阻断启动
    /// (reindex 是显式操作,应用层决定何时触发)。
    fn check_model_versions(&self) -> Result<(), DatabaseError> {
        let conn = self.pool.get()?;
        let mut stmt = conn.prepare("SELECT DISTINCT model_version FROM face_embeddings")?;
        let versions: Vec<String> = stmt
            .query_map([], |r| {
                let v: Option<String> = r.get(0)?;
                Ok(v.unwrap_or_else(|| "<NULL>".to_string()))
            })?
            .collect::<Result<Vec<_>, rusqlite::Error>>()?;
        if versions.len() > 1 {
            tracing::warn!(
                versions = ?versions,
                "face_embeddings has {} distinct versions — HNSW index may be inconsistent. \
                 Run reindex_all to reconcile.",
                versions.len()
            );
        } else if versions.len() == 1 {
            debug!(version = %versions[0], "face_embeddings single model version OK");
        } else {
            debug!("face_embeddings empty (no faces indexed yet)");
        }
        Ok(())
    }

    /// 执行 migration。
    fn run_migrations(&self) -> Result<(), DatabaseError> {
        let mut conn = self.pool.get()?;
        crate::migration::run(&mut conn, &self.migrations)?;
        Ok(())
    }

    /// 获取连接。
    pub fn connection(&self) -> Result<PooledConnection, DatabaseError> {
        Ok(PooledConnection {
            inner: self.pool.get()?,
        })
    }

    /// 在事务中执行。
    ///
    /// 使用 `BEGIN IMMEDIATE` 而非默认 `BEGIN DEFERRED`：避免多个并发事务都
    /// 先以共享锁启动，在首次写入时才升级到 reserved 而相互冲突导致
    /// `SQLITE_BUSY`（`busy_timeout` 在该 lock-upgrade 路径上不生效）。
    /// `IMMEDIATE` 在事务起点直接获取 reserved 锁，配合 `busy_timeout=5000`
    /// 让竞争事务等到当前事务结束。
    pub fn transaction<F, R>(&self, f: F) -> Result<R, DatabaseError>
    where
        F: FnOnce(&mut Transaction) -> Result<R, DatabaseError>,
    {
        let mut conn = self.pool.get()?;
        let rusqlite_tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let mut tx = Transaction { inner: rusqlite_tx };
        let result = f(&mut tx)?;
        tx.inner.commit()?;
        Ok(result)
    }

    /// 在事务中执行，对 `SQLITE_BUSY` 自动重试。
    ///
    /// 兜底：正常情况下 `BEGIN IMMEDIATE` + `busy_timeout=5000` 已经够用，
    /// 但极长事务（>5s）仍可能触发 BUSY。重试 3 次，每次间隔 50ms。
    pub fn transaction_with_retry<F, R>(&self, mut f: F) -> Result<R, DatabaseError>
    where
        F: FnMut(&mut Transaction) -> Result<R, DatabaseError>,
    {
        const MAX_ATTEMPTS: u32 = 3;
        const RETRY_DELAY_MS: u64 = 50;
        let mut attempt = 0;
        loop {
            attempt += 1;
            match self.transaction(&mut f) {
                Ok(r) => return Ok(r),
                Err(DatabaseError::Sqlite(e))
                    if is_sqlite_busy(&e) && attempt < MAX_ATTEMPTS =>
                {
                    tracing::debug!(
                        attempt,
                        error = %e,
                        "SQLITE_BUSY, retrying after {}ms",
                        RETRY_DELAY_MS
                    );
                    std::thread::sleep(std::time::Duration::from_millis(RETRY_DELAY_MS));
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// 重置整个 DB（清空所有表 + re-run migrations）。
    ///
    /// 动态从 `sqlite_master` 取所有用户对象,避免:
    /// - 漏 drop 新 migration 加进来的表/触发器/索引(老 reset 是硬编码 list,
    ///   014 加了 `face_person_assignments` + `_person_prototypes_legacy` + 3 个
    ///   trigger,reset 后再 re-run migration 会因 FK 目标残留而失败)
    /// - 误删 `schema_migrations` 元表导致 migration 状态丢失
    pub fn reset(&self) -> Result<(), DatabaseError> {
        let mut conn = self.pool.get()?;
        // 禁用外键约束以便正确删除所有表
        conn.execute_batch("PRAGMA foreign_keys = OFF;")?;

        // 1. drop 所有 trigger (TRG 引用被 drop 的表会失败)
        let triggers: Vec<String> = conn
            .prepare(
                "SELECT name FROM sqlite_master
                 WHERE type = 'trigger' AND name NOT LIKE 'sqlite_%'",
            )?
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        for name in &triggers {
            conn.execute_batch(&format!("DROP TRIGGER IF EXISTS \"{name}\";"))?;
        }

        // 2. drop 所有 view (视图可能依赖表)
        let views: Vec<String> = conn
            .prepare(
                "SELECT name FROM sqlite_master
                 WHERE type = 'view' AND name NOT LIKE 'sqlite_%'",
            )?
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        for name in &views {
            conn.execute_batch(&format!("DROP VIEW IF EXISTS \"{name}\";"))?;
        }

        // 3. drop 所有 index (index 跟表走其实可以省,但有些 migration 用
        //    CREATE INDEX 不带 IF EXISTS + 表名,显式 drop 更稳)
        let indexes: Vec<String> = conn
            .prepare(
                "SELECT name FROM sqlite_master
                 WHERE type = 'index' AND name NOT LIKE 'sqlite_%'",
            )?
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        for name in &indexes {
            conn.execute_batch(&format!("DROP INDEX IF EXISTS \"{name}\";"))?;
        }

        // 4. drop 所有 table (保留 schema_migrations 元表,这样如果 reset 中途
        //    出错,不会把已应用的 migration 记录搞丢,下次手动补完迁移)
        let tables: Vec<String> = conn
            .prepare(
                "SELECT name FROM sqlite_master
                 WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                   AND name != 'schema_migrations'",
            )?
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        for name in &tables {
            conn.execute_batch(&format!("DROP TABLE IF EXISTS \"{name}\";"))?;
        }

        // 清空 schema_migrations 记录,让所有 migration 重新执行
        conn.execute_batch("DELETE FROM schema_migrations;")?;

        // 重新启用外键约束
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        crate::migration::run(&mut conn, &self.migrations)?;
        debug!(
            triggers = triggers.len(),
            views = views.len(),
            indexes = indexes.len(),
            tables = tables.len(),
            "database reset"
        );
        Ok(())
    }
}

/// 连接包装（带 Deref）。
pub struct PooledConnection {
    inner: r2d2::PooledConnection<SqliteConnectionManager>,
}

impl std::ops::Deref for PooledConnection {
    type Target = Connection;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for PooledConnection {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// 事务包装。
pub struct Transaction<'a> {
    inner: SqliteTransaction<'a>,
}

impl<'a> std::ops::Deref for Transaction<'a> {
    type Target = SqliteTransaction<'a>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a> std::ops::DerefMut for Transaction<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<'a> Transaction<'a> {
    /// 提交事务。
    pub fn commit(self) -> Result<(), DatabaseError> {
        self.inner.commit()?;
        Ok(())
    }

    /// 获取 `image` Repository。
    pub fn images(&mut self) -> crate::repositories::image::ImageRepository<'_, 'a> {
        crate::repositories::image::ImageRepository { tx: self }
    }

    /// 获取 `face` Repository。
    pub fn faces(&mut self) -> crate::repositories::face::FaceRepository<'_, 'a> {
        crate::repositories::face::FaceRepository { tx: self }
    }

    /// 获取 `object` Repository。
    pub fn objects(&mut self) -> crate::repositories::object::ObjectRepository<'_, 'a> {
        crate::repositories::object::ObjectRepository { tx: self }
    }

    /// 获取 `person` Repository。
    pub fn persons(&mut self) -> crate::repositories::person::PersonRepository<'_, 'a> {
        crate::repositories::person::PersonRepository { tx: self }
    }

    /// 获取 `task` Repository。
    pub fn tasks(&mut self) -> crate::repositories::task::TaskRepository<'_, 'a> {
        crate::repositories::task::TaskRepository { tx: self }
    }

    /// 获取 `person_prototypes` Repository。
    pub fn person_prototypes(
        &mut self,
    ) -> crate::repositories::person_prototypes::PersonPrototypeRepository<'_, 'a> {
        crate::repositories::person_prototypes::PersonPrototypeRepository { tx: self }
    }

    /// 获取 `face_person_assignments` Repository。
    ///
    /// 014 后:旧 `face_person_matches` 已 rename 为 `face_person_assignments`。
    /// 这里保留 `face_person_matches` 别名做软迁移过渡,新代码应直接用 `face_person_assignments`。
    pub fn face_person_assignments(
        &mut self,
    ) -> crate::repositories::face_person_matches::FacePersonAssignmentRepository<'_, 'a> {
        crate::repositories::face_person_matches::FacePersonAssignmentRepository { tx: self }
    }

    /// 旧别名(临时,Phase 2 内删除)。
    #[deprecated(
        since = "0.3.0",
        note = "use face_person_assignments() instead"
    )]
    pub fn face_person_matches(
        &mut self,
    ) -> crate::repositories::face_person_matches::FacePersonAssignmentRepository<'_, 'a> {
        self.face_person_assignments()
    }

    /// 获取 `bodies` Repository（V2）。
    pub fn bodies(&mut self) -> crate::repositories::bodies::BodiesRepository<'_, 'a> {
        crate::repositories::bodies::BodiesRepository { tx: self }
    }

    /// 获取 `person_body_prototypes` Repository（V2）。
    pub fn person_body_prototypes(
        &mut self,
    ) -> crate::repositories::person_body_prototypes::PersonBodyPrototypeRepository<'_, 'a> {
        crate::repositories::person_body_prototypes::PersonBodyPrototypeRepository { tx: self }
    }

    /// 获取 `identity_shadow_records` Repository（Phase 30）。
    pub fn shadow_records(
        &mut self,
    ) -> crate::repositories::shadow_records::ShadowRecordsRepository<'_, 'a> {
        crate::repositories::shadow_records::ShadowRecordsRepository { tx: self }
    }

    /// 直接查 patches 表计数(BUG #10 修复:之前统计接口偷懒用 objects 表,
    /// 导致 patch_count 与真实 patches 行数完全不符)。
    pub fn count_patches(&mut self) -> Result<i64, DatabaseError> {
        let n: i64 = self
            .inner
            .query_row("SELECT count(*) FROM patches", [], |r| r.get(0))?;
        Ok(n)
    }
}

impl PooledConnection {
    /// 单查询取一行的便捷方法。
    pub fn query_row_optional<T, F>(
        &self,
        sql: &str,
        params: &[&dyn rusqlite::ToSql],
        f: F,
    ) -> Result<Option<T>, DatabaseError>
    where
        F: FnOnce(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
    {
        Ok(self
            .inner
            .query_row(sql, params, f)
            .optional()?)
    }
}

/// 判断 rusqlite 错误是否为 `SQLITE_BUSY` / `SQLITE_LOCKED`。
fn is_sqlite_busy(e: &rusqlite::Error) -> bool {
    use rusqlite::ErrorCode;
    matches!(
        e.sqlite_error_code(),
        Some(ErrorCode::DatabaseBusy) | Some(ErrorCode::DatabaseLocked)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::migration::builtin_migrations;

    /// Bug #12 回归测试:Database::reset() 必须能 drop 所有迁移创建的对象
    /// (含 012 改名后的 face_person_assignments、014 的 _person_prototypes_legacy、
    /// 全部 trigger 和 index),然后干净地重跑全部 migrations。
    ///
    /// 之前的硬编码 DROP 列表漏了 014 的 _person_prototypes_legacy、3 个
    /// trg_persons_face_count_* trigger、以及 face_person_assignments。
    #[test]
    fn reset_drops_all_schema_then_reruns_migrations() {
        let db = Database::open_in_memory(builtin_migrations()).expect("open");

        // sanity: 跑完 migrations 后这些对象应该都在
        let n_objects_before: i64 = {
            let conn = db.connection().expect("conn");
            conn.query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type IN ('table','trigger','index') AND name NOT LIKE 'sqlite_%'",
                [],
                |r| r.get(0),
            )
            .expect("count")
        };
        assert!(n_objects_before > 10, "fresh DB should have >10 schema objects, got {n_objects_before}");

        // 调用 reset
        db.reset().expect("reset must succeed");

        // reset 后:migrations 应已全部重跑 — 全部对象应重新存在(数量相同)
        let conn = db.connection().expect("conn");
        let n_objects_after: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type IN ('table','trigger','index') AND name NOT LIKE 'sqlite_%'",
                [],
                |r| r.get(0),
            )
            .expect("count");
        assert_eq!(
            n_objects_after, n_objects_before,
            "after reset+rerun, schema object count must match initial state"
        );

        // reset 后:含 014 的 _person_prototypes_legacy 临时表
        let has_legacy: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type='table' AND name='_person_prototypes_legacy'",
                [],
                |r| r.get(0),
            )
            .expect("count");
        assert_eq!(has_legacy, 1, "_person_prototypes_legacy must be recreated by 014");

        // reset 后:face_person_assignments 必须存在(014 rename 后)
        let has_fpa: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type='table' AND name='face_person_assignments'",
                [],
                |r| r.get(0),
            )
            .expect("count");
        assert_eq!(has_fpa, 1, "face_person_assignments must be created (post-014)");

        // reset 后:014 的 3 个 trigger 必须重建
        let n_trg: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name LIKE 'trg_persons_face_count_%'",
                [],
                |r| r.get(0),
            )
            .expect("count triggers");
        assert_eq!(n_trg, 3, "3 face_count triggers must be recreated by 014");

        // reset 后再调用 reset() 也应能成功(幂等)
        drop(conn);
        db.reset().expect("reset must be idempotent");
    }

    /// Bug #12 边界:reset() 后 Database 仍可用 — 可正常跑 transaction。
    #[test]
    fn reset_then_transaction_works() {
        let db = Database::open_in_memory(builtin_migrations()).expect("open");
        db.reset().expect("reset");

        // reset 后事务可正常 BEGIN/COMMIT
        db.transaction(|tx| {
            tx.images(); // 取 repository 不应 panic
            Ok(())
        })
        .expect("transaction after reset");

        // reset 后可直接 query schema
        let conn = db.connection().expect("conn");
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM schema_migrations", [], |r| r.get(0))
            .expect("query");
        assert!(n > 0, "reset must re-apply all builtin migrations");
    }
}