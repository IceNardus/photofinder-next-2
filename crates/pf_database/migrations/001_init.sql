-- 001_init.sql — 基础 schema（空 schema_migrations 表在 runner 里建）

-- 这条 migration 只建元数据表，迁移到 schema_migrations 的逻辑由 migration.rs 处理。
-- 实际业务表在后续 migration 中创建。

CREATE TABLE IF NOT EXISTS schema_migrations (
    id TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- 预留：apps 可在此扩展（PRAGMA、默认数据等）
-- 此 migration 必须存在并最早应用