-- 003_tasks.sql — tasks 表
--
-- 持久化后台任务队列。
-- pf_task::Task 结构的直接映射。

CREATE TABLE IF NOT EXISTS tasks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kind_json       TEXT NOT NULL,                        -- TaskKind 的 JSON 表示
    priority        INTEGER NOT NULL DEFAULT 1,           -- 0..3
    status          TEXT NOT NULL DEFAULT 'pending',      -- pending|running|completed|failed|cancelled
    retry_count     INTEGER NOT NULL DEFAULT 0,
    max_retries     INTEGER NOT NULL DEFAULT 3,
    error           TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    started_at      TEXT,
    completed_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_status          ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_priority        ON tasks(priority DESC, created_at);