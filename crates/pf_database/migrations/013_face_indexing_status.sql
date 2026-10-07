-- 013_face_indexing_status.sql — face 索引状态机
--
-- 目标（Phase 2 plan）：
--   1. faces 加 status 列(pending / indexed / failed),跟踪索引生命周期
--   2. 加 indexed_at / error_message / last_attempt_at 用于诊断与重建
--   3. 现状数据回填:Phase 1 写入流程要么整体成功要么整体回滚,不会有 pending 残留;
--      vector_id 非空的 face 视为 indexed;vector_id 为空的(理论上不存在)视为 failed。
--   4. 索引 idx_faces_status 加速诊断查询。
--
-- 不修改:HNSW 索引行为、clustering、search。

-- 步骤 1: 加列。DEFAULT 'indexed' 让现存 face 自动标 indexed(回填)。
ALTER TABLE faces ADD COLUMN status TEXT NOT NULL DEFAULT 'indexed';
ALTER TABLE faces ADD COLUMN indexed_at      TEXT;
ALTER TABLE faces ADD COLUMN error_message   TEXT;
ALTER TABLE faces ADD COLUMN last_attempt_at TEXT;

-- 步骤 2: 回填 indexed_at / last_attempt_at 为 created_at(已知成功索引的时间近似)
UPDATE faces SET indexed_at = created_at, last_attempt_at = created_at
WHERE indexed_at IS NULL OR last_attempt_at IS NULL;

-- 步骤 3: 修正异常 — vector_id 为 NULL 的 face 不可能是 indexed(理论上不应存在,防御性)
UPDATE faces SET status = 'failed', error_message = 'orphan: vector_id is NULL after migration 013'
WHERE status = 'indexed' AND vector_id IS NULL;

-- 步骤 4: 状态索引(加速 list_by_status / count_by_status)
CREATE INDEX IF NOT EXISTS idx_faces_status ON faces(status);