-- 016_person_body_prototypes.sql — Person Body Prototype Architecture
--
-- 设计目标（Person Prototype V2）：
--   1. person_body_prototypes 表：存储 person 的 body prototype（768D YouTu Re-ID）
--   2. 与 person_prototypes（512D ArcFace）并列，构成双信号
--   3. Body prototype 用于 candidate retrieval（FAR=29%太高，不能用于 verification）
--   4. Face prototype 用于 final verification
--
-- body prototype 选择策略：
--   - crop_type 分桶（HeadBody / UpperBody）
--   - 桶内 greedy single-linkage 去重（cosine ≥ 0.90 视为近重复）
--   - cap 8 个 prototype

-- =============================================================================
-- 步骤 1: person_body_prototypes 表
-- =============================================================================

CREATE TABLE person_body_prototypes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id       INTEGER NOT NULL,
    body_id         INTEGER NOT NULL,                              -- 来源 body（审计）
    embedding       BLOB NOT NULL,                                  -- 小端 f32 BLOB, 768D
    dim             INTEGER NOT NULL DEFAULT 768,
    crop_type       TEXT NOT NULL,                                 -- head_body / upper_body
    quality_score   REAL,
    weight          REAL NOT NULL DEFAULT 1.0,
    model_version   TEXT NOT NULL DEFAULT 'ytu_reid@v1',
    is_active       INTEGER NOT NULL DEFAULT 1,                   -- 软删标记
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE CASCADE,
    FOREIGN KEY(body_id)   REFERENCES bodies(id)   ON DELETE SET NULL
);

CREATE INDEX idx_person_body_prototypes_person_active
    ON person_body_prototypes(person_id, is_active);
CREATE INDEX idx_person_body_prototypes_body
    ON person_body_prototypes(body_id);

-- =============================================================================
-- 步骤 2: persons 表加 identity_status 列
-- =============================================================================

ALTER TABLE persons ADD COLUMN identity_status TEXT NOT NULL DEFAULT 'unknown';

-- =============================================================================
-- 步骤 3: body_embeddings 表加 model_name 索引（查询优化）
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_body_embeddings_model_name
    ON body_embeddings(body_id, model_name);
