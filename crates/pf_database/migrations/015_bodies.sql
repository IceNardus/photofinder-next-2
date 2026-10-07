-- 015_bodies.sql — body embedding architecture
--
-- 设计目标：
--   1. bodies 表：存储每张图的 body crop 元数据（bbox, crop_type, quality）
--   2. body_embeddings 表：存储 YouTu Re-ID 768D embedding（BLOB）
--   3. body_vectors 复用现有 HNSW 抽象（pf_vector），路径 <data_dir>/index/body.hnsw.*
--   4. bodies.person_id → persons.id（与 faces.person_id 并列）
--   5. body_person_assignments 表：body → person 候选打分（与 face_person_assignments 对应）
--
-- 不修改：faces / persons / face_embeddings 表

-- =============================================================================
-- 步骤 1: bodies 表
-- =============================================================================

CREATE TABLE bodies (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id            INTEGER NOT NULL,
    person_id           INTEGER,

    -- Bounding box（body crop 区域，基于 face bbox 扩展）
    bbox_x              REAL NOT NULL,
    bbox_y              REAL NOT NULL,
    bbox_w              REAL NOT NULL,
    bbox_h              REAL NOT NULL,

    -- Crop 策略
    crop_type           TEXT NOT NULL DEFAULT 'head_body',

    -- Embedding 元数据
    embedding_model     TEXT NOT NULL DEFAULT 'ytu_reid',
    model_version      TEXT,

    -- HNSW handle
    vector_id          INTEGER,

    -- Quality score
    quality_score      REAL,

    -- Status
    status             TEXT NOT NULL DEFAULT 'pending',

    -- Timestamps
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),

    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE SET NULL
);

CREATE INDEX idx_bodies_image_id ON bodies(image_id);
CREATE INDEX idx_bodies_person_id ON bodies(person_id);
CREATE INDEX idx_bodies_status ON bodies(status);
CREATE INDEX idx_bodies_vector_id ON bodies(vector_id);

-- =============================================================================
-- 步骤 2: body_embeddings 表（可选多模型存储）
-- =============================================================================

CREATE TABLE body_embeddings (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    body_id            INTEGER NOT NULL,
    model_name         TEXT NOT NULL DEFAULT 'ytu_reid',
    model_version     TEXT,
    dimension         INTEGER NOT NULL DEFAULT 768,
    vector            BLOB NOT NULL,
    vector_id         INTEGER,
    normalized        INTEGER NOT NULL DEFAULT 1,
    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE(body_id, model_name, model_version),
    FOREIGN KEY (body_id) REFERENCES bodies(id) ON DELETE CASCADE
);

CREATE INDEX idx_body_embeddings_body_id ON body_embeddings(body_id);
CREATE INDEX idx_body_embeddings_model ON body_embeddings(model_name);
CREATE INDEX idx_body_embeddings_vector_id ON body_embeddings(vector_id);

-- =============================================================================
-- 步骤 3: body_person_assignments 表（body → person 候选）
-- =============================================================================

CREATE TABLE body_person_assignments (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    body_id            INTEGER NOT NULL,
    person_id          INTEGER NOT NULL,
    similarity         REAL,
    confidence         REAL,
    method             TEXT,
    status             TEXT NOT NULL DEFAULT 'candidate',
    model_version      TEXT,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    FOREIGN KEY (body_id)  REFERENCES bodies(id)  ON DELETE CASCADE,
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
);

CREATE INDEX idx_bpa_body ON body_person_assignments(body_id);
CREATE INDEX idx_bpa_person ON body_person_assignments(person_id);
CREATE INDEX idx_bpa_status ON body_person_assignments(status);
CREATE INDEX idx_bpa_body_person ON body_person_assignments(body_id, person_id);
CREATE INDEX idx_bpa_person_status_sim ON body_person_assignments(person_id, status, similarity DESC);

-- =============================================================================
-- 步骤 4: persons.body_count trigger 自动同步
-- =============================================================================

CREATE TRIGGER IF NOT EXISTS trg_persons_body_count_insert
AFTER INSERT ON bodies
WHEN NEW.person_id IS NOT NULL
BEGIN
    UPDATE persons SET body_count = (
        SELECT COUNT(*) FROM bodies WHERE person_id = NEW.person_id
    ), updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
    WHERE id = NEW.person_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_persons_body_count_update
AFTER UPDATE OF person_id ON bodies
WHEN NEW.person_id IS NOT NULL OR OLD.person_id IS NOT NULL
BEGIN
    UPDATE persons SET body_count = (
        SELECT COUNT(*) FROM bodies WHERE person_id = NEW.person_id
    ), updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
    WHERE id = NEW.person_id AND NEW.person_id IS NOT NULL;
    UPDATE persons SET body_count = (
        SELECT COUNT(*) FROM bodies WHERE person_id = OLD.person_id
    ), updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
    WHERE id = OLD.person_id AND OLD.person_id IS NOT NULL;
END;

CREATE TRIGGER IF NOT EXISTS trg_persons_body_count_delete
AFTER DELETE ON bodies
WHEN OLD.person_id IS NOT NULL
BEGIN
    UPDATE persons SET body_count = (
        SELECT COUNT(*) FROM bodies WHERE person_id = OLD.person_id
    ), updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
    WHERE id = OLD.person_id;
END;

-- =============================================================================
-- 步骤 5: persons 表加 body_count 列（如果没有）
-- =============================================================================

-- 检查 persons 是否有 body_count 列
-- （如果 migration 顺序混乱可能导致列已存在）
-- SQLite 没有 IF NOT EXISTS COLUMN，使用 PRAGMA 查询
-- 但 migration runner 通常按顺序，所以简单 ADD
-- 安全做法：如果列存在会报错，应用层处理

-- 尝试添加 body_count 列
-- 注意：SQLite 3.25+ 支持 IF NOT EXISTS 但对列不支持
-- 这里简单加，应用层如果有错误可手动处理
ALTER TABLE persons ADD COLUMN body_count INTEGER NOT NULL DEFAULT 0;
