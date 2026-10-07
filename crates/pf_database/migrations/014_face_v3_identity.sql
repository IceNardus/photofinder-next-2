-- 014_face_v3_identity.sql — face identity v3 architecture
--
-- 设计目标（face recognition v3）：
--   1. vector_id 与 face_id 解耦 — faces 加 hnsw_handle 列,与 vector_id 并存
--      但语义清晰：vector_id 是最后已知的 hnsw handle,hnsw_handle 是下一次写入目标
--   2. faces 状态机拆分为 status (历史兼容) + index_status (当前权威) + index_generation
--   3. face_person_matches rename + 扩展 → face_person_assignments
--      + similarity / confidence / model_version 字段
--   4. person_prototypes 内嵌 embedding blob + face_id + weight + model_version
--      + pose_score / pose_yaw / is_active (替换 m012 的 FK 引用形式)
--   5. persons.face_count 自动同步 trigger
--
-- 全部变更 ADDitive（m013 同款容错）：
--   - 旧 vector_id / status 列保留,新代码读 hnsw_handle / index_status
--   - 旧 face_person_matches rename,不删除旧列(score)
--   - 旧 person_prototypes rename 为 _person_prototypes_legacy 后保留,数据复制到新表
--
-- 不修改:HNSW 接口、clustering 算法、search 算法（算法层在 P1+）
-- 不修改:faces.status / faces.vector_id 列（保留以保证旧代码与外部工具读得通）

-- =============================================================================
-- 步骤 1: faces 加 hnsw_handle + index_status + index_generation
-- =============================================================================

-- hnsw_handle: HNSW 内部 node_id 的最新已知值
--   - 与 vector_id 不同: vector_id 是历史上一次成功的 handle
--   - 新代码写 HNSW 时同时更新两列,旧代码读 vector_id 不报错
ALTER TABLE faces ADD COLUMN hnsw_handle INTEGER;
UPDATE faces SET hnsw_handle = vector_id
    WHERE hnsw_handle IS NULL AND vector_id IS NOT NULL;

-- index_status: 当前权威索引状态
--   - 与 status 等价起步,后续通过 mark_index_status 写
--   - status 列保留(Phase 阶段 UI 旧字段读取)
ALTER TABLE faces ADD COLUMN index_status TEXT NOT NULL DEFAULT 'indexed';
UPDATE faces SET index_status = status WHERE index_status IS NULL OR index_status = '';

-- index_generation: 用于追踪重建世代
--   - 启动时 SELECT MAX(index_generation) FROM faces
--   - reindex_all / 模型升级时 += 1
--   - 异步 rebuild_face_index 按 generation 过滤,避免旧 rebuild 覆盖新 face
ALTER TABLE faces ADD COLUMN index_generation INTEGER NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_faces_index_status    ON faces(index_status);
CREATE INDEX IF NOT EXISTS idx_faces_index_generation ON faces(index_generation);

-- =============================================================================
-- 步骤 2: face_person_matches → face_person_assignments
-- =============================================================================
-- SQLite 支持 RENAME TABLE,旧表的 score 列保留
ALTER TABLE face_person_matches RENAME TO face_person_assignments;

-- 新增列(user spec §4)
ALTER TABLE face_person_assignments ADD COLUMN similarity   REAL;
ALTER TABLE face_person_assignments ADD COLUMN confidence   REAL;
ALTER TABLE face_person_assignments ADD COLUMN model_version TEXT;

-- 回填:旧 score 当 similarity;confidence 与 model_version 默认值
UPDATE face_person_assignments
SET similarity = score,
    model_version = 'w600k_r50_v1'
WHERE similarity IS NULL;

-- 重建索引(原 idx_face_person_matches_* 仍然存在,因为 SQLite rename 表后索引名自动跟表名更新)
DROP INDEX IF EXISTS idx_face_person_matches_face;
DROP INDEX IF EXISTS idx_face_person_matches_person;
CREATE INDEX IF NOT EXISTS idx_fpa_face              ON face_person_assignments(face_id);
CREATE INDEX IF NOT EXISTS idx_fpa_person            ON face_person_assignments(person_id);
CREATE INDEX IF NOT EXISTS idx_fpa_status            ON face_person_assignments(status);
CREATE INDEX IF NOT EXISTS idx_fpa_face_person       ON face_person_assignments(face_id, person_id);
CREATE INDEX IF NOT EXISTS idx_fpa_person_status_sim ON face_person_assignments(person_id, status, similarity DESC);

-- =============================================================================
-- 步骤 3: person_prototypes → 新 schema (内嵌 embedding blob)
-- =============================================================================
-- 旧表(m012): id / person_id / prototype_type / embedding_id (FK) / vector_id / quality_score / face_count / created_at
-- 新表(v3):   id / person_id / face_id (来源 face,审计) / embedding (BLOB) / dim
--              / pose_score / pose_yaw / prototype_type / weight / model_version
--              / face_count / is_active / created_at

-- 3a. 先把旧表 rename 保留(不动旧数据)
ALTER TABLE person_prototypes RENAME TO _person_prototypes_legacy;

-- 3b. 用同名 person_prototypes 重建新 schema
CREATE TABLE person_prototypes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id       INTEGER NOT NULL,                              -- 来源 face (审计)
    face_id         INTEGER NOT NULL,                              -- 来源 face (审计)
    embedding       BLOB NOT NULL,                                  -- 小端 f32 BLOB
    dim             INTEGER NOT NULL,
    pose_score      REAL,                                            -- 来源 face 的 pose_score
    pose_yaw        REAL,                                            -- 用于 prototype_type 分桶
    quality_score   REAL,
    prototype_type  TEXT NOT NULL,                                   -- frontal/left_profile/right_profile/high_quality/general
    weight          REAL NOT NULL DEFAULT 1.0,                      -- 投票权重
    model_version   TEXT NOT NULL DEFAULT 'w600k_r50_v1',
    face_count      INTEGER NOT NULL DEFAULT 0,                     -- 该 prototype 覆盖的 face 数
    is_active       INTEGER NOT NULL DEFAULT 1,                     -- 软删标记(0=deleted)
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE CASCADE,
    FOREIGN KEY(face_id)   REFERENCES faces(id)   ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_person_prototypes_person_active
    ON person_prototypes(person_id, is_active);
CREATE INDEX IF NOT EXISTS idx_person_prototypes_face
    ON person_prototypes(face_id);
CREATE INDEX IF NOT EXISTS idx_person_prototypes_type
    ON person_prototypes(person_id, prototype_type) WHERE is_active = 1;

-- 3c. 数据迁移:旧 _person_prototypes_legacy → 新 person_prototypes
-- 旧表可能在 m012 之后没有写过任何数据(migration runner 是按顺序的),
-- 此时 LEFT JOIN 会返回 NULL,WHERE fe.vector IS NOT NULL 跳过。
INSERT INTO person_prototypes
    (person_id, face_id, embedding, dim, pose_score, pose_yaw,
     quality_score, prototype_type, weight, model_version, face_count, is_active, created_at)
SELECT
    pp.person_id,
    COALESCE(fe.face_id, 0),
    fe.vector,
    fe.dimension,
    f.pose_score,
    f.yaw,
    f.quality_score,
    pp.prototype_type,
    1.0,
    COALESCE(fe.model_version, 'w600k_r50_v1'),
    pp.face_count,
    1,
    pp.created_at
FROM _person_prototypes_legacy pp
LEFT JOIN face_embeddings fe ON fe.id = pp.embedding_id
LEFT JOIN faces f             ON f.id = fe.face_id
WHERE fe.vector IS NOT NULL;

-- =============================================================================
-- 步骤 4: persons.face_count trigger 自动同步
-- =============================================================================
-- SQLite ≥ 3.30 支持 trigger WHEN;检测版本决定是否启用 trigger
-- 当前 codebase 的 rusqlite 默认 SQLite 3.40+,trigger 稳定

CREATE TRIGGER IF NOT EXISTS trg_persons_face_count_insert
AFTER INSERT ON faces
WHEN NEW.person_id IS NOT NULL
BEGIN
    UPDATE persons SET face_count = (
        SELECT COUNT(*) FROM faces WHERE person_id = NEW.person_id
    ), updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
    WHERE id = NEW.person_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_persons_face_count_update
AFTER UPDATE OF person_id ON faces
WHEN NEW.person_id IS NOT NULL OR OLD.person_id IS NOT NULL
BEGIN
    UPDATE persons SET face_count = (
        SELECT COUNT(*) FROM faces WHERE person_id = NEW.person_id
    ), updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
    WHERE id = NEW.person_id AND NEW.person_id IS NOT NULL;
    UPDATE persons SET face_count = (
        SELECT COUNT(*) FROM faces WHERE person_id = OLD.person_id
    ), updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
    WHERE id = OLD.person_id AND OLD.person_id IS NOT NULL;
END;

CREATE TRIGGER IF NOT EXISTS trg_persons_face_count_delete
AFTER DELETE ON faces
WHEN OLD.person_id IS NOT NULL
BEGIN
    UPDATE persons SET face_count = (
        SELECT COUNT(*) FROM faces WHERE person_id = OLD.person_id
    ), updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
    WHERE id = OLD.person_id;
END;

-- =============================================================================
-- 步骤 5: model_version 一致性校验辅助(应用层 SELECT helper,SQLite 无 CHECK 跨表)
-- 启动时应用层做:
--   SELECT DISTINCT model_version FROM face_embeddings
-- 若返回多版本 → 不允许混合 HNSW, 需要 reindex
-- =============================================================================