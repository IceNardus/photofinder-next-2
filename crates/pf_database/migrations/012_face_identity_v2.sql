-- 012_face_identity_v2.sql — 人脸身份模型 v2
--
-- 目标（与 plan §3-§10 一致）：
--   1. faces 拆 1:1 → 1:N（允许多 embedding / 多 detector / 多 alignment）
--   2. face_embeddings 增加 model_name + model_version + dimension + vector_id + normalized
--   3. persons 增加 status + updated_at,不再物理删除
--   4. 新增 person_prototypes + face_person_matches
--   5. faces.person_id 加真正 FK（ON DELETE SET NULL）
--   6. faces.vector_id 改为可空（pending 状态可写入）
--
-- 不修改:识别算法、clustering 算法、search 算法（这些在 Phase 2-5）。

-- 步骤 1: 清孤儿 person_id,防止后续 FK 验证失败
UPDATE faces
SET person_id = NULL
WHERE person_id IS NOT NULL
  AND person_id NOT IN (SELECT id FROM persons);

-- 步骤 2: 重建 faces 表（SQLite ALTER TABLE 不能改 NOT NULL 与 FK,必须新建表）
CREATE TABLE faces_new (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id            INTEGER NOT NULL,
    person_id           INTEGER,
    bbox_x              REAL NOT NULL,
    bbox_y              REAL NOT NULL,
    bbox_w              REAL NOT NULL,
    bbox_h              REAL NOT NULL,
    detector_score      REAL NOT NULL,
    detector_model      TEXT NOT NULL DEFAULT 'scrfd-500m-bnkps',
    keypoints_json      TEXT,
    yaw                 REAL,
    pitch               REAL,
    roll                REAL,
    quality_score       REAL,
    blur_score          REAL,
    pose_score          REAL,
    face_area_score     REAL,
    alignment_version   TEXT,
    embedding_model     TEXT,
    model_version       TEXT NOT NULL,
    vector_id           INTEGER,                                  -- 改为 nullable（pending 状态）
    cluster_score       REAL,
    cluster_method      TEXT,
    created_at          TEXT NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE,
    FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE SET NULL
);

INSERT INTO faces_new
    (id, image_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h,
     detector_score, detector_model, keypoints_json,
     yaw, pitch, roll,
     quality_score, blur_score, pose_score, face_area_score,
     alignment_version, embedding_model, model_version,
     vector_id, cluster_score, cluster_method, created_at)
SELECT
    id, image_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h,
    detector_score, 'scrfd-500m-bnkps', keypoints_json,
    yaw, pitch, roll,
    quality, blur_score, pose_score, face_area_score,
    NULL, NULL, model_version,
    vector_id, NULL, NULL, created_at
FROM faces;

DROP TABLE faces;
ALTER TABLE faces_new RENAME TO faces;

-- 步骤 3: faces 索引（保持原有）
CREATE INDEX IF NOT EXISTS idx_faces_image_id        ON faces(image_id);
CREATE INDEX IF NOT EXISTS idx_faces_person_id       ON faces(person_id);
CREATE INDEX IF NOT EXISTS idx_faces_vector_id       ON faces(vector_id);

-- 步骤 4: 重建 face_embeddings 表（拆 1:1 → 1:N）
CREATE TABLE face_embeddings_new (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id       INTEGER NOT NULL,
    model_name    TEXT NOT NULL DEFAULT 'arcface',
    model_version TEXT NOT NULL DEFAULT 'w600k_r50_v1',
    dimension     INTEGER NOT NULL,
    vector        BLOB NOT NULL,
    vector_id     INTEGER,                                       -- nullable,等 HNSW 写入后回填
    normalized    INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    FOREIGN KEY(face_id) REFERENCES faces(id) ON DELETE CASCADE,
    UNIQUE(face_id, model_name, model_version)
);

-- 旧数据全部是 ArcFace w600k r50(已知)
INSERT INTO face_embeddings_new
    (face_id, model_name, model_version, dimension, vector, vector_id, normalized, created_at)
SELECT
    face_id, 'arcface', 'w600k_r50_v1', dim, vector, NULL, 1, created_at
FROM face_embeddings;

DROP TABLE face_embeddings;
ALTER TABLE face_embeddings_new RENAME TO face_embeddings;

-- 步骤 5: face_embeddings 索引
-- 注:历史版本曾有 idx_face_embeddings_image_id ON face_embeddings(image_id) 一行,
-- 但 face_embeddings 步骤 4 重建后已无 image_id 列(只有 face_id),所以该 CREATE 永远失败。
-- 之前被 is_ignorable_error('no such column') 静默吞掉,Bug #13 修复后必须移除。
-- 正确的 image→embeddings 反查路径是 face_embeddings.face_id → faces.id → faces.image_id,
-- 由 idx_face_embeddings_face_id(下方)+ idx_faces_image_id(004) 覆盖。
CREATE INDEX IF NOT EXISTS idx_face_embeddings_face_id  ON face_embeddings(face_id);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_model    ON face_embeddings(model_name, model_version);

-- 步骤 6: persons 增加 status + updated_at
ALTER TABLE persons ADD COLUMN status     TEXT NOT NULL DEFAULT 'active';
ALTER TABLE persons ADD COLUMN updated_at TEXT NOT NULL
    DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'));

-- 步骤 7: person_prototypes 表(plan §8)
CREATE TABLE IF NOT EXISTS person_prototypes (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id      INTEGER NOT NULL,
    prototype_type TEXT NOT NULL,                               -- frontal / left_profile / right_profile / high_quality / glasses / general
    embedding_id   INTEGER,
    vector_id      INTEGER,
    quality_score  REAL,
    face_count     INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    FOREIGN KEY(person_id)    REFERENCES persons(id)         ON DELETE CASCADE,
    FOREIGN KEY(embedding_id) REFERENCES face_embeddings(id)  ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_person_prototypes_person ON person_prototypes(person_id);

-- 步骤 8: face_person_matches 表(plan §10,可第二阶段使用)
CREATE TABLE IF NOT EXISTS face_person_matches (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id    INTEGER NOT NULL,
    person_id  INTEGER NOT NULL,
    score      REAL NOT NULL,
    method     TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'candidate',              -- candidate / confirmed / rejected
    created_at TEXT NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    FOREIGN KEY(face_id)   REFERENCES faces(id)   ON DELETE CASCADE,
    FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_face_person_matches_face   ON face_person_matches(face_id);
CREATE INDEX IF NOT EXISTS idx_face_person_matches_person ON face_person_matches(person_id);
