-- 004_faces.sql — faces 表
--
-- 人脸检测 + embedding 后的记录。
-- vector embedding 本身存 pf_vector（mmap 文件），DB 只存 metadata。

CREATE TABLE IF NOT EXISTS faces (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id            INTEGER NOT NULL,
    person_id           INTEGER,                          -- 聚类后填
    bbox_x              REAL NOT NULL,
    bbox_y              REAL NOT NULL,
    bbox_w              REAL NOT NULL,
    bbox_h              REAL NOT NULL,
    detector_score      REAL NOT NULL,
    quality             REAL NOT NULL,
    yaw                 REAL,
    pitch               REAL,
    roll                REAL,
    keypoints_json      TEXT,                             -- 5 个关键点 JSON
    model_version       TEXT NOT NULL,
    vector_id           INTEGER NOT NULL,                 -- pf_vector::VectorIndex 中的 id
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_faces_image_id        ON faces(image_id);
CREATE INDEX IF NOT EXISTS idx_faces_person_id       ON faces(person_id);
CREATE INDEX IF NOT EXISTS idx_faces_vector_id       ON faces(vector_id);