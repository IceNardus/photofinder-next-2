-- 007_patches.sql — patches 表（Phase 3）
--
-- SuperPoint 关键点 + 描述子 + VLAD 聚合。
-- 描述子本体存 pf_vector，DB 只存 metadata。

CREATE TABLE IF NOT EXISTS patches (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id        INTEGER NOT NULL,
    patch_x         REAL NOT NULL,
    patch_y         REAL NOT NULL,
    patch_w         REAL NOT NULL,
    patch_h         REAL NOT NULL,
    keypoint_count  INTEGER NOT NULL,
    vector_id       INTEGER NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_patches_image_id     ON patches(image_id);