-- 008_face_embeddings.sql — face embeddings 表
--
-- 持久化 ArcFace 512-d embeddings，供 `PersonService::cluster_all` 反查。
-- 替代 `pf_vector::VectorIndex` 不能 reverse-lookup 的退化路径（曾经用 `vec![detector_score; 512]` 凑数）。
--
-- dim = 512 (ArcFace-w600k-r50)
-- vector 存 little-endian f32 BLOB（size = dim * 4 bytes）

CREATE TABLE IF NOT EXISTS face_embeddings (
    face_id    INTEGER PRIMARY KEY,
    image_id   INTEGER NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (face_id) REFERENCES faces(id) ON DELETE CASCADE,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_face_embeddings_image_id ON face_embeddings(image_id);
