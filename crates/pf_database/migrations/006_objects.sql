-- 006_objects.sql — objects 表
--
-- 检测到的对象（dog / cat / car / 等）。

CREATE TABLE IF NOT EXISTS objects (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id        INTEGER NOT NULL,
    class_id        INTEGER NOT NULL,
    class_name      TEXT NOT NULL,
    confidence      REAL NOT NULL,
    bbox_x          REAL NOT NULL,
    bbox_y          REAL NOT NULL,
    bbox_w          REAL NOT NULL,
    bbox_h          REAL NOT NULL,
    model_version   TEXT NOT NULL,
    vector_id       INTEGER NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_objects_image_id      ON objects(image_id);
CREATE INDEX IF NOT EXISTS idx_objects_class_id      ON objects(class_id);