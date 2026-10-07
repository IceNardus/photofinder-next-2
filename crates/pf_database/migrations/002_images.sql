-- 002_images.sql — images 表
--
-- 扫描发现的图片主表。
-- 使用 CREATE TABLE IF NOT EXISTS 以支持全新 DB；
-- 使用 ADD COLUMN 兼容已有表（旧 DB 的 images 表结构不同）。
-- SQLite ADD COLUMN 是 online migration，兼容已有数据。

-- 创建 images 表（全新 DB 时执行）
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    hash TEXT NOT NULL,
    size INTEGER NOT NULL,
    modified_time INTEGER NOT NULL DEFAULT 0,
    width INTEGER,
    height INTEGER,
    captured_at TEXT,
    thumbnail_path TEXT,
    thumbnail_status TEXT NOT NULL DEFAULT 'pending',
    scan_status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    face_count INTEGER NOT NULL DEFAULT 0,
    object_count INTEGER NOT NULL DEFAULT 0
);

-- 后续迁移只增新表/新索引，不修改已有表结构。
