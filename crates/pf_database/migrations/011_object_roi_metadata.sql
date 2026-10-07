-- 011_object_roi_metadata.sql — Phase 2 新增 ROI 元数据
--
-- roi_scale:  在哪些 scale 上提取（用于 rank 调权）
-- roi_type:   ROI 来源（FullImage / SlidingWindow），未来可扩展
--             DetectedObject / Saliency / RegionProposal
--
-- 兼容旧库：列不存在时跳过 ALTER 失败（migration.rs 视 duplicate column 为 ignorable）

ALTER TABLE objects ADD COLUMN roi_scale REAL NOT NULL DEFAULT 1.0;
ALTER TABLE objects ADD COLUMN roi_type   TEXT NOT NULL DEFAULT 'FullImage';

CREATE INDEX IF NOT EXISTS idx_objects_roi_type ON objects(roi_type);