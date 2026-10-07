-- 009_faces_extended.sql — 添加详细质量分列
--
-- 对齐 ai-next faces 表的 blur_score / pose_score / face_area_score 字段。
-- 这些分数在 `pf_ai::quality::QualityBreakdown` 中独立计算，
-- 比单一 quality 值更适合做质量分析和阈值调优。

ALTER TABLE faces ADD COLUMN blur_score       REAL;
ALTER TABLE faces ADD COLUMN pose_score       REAL;
ALTER TABLE faces ADD COLUMN face_area_score  REAL;
