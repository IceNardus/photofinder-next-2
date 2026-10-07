-- 018_shadow_enhanced_fields.sql — Enhanced Shadow Record Fields
--
-- Phase 31.5 T1: Shadow / Legacy Result Explainability
--
-- Adds fields needed to explain WHY Legacy and Shadow disagree:
--   - legacy_score: Legacy's prototype verification score
--   - legacy_margin: Legacy's top-2 candidate margin (prototype_score_top1 - prototype_score_top2)
--   - candidate_count: Number of distinct candidate persons Legacy evaluated
--   - shadow_face_score: Shadow's face evidence score (clarified naming)
--   - shadow_face_margin: Shadow's face evidence margin
--   - face_size, yaw, pitch, roll, blur_score: Face quality components
--
-- Note: face_score/face_margin columns already exist but named generically.
-- This migration adds shadow_face_score/margin as the canonical Shadow field names.
-- Legacy evaluation uses legacy_score/legacy_margin.

-- Add Legacy decision details
ALTER TABLE identity_shadow_records ADD COLUMN legacy_score REAL;
ALTER TABLE identity_shadow_records ADD COLUMN legacy_margin REAL;
ALTER TABLE identity_shadow_records ADD COLUMN candidate_count INTEGER;

-- Add Shadow-specific score fields (face evidence)
ALTER TABLE identity_shadow_records ADD COLUMN shadow_face_score REAL;
ALTER TABLE identity_shadow_records ADD COLUMN shadow_face_margin REAL;

-- Add face quality components for detailed analysis
ALTER TABLE identity_shadow_records ADD COLUMN face_size REAL;
ALTER TABLE identity_shadow_records ADD COLUMN yaw REAL;
ALTER TABLE identity_shadow_records ADD COLUMN pitch REAL;
ALTER TABLE identity_shadow_records ADD COLUMN roll REAL;
ALTER TABLE identity_shadow_records ADD COLUMN blur_score REAL;
