-- 019_face_cascade_roll_correct.sql
--
-- Face Cascade + RollCorrect production migration.
-- Adds detector_origin enum-as-text column and face_roll_corrections audit table.

-- detector_origin records which detector(s) produced this face.
-- Values: 'Primary' (500M only) | 'Secondary' (10G only, cascade recovery) | 'Both' (dedup'd match)
-- Existing rows default to 'Primary' (no cascade was running before this migration).
ALTER TABLE faces ADD COLUMN detector_origin TEXT NOT NULL DEFAULT 'Primary';

-- face_roll_corrections: audit table for RollCorrect replacements.
-- A row exists only for faces where RollCorrect embedding REPLACED baseline
-- (triggered by |roll|>threshold at detection time).
CREATE TABLE face_roll_corrections (
    face_id          INTEGER PRIMARY KEY,
    roll_deg         REAL    NOT NULL,
    baseline_replaced INTEGER NOT NULL DEFAULT 1,
    created_at       INTEGER NOT NULL,
    FOREIGN KEY (face_id) REFERENCES faces(id) ON DELETE CASCADE
);

CREATE INDEX idx_face_rc_created ON face_roll_corrections(created_at);
