-- 017_identity_shadow_records.sql — Identity Shadow Mode Records
--
-- 设计目标（Phase 30 Shadow Mode）：
--   1. 记录 shadow pipeline vs legacy pipeline 的决策差异
--   2. Shadow mode 默认不修改任何生产数据（faces, persons, prototypes）
--   3. 仅记录 shadow decision 用于分析和调试
--   4. 支持 disagreement 追踪和 metrics 计算
--
-- 关键约束：
--   - Shadow 记录不影响任何生产表
--   - (image_id, pipeline_version, threshold_version) 作为唯一约束
--   - 避免重复写入（idempotent）

-- =============================================================================
-- 步骤 1: identity_shadow_records 表
-- =============================================================================

CREATE TABLE identity_shadow_records (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,

    -- 基础标识
    image_id            INTEGER NOT NULL,
    face_id             INTEGER,                                      -- 可选（body-only 时为空）
    body_id             INTEGER,                                      -- 可选（face-only 时为空）

    -- Legacy Pipeline 决策
    legacy_person_id     INTEGER,                                      -- Legacy 分配的 person（null = NewPerson）
    legacy_decision     TEXT NOT NULL,                                 -- legacy 决策结果

    -- Shadow Pipeline 决策
    shadow_person_id    INTEGER,                                      -- Shadow 分配的 person（null = NewPerson）
    shadow_decision     TEXT NOT NULL,                                 -- shadow 决策结果

    -- 分数详情
    face_score         REAL,                                          -- Face cosine 分数
    face_margin        REAL,                                          -- Face margin (best - second)
    body_score          REAL,                                          -- Body cosine 分数
    body_margin         REAL,                                          -- Body margin

    -- 置信度和质量
    confidence          REAL NOT NULL,                                  -- 综合置信度
    face_quality       REAL,                                          -- FaceQuality composite
    body_quality        REAL,                                          -- Body quality

    -- 状态标记
    anti_chain_status   TEXT,                                         -- 'passed' / 'blocked' / 'N/A'
    pollution_status    TEXT,                                          -- 'passed' / 'blocked' / 'N/A'
    quality_gate_status TEXT,                                          -- 'passed' / 'rejected'
    disagreement_type   TEXT,                                          -- Agree/Disagree 类型

    -- 版本信息（用于一致性检查）
    pipeline_version    TEXT NOT NULL,                                 -- scrfd_500m_bnkps + arcface_w600k_r50
    threshold_version   TEXT NOT NULL DEFAULT 'v1',                   -- 阈值版本
    model_version       TEXT NOT NULL,                                 -- 完整模型版本字符串

    -- 时间戳
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),

    -- 唯一约束（避免重复写入）
    UNIQUE(image_id, face_id, pipeline_version, threshold_version)
);

-- 索引
CREATE INDEX idx_shadow_image ON identity_shadow_records(image_id);
CREATE INDEX idx_shadow_face  ON identity_shadow_records(face_id);
CREATE INDEX idx_shadow_legacy_person ON identity_shadow_records(legacy_person_id);
CREATE INDEX idx_shadow_shadow_person ON identity_shadow_records(shadow_person_id);
CREATE INDEX idx_shadow_decision ON identity_shadow_records(shadow_decision);
-- 注意: disagreement 追踪通过 count_disagreements() 查询实现，不需要 partial index

-- =============================================================================
-- 步骤 2: 定期清理策略（可选）
-- =============================================================================

-- 建议：shadow_records 保留最近 30 天数据，定期归档清理
-- CREATE EVENT IF NOT EXISTS cleanup_shadow_records ...
