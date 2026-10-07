//! Identity Evidence — 多证据身份判定系统 (Phase 14, Phase 29 Production Hardening)
//!
//! 核心原则：
//! - Face 是主要身份证据
//! - Body 是辅助召回证据，不能单独确认身份
//! - Cluster 阈值严于 Search 阈值
//! - Anti-chaining 分别计算 Face/Body 的 top1-top2 差距
//!
//! Phase 29 Production Hardening:
//! - T1: Model Version Freeze (scrfd_500m_bnkps, arcface_w600k_r50, youtu_reid, margin_v1)
//! - T2-T3: FaceQuality struct + FaceQualityGate
//! - T4: Evidence v2 with quality-aware confidence
//! - T5: Threshold Separation (PAIRWISE/PROTOTYPE/AUTO_ASSIGN/REVIEW/CLUSTER_MERGE)
//! - T6: Decision Policy (AUTO/REVIEW/UNKNOWN)
//! - T7-T9: Multi-Prototype + Anti-Chaining + Pollution Protection
//! - T10: Shadow Mode
//! - T11-T14: Production Metrics, Regression, Efficiency, Gate
//!
//! Phase 30 Shadow Mode Integration:
//! - T2: IdentityPipeline Adapter (unified entry point)
//! - T3: ExecutionMode enum (Legacy/Shadow/NewPipeline)
//! - T10: Shadow database storage

use std::sync::Arc;

use pf_ai::face::traits::FaceFeature;
use pf_vector::{SearchHit, VectorIndex};
use serde::{Serialize, Deserialize};

use crate::prototype_service::PrototypeService;

// ============================================================================
// T1: Model Version Freeze
// ============================================================================

/// Identity Pipeline 版本信息（Phase 29 T1）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityPipelineVersion {
    /// Face detector model
    pub face_detector: String,
    /// Face embedding model
    pub face_embedding: String,
    /// Body embedding model
    pub body_embedding: String,
    /// Fusion strategy version
    pub fusion: String,
    /// Pipeline creation timestamp (UNIX epoch seconds)
    pub created_at: u64,
}

impl IdentityPipelineVersion {
    pub fn current() -> Self {
        Self {
            face_detector: "scrfd_500m_bnkps".to_string(),
            face_embedding: "arcface_w600k_r50".to_string(),
            body_embedding: "person_reid_youtu_2021nov".to_string(),
            fusion: "margin_v1".to_string(),
            created_at: current_unix_time(),
        }
    }

    pub fn is_compatible(&self, other: &IdentityPipelineVersion) -> bool {
        self.face_detector == other.face_detector
            && self.face_embedding == other.face_embedding
            && self.body_embedding == other.body_embedding
            && self.fusion == other.fusion
    }
}

fn current_unix_time() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ============================================================================
// T2: Face Quality Struct
// ============================================================================

/// Face 质量指标（Phase 29 T2）
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FaceQuality {
    /// Face 大小（宽高的最小值，像素）
    pub face_size: f32,
    /// Face 检测置信度
    pub detection_score: f32,
    /// 对齐质量分数（alignment pipeline 输出）
    pub alignment_score: f32,
    /// Yaw 角度（度，负值=左侧，正值=右侧）
    pub yaw: f32,
    /// Pitch 角度（度，负值=俯视，正值=仰视）
    pub pitch: f32,
    /// Roll 角度（度）
    pub roll: f32,
    /// 模糊分数（0=模糊，1=清晰）
    pub blur_score: f32,
}

impl FaceQuality {
    /// 计算综合质量分数（用于 prototype 构建）
    pub fn composite_quality(&self) -> f32 {
        let size_factor = (self.face_size / 120.0).min(1.0);
        let detection_factor = self.detection_score;
        let alignment_factor = self.alignment_score;
        let blur_factor = self.blur_score;

        // Pose penalty：越大角度，质量越低
        let yaw_penalty = 1.0 - (self.yaw.abs() / 90.0).min(1.0) * 0.3;
        let pitch_penalty = 1.0 - (self.pitch.abs() / 90.0).min(1.0) * 0.2;
        let roll_penalty = 1.0 - (self.roll.abs() / 45.0).min(1.0) * 0.1;
        let pose_factor = yaw_penalty * pitch_penalty * roll_penalty;

        size_factor * detection_factor * alignment_factor * blur_factor * pose_factor
    }

    /// Face 是否为"高质量"（可用于 prototype 构建）
    pub fn is_high_quality(&self, min_quality: f32) -> bool {
        self.composite_quality() >= min_quality
    }

    // =========================================================================
    // Phase 36: Identity Eligibility
    // =========================================================================

    /// Face 是否可以用于身份识别（可以搜索，但可能不能确认身份）
    ///
    /// 条件：
    /// - face_size >= 32 (小脸检测有效但不自动确认)
    /// - detection_score >= 0.30
    ///
    /// 注意：这只是"可以使用"，不代表"可以自动确认身份"
    pub fn identity_eligible(&self) -> bool {
        self.face_size >= 32.0 && self.detection_score >= 0.30
    }

    /// Face 是否可以用于更新 Person Prototype
    ///
    /// 条件：
    /// - face_size >= 60
    /// - composite_quality >= 0.45
    /// - |yaw| <= 50°
    /// - blur_score >= 0.3
    ///
    /// 只有高质量脸才有权限创建/污染 Person Prototype
    pub fn prototype_eligible(&self) -> bool {
        let yaw_ok = self.yaw.abs() <= 50.0;
        let blur_ok = self.blur_score >= 0.3;
        let quality_ok = self.composite_quality() >= 0.45;
        let size_ok = self.face_size >= 60.0;

        size_ok && quality_ok && yaw_ok && blur_ok
    }

    /// Face 姿态分类（用于决策）
    ///
    /// - NORMAL: |yaw| < 30°
    /// - MODERATE: 30° <= |yaw| < 45°
    /// - HARD: 45° <= |yaw| < 60°
    /// - EXTREME: |yaw| >= 60°
    pub fn pose_category(&self) -> &'static str {
        let yaw_abs = self.yaw.abs();
        if yaw_abs < 30.0 {
            "NORMAL"
        } else if yaw_abs < 45.0 {
            "MODERATE"
        } else if yaw_abs < 60.0 {
            "HARD"
        } else {
            "EXTREME"
        }
    }
}

impl Default for FaceQuality {
    fn default() -> Self {
        Self {
            face_size: 0.0,
            detection_score: 0.0,
            alignment_score: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            blur_score: 0.0,
        }
    }
}

impl From<FaceFeature> for FaceQuality {
    /// Extract FaceQuality from FaceFeature produced by FacePipeline
    fn from(feature: FaceFeature) -> Self {
        let (yaw, pitch, roll) = feature
            .yaw_pitch_roll
            .unwrap_or((0.0, 0.0, 0.0));
        let face_size = feature.detection.bbox.w.min(feature.detection.bbox.h);
        Self {
            face_size,
            detection_score: feature.detection.score,
            alignment_score: feature.pose_score,
            yaw,
            pitch,
            roll,
            blur_score: feature.blur_score,
        }
    }
}

// ============================================================================
// T3: Face Quality Gate
// ============================================================================

/// FaceQualityGate 配置（Phase 29 T3）
#[derive(Debug, Clone, Copy)]
pub struct FaceQualityGateConfig {
    /// 最小 face size（像素）
    pub min_face_size: f32,
    /// 最小检测分数
    pub min_detection_score: f32,
    /// 最小对齐分数
    pub min_alignment_score: f32,
    /// 最大 yaw 角度（度）
    pub max_yaw: f32,
    /// 最大 pitch 角度（度）
    pub max_pitch: f32,
    /// 最小模糊分数
    pub min_blur_score: f32,
    /// 最小综合质量
    pub min_composite_quality: f32,
}

impl Default for FaceQualityGateConfig {
    fn default() -> Self {
        Self {
            min_face_size: 40.0,        // 过滤远景小人脸
            min_detection_score: 0.5,   // 检测置信度
            min_alignment_score: 0.3,   // 对齐质量
            max_yaw: 45.0,             // 大角度侧脸
            max_pitch: 30.0,
            min_blur_score: 0.5,        // 模糊过滤
            min_composite_quality: 0.25, // 综合质量阈值
        }
    }
}

/// Face Quality Gate — 低质量 face 不贡献强证据（Phase 29 T3）
pub struct FaceQualityGate {
    config: FaceQualityGateConfig,
}

impl FaceQualityGate {
    pub fn new(config: FaceQualityGateConfig) -> Self {
        Self { config }
    }

    pub fn from_default() -> Self {
        Self::new(FaceQualityGateConfig::default())
    }

    /// 检查 face 是否通过 gate
    pub fn passes(&self, quality: FaceQuality) -> bool {
        if quality.face_size < self.config.min_face_size {
            return false;
        }
        if quality.detection_score < self.config.min_detection_score {
            return false;
        }
        if quality.alignment_score < self.config.min_alignment_score {
            return false;
        }
        if quality.yaw.abs() > self.config.max_yaw {
            return false;
        }
        if quality.pitch.abs() > self.config.max_pitch {
            return false;
        }
        if quality.blur_score < self.config.min_blur_score {
            return false;
        }
        if quality.composite_quality() < self.config.min_composite_quality {
            return false;
        }
        true
    }

    /// 返回 gate 失败原因（用于调试）
    pub fn failure_reason(&self, quality: FaceQuality) -> Option<&'static str> {
        if quality.face_size < self.config.min_face_size {
            return Some("face_size_too_small");
        }
        if quality.detection_score < self.config.min_detection_score {
            return Some("detection_score_too_low");
        }
        if quality.alignment_score < self.config.min_alignment_score {
            return Some("alignment_score_too_low");
        }
        if quality.yaw.abs() > self.config.max_yaw {
            return Some("yaw_too_large");
        }
        if quality.pitch.abs() > self.config.max_pitch {
            return Some("pitch_too_large");
        }
        if quality.blur_score < self.config.min_blur_score {
            return Some("blur_score_too_low");
        }
        if quality.composite_quality() < self.config.min_composite_quality {
            return Some("composite_quality_too_low");
        }
        None
    }

    /// 计算 quality weight（用于 prototype 构建时的加权）
    /// 返回 [0.0, 1.0] 的权重，低质量 face 权重降低
    pub fn quality_weight(&self, quality: FaceQuality) -> f32 {
        if !self.passes(quality) {
            return 0.0;
        }
        // 质量越高权重越高，但平滑处理
        let q = quality.composite_quality();
        // 映射到 [0.5, 1.0] 范围，避免完全丢弃中等质量 face
        0.5 + 0.5 * (q / self.config.min_composite_quality).min(2.0)
    }
}

/// 身份判定结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdentityDecision {
    /// 高置信匹配：Face 强 + Margin 足够
    ConfidentMatch,
    /// 高分 + 小 Margin：存在竞争候选人，但仍分配 Top1
    StrongMatchWithCompetitor,
    /// 弱匹配：Face 中等 + Margin 足够
    WeakMatch,
    /// 歧义：无法确定，暂不分配
    Ambiguous,
    /// 新人：高质量人脸且所有已有 Person 分数都很低
    NewPerson,
    /// 不可靠：人脸质量不足，不参与自动决策
    Unreliable,
    /// Gray Zone 多脸支持匹配：prototype_score 在 0.70-0.78 之间，但有多个已有 face 独立支持
    /// Phase 36.1: 用于处理连续拍摄同一人但 prototype 分数不够高的情况
    SupportedMatch,
}

impl IdentityDecision {
    pub fn as_str(&self) -> &'static str {
        match self {
            IdentityDecision::ConfidentMatch => "confirmed",
            IdentityDecision::StrongMatchWithCompetitor => "strong_with_competitor",
            IdentityDecision::WeakMatch => "probable",
            IdentityDecision::Ambiguous => "conflict",
            IdentityDecision::NewPerson => "unknown",
            IdentityDecision::Unreliable => "unreliable",
            IdentityDecision::SupportedMatch => "supported",
        }
    }
}

/// 证据等级（用于标识有多少通道可用）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceLevel {
    /// 两个通道都可用
    Dual,
    /// 只有 Face 通道
    FaceOnly,
    /// 只有 Body 通道
    BodyOnly,
    /// 没有通道可用
    None,
}

impl EvidenceLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            EvidenceLevel::Dual => "DUAL",
            EvidenceLevel::FaceOnly => "FACE_ONLY",
            EvidenceLevel::BodyOnly => "BODY_ONLY",
            EvidenceLevel::None => "NONE",
        }
    }
}

/// 单个人脸/物体的 HNSW 搜索命中。
/// 用于携带原始分数信息。
#[derive(Debug, Clone)]
pub struct SearchHitWithScore {
    pub vector_id: i64,
    pub score: f32,
}

/// 候选 person 集合（双通道 candidate retrieval 结果）。
#[derive(Debug, Clone)]
pub struct CandidateSet {
    /// 候选 person 列表（按 face_rank 排序）
    pub candidates: Vec<PersonCandidate>,
    /// 原始 face HNSW hits（用于调试/分析）
    pub face_hits: Vec<SearchHitWithScore>,
    /// 原始 body HNSW hits（用于调试/分析）
    pub body_hits: Vec<SearchHitWithScore>,
}

/// 单个候选 person 的基本信息。
#[derive(Debug, Clone)]
pub struct PersonCandidate {
    pub person_id: i64,
    /// Face 最佳分数
    pub face_best_score: f32,
    /// Body 最佳分数
    pub body_best_score: f32,
    /// Face 排名（1-indexed）
    pub face_rank: usize,
    /// Body 排名（1-indexed）
    pub body_rank: usize,
}

impl PersonCandidate {
    pub fn new(person_id: i64) -> Self {
        Self {
            person_id,
            face_best_score: 0.0,
            body_best_score: 0.0,
            face_rank: 0,
            body_rank: 0,
        }
    }

    pub fn update_face(&mut self, score: f32, rank: usize) {
        if rank < self.face_rank || self.face_rank == 0 {
            self.face_best_score = score;
            self.face_rank = rank;
        }
    }

    pub fn update_body(&mut self, score: f32, rank: usize) {
        if rank < self.body_rank || self.body_rank == 0 {
            self.body_best_score = score;
            self.body_rank = rank;
        }
    }
}

/// 身份证据（用于多证据评分决策）。
#[derive(Debug, Clone)]
pub struct IdentityEvidence {
    pub person_id: i64,

    // ========== Face evidence ==========
    /// Face 与该 person face prototypes 的最大 cosine 相似度
    pub face_score: f32,
    /// 最佳 prototype 的索引
    pub face_best_proto_idx: usize,
    /// 第二高 prototype 分数（用于 margin 计算）
    pub face_second_score: f32,
    /// 该 person 的 face prototype 数量
    pub face_prototype_count: usize,
    /// Face 排名
    pub face_rank: usize,
    /// Face margin: face_score - max(other_person_face_score). None if candidate_count < 2.
    pub face_margin: Option<f32>,

    // ========== Body evidence ==========
    /// Body 与该 person body prototypes 的最大 cosine 相似度
    pub body_score: f32,
    /// 最佳 prototype 的索引
    pub body_best_proto_idx: usize,
    /// 第二高 prototype 分数（用于 margin 计算）
    pub body_second_score: f32,
    /// 该 person 的 body prototype 数量
    pub body_prototype_count: usize,
    /// Body 排名
    pub body_rank: usize,
    /// Body margin: body_score - max(other_person_body_score). None if candidate_count < 2.
    pub body_margin: Option<f32>,

    // ========== Quality ==========
    pub face_quality: f32,
    pub body_quality: f32,

    // ========== Support count ==========
    /// 支持该 assignment 的样本数量（Face）
    pub face_support_count: usize,
    /// 支持该 assignment 的样本数量（Body）
    pub body_support_count: usize,

    // ========== Channel availability ==========
    /// Face 通道是否可用（检测到 face 且有 prototypes）
    pub face_available: bool,
    /// Body 通道是否可用（检测到 body 且有 prototypes）
    pub body_available: bool,

    // ========== Fusion ==========
    /// 综合 margin（使用 margin-based fusion 处理 missing channels）
    pub fusion_margin: f32,
}

impl IdentityEvidence {
    pub fn new(person_id: i64) -> Self {
        Self {
            person_id,
            face_score: 0.0,
            face_best_proto_idx: 0,
            face_second_score: 0.0,
            face_prototype_count: 0,
            face_rank: 0,
            face_margin: None,
            body_score: 0.0,
            body_best_proto_idx: 0,
            body_second_score: 0.0,
            body_prototype_count: 0,
            body_rank: 0,
            body_margin: None,
            face_quality: 0.0,
            body_quality: 0.0,
            face_support_count: 0,
            body_support_count: 0,
            face_available: false,
            body_available: false,
            fusion_margin: 0.0,
        }
    }

    /// 计算综合置信度（基于 margin 的 fusion）
    ///
    /// 使用 margin-based fusion 处理 missing channels：
    /// - 如果只有 face：fusion = face_margin
    /// - 如果只有 body：fusion = body_margin
    /// - 如果都有：fusion = 0.5 * face_margin + 0.5 * body_margin
    ///
    /// Phase 36.2 T2: 如果 margin 是 None（candidate_count < 2），视为 0.0
    pub fn fusion_margin(&self, face_weight: f32) -> f32 {
        let face_m = self.face_margin.unwrap_or(0.0);
        let body_m = self.body_margin.unwrap_or(0.0);
        if self.face_available && self.body_available {
            face_weight * face_m + (1.0 - face_weight) * body_m
        } else if self.face_available {
            face_m
        } else if self.body_available {
            body_m
        } else {
            0.0
        }
    }

    /// 计算综合置信度（用于搜索排序）
    ///
    /// Phase 36.3 T7: Gray-zone body channel fusion
    /// - face_score >= 0.70: confidence = face_score (face alone is sufficient)
    /// - 0.55 <= face_score < 0.70: confidence = 0.85 * face_score + 0.15 * body_score (gray zone)
    /// - face_score < 0.55: confidence = face_score (body not enough to help)
    /// - body unavailable: confidence = face_score
    pub fn confidence(&self) -> f32 {
        if !self.face_available {
            return 0.0;
        }

        // Gray zone thresholds
        let high_face_threshold = 0.70;
        let gray_zone_lower = 0.55;

        if self.face_score >= high_face_threshold {
            // Face alone is sufficient for high confidence
            self.face_score
        } else if self.face_score >= gray_zone_lower && self.body_available {
            // Gray zone: body can help boost confidence
            0.85 * self.face_score + 0.15 * self.body_score
        } else {
            // Low face score or body unavailable: face alone
            self.face_score
        }
    }

    /// 证据等级
    pub fn evidence_level(&self) -> EvidenceLevel {
        if self.face_available && self.body_available {
            EvidenceLevel::Dual
        } else if self.face_available {
            EvidenceLevel::FaceOnly
        } else if self.body_available {
            EvidenceLevel::BodyOnly
        } else {
            EvidenceLevel::None
        }
    }

    /// 多证据决策（核心算法）
    ///
    /// 使用 margin-based fusion + missing channel 处理：
    /// - 如果只有 face 可用：fusion = face_margin（只用 face 决策）
    /// - 如果只有 body 可用：fusion = body_margin（只用 body 决策）
    /// - 如果都有：fusion = 0.5 * face_margin + 0.5 * body_margin
    ///
    /// 决策树（保持与原逻辑兼容）：
    /// 1. Face >= 0.75 AND face_margin >= 0.03 → ConfidentMatch
    /// 2. Face >= 0.65 AND face_margin >= 0.0 AND Body >= 0.70 AND body_margin >= 0.0 → ConfidentMatch（仅 DUAL）
    /// 3. Face >= 0.55 AND face_margin >= 0.0 AND Body >= 0.75 AND body_margin >= 0.05 → WeakMatch（仅 DUAL）
    /// 4. Face ambiguous but Body strong → WeakMatch（仅 DUAL when face alone insufficient）
    /// 5. Face strong alone → ConfidentMatch（FACE_ONLY）
    /// 6. Body strong alone → WeakMatch（BODY_ONLY）
    /// 7. else → NewPerson
    ///
    /// Phase 36.2 T2: margin 是 Option<f32>。None 视为 0.0（insufficient competition）
    pub fn decide(&self) -> IdentityDecision {
        // 如果没有任何证据，视为新人
        if !self.face_available && !self.body_available {
            return IdentityDecision::NewPerson;
        }

        // 计算 fusion margin（处理 missing channels）
        let fusion = self.fusion_margin(0.5);

        // Phase 36.2 T2: None margin 视为 0.0
        let face_m = self.face_margin.unwrap_or(0.0);
        let body_m = self.body_margin.unwrap_or(0.0);

        // Rule 1: Face 非常强（单通道也接受）
        if self.face_available && self.face_score >= 0.75 && face_m >= 0.03 {
            return IdentityDecision::ConfidentMatch;
        }

        // Rule 2: DUAL channel - Face 强 + Body 支持
        if self.face_available && self.body_available {
            if self.face_score >= 0.65
                && face_m >= 0.0
                && self.body_score >= 0.70
                && body_m >= 0.0
            {
                return IdentityDecision::ConfidentMatch;
            }

            // Rule 3: Face 中等 + Body 强支持
            if self.face_score >= 0.55
                && face_m >= 0.0
                && self.body_score >= 0.75
                && body_m >= 0.05
            {
                return IdentityDecision::WeakMatch;
            }

            // Rule 4: Face ambiguous but Body provides discrimination
            if face_m.abs() < 0.02 // Face ambiguous
                && body_m >= 0.05   // Body 明确区分
                && self.body_score >= 0.70
            {
                return IdentityDecision::WeakMatch;
            }
        }

        // Rule 5: FACE_ONLY - Face 强但没有 body
        if self.face_available && !self.body_available {
            if self.face_score >= 0.65 && face_m >= 0.03 {
                return IdentityDecision::ConfidentMatch;
            }
            if self.face_score >= 0.55 && face_m >= 0.0 {
                return IdentityDecision::WeakMatch;
            }
        }

        // Rule 6: BODY_ONLY - Body 强但没有 face
        if !self.face_available && self.body_available {
            if self.body_score >= 0.75 && body_m >= 0.05 {
                return IdentityDecision::WeakMatch;
            }
            if self.body_score >= 0.70 && body_m >= 0.0 {
                return IdentityDecision::WeakMatch;
            }
        }

        // 如果两个 margin 都很小但为正 → 歧义
        if face_m < 0.03 && body_m < 0.03 {
            return IdentityDecision::Ambiguous;
        }

        // Face 明确拒绝（margin < 0）→ Body 无法救援，返回 NewPerson
        // 这是关键安全规则：Face 负 margin 意味着 Face 明确拒绝，Body 不能覆盖
        if self.face_available && face_m < 0.0 {
            return IdentityDecision::NewPerson;
        }

        // 如果 fusion margin 为正但没有达到上述条件
        if fusion > 0.0 {
            return IdentityDecision::WeakMatch;
        }

        IdentityDecision::NewPerson
    }

    // =========================================================================
    // Phase 36: Production Identity Decision
    // =========================================================================

    /// Phase 36 决策方法 - 考虑质量、姿态和 anti-chain
    ///
    /// 决策规则：
    /// - EXTREME profile (|yaw| >= 60°) → UNKNOWN (NewPerson)
    /// - LOW quality (composite < 0.30) → UNKNOWN (NewPerson)
    /// Phase 36 Production Fix: 保守决策，Precision > Recall
    ///
    /// 核心原则：
    /// - 只有高度确定是同一个人，才允许自动合并
    /// - WeakMatch 永不自动合并，必须创建新人
    /// - 只有 1 个 candidate 时 margin=None，不能降低标准
    ///
    /// 决策条件（必须同时满足）：
    /// A. score >= 0.78 (Strong Match)
    /// B. candidate_count >= 2
    /// C. margin >= 0.05
    /// D. quality >= 0.45
    /// E. anti_chain == UseTop1
    ///
    /// Phase 36.1: Gray Zone Support Match
    /// 如果 prototype_score 在 0.70-0.78 之间，但有多个已有 face 独立支持（>=0.78），
    /// 则允许 SupportedMatch。
    ///
    /// Phase 36.2 T3/T4: Bootstrap Protection
    /// 如果 candidate 的 person 处于 bootstrap 状态（face_count < 3），使用更严格的阈值。
    pub fn decide_phase36(
        &self,
        yaw: f32,
        anti_chain: &AntiChainingResultV2,
        individual_support: Option<&IndividualSupportEvidence>,
        candidate_face_count: usize,
    ) -> IdentityDecision {
        // 规则 0: NO EVIDENCE → NewPerson
        if !self.face_available && !self.body_available {
            return IdentityDecision::NewPerson;
        }

        // 规则 1: EXTREME PROFILE → NewPerson
        if yaw.abs() >= 60.0 {
            return IdentityDecision::NewPerson;
        }

        // Phase 36: 生产环境严格参数
        const MIN_CLUSTER_QUALITY: f32 = 0.30;      // 低于此值 = Unreliable
        const AUTO_ASSIGN_SCORE: f32 = 0.78;         // 自动合并阈值（提高）
        const WEAK_MATCH_MIN: f32 = 0.60;           // Weak Match 最低分数
        const AUTO_ASSIGN_MARGIN: f32 = 0.05;        // 自动合并 margin 阈值（提高）
        const MIN_CANDIDATE_COUNT: f32 = 2.0;        // 最少候选人数
        const MIN_NEW_PERSON_QUALITY: f32 = 0.45;    // 创建新人的最低质量

        // Phase 36.2 T3/T4: Bootstrap Protection
        let is_bootstrap = candidate_face_count < BOOTSTRAP_FACE_COUNT;
        let bootstrap_threshold = if is_bootstrap {
            BOOTSTRAP_CONFIDENT_THRESHOLD
        } else {
            AUTO_ASSIGN_SCORE
        };
        let bootstrap_quality = if is_bootstrap {
            BOOTSTRAP_QUALITY_THRESHOLD
        } else {
            MIN_NEW_PERSON_QUALITY
        };

        // 规则 2: LOW QUALITY → Unreliable
        if self.face_quality < MIN_CLUSTER_QUALITY {
            return IdentityDecision::Unreliable;
        }

        // 规则 3: 检查候选人数（anti_chain 携带信息）
        // 如果只有 1 个 candidate，严格限制自动合并
        let candidate_count = anti_chain.candidate_count.unwrap_or(1) as f32;
        let has_multiple_candidates = candidate_count >= MIN_CANDIDATE_COUNT;

        // 规则 4: STRONG MATCH (score >= threshold + margin >= 0.05 + candidate_count >= 2)
        // Phase 36: 更严格的条件，必须全部满足
        // Phase 36.2 T2: margin 是 Option<f32>。如果 None（candidate_count < 2），不能自动合并
        // Phase 36.2 T3/T4: Bootstrap persons use higher threshold
        if self.face_score >= bootstrap_threshold {
            // Anti-chain conflict prevents confident match
            if anti_chain.suggested_decision == AntiChainingDecision::Conflict {
                // Fall through to rule 6/7
            } else {
                // margin 必须 >= 0.05（不是 0.03）
                // candidate_count 必须 >= 2
                // quality 必须 >= bootstrap_quality (higher for bootstrap)
                let margin_sufficient = self.face_margin
                    .map(|m| m >= AUTO_ASSIGN_MARGIN)
                    .unwrap_or(false); // None = insufficient competition

                if margin_sufficient
                    && has_multiple_candidates
                    && self.face_quality >= bootstrap_quality
                {
                    return IdentityDecision::ConfidentMatch;
                } else if has_multiple_candidates {
                    // 有多个 candidate 但 margin 不足 → StrongMatchWithCompetitor
                    // （仍分配但记录存在竞争）
                    return IdentityDecision::StrongMatchWithCompetitor;
                }
                // 只有 1 个 candidate：即使高分也降级为 WeakMatch
                // 不能因为"只有一个候选人"就降低标准
            }
        }

        // 规则 5: GRAY ZONE + MULTI-FACE SUPPORT (Phase 36.1)
        // prototype_score 在 0.70-0.78 之间，但有多个已有 face 独立支持
        // Phase 36.2 T3/T4: Bootstrap persons cannot use gray zone support
        if !is_bootstrap && self.face_score >= GRAY_ZONE_MIN_SCORE && self.face_score < bootstrap_threshold {
            if let Some(support) = individual_support {
                // Phase 36.1: 检查是否有足够的独立 face 支持
                // 要求：>= 2 个已有 face 相似度 >= 0.78
                let has_strong_support = support.support_count >= MIN_SUPPORT_COUNT;

                // anti-chain 必须通过（不是歧义）
                let anti_chain_pass = !anti_chain.is_ambiguous && anti_chain.face_gap >= AUTO_ASSIGN_MARGIN;

                // quality 必须 >= 0.40
                let quality_ok = self.face_quality >= MIN_QUALITY;

                if has_strong_support && anti_chain_pass && quality_ok {
                    return IdentityDecision::SupportedMatch;
                }
            }
        }

        // 规则 6: WEAK MATCH (0.60 <= score < 0.78)
        // Phase 36: WeakMatch 永不自动合并！
        // 只记录 suggested_person_id，不自动分配
        // Phase 36.2: Anti-chain CONFLICT takes precedence - return NewPerson for conflicts
        if self.face_score >= WEAK_MATCH_MIN {
            if anti_chain.suggested_decision == AntiChainingDecision::Conflict {
                // Anti-chain conflict: don't even assign to weak match, create new person
                return IdentityDecision::NewPerson;
            }
            // 不再检查 margin 并自动分配
            // 返回 WeakMatch 由调用方决定（调用方应创建新人）
            return IdentityDecision::WeakMatch;
        }

        // 规则 7: LOW SCORE (< 0.60) + HIGH QUALITY → NewPerson
        if self.face_quality >= MIN_NEW_PERSON_QUALITY {
            return IdentityDecision::NewPerson;
        }

        // 规则 8: LOW SCORE + LOW QUALITY → Unreliable
        IdentityDecision::Unreliable
    }
}
#[derive(Debug, Clone)]
pub struct AntiChainingResult {
    pub is_ambiguous: bool,
    pub face_gap: f32,
    pub body_gap: f32,
    pub primary_signal: PrimarySignal,
    /// Number of candidates considered (from AntiChainingResultV2)
    pub candidate_count: Option<usize>,
}

/// 主要信号来源。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrimarySignal {
    Face,
    Body,
    Both,
}

impl AntiChainingResult {
    /// 从候选 evidence 列表检查 anti-chaining。
    ///
    /// 如果 face_gap 和 body_gap 都小于 threshold，则为歧义。
    pub fn check(evidence: &[IdentityEvidence], threshold: f32) -> Self {
        let candidate_count = Some(evidence.len());
        if evidence.len() < 2 {
            return Self {
                is_ambiguous: false,
                face_gap: 0.0,
                body_gap: 0.0,
                primary_signal: PrimarySignal::Both,
                candidate_count,
            };
        }

        // 按 face_score 排序
        let mut by_face = evidence.to_vec();
        by_face.sort_by(|a, b| b.face_score.partial_cmp(&a.face_score).unwrap());
        let face_gap = by_face[0].face_score - by_face[1].face_score;

        // 按 body_score 排序
        let mut by_body = evidence.to_vec();
        by_body.sort_by(|a, b| b.body_score.partial_cmp(&a.body_score).unwrap());
        let body_gap = by_body[0].body_score - by_body[1].body_score;

        let is_ambiguous = face_gap < threshold && body_gap < threshold;

        let primary_signal = if face_gap > body_gap {
            PrimarySignal::Face
        } else if body_gap > face_gap {
            PrimarySignal::Body
        } else {
            PrimarySignal::Both
        };

        Self {
            is_ambiguous,
            face_gap,
            body_gap,
            primary_signal,
            candidate_count,
        }
    }
}

impl From<AntiChainingResultV2> for AntiChainingResult {
    fn from(v2: AntiChainingResultV2) -> Self {
        Self {
            is_ambiguous: v2.is_ambiguous,
            face_gap: v2.face_gap,
            body_gap: v2.body_gap,
            primary_signal: v2.primary_signal,
            candidate_count: v2.candidate_count,
        }
    }
}

/// 阈值配置。
#[derive(Debug, Clone, Copy)]
pub struct DecisionThresholds {
    /// Face 强匹配阈值
    pub face_strong: f32,
    /// Face 中等阈值
    pub face_medium: f32,
    /// Face weak 阈值（用于辅助判断）
    pub face_weak: f32,
    /// Body 阈值
    pub body_threshold: f32,
    /// Body 强阈值
    pub body_strong: f32,
    /// Margin 阈值
    pub margin_threshold: f32,
    /// Anti-chaining threshold
    pub anti_chain_threshold: f32,
}

impl Default for DecisionThresholds {
    fn default() -> Self {
        Self {
            face_strong: 0.75,
            face_medium: 0.65,
            face_weak: 0.55,
            body_threshold: 0.70,
            body_strong: 0.75,
            margin_threshold: 0.03,
            anti_chain_threshold: 0.05,
        }
    }
}

/// 生产准入条件。
#[derive(Debug, Clone, Copy)]
pub struct ProductionRequirements {
    /// Top-1 准确率要求
    pub min_top1: f32,
    /// FAR 上限
    pub max_far: f32,
    /// FRR 上限
    pub max_frr: f32,
    /// LOO pass rate 下限
    pub min_loo_pass_rate: f32,
    /// 最小 margin（positive 必须 > negative）
    pub min_margin: f32,
}

impl Default for ProductionRequirements {
    fn default() -> Self {
        Self {
            min_top1: 0.95,
            max_far: 0.01,   // FAR <= 1%
            max_frr: 0.10,    // FRR <= 10%
            min_loo_pass_rate: 0.90,
            min_margin: 0.0,  // positive margin 必须 > 0
        }
    }
}

impl ProductionRequirements {
    /// 检查 benchmark 结果是否满足生产准入条件。
    pub fn check(&self, result: &BenchmarkMetrics) -> ProductionCheckResult {
        let top1_ok = result.top1 >= self.min_top1;
        let far_ok = result.far <= self.max_far;
        let frr_ok = result.frr <= self.max_frr;
        let loo_ok = result.loo_pass_rate >= self.min_loo_pass_rate;
        let margin_ok = result.margin > self.min_margin;

        let is_ready = top1_ok && far_ok && frr_ok && loo_ok && margin_ok;

        ProductionCheckResult {
            is_ready,
            top1_ok,
            far_ok,
            frr_ok,
            loo_ok,
            margin_ok,
            details: format!(
                "top1={:.2}%({:.2}%) far={:.2}%({:.2}%) frr={:.2}%({:.2}%) loo={:.2}%({:.2}%) margin={:.3}(>{:.3})",
                result.top1 * 100.0,
                self.min_top1 * 100.0,
                result.far * 100.0,
                self.max_far * 100.0,
                result.frr * 100.0,
                self.max_frr * 100.0,
                result.loo_pass_rate * 100.0,
                self.min_loo_pass_rate * 100.0,
                result.margin,
                self.min_margin,
            ),
        }
    }
}

/// Benchmark 指标（用于生产准入检查）。
#[derive(Debug, Clone)]
pub struct BenchmarkMetrics {
    pub top1: f32,
    pub far: f32,
    pub frr: f32,
    pub loo_pass_rate: f32,
    pub margin: f32,
}

/// 生产准入检查结果。
#[derive(Debug, Clone)]
pub struct ProductionCheckResult {
    pub is_ready: bool,
    pub top1_ok: bool,
    pub far_ok: bool,
    pub frr_ok: bool,
    pub loo_ok: bool,
    pub margin_ok: bool,
    pub details: String,
}

/// 身份搜索结果（包含决策）。
#[derive(Debug, Clone)]
pub struct IdentitySearchResult {
    pub person_id: i64,
    pub decision: IdentityDecision,
    pub confidence: f32,
    pub face_score: f32,
    pub body_score: f32,
    /// Face margin: top1 - top2. None if candidate_count < 2.
    pub face_margin: Option<f32>,
    /// Body margin: top1 - top2. None if candidate_count < 2.
    pub body_margin: Option<f32>,
    pub face_rank: usize,
    pub body_rank: usize,
    pub is_ambiguous: bool,
}

// ============================================================================
// T4: Evidence v2 — Quality-aware evidence
// ============================================================================

/// IdentityEvidence v2 — 包含详细 FaceQuality 和 BodyQuality（Phase 29 T4）
#[derive(Debug, Clone)]
pub struct IdentityEvidenceV2 {
    pub person_id: i64,

    // Face evidence
    pub face_score: f32,
    pub face_second_score: f32,
    pub face_rank: usize,
    pub face_margin: f32,
    pub face_prototype_count: usize,
    pub face_support_count: usize,
    pub face_available: bool,

    // Face quality (detailed, Phase 29 T2)
    pub face_quality: FaceQuality,

    // Body evidence
    pub body_score: f32,
    pub body_second_score: f32,
    pub body_rank: usize,
    pub body_margin: f32,
    pub body_prototype_count: usize,
    pub body_support_count: usize,
    pub body_available: bool,

    // Body quality (simplified for now)
    pub body_quality: f32,

    // Fusion
    pub fusion_margin: f32,
}

impl IdentityEvidenceV2 {
    /// 计算质量加权的 confidence
    /// 质量高的 face 贡献更大的 confidence
    pub fn quality_weighted_confidence(&self) -> f32 {
        let face_w = 0.7;
        let body_w = 0.3;

        let face_quality_factor = if self.face_available {
            self.face_quality.composite_quality().max(0.3)
        } else {
            0.0
        };

        let body_quality_factor = if self.body_available {
            self.body_quality.max(0.3)
        } else {
            0.0
        };

        let base_confidence = face_w * self.face_score + body_w * self.body_score;
        let quality_boost = 0.2 * (face_quality_factor + body_quality_factor) / 2.0;

        (base_confidence + quality_boost).min(1.0)
    }
}

impl Default for IdentityEvidenceV2 {
    fn default() -> Self {
        Self {
            person_id: 0,
            face_score: 0.0,
            face_second_score: 0.0,
            face_rank: 0,
            face_margin: 0.0,
            face_prototype_count: 0,
            face_support_count: 0,
            face_available: false,
            face_quality: FaceQuality::default(),
            body_score: 0.0,
            body_second_score: 0.0,
            body_rank: 0,
            body_margin: 0.0,
            body_prototype_count: 0,
            body_support_count: 0,
            body_available: false,
            body_quality: 0.0,
            fusion_margin: 0.0,
        }
    }
}

// ============================================================================
// T5: Threshold Separation (Phase 29 T5)
// ============================================================================

/// 身份判断阈值集合 — 不同场景使用不同阈值（Phase 29 T5）
#[derive(Debug, Clone, Copy)]
pub struct IdentityThresholdsV2 {
    // PAIRWISE: Face↔Face direct verification
    pub pairwise_face_threshold: f32,
    pub pairwise_body_threshold: f32,

    // PROTOTYPE: Query↔PersonPrototype
    pub prototype_face_threshold: f32,
    pub prototype_body_threshold: f32,

    // AUTO_ASSIGN: 用于自动分配决策
    pub auto_assign_face_threshold: f32,
    pub auto_assign_body_threshold: f32,

    // REVIEW: 进入人工审核的阈值
    pub review_face_threshold: f32,

    // CLUSTER_MERGE: 聚类合并时的阈值（最严格）
    pub cluster_merge_face_threshold: f32,
    pub cluster_merge_body_threshold: f32,

    // Margin thresholds
    pub face_margin_threshold: f32,
    pub body_margin_threshold: f32,
    pub ambiguity_margin_threshold: f32,
}

impl Default for IdentityThresholdsV2 {
    fn default() -> Self {
        Self {
            // PAIRWISE: 直接验证，用较高阈值
            pairwise_face_threshold: 0.75,
            pairwise_body_threshold: 0.70,

            // PROTOTYPE: Prototype 匹配，用校准后的值
            prototype_face_threshold: 0.55,
            prototype_body_threshold: 0.60,

            // AUTO_ASSIGN: 自动分配，比 prototype 略宽松
            auto_assign_face_threshold: 0.55,
            auto_assign_body_threshold: 0.60,

            // REVIEW: 进入审核的阈值
            review_face_threshold: 0.50,

            // CLUSTER_MERGE: 最严格，防止错误合并
            cluster_merge_face_threshold: 0.75,
            cluster_merge_body_threshold: 0.70,

            // Margins
            face_margin_threshold: 0.03,
            body_margin_threshold: 0.03,
            ambiguity_margin_threshold: 0.02,
        }
    }
}

impl IdentityThresholdsV2 {
    /// 获取对应场景的阈值
    pub fn for_mode(&self, mode: IdentitySearchMode) -> (f32, f32, f32, f32) {
        match mode {
            IdentitySearchMode::Pairwise => (
                self.pairwise_face_threshold,
                self.pairwise_body_threshold,
                self.face_margin_threshold,
                self.body_margin_threshold,
            ),
            IdentitySearchMode::Prototype => (
                self.prototype_face_threshold,
                self.prototype_body_threshold,
                self.face_margin_threshold,
                self.body_margin_threshold,
            ),
            IdentitySearchMode::AutoAssign => (
                self.auto_assign_face_threshold,
                self.auto_assign_body_threshold,
                self.face_margin_threshold,
                self.body_margin_threshold,
            ),
            IdentitySearchMode::Review => (
                self.review_face_threshold,
                self.prototype_body_threshold,
                self.face_margin_threshold,
                self.body_margin_threshold,
            ),
            IdentitySearchMode::ClusterMerge => (
                self.cluster_merge_face_threshold,
                self.cluster_merge_body_threshold,
                self.face_margin_threshold,
                self.body_margin_threshold,
            ),
        }
    }
}

/// 身份搜索模式（Phase 29 T5）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentitySearchMode {
    /// Face↔Face direct verification
    Pairwise,
    /// Query↔PersonPrototype
    Prototype,
    /// Auto assignment decision
    AutoAssign,
    /// Human review
    Review,
    /// Cluster merge (strictest)
    ClusterMerge,
}

// ============================================================================
// T6: Decision Policy (Phase 29 T6)
// ============================================================================

/// 身份决策策略（Phase 29 T6）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionPolicy {
    /// 自动决策（高置信直接分配，低置信进入审核）
    Auto,
    /// 所有决策都进入人工审核
    Review,
    /// 不自动分配，只记录 pipeline 决策供分析
    Unknown,
}

impl Default for DecisionPolicy {
    fn default() -> Self {
        DecisionPolicy::Auto
    }
}

impl DecisionPolicy {
    /// 根据 evidence 和策略决定是否自动分配
    pub fn should_auto_assign(&self, evidence: &IdentityEvidence) -> bool {
        match self {
            DecisionPolicy::Auto => {
                let decision = evidence.decide();
                decision == IdentityDecision::ConfidentMatch
            }
            DecisionPolicy::Review | DecisionPolicy::Unknown => false,
        }
    }

    /// 根据 evidence 和策略决定是否进入审核队列
    pub fn should_review(&self, evidence: &IdentityEvidence) -> bool {
        match self {
            DecisionPolicy::Auto => {
                let decision = evidence.decide();
                decision == IdentityDecision::WeakMatch || decision == IdentityDecision::Ambiguous
            }
            DecisionPolicy::Review => {
                let decision = evidence.decide();
                decision != IdentityDecision::NewPerson
            }
            DecisionPolicy::Unknown => false,
        }
    }
}

// ============================================================================
// T7: Multi-Prototype Preparation (Phase 29 T7)
// ============================================================================

/// 单个人的 Face Prototype 集合（支持多个 prototype，Phase 29 T7）
#[derive(Debug, Clone)]
pub struct PersonFacePrototypeSet {
    pub person_id: i64,
    /// 多个 prototypes（支持 crop_type bucketing）
    pub prototypes: Vec<FacePrototype>,
    /// 最后更新时间
    pub updated_at: u64,
}

impl PersonFacePrototypeSet {
    pub fn new(person_id: i64) -> Self {
        Self {
            person_id,
            prototypes: Vec::new(),
            updated_at: current_unix_time(),
        }
    }

    pub fn add_prototype(&mut self, proto: FacePrototype) {
        self.updated_at = current_unix_time();
        self.prototypes.push(proto);
    }

    /// 获取最佳匹配的 prototype
    pub fn best_match(&self, embedding: &[f32]) -> Option<(usize, f32)> {
        if self.prototypes.is_empty() {
            return None;
        }
        let mut best_idx = 0;
        let mut best_score = cosine_similarity(embedding, &self.prototypes[0].embedding);
        for (i, proto) in self.prototypes.iter().enumerate().skip(1) {
            let score = cosine_similarity(embedding, &proto.embedding);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        Some((best_idx, best_score))
    }

    /// 计算与另一个 prototype set 的相似度
    pub fn similarity_to(&self, other: &PersonFacePrototypeSet) -> f32 {
        let mut max_sim = 0.0f32;
        for a in &self.prototypes {
            for b in &other.prototypes {
                let sim = cosine_similarity(&a.embedding, &b.embedding);
                if sim > max_sim {
                    max_sim = sim;
                }
            }
        }
        max_sim
    }
}

/// Face Prototype（单个 prototype）
#[derive(Debug, Clone)]
pub struct FacePrototype {
    /// L2-normalized embedding vector
    pub embedding: Vec<f32>,
    /// 创建该 prototype 的 face 质量
    pub quality: FaceQuality,
    /// 该 prototype 关联的 face 数量
    pub face_count: usize,
    /// 该 prototype 的 crop_type（0=unknown, 1=face, 2=upper_body, 3=full_body）
    pub crop_type: u8,
}

impl FacePrototype {
    pub fn new(embedding: Vec<f32>, quality: FaceQuality, crop_type: u8) -> Self {
        Self {
            embedding,
            quality,
            face_count: 1,
            crop_type,
        }
    }
}

/// 单个人的 Body Prototype 集合（Phase 29 T7）
#[derive(Debug, Clone)]
pub struct PersonBodyPrototypeSet {
    pub person_id: i64,
    pub prototypes: Vec<BodyPrototype>,
    pub updated_at: u64,
}

impl PersonBodyPrototypeSet {
    pub fn new(person_id: i64) -> Self {
        Self {
            person_id,
            prototypes: Vec::new(),
            updated_at: current_unix_time(),
        }
    }

    pub fn add_prototype(&mut self, proto: BodyPrototype) {
        self.updated_at = current_unix_time();
        self.prototypes.push(proto);
    }

    pub fn best_match(&self, embedding: &[f32]) -> Option<(usize, f32)> {
        if self.prototypes.is_empty() {
            return None;
        }
        let mut best_idx = 0;
        let mut best_score = cosine_similarity(embedding, &self.prototypes[0].embedding);
        for (i, proto) in self.prototypes.iter().enumerate().skip(1) {
            let score = cosine_similarity(embedding, &proto.embedding);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        Some((best_idx, best_score))
    }
}

/// Body Prototype
#[derive(Debug, Clone)]
pub struct BodyPrototype {
    pub embedding: Vec<f32>,
    pub quality: f32,
    pub body_count: usize,
    pub crop_type: u8,
}

impl BodyPrototype {
    pub fn new(embedding: Vec<f32>, quality: f32, crop_type: u8) -> Self {
        Self {
            embedding,
            quality,
            body_count: 1,
            crop_type,
        }
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ============================================================================
// T8: Anti-Chaining Strengthening (Phase 29 T8)
// ============================================================================

/// Anti-Chaining 结果 v2 — 更详细的歧义分析（Phase 29 T8, Phase 36 Fix）
#[derive(Debug, Clone)]
pub struct AntiChainingResultV2 {
    pub is_ambiguous: bool,
    pub face_gap: f32,
    pub body_gap: f32,
    pub face_gap_sufficient: bool,
    pub body_gap_sufficient: bool,
    pub primary_signal: PrimarySignal,
    /// 建议的决策
    pub suggested_decision: AntiChainingDecision,
    /// Phase 36: 候选人数（用于判断 margin 是否有效）
    pub candidate_count: Option<usize>,
}

/// Anti-Chaining 决策建议
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntiChainingDecision {
    /// Top-1 足够清晰，可以使用
    UseTop1,
    /// 需要进入审核
    Review,
    /// 歧义太严重，需要更多信息
    Conflict,
}

impl AntiChainingResultV2 {
    /// 从候选 evidence 列表检查 anti-chaining v2
    pub fn check(evidence: &[IdentityEvidence], thresholds: &IdentityThresholdsV2) -> Self {
        // Phase 36: 记录 candidate_count
        let candidate_count = Some(evidence.len());

        if evidence.len() < 2 {
            return Self {
                is_ambiguous: false,
                face_gap: 0.0,
                body_gap: 0.0,
                face_gap_sufficient: true,
                body_gap_sufficient: true,
                primary_signal: PrimarySignal::Both,
                suggested_decision: AntiChainingDecision::UseTop1,
                candidate_count,
            };
        }

        // 按 face_score 排序
        let mut by_face = evidence.to_vec();
        by_face.sort_by(|a, b| b.face_score.partial_cmp(&a.face_score).unwrap());
        let face_gap = by_face[0].face_score - by_face[1].face_score;

        // 按 body_score 排序
        let mut by_body = evidence.to_vec();
        by_body.sort_by(|a, b| b.body_score.partial_cmp(&a.body_score).unwrap());
        let body_gap = by_body[0].body_score - by_body[1].body_score;

        let face_gap_sufficient = face_gap >= thresholds.face_margin_threshold;
        let body_gap_sufficient = body_gap >= thresholds.body_margin_threshold;

        // 歧义判定：两个 gap 都不足
        let is_ambiguous = !face_gap_sufficient && !body_gap_sufficient;

        let primary_signal = if face_gap > body_gap {
            PrimarySignal::Face
        } else if body_gap > face_gap {
            PrimarySignal::Body
        } else {
            PrimarySignal::Both
        };

        // 决策建议
        let suggested_decision = if face_gap_sufficient || body_gap_sufficient {
            AntiChainingDecision::UseTop1
        } else if face_gap >= thresholds.ambiguity_margin_threshold
                  || body_gap >= thresholds.ambiguity_margin_threshold {
            AntiChainingDecision::Review
        } else {
            AntiChainingDecision::Conflict
        };

        Self {
            is_ambiguous,
            face_gap,
            body_gap,
            face_gap_sufficient,
            body_gap_sufficient,
            primary_signal,
            suggested_decision,
            candidate_count,
        }
    }
}

// ============================================================================
// T9: Prototype Pollution Protection (Phase 29 T9)
// ============================================================================

/// Prototype Pollution 保护配置（Phase 29 T9）
#[derive(Debug, Clone, Copy)]
pub struct PollutionProtectionConfig {
    /// 单个 prototype 允许的最大 face 数量
    pub max_face_count_per_proto: usize,
    /// Prototype 允许的最小质量分数
    pub min_proto_quality: f32,
    /// 与该 prototype 相似度超过此值的 embedding 不能加入
    pub similarity_reject_threshold: f32,
    /// 检测到污染后，是否自动触发 rebuild
    pub auto_rebuild: bool,
}

impl Default for PollutionProtectionConfig {
    fn default() -> Self {
        Self {
            max_face_count_per_proto: 1000,
            min_proto_quality: 0.20,
            similarity_reject_threshold: 0.85,
            auto_rebuild: true,
        }
    }
}

/// Prototype Pollution 检测结果
#[derive(Debug, Clone)]
pub struct PollutionCheckResult {
    pub is_polluted: bool,
    pub polluted_prototype_indices: Vec<usize>,
    pub avg_quality_too_low: bool,
    pub has_duplicate_faces: bool,
    pub reason: &'static str,
}

/// 检查 prototype 是否被污染
pub struct PollutionChecker {
    config: PollutionProtectionConfig,
}

impl PollutionChecker {
    pub fn new(config: PollutionProtectionConfig) -> Self {
        Self { config }
    }

    pub fn from_default() -> Self {
        Self::new(PollutionProtectionConfig::default())
    }

    /// 检查 prototype 集合是否被污染
    pub fn check(&self, proto_set: &PersonFacePrototypeSet) -> PollutionCheckResult {
        // 检查 face count 膨胀
        let polluted_indices: Vec<usize> = proto_set
            .prototypes
            .iter()
            .enumerate()
            .filter(|(_, p)| p.face_count > self.config.max_face_count_per_proto)
            .map(|(i, _)| i)
            .collect();

        if !polluted_indices.is_empty() {
            return PollutionCheckResult {
                is_polluted: true,
                polluted_prototype_indices: polluted_indices,
                avg_quality_too_low: false,
                has_duplicate_faces: false,
                reason: "face_count_exceeded",
            };
        }

        // 检查平均质量过低
        if !proto_set.prototypes.is_empty() {
            let avg_quality: f32 = proto_set
                .prototypes
                .iter()
                .map(|p| p.quality.composite_quality())
                .sum::<f32>()
                / proto_set.prototypes.len() as f32;

            if avg_quality < self.config.min_proto_quality {
                return PollutionCheckResult {
                    is_polluted: true,
                    polluted_prototype_indices: Vec::new(),
                    avg_quality_too_low: true,
                    has_duplicate_faces: false,
                    reason: "avg_quality_too_low",
                };
            }
        }

        PollutionCheckResult {
            is_polluted: false,
            polluted_prototype_indices: Vec::new(),
            avg_quality_too_low: false,
            has_duplicate_faces: false,
            reason: "",
        }
    }

    /// 检查新加入的 embedding 是否可能导致污染
    pub fn check_new_embedding(&self, proto_set: &PersonFacePrototypeSet, embedding: &[f32]) -> bool {
        if let Some((_, similarity)) = proto_set.best_match(embedding) {
            if similarity > self.config.similarity_reject_threshold {
                return true; // 太相似，可能导致污染
            }
        }
        false
    }
}

// ============================================================================
// T10: Shadow Mode (Phase 29 T10)
// ============================================================================

/// Shadow Mode 配置（Phase 29 T10）
#[derive(Debug, Clone)]
pub struct ShadowModeConfig {
    /// 是否启用 shadow mode
    pub enabled: bool,
    /// 记录所有决策（不只是需要审核的）
    pub log_all_decisions: bool,
    /// 影子 pipeline 的版本
    pub shadow_pipeline_version: IdentityPipelineVersion,
}

impl Default for ShadowModeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            log_all_decisions: false,
            shadow_pipeline_version: IdentityPipelineVersion::current(),
        }
    }
}

/// Shadow Mode 记录条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowModeRecord {
    /// 记录时间
    pub timestamp: u64,
    /// 图片 ID
    pub image_id: i64,
    /// 影子 pipeline 决策
    pub shadow_decision: IdentityDecision,
    /// 生产 pipeline 决策（如果有）
    pub production_decision: Option<IdentityDecision>,
    /// 影子 pipeline 的置信度
    pub shadow_confidence: f32,
    /// 生产 pipeline 的置信度（如果有）
    pub production_confidence: Option<f32>,
    /// 影子决策与生产决策是否一致
    pub disagreement: bool,
}

/// Shadow Mode 状态
#[derive(Debug, Clone, Default)]
pub struct ShadowModeState {
    pub records: Vec<ShadowModeRecord>,
    pub total_decisions: usize,
    pub disagreements: usize,
}

impl ShadowModeState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, record: ShadowModeRecord) {
        self.total_decisions += 1;
        if record.disagreement {
            self.disagreements += 1;
        }
        self.records.push(record);
    }

    pub fn disagreement_rate(&self) -> f32 {
        if self.total_decisions == 0 {
            return 0.0;
        }
        self.disagreements as f32 / self.total_decisions as f32
    }
}

// ============================================================================
// T11: Production Metrics (Phase 29 T11)
// ============================================================================

/// 身份系统生产指标（Phase 29 T11）
#[derive(Debug, Clone, Default)]
pub struct ProductionMetrics {
    /// 自动分配Precision (TP / (TP + FP))
    pub auto_assign_precision: f32,
    /// 自动分配Recall (TP / (TP + FN))
    pub auto_assign_recall: f32,
    /// 审核率 (review_count / total_assignments)
    pub review_rate: f32,
    /// 审核后准确率 (review_correct / review_total)
    pub review_precision: f32,
    /// 冲突率 (conflict_count / total)
    pub conflict_rate: f32,
    /// Shadow mode 不一致率
    pub shadow_disagreement_rate: f32,
    /// 当前时间
    pub timestamp: u64,
}

impl ProductionMetrics {
    pub fn is_healthy(&self) -> bool {
        // 自动分配precision >= 95%
        // 审核率 <= 20%
        // 冲突率 <= 5%
        self.auto_assign_precision >= 0.95
            && self.review_rate <= 0.20
            && self.conflict_rate <= 0.05
    }

    pub fn summary(&self) -> String {
        format!(
            "auto_precision={:.1}% review_rate={:.1}% conflict_rate={:.1}% shadow_disagree={:.1}%",
            self.auto_assign_precision * 100.0,
            self.review_rate * 100.0,
            self.conflict_rate * 100.0,
            self.shadow_disagreement_rate * 100.0,
        )
    }
}

// ============================================================================
// T13: Benchmark Efficiency (Phase 29 T13)
// ============================================================================

/// Benchmark 效率要求（Phase 29 T13）
#[derive(Debug, Clone, Copy)]
pub struct BenchmarkEfficiencyReq {
    /// 最大 embedding 提取时间（ms per image）
    pub max_embedding_ms: f32,
    /// 最大搜索时间（ms per query）
    pub max_search_ms: f32,
    /// 必须使用缓存的 embeddings
    pub must_use_cache: bool,
}

impl Default for BenchmarkEfficiencyReq {
    fn default() -> Self {
        Self {
            max_embedding_ms: 100.0,
            max_search_ms: 50.0,
            must_use_cache: true,
        }
    }
}

// ============================================================================
// T14: Production Gate (Phase 29 T14)
// ============================================================================

/// Production Gate 判定结果（Phase 29 T14）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductionGate {
    /// 可以投入生产
    ProductionReady,
    /// 可以小批量试运行
    ProductionBeta,
    /// 仍在实验阶段，不建议生产使用
    Experimental,
}

impl ProductionGate {
    pub fn from_metrics(
        metrics: &ProductionMetrics,
        benchmark: &BenchmarkMetrics,
        requirements: &ProductionRequirements,
    ) -> Self {
        let metrics_ok = metrics.is_healthy();
        let benchmark_ok = requirements.check(benchmark).is_ready;

        if metrics_ok && benchmark_ok {
            ProductionGate::ProductionReady
        } else if benchmark_ok {
            // Benchmark OK but metrics not healthy yet
            ProductionGate::ProductionBeta
        } else {
            ProductionGate::Experimental
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ProductionGate::ProductionReady => "PRODUCTION_READY",
            ProductionGate::ProductionBeta => "PRODUCTION_BETA",
            ProductionGate::Experimental => "EXPERIMENTAL",
        }
    }
}

// ============================================================================
// Phase 30: Identity Pipeline Adapter (Unified Entry Point)
// ============================================================================

/// Identity Query — 输入到 IdentityPipeline::evaluate()
#[derive(Debug, Clone)]
pub struct IdentityQuery {
    /// Face ID（来自数据库）
    pub face_id: i64,
    /// Image ID
    pub image_id: i64,
    /// Face embedding (512-d ArcFace, L2 normalized)
    pub face_embedding: Vec<f32>,
    /// Face quality metrics
    pub face_quality: FaceQuality,
    /// Body embedding (768-d YouTu, optional)
    pub body_embedding: Option<Vec<f32>>,
    /// Body quality (optional)
    pub body_quality: f32,
    /// Whether body is available
    pub body_available: bool,
}

/// Candidate result with scores
#[derive(Debug, Clone)]
pub struct CandidateResult {
    pub person_id: i64,
    pub face_score: f32,
    pub face_rank: usize,
    pub body_score: f32,
    pub body_rank: usize,
    /// Face count of this person (for bootstrap detection, Phase 36.2 T3/T4)
    pub face_count: usize,
}

/// Identity Decision Result — 输出
#[derive(Debug, Clone)]
pub struct IdentityDecisionResult {
    /// Production candidate (legacy pipeline result)
    pub production_candidate: Option<CandidateResult>,
    /// Shadow candidate (new pipeline result)
    pub shadow_candidate: Option<CandidateResult>,
    /// Face evidence for shadow
    pub face_evidence: Option<FaceEvidence>,
    /// Body evidence for shadow
    pub body_evidence: Option<BodyEvidence>,
    /// Fusion margin
    pub fusion_margin: f32,
    /// Shadow decision
    pub shadow_decision: IdentityDecision,
    /// Shadow confidence
    pub shadow_confidence: f32,
    /// Anti-chaining result
    pub anti_chain: Option<AntiChainingResult>,
    /// Pollution check result
    pub pollution_result: Option<PollutionCheckResult>,
    /// Pipeline version used
    pub pipeline_version: IdentityPipelineVersion,
    /// Quality gate result
    pub quality_gate_passed: bool,
    /// Quality gate failure reason (if any)
    pub quality_gate_failure: Option<&'static str>,
}

/// Face evidence for identity
#[derive(Debug, Clone)]
pub struct FaceEvidence {
    pub score: f32,
    pub second_score: f32,
    /// Margin: top1 - top2 score. None if candidate_count < 2 (insufficient competition).
    pub margin: Option<f32>,
    pub rank: usize,
    pub quality: FaceQuality,
    pub evidence_strength: EvidenceStrength,
}

/// Body evidence for identity
#[derive(Debug, Clone)]
pub struct BodyEvidence {
    pub score: f32,
    pub second_score: f32,
    /// Margin: top1 - top2 score. None if candidate_count < 2 (insufficient competition).
    pub margin: Option<f32>,
    pub rank: usize,
    pub quality: f32,
    pub evidence_strength: EvidenceStrength,
}

/// Phase 36.1: Individual face support evidence for gray-zone matching
///
/// Computes cosine similarity between query embedding and individual faces in a person,
/// not just the prototype. This helps detect when a query face matches multiple
/// distinct faces within the same person, providing stronger evidence than prototype alone.
#[derive(Debug, Clone)]
pub struct IndividualSupportEvidence {
    /// Top-1 individual face similarity
    pub top1_score: f32,
    /// Top-2 individual face similarity (None if person has only 1 face)
    pub top2_score: Option<f32>,
    /// Number of faces in person with similarity >= SUPPORT_SCORE threshold
    pub support_count: usize,
    /// All computed individual scores
    pub scores: Vec<f32>,
    /// Face IDs of supporting faces (for debugging)
    pub supporting_face_ids: Vec<i64>,
}

/// Constants for Individual Support matching
mod support_constants {
    /// Minimum similarity to count as a supporting face
    pub const SUPPORT_SCORE: f32 = 0.78;
    /// Minimum number of supporting faces required
    pub const MIN_SUPPORT_COUNT: usize = 2;
    /// Minimum support margin between top candidate and second-best person
    pub const SUPPORT_MARGIN_THRESHOLD: f32 = 0.02;
    /// Gray zone minimum prototype score (0.70-0.78 is gray zone)
    pub const GRAY_ZONE_MIN_SCORE: f32 = 0.70;
    /// Strict confident match threshold
    pub const AUTO_ASSIGN_SCORE: f32 = 0.78;
    /// Minimum quality for supported match
    pub const MIN_QUALITY: f32 = 0.40;
    /// Maximum number of representative faces to evaluate per person
    pub const MAX_SUPPORT_FACES: usize = 5;
}

/// Phase 36.2 T3/T4: Bootstrap Protection Constants
///
/// When a Person has only 1 face (early-stage / bootstrap state),
/// its prototype is very unstable and errors can directly pollute the Person.
/// We require stricter evidence for auto-assignment to bootstrap persons.
mod bootstrap_constants {
    /// Face count threshold below which a Person is considered "bootstrap"
    pub const BOOTSTRAP_FACE_COUNT: usize = 3;
    /// Minimum face count to transition from Bootstrap to Stable
    pub const STABLE_FACE_COUNT: usize = 3;
    /// Bootstrap person: minimum score for ConfidentMatch (higher than normal 0.78)
    pub const BOOTSTRAP_CONFIDENT_THRESHOLD: f32 = 0.88;
    /// Bootstrap person: minimum quality for ConfidentMatch
    pub const BOOTSTRAP_QUALITY_THRESHOLD: f32 = 0.50;
}

pub use support_constants::*;
pub use bootstrap_constants::*;

/// Evidence strength classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceStrength {
    /// High quality, strong evidence
    Strong,
    /// Medium quality, usable but weak evidence
    Weak,
    /// Too low quality, not usable for identity decision
    Rejected,
}

impl EvidenceStrength {
    pub fn from_quality(quality: f32) -> Self {
        if quality >= 0.5 {
            EvidenceStrength::Strong
        } else if quality >= 0.25 {
            EvidenceStrength::Weak
        } else {
            EvidenceStrength::Rejected
        }
    }
}

/// Identity Pipeline — 统一入口（Phase 30 T2, Phase 32 修复）
///
/// Phase 32 修复：
/// - retrieve_face_candidates 现在正确映射 vector_id → person_id
/// - compute_face_evidence 使用 prototype-based scoring 而非 raw HNSW cosine
/// - 自匹配过滤（query face 不会作为 candidate 返回）
pub struct IdentityPipeline {
    face_index: Arc<dyn VectorIndex>,
    body_index: Option<Arc<dyn VectorIndex>>,
    db: Arc<pf_database::Database>,
    prototype_service: Option<Arc<PrototypeService>>,
    quality_gate: FaceQualityGate,
    thresholds_v2: IdentityThresholdsV2,
    pollution_checker: PollutionChecker,
    decision_policy: DecisionPolicy,
}

impl IdentityPipeline {
    pub fn new(
        face_index: Arc<dyn VectorIndex>,
        body_index: Option<Arc<dyn VectorIndex>>,
        db: Arc<pf_database::Database>,
        prototype_service: Option<Arc<PrototypeService>>,
    ) -> Self {
        Self {
            face_index,
            body_index,
            db,
            prototype_service,
            quality_gate: FaceQualityGate::from_default(),
            thresholds_v2: IdentityThresholdsV2::default(),
            pollution_checker: PollutionChecker::from_default(),
            decision_policy: DecisionPolicy::default(),
        }
    }

    /// 评估单个 identity query
    ///
    /// 1. Quality gate check
    /// 2. Face candidate retrieval
    /// 3. Body candidate retrieval (if available)
    /// 4. Compute evidence
    /// 5. Anti-chaining check
    /// 6. Decision
    pub fn evaluate(&self, query: &IdentityQuery) -> IdentityDecisionResult {
        let pipeline_version = IdentityPipelineVersion::current();

        // 1. Quality Gate
        let quality_gate_passed = self.quality_gate.passes(query.face_quality);
        let quality_gate_failure = if quality_gate_passed {
            None
        } else {
            self.quality_gate.failure_reason(query.face_quality)
        };

        // 2. Face candidate retrieval (Phase 32: now with self-match filtering)
        let face_candidates = self.retrieve_face_candidates(
            &query.face_embedding,
            10,
            query.face_id,
            query.image_id,
        );

        // 3. Body candidate retrieval (if available)
        let body_candidates = if query.body_available {
            self.retrieve_body_candidates(&query.body_embedding.as_ref().unwrap(), 10)
        } else {
            Vec::new()
        };

        // 4. Compute evidence for each candidate (Phase 32: now uses prototype scoring)
        // Phase 2 LOO Fix: 传递 query.image_id 用于 LOO exclusion
        let mut all_evidence = Vec::new();
        for face_cand in &face_candidates {
            let face_evidence = self.compute_face_evidence(
                &query.face_embedding,
                face_cand,
                &face_candidates,
                query.face_quality,
                Some(query.image_id),
            );

            let body_evidence = if query.body_available {
                self.compute_body_evidence(&query.body_embedding.as_ref().unwrap(), face_cand.person_id)
            } else {
                None
            };

            all_evidence.push((face_cand.clone(), face_evidence, body_evidence));
        }

        // 5. Anti-chaining check
        let evidence_list: Vec<IdentityEvidence> = all_evidence
            .iter()
            .map(|(c, fe, be)| {
                let mut e = IdentityEvidence::new(c.person_id);
                e.face_available = true;
                e.face_score = fe.score;
                e.face_margin = fe.margin;
                e.face_rank = c.face_rank;
                if let Some(be) = be {
                    e.body_available = true;
                    e.body_score = be.score;
                    e.body_margin = be.margin;
                    e.body_rank = be.rank;
                }
                e
            })
            .collect();

        let anti_chain = AntiChainingResultV2::check(&evidence_list, &self.thresholds_v2);

        // 6. Build shadow candidate (best from new pipeline)
        let shadow_candidate = face_candidates.first().map(|c| CandidateResult {
            person_id: c.person_id,
            face_score: c.face_score,
            face_rank: c.face_rank,
            body_score: body_candidates
                .iter()
                .find(|b| b.person_id == c.person_id)
                .map(|b| b.body_score)
                .unwrap_or(0.0),
            body_rank: body_candidates
                .iter()
                .find(|b| b.person_id == c.person_id)
                .map(|b| b.body_rank)
                .unwrap_or(0),
            face_count: 0,
        });

        // 7. Compute shadow decision
        // Phase 36.2 T3/T4: Look up candidate's face_count for bootstrap protection
        let candidate_face_count = face_candidates.first()
            .and_then(|c| {
                self.db.transaction(|tx| tx.persons().get_by_id(c.person_id))
                    .ok()
                    .flatten()
                    .map(|p| p.face_count as usize)
            })
            .unwrap_or(0);

        let best_evidence = all_evidence.first();
        let (shadow_decision, shadow_confidence, fusion_margin) = if let Some((_, fe, be)) = best_evidence {
            let mut identity_ev = IdentityEvidence::new(face_candidates[0].person_id);
            identity_ev.face_available = true;
            identity_ev.face_score = fe.score;
            identity_ev.face_margin = fe.margin;
            identity_ev.face_rank = fe.rank;
            identity_ev.face_quality = query.face_quality.composite_quality();

            if let Some(be) = be {
                identity_ev.body_available = true;
                identity_ev.body_score = be.score;
                identity_ev.body_margin = be.margin;
                identity_ev.body_rank = be.rank;
            }

            // Phase 36.1: Compute individual support for gray-zone matching
            let individual_support = if identity_ev.face_score >= GRAY_ZONE_MIN_SCORE
                && identity_ev.face_score < 0.78
            {
                Some(self.compute_individual_support(&query.face_embedding, face_candidates[0].person_id))
            } else {
                None
            };

            let fusion = identity_ev.fusion_margin(0.5);
            let confidence = identity_ev.confidence();
            // Phase 36.1: Use new decision logic with individual support for gray-zone
            // Phase 36.2 T3/T4: Pass candidate_face_count for bootstrap protection
            let decision = identity_ev.decide_phase36(query.face_quality.yaw, &anti_chain, individual_support.as_ref(), candidate_face_count);

            (decision, confidence, fusion)
        } else {
            (IdentityDecision::NewPerson, 0.0, 0.0)
        };

        // 8. Pollution check (placeholder - needs prototype service injection in full impl)
        let pollution_result = None;

        IdentityDecisionResult {
            production_candidate: None, // Set by caller if running in legacy mode
            shadow_candidate,
            face_evidence: best_evidence.map(|(_, fe, _)| fe.clone()),
            body_evidence: best_evidence.and_then(|(_, _, be)| be.clone()),
            fusion_margin,
            shadow_decision,
            shadow_confidence,
            anti_chain: Some(anti_chain.into()),
            pollution_result,
            pipeline_version,
            quality_gate_passed,
            quality_gate_failure,
        }
    }

    /// 使用预先计算的 candidates 评估 identity（用于 Shadow Mode）
    ///
    /// 与 `evaluate()` 不同，这个方法不重新进行 HNSW search，
    /// 而是直接使用 Legacy Pipeline 已经找到并解析好的 candidates。
    /// 这样可以确保 Shadow 和 Legacy 使用完全相同的 candidate set，
    /// 从而使 disagreement 分析有意义（candidate retrieval 问题 vs decision policy 问题）。
    pub fn evaluate_with_candidates(
        &self,
        query: &IdentityQuery,
        face_candidates: Vec<CandidateResult>,
        body_candidates: Vec<CandidateResult>,
    ) -> IdentityDecisionResult {
        let pipeline_version = IdentityPipelineVersion::current();

        // 1. Quality Gate
        let quality_gate_passed = self.quality_gate.passes(query.face_quality);
        let quality_gate_failure = if quality_gate_passed {
            None
        } else {
            self.quality_gate.failure_reason(query.face_quality)
        };

        // 2. Compute evidence for each candidate using pre-computed candidates
        // Phase 2 LOO Fix: 传递 query.image_id 用于 LOO exclusion
        let mut all_evidence = Vec::new();
        for face_cand in &face_candidates {
            let face_evidence = self.compute_face_evidence(
                &query.face_embedding,
                face_cand,
                &face_candidates,
                query.face_quality,
                Some(query.image_id),
            );

            // Body evidence: for now, use body_candidates scores directly if available
            // TODO: proper body evidence computation when BodyPipeline is fully integrated
            let body_evidence = if query.body_available {
                body_candidates
                    .iter()
                    .find(|b| b.person_id == face_cand.person_id)
                    .map(|b| BodyEvidence {
                        score: b.body_score,
                        second_score: 0.0,
                        margin: None, // Phase 36.2 T2: margin is Option<f32>
                        rank: b.body_rank,
                        quality: query.body_quality,
                        evidence_strength: EvidenceStrength::Weak,
                    })
            } else {
                None
            };

            all_evidence.push((face_cand.clone(), face_evidence, body_evidence));
        }

        // 3. Anti-chaining check
        let evidence_list: Vec<IdentityEvidence> = all_evidence
            .iter()
            .map(|(c, fe, be)| {
                let mut e = IdentityEvidence::new(c.person_id);
                e.face_available = true;
                e.face_score = fe.score;
                e.face_margin = fe.margin;
                e.face_rank = c.face_rank;
                if let Some(be) = be {
                    e.body_available = true;
                    e.body_score = be.score;
                    e.body_margin = be.margin;
                    e.body_rank = be.rank;
                }
                e
            })
            .collect();

        let anti_chain = AntiChainingResultV2::check(&evidence_list, &self.thresholds_v2);

        // 4. Build shadow candidate (best from provided candidates)
        let shadow_candidate = face_candidates.first().map(|c| CandidateResult {
            person_id: c.person_id,
            face_score: c.face_score,
            face_rank: c.face_rank,
            body_score: body_candidates
                .iter()
                .find(|b| b.person_id == c.person_id)
                .map(|b| b.body_score)
                .unwrap_or(0.0),
            body_rank: body_candidates
                .iter()
                .find(|b| b.person_id == c.person_id)
                .map(|b| b.body_rank)
                .unwrap_or(0),
            face_count: 0,
        });

        // 5. Compute shadow decision
        // Phase 36.2 T3/T4: Use candidate's face_count from pre-computed candidates
        let candidate_face_count = face_candidates.first().map(|c| c.face_count).unwrap_or(0);

        let best_evidence = all_evidence.first();
        let (shadow_decision, shadow_confidence, fusion_margin) = if let Some((_, fe, be)) = best_evidence {
            let mut identity_ev = IdentityEvidence::new(face_candidates[0].person_id);
            identity_ev.face_available = true;
            identity_ev.face_score = fe.score;
            identity_ev.face_margin = fe.margin;
            identity_ev.face_rank = fe.rank;
            identity_ev.face_quality = query.face_quality.composite_quality();

            if let Some(be) = be {
                identity_ev.body_available = true;
                identity_ev.body_score = be.score;
                identity_ev.body_margin = be.margin;
                identity_ev.body_rank = be.rank;
            }

            // Phase 36.1: Compute individual support for gray-zone matching
            let individual_support = if identity_ev.face_score >= GRAY_ZONE_MIN_SCORE
                && identity_ev.face_score < 0.78
            {
                Some(self.compute_individual_support(&query.face_embedding, face_candidates[0].person_id))
            } else {
                None
            };

            let fusion = identity_ev.fusion_margin(0.5);
            let confidence = identity_ev.confidence();
            // Phase 36.1: Use new decision logic with individual support for gray-zone
            // Phase 36.2 T3/T4: Pass candidate_face_count for bootstrap protection
            let decision = identity_ev.decide_phase36(query.face_quality.yaw, &anti_chain, individual_support.as_ref(), candidate_face_count);

            (decision, confidence, fusion)
        } else {
            (IdentityDecision::NewPerson, 0.0, 0.0)
        };

        // 6. Pollution check (placeholder)
        let pollution_result = None;

        IdentityDecisionResult {
            production_candidate: None,
            shadow_candidate,
            face_evidence: best_evidence.map(|(_, fe, _)| fe.clone()),
            body_evidence: best_evidence.and_then(|(_, _, be)| be.clone()),
            fusion_margin,
            shadow_decision,
            shadow_confidence,
            anti_chain: Some(anti_chain.into()),
            pollution_result,
            pipeline_version,
            quality_gate_passed,
            quality_gate_failure,
        }
    }

    /// Phase 32 修复：retrieve_face_candidates 现在正确：
    /// Phase 36.2 T5: HNSW retrieval uses fixed k=30 to ensure candidate recall.
    /// Top 10 persons are returned after grouping by person_id.
    const FACE_RETRIEVAL_K: usize = 30;

    /// 1. 将 HNSW vector_id 映射到 face.person_id
    /// 2. 过滤自匹配（query face 不会作为 candidate）
    /// 3. 按 person_id 聚合分数（取 cum_score 和 top_score）
    fn retrieve_face_candidates(
        &self,
        embedding: &[f32],
        top_k: usize,
        query_face_id: i64,
        query_image_id: i64,
    ) -> Vec<CandidateResult> {
        use std::collections::HashMap;

        // Phase 36.2 T5: Use fixed k=30 for HNSW retrieval to ensure candidate recall
        let retrieval_k = Self::FACE_RETRIEVAL_K.max(top_k * 3);
        let hits = match self.face_index.search(embedding, retrieval_k) {
            Ok(h) => h,
            Err(_) => return Vec::new(),
        };

        // Phase 32 修复：HNSW 返回的是 vector_id，需要映射到 face_id → person_id
        // 同时聚合同一 person 的多个 face hits
        let mut votes: HashMap<i64, (f32, f32)> = HashMap::new(); // pid → (cum_score, top_score)
        let mut face_to_person: HashMap<i64, i64> = HashMap::new(); // face_id → person_id (for self-match check)

        for hit in &hits {
            let vector_id = hit.id as i64;
            // Look up face by vector_id
            if let Ok(Some(face)) = self.db.transaction(|tx| tx.faces().get_by_vector_id(vector_id)) {
                let face_id = face.id;
                let person_id = face.person_id.unwrap_or(0);

                // Skip self-match: same face_id or same image_id
                if face_id == query_face_id || face.image_id == query_image_id {
                    continue;
                }

                // Skip faces without person_id (unassigned faces still in index)
                if person_id == 0 {
                    continue;
                }

                face_to_person.insert(face_id, person_id);

                let entry = votes.entry(person_id).or_insert((0.0, 0.0));
                entry.0 += hit.score; // cumulative score
                entry.1 = entry.1.max(hit.score); // top score for this person
            }
        }

        // Convert to sorted CandidateResult (sorted by cumulative score, like Legacy)
        let mut candidates: Vec<(i64, f32, f32)> = votes
            .into_iter()
            .map(|(pid, (cum, top))| (pid, cum, top))
            .collect();

        candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates
            .into_iter()
            .take(top_k)
            .enumerate()
            .map(|(i, (pid, cum_score, top_score))| CandidateResult {
                person_id: pid,
                face_score: top_score, // Use top_score for the candidate's representative score
                face_rank: i + 1,
                body_score: 0.0,
                body_rank: 0,
                face_count: 0,
            })
            .collect()
    }

    fn retrieve_body_candidates(&self, embedding: &[f32], top_k: usize) -> Vec<CandidateResult> {
        let Some(ref bi) = self.body_index else {
            return Vec::new();
        };

        let hits = match bi.search(embedding, top_k) {
            Ok(h) => h,
            Err(_) => return Vec::new(),
        };

        hits.into_iter()
            .enumerate()
            .map(|(i, h)| CandidateResult {
                person_id: h.id as i64,
                body_score: h.score,
                body_rank: i + 1,
                face_score: 0.0,
                face_rank: 0,
                face_count: 0,
            })
            .collect()
    }

    /// Phase 32 修复：compute_face_evidence 现在使用 prototype-based scoring
    ///
    /// 与 Legacy 的 decide_assignment 相同：
    /// 1. 获取 candidate person 的 prototype set
    /// 2. 计算 query embedding vs prototypes 的最大 cosine similarity
    /// 3. margin = top1_score - top2_score (person-level，而非 HNSW neighbor-level)
    ///
    /// Phase 36.2 T2: margin 是 Option<f32>。如果 candidate_count < 2，返回 None。
    ///
    /// Phase 2 (LOO Fix): query_image_id 用于 LOO exclusion。
    /// 当对 query 自身的 person 计分时，排除 query 所在的 image，避免自泄漏。
    fn compute_face_evidence(
        &self,
        embedding: &[f32],
        candidate: &CandidateResult,
        all_candidates: &[CandidateResult],
        face_quality: FaceQuality,
        query_image_id: Option<i64>,
    ) -> FaceEvidence {
        use crate::cluster_v2::cosine_similarity;

        // Helper: compute prototype score for a single person
        // Phase 2 LOO Fix: 使用 exclude_image_id 排除 query 自身的 image
        let compute_proto_score = |person_id: i64| -> f32 {
            if let Some(ref ps) = self.prototype_service {
                if let Ok(protos) = ps.list_for_person(person_id, query_image_id) {
                    if !protos.is_empty() {
                        return protos
                            .iter()
                            .map(|(_, proto_emb)| cosine_similarity(embedding, proto_emb))
                            .fold(0.0f32, f32::max);
                    }
                }
            }
            0.0 // No prototypes = 0 score
        };

        // Phase 36 Fix: compute prototype scores for ALL candidates to get accurate margin
        // All candidates are scored by their prototype similarity, then sorted
        let mut scored: Vec<(i64, f32)> = all_candidates
            .iter()
            .map(|c| (c.person_id, compute_proto_score(c.person_id)))
            .collect();

        // Sort by prototype score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let best_score = scored.first().map(|(_, s)| *s).unwrap_or(0.0);
        let second_score = scored.get(1).map(|(_, s)| *s).unwrap_or(0.0);

        // Phase 36.2 T2: margin 是 Option<f32>。只有当 candidate_count >= 2 时才计算 margin
        let margin = if scored.len() >= 2 {
            Some(best_score - second_score)
        } else {
            None // Single candidate - insufficient competition for margin
        };

        // Get prototype score for the specific candidate we're evaluating
        let prototype_score = compute_proto_score(candidate.person_id);

        FaceEvidence {
            score: prototype_score,
            second_score,
            margin,
            rank: candidate.face_rank,
            quality: face_quality,
            evidence_strength: EvidenceStrength::from_quality(face_quality.composite_quality()),
        }
    }

    /// Phase 36.1: Compute individual face support for a person
    ///
    /// Computes cosine similarity between query embedding and individual face embeddings in a person.
    /// Returns evidence for gray-zone matching when prototype_score alone is insufficient.
    ///
    /// Only examines up to MAX_SUPPORT_FACES representative faces per person to limit compute.
    pub fn compute_individual_support(
        &self,
        embedding: &[f32],
        person_id: i64,
    ) -> IndividualSupportEvidence {
        use crate::cluster_v2::cosine_similarity;

        let mut scores: Vec<f32> = Vec::new();

        if let Some(ref ps) = self.prototype_service {
            if let Ok(protos) = ps.list_for_person(person_id, None) {
                // Compute similarity for each prototype (individual face embedding)
                for (_, proto_emb) in &protos {
                    let sim = cosine_similarity(embedding, proto_emb);
                    scores.push(sim);
                }
            }
        }

        // Sort scores descending
        scores.sort_by(|a, b| b.partial_cmp(a).unwrap());

        // Take top MAX_SUPPORT_FACES
        scores.truncate(MAX_SUPPORT_FACES);

        let top1_score = scores.first().copied().unwrap_or(0.0);
        let top2_score = scores.get(1).copied();
        let support_count = scores.iter().filter(|&&s| s >= SUPPORT_SCORE).count();

        IndividualSupportEvidence {
            top1_score,
            top2_score,
            support_count,
            scores: scores.clone(),
            supporting_face_ids: Vec::new(), // Face IDs not easily available here, for debugging only
        }
    }

    /// Phase 34 T6: Replay Evaluation
    ///
    /// Evaluates a face against pre-computed candidates using BOTH Legacy and Shadow scoring.
    /// This ensures fair comparison by using the SAME candidate set.
    ///
    /// Returns a ReplayComparisonResult that compares Legacy vs Shadow decisions.
    pub fn evaluate_replay(
        &self,
        face_id: i64,
        image_id: i64,
        embedding: &[f32],
        face_quality: FaceQuality,
        candidates: &[PreComputedCandidate],
    ) -> ReplayComparisonResult {
        use crate::cluster_v2::cosine_similarity;

        // Score each candidate with Shadow's prototype-based scoring
        let mut scored: Vec<(PreComputedCandidate, f32)> = candidates
            .iter()
            .map(|c| {
                let shadow_score = if let Some(ref ps) = self.prototype_service {
                    match ps.list_for_person(c.person_id, None) {
                        Ok(protos) if !protos.is_empty() => {
                            let max_sim = protos
                                .iter()
                                .map(|(_, proto_emb)| cosine_similarity(embedding, proto_emb))
                                .fold(0.0f32, f32::max);
                            max_sim
                        }
                        _ => c.legacy_score, // Fallback
                    }
                } else {
                    c.legacy_score
                };
                (c.clone(), shadow_score)
            })
            .collect();

        // Sort by shadow score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Get top candidates
        let shadow_top1 = scored.first().map(|(c, _)| c.clone());
        let legacy_top1 = candidates.first().cloned();

        // Score for Shadow on Legacy's top-1
        let shadow_score_on_legacy_top1 = legacy_top1.as_ref().map(|c| {
            if let Some(ref ps) = self.prototype_service {
                match ps.list_for_person(c.person_id, None) {
                    Ok(protos) if !protos.is_empty() => {
                        protos
                            .iter()
                            .map(|(_, proto_emb)| cosine_similarity(embedding, proto_emb))
                            .fold(0.0f32, f32::max)
                    }
                    _ => c.legacy_score,
                }
            } else {
                c.legacy_score
            }
        });

        // Score for Shadow on Shadow's top-1
        let shadow_score_on_shadow_top1 = scored.first().map(|(c, s)| *s);

        // Compute score delta
        let score_delta = match (shadow_score_on_legacy_top1, shadow_score_on_shadow_top1) {
            (Some(l), Some(s)) => Some((l - s).abs()),
            _ => None,
        };

        // Determine decisions
        let legacy_decision = if legacy_top1.is_some() {
            IdentityDecision::ConfidentMatch
        } else {
            IdentityDecision::NewPerson
        };

        let shadow_decision = if let Some((c, score)) = scored.first() {
            if *score >= 0.70 {
                IdentityDecision::ConfidentMatch
            } else if *score >= 0.50 {
                IdentityDecision::WeakMatch
            } else {
                IdentityDecision::NewPerson
            }
        } else {
            IdentityDecision::NewPerson
        };

        ReplayComparisonResult {
            face_id,
            image_id,
            legacy_top1,
            shadow_top1,
            legacy_decision,
            shadow_decision,
            shadow_score_on_legacy_top1,
            shadow_score_on_shadow_top1,
            score_delta,
            face_quality,
            anti_chain_passed: true, // Will be set by caller if needed
        }
    }

    fn compute_body_evidence(&self, embedding: &[f32], person_id: i64) -> Option<BodyEvidence> {
        // Simplified body evidence computation
        None
    }
}

// ============================================================================
// Phase 30 T3: Execution Mode Enum
// ============================================================================

/// Identity Pipeline 执行模式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdentityExecutionMode {
    /// Legacy pipeline only (no shadow)
    Legacy,
    /// Shadow mode: run new pipeline alongside legacy, record only
    Shadow,
    /// New pipeline (production, no legacy)
    NewPipeline,
}

impl Default for IdentityExecutionMode {
    fn default() -> Self {
        IdentityExecutionMode::Shadow
    }
}

impl IdentityExecutionMode {
    pub fn is_shadow(&self) -> bool {
        matches!(self, IdentityExecutionMode::Shadow)
    }

    pub fn should_write_production(&self) -> bool {
        matches!(self, IdentityExecutionMode::Legacy | IdentityExecutionMode::NewPipeline)
    }
}

// ============================================================================
// Phase 30: Shadow Mode Types (T7, T17)
// ============================================================================

/// Shadow Disagreement Type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShadowDisagreementType {
    /// Legacy and Shadow agree
    Agree,
    /// Same candidate universe but different decision
    DecisionDifferent,
    /// Legacy assigns but Shadow says Unknown/Review
    LegacyAssignShadowUnknown,
    /// Legacy assigns but Shadow wants different person
    LegacyAssignShadowDifferentPerson,
    /// Legacy NewPerson but Shadow assigns
    LegacyNewPersonShadowAssign,
    /// Legacy assigns but Shadow suggests Review
    LegacyAssignShadowReview,
    /// Score-only difference (margin difference)
    ScoreOnlyDifference,
}

impl ShadowDisagreementType {
    pub fn from_comparison(
        legacy_decision: IdentityDecision,
        legacy_candidate: Option<i64>,
        shadow_decision: IdentityDecision,
        shadow_candidate: Option<i64>,
    ) -> Self {
        if legacy_decision == shadow_decision {
            if legacy_candidate == shadow_candidate {
                return ShadowDisagreementType::Agree;
            }
            // Same decision but different person IDs = serious disagreement
            return ShadowDisagreementType::LegacyAssignShadowDifferentPerson;
        }

        match (legacy_decision, shadow_decision) {
            (IdentityDecision::ConfidentMatch | IdentityDecision::WeakMatch, IdentityDecision::NewPerson) => {
                ShadowDisagreementType::LegacyAssignShadowUnknown
            }
            (IdentityDecision::ConfidentMatch | IdentityDecision::WeakMatch, IdentityDecision::Ambiguous) => {
                ShadowDisagreementType::LegacyAssignShadowReview
            }
            (IdentityDecision::NewPerson, IdentityDecision::ConfidentMatch | IdentityDecision::WeakMatch) => {
                ShadowDisagreementType::LegacyNewPersonShadowAssign
            }
            _ => {
                if legacy_candidate != shadow_candidate {
                    ShadowDisagreementType::LegacyAssignShadowDifferentPerson
                } else {
                    ShadowDisagreementType::DecisionDifferent
                }
            }
        }
    }
}

// ============================================================================
// Phase 34: Candidate-Set Replay Types
// ============================================================================

/// Pre-computed candidate from Legacy state (Phase 34 T1-T2)
///
/// Contains the candidate set that BOTH Legacy and Shadow will score.
/// This ensures fair comparison by using the SAME candidate set.
#[derive(Debug, Clone)]
pub struct PreComputedCandidate {
    pub person_id: i64,
    /// Legacy's prototype score for this candidate
    pub legacy_score: f32,
    /// Legacy's margin (top1 - top2)
    pub legacy_margin: f32,
    /// Number of prototypes for this person
    pub prototype_count: usize,
}

/// Result of Phase 34 Replay evaluation (Phase 34 T3-T4)
#[derive(Debug, Clone)]
pub struct ReplayComparisonResult {
    /// Face ID
    pub face_id: i64,
    /// Image ID
    pub image_id: i64,

    /// Legacy's top-1 candidate
    pub legacy_top1: Option<PreComputedCandidate>,
    /// Shadow's top-1 candidate (based on Shadow scoring)
    pub shadow_top1: Option<PreComputedCandidate>,

    /// Legacy's decision
    pub legacy_decision: IdentityDecision,
    /// Shadow's decision
    pub shadow_decision: IdentityDecision,

    /// Shadow's score for Legacy's top-1 person
    pub shadow_score_on_legacy_top1: Option<f32>,
    /// Shadow's score for Shadow's top-1 person
    pub shadow_score_on_shadow_top1: Option<f32>,

    /// Score delta (|legacy_score - shadow_score|)
    pub score_delta: Option<f32>,

    /// Face quality metrics
    pub face_quality: FaceQuality,

    /// Anti-chain status
    pub anti_chain_passed: bool,
}

/// Phase 34 T6: Replay Statistics
#[derive(Debug, Clone, Default)]
pub struct ReplayStats {
    pub total_faces: usize,
    /// Candidate Agreement: Legacy Top1 == Shadow Top1
    pub candidate_agreement: usize,
    /// Candidate Recall: Shadow Top1 in Legacy Top-3
    pub candidate_recall_at_3: usize,
    /// Candidate Recall: Shadow Top1 in Legacy Top-5
    pub candidate_recall_at_5: usize,

    /// Decision Agreement
    pub decision_agree: usize,
    /// Decision: Shadow more aggressive (found match Legacy missed)
    pub shadow_more_aggressive: usize,
    /// Decision: Shadow more conservative (rejected valid Legacy match)
    pub shadow_more_conservative: usize,
    /// Decision: Different person assigned
    pub different_person: usize,

    /// Score statistics
    pub legacy_mean_score: f32,
    pub shadow_mean_score: f32,
    pub mean_score_delta: f32,

    /// Anti-chain
    pub anti_chain_passed: usize,
    pub anti_chain_blocked: usize,
}

impl ReplayStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, result: &ReplayComparisonResult) {
        self.total_faces += 1;

        // Candidate agreement
        let legacy_top1_pid = result.legacy_top1.as_ref().map(|c| c.person_id);
        let shadow_top1_pid = result.shadow_top1.as_ref().map(|c| c.person_id);

        if legacy_top1_pid == shadow_top1_pid {
            self.candidate_agreement += 1;
        }

        // Score stats
        if let Some(ref lc) = result.legacy_top1 {
            self.legacy_mean_score += lc.legacy_score;
        }
        if let Some(ref sc) = result.shadow_top1 {
            self.shadow_mean_score += sc.legacy_score; // Shadow uses same scoring
        }

        // Decision agreement
        if result.legacy_decision == result.shadow_decision {
            self.decision_agree += 1;
        }

        // Anti-chain
        if result.anti_chain_passed {
            self.anti_chain_passed += 1;
        } else {
            self.anti_chain_blocked += 1;
        }

        // Score delta
        if let Some(delta) = result.score_delta {
            self.mean_score_delta += delta;
        }
    }

    pub fn finalize(&mut self) {
        if self.total_faces > 0 {
            self.legacy_mean_score /= self.total_faces as f32;
            self.shadow_mean_score /= self.total_faces as f32;
            self.mean_score_delta /= self.total_faces as f32;
        }
    }
}

/// Shadow Record for database storage (Phase 30 T7)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowRecord {
    pub image_id: i64,
    pub face_id: Option<i64>,
    pub body_id: Option<i64>,
    pub legacy_person_id: Option<i64>,
    pub legacy_decision: String,
    pub shadow_person_id: Option<i64>,
    pub shadow_decision: String,
    pub face_score: Option<f32>,
    pub face_margin: Option<f32>,
    pub body_score: Option<f32>,
    pub body_margin: Option<f32>,
    pub confidence: f32,
    pub face_quality: Option<f32>,
    pub body_quality: Option<f32>,
    pub anti_chain_status: String,
    pub pollution_status: String,
    pub quality_gate_status: String,
    pub disagreement_type: String,
    pub pipeline_version: String,
    pub threshold_version: String,
    pub model_version: String,
    pub execution_time_ms: u64,
}

impl ShadowRecord {
    pub fn from_result(
        result: &IdentityDecisionResult,
        face_id: Option<i64>,
        body_id: Option<i64>,
        legacy_person_id: Option<i64>,
        legacy_decision: &str,
        execution_time_ms: u64,
    ) -> Self {
        let shadow_decision_str = match result.shadow_decision {
            IdentityDecision::ConfidentMatch => "confirmed",
            IdentityDecision::SupportedMatch => "supported",
            IdentityDecision::StrongMatchWithCompetitor => "strong_with_competitor",
            IdentityDecision::WeakMatch => "probable",
            IdentityDecision::Ambiguous => "conflict",
            IdentityDecision::NewPerson => "unknown",
            IdentityDecision::Unreliable => "unreliable",
        };

        let disagreement = ShadowDisagreementType::from_comparison(
            Self::parse_decision(legacy_decision),
            legacy_person_id,
            result.shadow_decision,
            result.shadow_candidate.as_ref().map(|c| c.person_id),
        );

        let anti_chain_status = result
            .anti_chain
            .as_ref()
            .map(|ac| {
                if ac.is_ambiguous { "blocked" } else { "passed" }
            })
            .unwrap_or("N/A");

        let pollution_status = result
            .pollution_result
            .as_ref()
            .map(|pr| {
                if pr.is_polluted { "blocked" } else { "passed" }
            })
            .unwrap_or("N/A");

        let quality_gate_status = if result.quality_gate_passed { "passed" } else { "rejected" };

        Self {
            image_id: 0, // Set by caller
            face_id,
            body_id,
            legacy_person_id,
            legacy_decision: legacy_decision.to_string(),
            shadow_person_id: result.shadow_candidate.as_ref().map(|c| c.person_id),
            shadow_decision: shadow_decision_str.to_string(),
            face_score: result.face_evidence.as_ref().map(|e| e.score),
            face_margin: result.face_evidence.as_ref().and_then(|e| e.margin),
            body_score: result.body_evidence.as_ref().map(|e| e.score),
            body_margin: result.body_evidence.as_ref().and_then(|e| e.margin),
            confidence: result.shadow_confidence,
            face_quality: result.face_evidence.as_ref().map(|e| e.quality.composite_quality()),
            body_quality: result.body_evidence.as_ref().map(|e| e.quality),
            anti_chain_status: anti_chain_status.to_string(),
            pollution_status: pollution_status.to_string(),
            quality_gate_status: quality_gate_status.to_string(),
            disagreement_type: format!("{:?}", disagreement),
            pipeline_version: format!(
                "{}/{}/{}/{}",
                result.pipeline_version.face_detector,
                result.pipeline_version.face_embedding,
                result.pipeline_version.body_embedding,
                result.pipeline_version.fusion
            ),
            threshold_version: "v1".to_string(),
            model_version: result.pipeline_version.face_embedding.clone(),
            execution_time_ms,
        }
    }

    fn parse_decision(s: &str) -> IdentityDecision {
        match s {
            "confirmed" => IdentityDecision::ConfidentMatch,
            "probable" => IdentityDecision::WeakMatch,
            "conflict" => IdentityDecision::Ambiguous,
            _ => IdentityDecision::NewPerson,
        }
    }
}

/// Shadow Metrics Aggregator (Phase 30 T17)
#[derive(Debug, Clone, Default)]
pub struct ShadowMetrics {
    pub total_queries: usize,
    pub agreement_count: usize,
    pub disagreement_count: usize,
    pub candidate_difference_count: usize,
    pub decision_difference_count: usize,
    pub legacy_assign_shadow_unknown: usize,
    pub legacy_assign_shadow_review: usize,
    pub legacy_new_person_shadow_assign: usize,
    pub legacy_assign_shadow_different_person: usize,
    pub anti_chain_blocks: usize,
    pub pollution_rejections: usize,
    pub face_only_count: usize,
    pub body_only_count: usize,
    pub dual_evidence_count: usize,
    pub shadow_errors: usize,
}

impl ShadowMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, record: &ShadowRecord) {
        self.total_queries += 1;

        let is_agreement = record.disagreement_type == "Agree";
        if is_agreement {
            self.agreement_count += 1;
        } else {
            self.disagreement_count += 1;
        }

        match record.disagreement_type.as_str() {
            "ScoreOnlyDifference" => self.candidate_difference_count += 1,
            "DecisionDifferent" => self.decision_difference_count += 1,
            "LegacyAssignShadowUnknown" => self.legacy_assign_shadow_unknown += 1,
            "LegacyAssignShadowReview" => self.legacy_assign_shadow_review += 1,
            "LegacyNewPersonShadowAssign" => self.legacy_new_person_shadow_assign += 1,
            "LegacyAssignShadowDifferentPerson" => self.legacy_assign_shadow_different_person += 1,
            _ => {}
        }

        if record.anti_chain_status == "blocked" {
            self.anti_chain_blocks += 1;
        }
        if record.pollution_status == "blocked" {
            self.pollution_rejections += 1;
        }
    }

    pub fn record_error(&mut self) {
        self.shadow_errors += 1;
    }

    pub fn agreement_rate(&self) -> f32 {
        if self.total_queries == 0 {
            return 0.0;
        }
        self.agreement_count as f32 / self.total_queries as f32
    }

    pub fn disagreement_rate(&self) -> f32 {
        if self.total_queries == 0 {
            return 0.0;
        }
        self.disagreement_count as f32 / self.total_queries as f32
    }

    pub fn summary(&self) -> String {
        format!(
            "total={} agree={} ({:.1}%) disagree={} ({:.1}%) \
             anti_chain_blocks={} pollution_rejects={} errors={}",
            self.total_queries,
            self.agreement_count,
            self.agreement_rate() * 100.0,
            self.disagreement_count,
            self.disagreement_rate() * 100.0,
            self.anti_chain_blocks,
            self.pollution_rejections,
            self.shadow_errors
        )
    }

    /// Phase 31.5 T5: Evaluate if shadow health passes promotion gate
    pub fn evaluate_promotion(&self, policy: &pf_config::ShadowPromotionPolicy) -> PromotionEvaluation {
        let total = self.total_queries;
        let different_person_count = self.legacy_assign_shadow_different_person;
        let different_person_rate = if total > 0 {
            different_person_count as f32 / total as f32
        } else {
            0.0
        };
        let decision_agreement = self.agreement_rate();

        let has_enough_samples = total >= policy.min_samples_for_evaluation;
        let agreement_passes = decision_agreement >= policy.min_decision_agreement;
        let different_person_passes = different_person_rate <= policy.max_different_person_rate;
        let anti_chain_clean = self.anti_chain_blocks == 0;
        let no_pollution = self.pollution_rejections == 0;
        let no_errors = self.shadow_errors == 0;

        let is_healthy = has_enough_samples
            && agreement_passes
            && different_person_passes
            && anti_chain_clean
            && no_pollution
            && no_errors;

        PromotionEvaluation {
            total_queries: total,
            decision_agreement,
            different_person_rate,
            agreement_passes,
            different_person_passes,
            anti_chain_clean,
            pollution_clean: no_pollution,
            no_errors,
            has_enough_samples,
            is_healthy,
            recommendation: if is_healthy {
                PromotionRecommendation::ReadyForPromotion
            } else if !has_enough_samples {
                PromotionRecommendation::NeedMoreSamples
            } else {
                PromotionRecommendation::NotReady
            },
        }
    }
}

/// Result of promotion evaluation (Phase 31.5 T5)
#[derive(Debug, Clone)]
pub struct PromotionEvaluation {
    pub total_queries: usize,
    pub decision_agreement: f32,
    pub different_person_rate: f32,
    pub agreement_passes: bool,
    pub different_person_passes: bool,
    pub anti_chain_clean: bool,
    pub pollution_clean: bool,
    pub no_errors: bool,
    pub has_enough_samples: bool,
    pub is_healthy: bool,
    pub recommendation: PromotionRecommendation,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PromotionRecommendation {
    ReadyForPromotion,
    NeedMoreSamples,
    NotReady,
}

/// Shadow execution result (Phase 30 T9)
#[derive(Debug, Clone)]
pub enum ShadowExecutionResult {
    /// Shadow evaluation succeeded
    Evaluated {
        decision: IdentityDecisionResult,
    },
    /// Shadow evaluation failed (best-effort, does not affect legacy)
    Failed {
        error: String,
    },
    /// Shadow was skipped (not an error)
    Skipped {
        reason: ShadowSkipReason,
    },
}

/// Reason why shadow evaluation was skipped (Phase 30 T9)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowSkipReason {
    /// Shadow mode is not enabled
    ExecutionModeDisabled,
    /// Face embedding not available
    MissingFaceEmbedding,
    /// IdentityPipeline not available
    PipelineUnavailable,
    /// No candidates found by legacy
    NoCandidates,
    /// Face quality too low
    QualityGateFailed,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confident_match_face_strong() {
        let mut e = IdentityEvidence::new(1);
        e.face_available = true;
        e.body_available = true;
        e.face_score = 0.78;
        e.face_margin = Some(0.05);
        e.body_score = 0.60;
        e.body_margin = Some(0.0);
        assert_eq!(e.decide(), IdentityDecision::ConfidentMatch);
    }

    #[test]
    fn test_confident_match_face_plus_body() {
        let mut e = IdentityEvidence::new(1);
        e.face_available = true;
        e.body_available = true;
        e.face_score = 0.68;
        e.face_margin = Some(0.02);
        e.body_score = 0.72;
        e.body_margin = Some(0.01);
        assert_eq!(e.decide(), IdentityDecision::ConfidentMatch);
    }

    #[test]
    fn test_weak_match_body_support() {
        let mut e = IdentityEvidence::new(1);
        e.face_available = true;
        e.body_available = true;
        e.face_score = 0.58;
        e.face_margin = Some(0.0);
        e.body_score = 0.78;
        e.body_margin = Some(0.06);
        assert_eq!(e.decide(), IdentityDecision::WeakMatch);
    }

    #[test]
    fn test_weak_match_face_ambiguous_body_strong() {
        let mut e = IdentityEvidence::new(1);
        e.face_available = true;
        e.body_available = true;
        e.face_score = 0.60;
        e.face_margin = Some(0.01); // ambiguous
        e.body_score = 0.73;
        e.body_margin = Some(0.06);
        assert_eq!(e.decide(), IdentityDecision::WeakMatch);
    }

    #[test]
    fn test_ambiguous() {
        let mut e = IdentityEvidence::new(1);
        e.face_available = true;
        e.body_available = true;
        e.face_score = 0.65;
        e.face_margin = Some(0.02);
        e.body_score = 0.68;
        e.body_margin = Some(0.02);
        assert_eq!(e.decide(), IdentityDecision::Ambiguous);
    }

    #[test]
    fn test_new_person() {
        let e = IdentityEvidence::new(1);
        assert_eq!(e.decide(), IdentityDecision::NewPerson);
    }

    #[test]
    fn test_anti_chaining() {
        let mut e1 = IdentityEvidence::new(1);
        e1.face_score = 0.75;
        e1.body_score = 0.80;

        let mut e2 = IdentityEvidence::new(2);
        e2.face_score = 0.73;
        e2.body_score = 0.82;

        let result = AntiChainingResult::check(&[e1, e2], 0.05);
        assert!(result.is_ambiguous); // face_gap=0.02 < 0.05, body_gap=0.02 < 0.05
    }

    #[test]
    fn test_anti_chaining_face_resolves() {
        let mut e1 = IdentityEvidence::new(1);
        e1.face_score = 0.78;
        e1.body_score = 0.80;

        let mut e2 = IdentityEvidence::new(2);
        e2.face_score = 0.68;
        e2.body_score = 0.82;

        let result = AntiChainingResult::check(&[e1, e2], 0.05);
        assert!(!result.is_ambiguous); // face_gap=0.10 > 0.05
        assert_eq!(result.primary_signal, PrimarySignal::Face);
    }

    // =========================================================================
    // Phase 36 Decision Tests
    // =========================================================================

    fn make_anti_chain_use_top1() -> AntiChainingResultV2 {
        AntiChainingResultV2 {
            is_ambiguous: false,
            face_gap: 0.10,
            body_gap: 0.10,
            face_gap_sufficient: true,
            body_gap_sufficient: true,
            primary_signal: PrimarySignal::Face,
            suggested_decision: AntiChainingDecision::UseTop1,
            candidate_count: Some(2),
        }
    }

    fn make_anti_chain_review() -> AntiChainingResultV2 {
        AntiChainingResultV2 {
            is_ambiguous: true,
            face_gap: 0.03,
            body_gap: 0.03,
            face_gap_sufficient: false,
            body_gap_sufficient: false,
            primary_signal: PrimarySignal::Both,
            suggested_decision: AntiChainingDecision::Review,
            candidate_count: Some(2),
        }
    }

    fn make_anti_chain_conflict() -> AntiChainingResultV2 {
        AntiChainingResultV2 {
            is_ambiguous: true,
            face_gap: 0.01,
            body_gap: 0.01,
            face_gap_sufficient: false,
            body_gap_sufficient: false,
            primary_signal: PrimarySignal::Both,
            suggested_decision: AntiChainingDecision::Conflict,
            candidate_count: Some(2),
        }
    }

    #[test]
    fn test_phase36_extreme_profile_returns_unknown() {
        // EXTREME profile (|yaw| >= 60°) → UNKNOWN
        let mut e = IdentityEvidence::new(1);
        e.face_available = true;
        e.face_score = 0.85;
        e.face_margin = Some(0.15);
        e.face_quality = 0.60;

        let anti_chain = make_anti_chain_use_top1();
        // yaw = 65° (EXTREME), candidate_face_count = 3 (stable)
        let decision = e.decide_phase36(65.0, &anti_chain, None, 3);
        assert_eq!(decision, IdentityDecision::NewPerson);
    }

    #[test]
    fn test_phase36_low_quality_returns_unreliable() {
        // LOW quality (composite < 0.30) → Unreliable
        let mut e = IdentityEvidence::new(1);
        e.face_available = true;
        e.face_score = 0.85;
        e.face_margin = Some(0.15);
        e.face_quality = 0.20; // LOW quality (< 0.30 MIN_CLUSTER_QUALITY)

        let anti_chain = make_anti_chain_use_top1();
        // NORMAL yaw, candidate_face_count = 3 (stable)
        let decision = e.decide_phase36(20.0, &anti_chain, None, 3);
        assert_eq!(decision, IdentityDecision::Unreliable);
    }

    #[test]
    fn test_phase36_high_quality_auto_assign() {
        // HIGH quality + score >= 0.70 + margin >= 0.05 + anti_chain.UseTop1 → AUTO_ASSIGN
        let mut e = IdentityEvidence::new(1);
        e.face_available = true;
        e.face_score = 0.78;
        e.face_margin = Some(0.08);
        e.face_quality = 0.55; // HIGH quality

        let anti_chain = make_anti_chain_use_top1();
        // NORMAL yaw, candidate_face_count = 3 (stable)
        let decision = e.decide_phase36(10.0, &anti_chain, None, 3);
        assert_eq!(decision, IdentityDecision::ConfidentMatch);
    }

    #[test]
    fn test_phase36_conflict_returns_unknown() {
        // Anti-chain CONFLICT → UNKNOWN
        let mut e = IdentityEvidence::new(1);
        e.face_available = true;
        e.face_score = 0.78;
        e.face_margin = Some(0.08);
        e.face_quality = 0.55;

        let anti_chain = make_anti_chain_conflict();
        let decision = e.decide_phase36(10.0, &anti_chain, None, 3);
        assert_eq!(decision, IdentityDecision::NewPerson);
    }

    #[test]
    fn test_phase36_review_goes_to_weak_match() {
        // MEDIUM quality or ANTI-CHAIN REVIEW → REVIEW (WeakMatch if body supports)
        let mut e = IdentityEvidence::new(1);
        e.face_available = true;
        e.body_available = true;
        e.face_score = 0.60;
        e.face_margin = Some(0.03);
        e.face_quality = 0.35; // MEDIUM quality
        e.body_score = 0.75;
        e.body_margin = Some(0.08);

        let anti_chain = make_anti_chain_review();
        let decision = e.decide_phase36(10.0, &anti_chain, None, 3);
        // Body provides enough support → WeakMatch
        assert_eq!(decision, IdentityDecision::WeakMatch);
    }

    #[test]
    fn test_phase36_no_evidence_returns_unknown() {
        let e = IdentityEvidence::new(1);
        let anti_chain = make_anti_chain_use_top1();
        let decision = e.decide_phase36(0.0, &anti_chain, None, 3);
        assert_eq!(decision, IdentityDecision::NewPerson);
    }

    #[test]
    fn test_phase36_dual_channel_strong_auto_assign() {
        // DUAL channel strong → ConfidentMatch when score >= threshold + margin sufficient
        let mut e = IdentityEvidence::new(1);
        e.face_available = true;
        e.body_available = true;
        e.face_score = 0.80; // Must be >= 0.78 (AUTO_ASSIGN_SCORE) for ConfidentMatch
        e.face_margin = Some(0.05); // Must be >= 0.05 (AUTO_ASSIGN_MARGIN)
        e.face_quality = 0.50;
        e.body_score = 0.75;
        e.body_margin = Some(0.05);

        let anti_chain = make_anti_chain_use_top1();
        let decision = e.decide_phase36(15.0, &anti_chain, None, 3);
        assert_eq!(decision, IdentityDecision::ConfidentMatch);
    }
}
