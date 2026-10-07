//! 配置根结构。
//!
//! 所有默认值与 legacy `ai-next` **逐位对齐**——`FaceConfig::cluster_threshold = 0.65`
//! （person_cluster.rs:12），`FaceConfig::similarity_threshold = 0.50`（person_search.rs:227），
//! `ObjectConfig::fusion_alpha/beta/gamma = 0.3/0.5/0.2`（object_search.rs:81-82），
//! `ObjectConfig::min_score = 0.5`（object_search.rs:576）等。

use serde::{Deserialize, Serialize};

/// 顶层 Config。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// 人脸相关
    pub face: FaceConfig,
    /// 对象相关
    pub object: ObjectConfig,
    /// 补丁（patch）相关
    pub patch: PatchConfig,
    /// ROI 提取相关
    pub roi: RoiConfig,
    /// 向量相关
    pub vector: VectorConfig,
    /// 扫描相关
    pub scanner: ScannerConfig,
    /// 身份识别相关（Phase 18+）
    pub identity: IdentityConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            face: FaceConfig::default(),
            object: ObjectConfig::default(),
            patch: PatchConfig::default(),
            roi: RoiConfig::default(),
            vector: VectorConfig::default(),
            scanner: ScannerConfig::default(),
            identity: IdentityConfig::default(),
        }
    }
}

/// 人脸配置。
///
/// **算法与 ai-next 逐位对齐**：
/// - `min_detector_score=0.50`：config 默认（`person_search.rs::KPS` 内部放宽为 0.30）
/// - `cluster_threshold=0.65`：`person_cluster.rs:12 MATCH_THRESHOLD`
/// - `similarity_threshold=0.50`：`person_search.rs:227`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceConfig {
    /// 检测器最小置信度（config 层）
    pub min_detector_score: f32,
    /// 最小 face 宽度（像素），过滤远景小人脸
    pub min_face_size: u32,
    /// 最小质量分数（QualityFilter pipeline 阈值，0.45 与 legacy `is_acceptable` 对齐）
    pub min_quality: f32,
    /// 最大 yaw / pitch（度），过滤大角度侧脸
    pub max_yaw: f32,
    /// 聚类距离阈值（cosine similarity，>= 视为同人）— 与 ai-next 0.65 对齐
    pub cluster_threshold: f32,
    /// 搜索结果相似度阈值（>= 视为命中）— 与 ai-next 0.50 对齐
    pub similarity_threshold: f32,
    /// Phase 1.2 新增：HNSW 粗筛 fetch_k 倍率
    /// (`fetch_k = max(top_k * coarse_fetch_multiplier, coarse_fetch_min)`)。
    /// 默认 4 — 保留之前硬编码 `fetch_k = top_k * 4` 的行为。
    pub coarse_fetch_multiplier: usize,
    /// Phase 1.2 新增：HNSW 粗筛 fetch_k 最小值。默认 4 — 小 top_k 时
    /// 仍能拿到 4 个候选供阈值过滤。
    pub coarse_fetch_min: usize,
    /// Phase 1.2 新增：粗筛 embedding 阈值，低于此值的 candidate 直接跳过。
    /// 默认 0.0 — face 搜索当前没有 per-candidate pre-filter。
    pub fine_match_min_embedding: f32,
    /// Phase Cascade: 启用 SCRFD 500M + 10G 双 detector cascade。
    /// 启用后加载第二个 SCRFD 10G 模型,IoU dedup 后取并集。
    /// 默认 false — 与 legacy 行为一致。
    #[serde(default)]
    pub enable_face_cascade: bool,
    /// Phase Cascade: IoU 阈值,primary 与 secondary 的 bbox 高于此值视为同人脸。
    /// 默认 0.30 (与 SCRFD 内部 soft_nms 0.35 接近,稍放宽处理 10G bbox 偏移)。
    #[serde(default)]
    pub cascade_iou_threshold: f32,
    /// Phase Cascade: secondary detector (10G) 的最小 score,
    /// 低于此值的 secondary-only detection 直接丢弃。
    /// 默认 0.05 — 10G 在小脸 / 正面照上噪声较多,需要兜底。
    #[serde(default)]
    pub cascade_10g_min_score: f32,
    /// Phase RollCorrect: 启用 in-plane 旋转校正 (Rotate by -roll)。
    /// 对 |roll|>threshold_deg 的人脸用旋转后的图像重做 align+embed,
    /// 替换 baseline embedding。
    /// 默认 false — 与 legacy 行为一致。
    #[serde(default)]
    pub enable_roll_correct: bool,
    /// Phase RollCorrect: 触发阈值 (度)。
    /// |roll| 超过此值才走 RollCorrect 路径 (绝大多数正面照不动)。
    /// 默认 5.0。
    #[serde(default)]
    pub roll_correct_threshold_deg: f32,
    /// SCRFD 10G 模型路径 (cascade 启用时加载)。None 时使用默认路径。
    #[serde(default)]
    pub scrfd_10g_model_path: Option<std::path::PathBuf>,
}

impl Default for FaceConfig {
    fn default() -> Self {
        Self {
            min_detector_score: 0.50,
            min_face_size: 20,
            min_quality: 0.45,
            max_yaw: 60.0,
            cluster_threshold: 0.65,
            similarity_threshold: 0.50,
            coarse_fetch_multiplier: 4,
            coarse_fetch_min: 4,
            fine_match_min_embedding: 0.0,
            enable_face_cascade: false,
            cascade_iou_threshold: 0.30,
            cascade_10g_min_score: 0.05,
            enable_roll_correct: false,
            roll_correct_threshold_deg: 5.0,
            scrfd_10g_model_path: None,
        }
    }
}

/// 对象配置（YOLOv8n + MobileCLIP + LightGlue 融合）。
///
/// **算法与 ai-next 逐位对齐**（`object_search.rs:64-87`）：
/// - `fusion_alpha/beta/gamma = 0.3 / 0.5 / 0.2`
/// - `min_bbox_overlap = 0.2`
/// - `min_score = 0.5`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectConfig {
    /// 检测器最小置信度
    pub min_detector_score: f32,
    /// 最小对象宽度（像素）
    pub min_object_size: u32,
    /// 融合权重：embedding_score
    pub fusion_alpha: f32,
    /// 融合权重：inlier_ratio
    pub fusion_beta: f32,
    /// 融合权重：color_similarity / bbox_overlap
    pub fusion_gamma: f32,
    /// bbox 重叠阈值（>= 视为空间命中）
    pub min_bbox_overlap: f32,
    /// 最小相似度/置信度（两个都需 >= 此值）— ai-next `object_search.rs:576`
    pub min_score: f32,
    /// LightGlue 几何校验最大候选数（Phase 7: 从 20 提到 50，确保更多 candidate 经过几何校验）
    pub hnsw_top_k: usize,
    /// Phase 3 新增：HNSW 粗筛 fetch_k 倍率（`fetch_k = max(top_k * coarse_fetch_multiplier, coarse_fetch_min)`）
    /// 与 ai-next hnsw_top_k=20 / `fetch_k=max(top_k*20, 200)` 对齐
    pub coarse_fetch_multiplier: usize,
    /// Phase 3 新增：HNSW 粗筛 fetch_k 最小值（保证小 top_k 也能拿到足够候选）
    pub coarse_fetch_min: usize,
    /// Phase 3 新增：FullImage ROI 调权（粗筛阶段按 roi_type 调整 score）
    /// 设计：full_image 是更 general 的信号，权重 < 1.0；sliding_window 是更 specific 的局部信号，权重 = 1.0
    pub full_image_weight: f32,
    /// Phase 3 新增：SlidingWindow ROI 调权（粗筛阶段按 roi_type 调整 score）
    pub sliding_window_weight: f32,
    /// Phase 5 新增：LightGlue 最小匹配数（少于这个数 → confidence=0，跳过该候选）
    /// 与 PatchConfig::min_lightglue_matches_object=3 对齐（场景用 15，对象更宽松）
    pub lightglue_min_matches: usize,
    /// Phase 7 新增：粗筛 embedding 阈值。低于此值的 candidate 直接跳过 LightGlue。
    /// 设计：保留明显相关的 candidate，过滤 trivial non-matches，节省 LightGlue 计算。
    pub fine_match_min_embedding: f32,
}

impl Default for ObjectConfig {
    fn default() -> Self {
        Self {
            min_detector_score: 0.4,
            min_object_size: 32,
            fusion_alpha: 0.3,
            fusion_beta: 0.5,
            fusion_gamma: 0.2,
            min_bbox_overlap: 0.2,
            min_score: 0.5,
            hnsw_top_k: 50,
            coarse_fetch_multiplier: 20,
            coarse_fetch_min: 200,
            full_image_weight: 0.85,
            sliding_window_weight: 1.0,
            lightglue_min_matches: 3,
            fine_match_min_embedding: 0.3,
        }
    }
}

/// 补丁搜索配置（SuperPoint + LightGlue + VLAD）。
///
/// **算法与 ai-next 逐位对齐**（`patch_search.rs:78-90`）：
/// - `vlad_k = 64`
/// - `max_keypoints_per_patch = 256`
/// - `min_lightglue_matches_scene = 15`
/// - `min_lightglue_matches_object = 3`
/// - `ransac_threshold = 4.0`
/// - `min_bbox_overlap = 0.3`
/// - `max_query_patches = 50`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchConfig {
    /// HNSW top_k
    pub hnsw_top_k: usize,
    /// 每 patch 最大 keypoints
    pub max_keypoints_per_patch: usize,
    /// 场景模式 LightGlue 最小匹配数
    pub min_lightglue_matches_scene: usize,
    /// 对象模式 LightGlue 最小匹配数
    pub min_lightglue_matches_object: usize,
    /// RANSAC inlier 阈值（像素）
    pub ransac_threshold: f32,
    /// 空间 bbox 重叠阈值
    pub min_bbox_overlap: f32,
    /// 描述子最小平均相似度（early exit）
    pub min_desc_similarity: f32,
    /// VLAD 聚类数（k-means k）
    pub vlad_k: usize,
    /// 单 query 最多取多少 patches
    pub max_query_patches: usize,
}

impl Default for PatchConfig {
    fn default() -> Self {
        Self {
            hnsw_top_k: 50,
            max_keypoints_per_patch: 256,
            min_lightglue_matches_scene: 15,
            min_lightglue_matches_object: 3,
            ransac_threshold: 4.0,
            min_bbox_overlap: 0.3,
            min_desc_similarity: 0.7,
            vlad_k: 64,
            max_query_patches: 50,
        }
    }
}

/// ROI 提取配置。
///
/// **算法与 ai-next 逐位对齐**（`roi_extractor.rs:18-86`）：
/// - `scales = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]`（用于 multi-scale sliding window）
/// - `min_size = 32`
/// - `stride_ratio = 0.5`
/// - `dynamic_max_rois = (area / 50000).clamp(10, 100)`
/// - `max_rois_per_image = 10`
/// - `min_roi_size = 64`（像素，过小 ROI 跳过）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoiConfig {
    /// multi-scale 比例
    pub scales: Vec<f32>,
    /// 最小 ROI 边长
    pub min_size: u32,
    /// 滑动步长比例（stride = scale * min_size * stride_ratio）
    pub stride_ratio: f32,
    /// 动态 ROI 上限：(area / area_per_roi).clamp(min_dynamic, max_dynamic)
    /// Phase 1+ 不再使用 global area-sorted truncate，保留字段以兼容旧调用
    pub area_per_roi: u32,
    pub min_dynamic_rois: u32,
    pub max_dynamic_rois: u32,
    /// 单图最大 ROI 数（绝对上限；Phase 1 起等于 `sum(quota_per_scale) + 1`）
    pub max_rois_per_image: u32,
    /// 小于此边长的 ROI 跳过
    pub min_roi_size: u32,
    /// Phase 1 新增：每个 scale 的 quota（与 scales 平行）。保证小 scale 不被大 scale 挤掉
    pub quota_per_scale: Vec<u32>,
    /// Phase 1 新增：scale 内 spatial NMS IoU 阈值（> 此值视为重复）
    pub iou_threshold: f32,
}

impl Default for RoiConfig {
    fn default() -> Self {
        let scales = vec![0.10, 0.20, 0.30, 0.50, 0.70, 1.00];
        // 总数约 91 (=20+20+20+15+10+5+1 full_image)
        let quota_per_scale = vec![20, 20, 20, 15, 10, 5];
        Self {
            scales,
            min_size: 32,
            stride_ratio: 0.5,
            area_per_roi: 50_000,
            min_dynamic_rois: 10,
            max_dynamic_rois: 100,
            max_rois_per_image: quota_per_scale.iter().sum::<u32>() + 1,
            min_roi_size: 32,
            quota_per_scale,
            iou_threshold: 0.7,
        }
    }
}

/// 向量配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorConfig {
    /// HNSW ef_construction
    pub ef_construction: usize,
    /// HNSW M
    pub m: usize,
    /// 默认 top_k
    pub default_top_k: usize,
    /// Phase 1.1 新增：搜索时 ef（越大越精确，越慢）。
    /// 默认 64 = 之前硬编码的 `pf_vector::DEFAULT_EF_SEARCH`;改这里
    /// 后,both face/object index 都会按新值搜。
    pub ef_search: usize,
}

impl Default for VectorConfig {
    fn default() -> Self {
        Self {
            ef_construction: 200,
            m: 16,
            default_top_k: 50,
            ef_search: 64,
        }
    }
}

/// 扫描配置。
///
/// **算法与 ai-next 逐位对齐**（`image_filter.rs:8-10, 38` + `scanner.rs:265-269`）：
/// - 最小尺寸 256x256
/// - 最小 bpp 0.05
/// - 最大宽高比 5.0
/// - 颜色熵阈值 3.5
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScannerConfig {
    /// 支持的图片扩展名（小写）
    pub supported_extensions: Vec<String>,
    /// 跳过隐藏文件 / dotfile
    pub skip_hidden: bool,
    /// 最小文件大小（bytes）
    pub min_file_size: u64,
    /// 最大文件大小（bytes）
    pub max_file_size: u64,
    /// 最小宽高（像素）
    pub min_width_height: u32,
    /// 最小 bytes/pixel（低于此视为图标/矢量）
    pub min_bpp: f32,
    /// 最大宽高比
    pub max_aspect_ratio: f32,
    /// 最低颜色熵（Shannon，简化 80x80 采样）
    pub min_color_entropy: f32,
}

impl Default for ScannerConfig {
    fn default() -> Self {
        Self {
            supported_extensions: vec![
                "jpg".into(),
                "jpeg".into(),
                "png".into(),
                "webp".into(),
                "heic".into(),
                "heif".into(),
                "bmp".into(),
                "gif".into(),
                "tiff".into(),
                "tif".into(),
                "avif".into(),
            ],
            skip_hidden: true,
            min_file_size: 1024,           // 1 KB
            max_file_size: 100 * 1024 * 1024, // 100 MB
            min_width_height: 256,
            min_bpp: 0.05,
            max_aspect_ratio: 5.0,
            min_color_entropy: 3.5,
        }
    }
}

/// 身份识别配置（Phase 18: Face + Body 双通道身份系统）。
///
/// 控制是否启用 body REID 和 identity fusion。
/// 两个 flag 都为 true 时才使用双通道决策；否则回退到纯 face 逻辑。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityConfig {
    /// 是否启用 Body Re-ID 通道（需要 body_pipeline + body_index 可用）
    pub enable_body_reid: bool,
    /// 是否启用 Identity Fusion 决策（margin-based fusion + 多证据决策）
    pub enable_identity_fusion: bool,
    /// Phase 26: 拆分后的独立阈值集合（v1）
    pub thresholds: IdentityThresholds,
    /// Phase 29 T5: 分离阈值集合（不同场景使用不同阈值）
    pub thresholds_v2: Option<IdentityThresholdsV2Config>,
    /// Phase 29 T6: 决策策略
    pub decision_policy: DecisionPolicyConfig,
    /// Phase 29 T10: Shadow Mode 配置
    pub shadow_mode: ShadowModeConfig,
    /// Phase 30 T3: 执行模式
    pub execution_mode: IdentityExecutionMode,
}

impl Default for IdentityConfig {
    fn default() -> Self {
        Self {
            enable_body_reid: false,
            enable_identity_fusion: false,
            thresholds: IdentityThresholds::default(),
            thresholds_v2: None,
            decision_policy: DecisionPolicyConfig::default(),
            shadow_mode: ShadowModeConfig::default(),
            execution_mode: IdentityExecutionMode::Legacy,
        }
    }
}

/// Phase 30 T3: 身份系统执行模式
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum IdentityExecutionMode {
    /// Legacy pipeline only - 不运行新 pipeline
    Legacy,
    /// Shadow mode - Legacy 是正式结果，New Pipeline 同时运行只记录
    Shadow,
    /// 新 pipeline 作为正式结果（实验模式，暂不启用）
    NewPipeline,
}

impl IdentityExecutionMode {
    pub fn is_shadow(&self) -> bool {
        matches!(self, IdentityExecutionMode::Shadow)
    }

    pub fn should_write_production(&self) -> bool {
        matches!(self, IdentityExecutionMode::Legacy | IdentityExecutionMode::NewPipeline)
    }
}

impl Default for IdentityExecutionMode {
    fn default() -> Self {
        IdentityExecutionMode::Legacy
    }
}

/// Phase 26: 独立的身份判断阈值
///
/// 关键设计原则：
/// - face_pairwise_threshold: 直接 Face↔Face 验证（传统方式）
/// - face_prototype_threshold: Query Face ↔ Person Prototype（LOO校准后）
/// - 这两个阈值不能混用！
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityThresholds {
    /// Face ↔ Face 直接验证阈值（传统 pairwise matching）
    /// Phase 25.2 确认：0.75 适用于 direct pairwise
    pub face_pairwise_threshold: f32,
    /// Query Face ↔ Person Prototype 阈值
    /// Phase 25.2 校准：positive median = 0.584，初始值设为 0.55
    pub face_prototype_threshold: f32,
    /// Body ↔ Body 直接验证阈值
    pub body_pairwise_threshold: f32,
    /// Query Body ↔ Person Prototype 阈值
    pub body_prototype_threshold: f32,
    /// 身份决策 margin 阈值：best - second >= margin
    pub identity_margin_threshold: f32,
    /// 模糊决策 margin 阈值：用于 ambiguous 判断
    pub ambiguity_margin_threshold: f32,
    /// Prototype strategy: 0=Mean, 1=NormalizedMean, 2=Medoid, 3+=MultiPrototype K
    pub prototype_strategy: usize,
}

impl Default for IdentityThresholds {
    fn default() -> Self {
        Self {
            // Pairwise thresholds (calibrated for direct matching)
            face_pairwise_threshold: 0.75,
            face_prototype_threshold: 0.55,  // Phase 25.2 calibration starting point
            body_pairwise_threshold: 0.70,
            body_prototype_threshold: 0.60,
            // Decision margins
            identity_margin_threshold: 0.03,
            ambiguity_margin_threshold: 0.02,
            // Default: Mean prototype
            prototype_strategy: 0,
        }
    }
}

// ============================================================================
// Phase 29 T5: Threshold Separation Config
// ============================================================================

/// Phase 29 T5: 分离阈值配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityThresholdsV2Config {
    pub pairwise_face_threshold: f32,
    pub pairwise_body_threshold: f32,
    pub prototype_face_threshold: f32,
    pub prototype_body_threshold: f32,
    pub auto_assign_face_threshold: f32,
    pub auto_assign_body_threshold: f32,
    pub review_face_threshold: f32,
    pub cluster_merge_face_threshold: f32,
    pub cluster_merge_body_threshold: f32,
    pub face_margin_threshold: f32,
    pub body_margin_threshold: f32,
    pub ambiguity_margin_threshold: f32,
}

impl Default for IdentityThresholdsV2Config {
    fn default() -> Self {
        Self {
            pairwise_face_threshold: 0.75,
            pairwise_body_threshold: 0.70,
            prototype_face_threshold: 0.55,
            prototype_body_threshold: 0.60,
            auto_assign_face_threshold: 0.55,
            auto_assign_body_threshold: 0.60,
            review_face_threshold: 0.50,
            cluster_merge_face_threshold: 0.75,
            cluster_merge_body_threshold: 0.70,
            face_margin_threshold: 0.03,
            body_margin_threshold: 0.03,
            ambiguity_margin_threshold: 0.02,
        }
    }
}

// ============================================================================
// Phase 29 T6: Decision Policy Config
// ============================================================================

/// Phase 29 T6: 决策策略配置
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DecisionPolicyConfig {
    Auto,
    Review,
    Unknown,
}

impl Default for DecisionPolicyConfig {
    fn default() -> Self {
        DecisionPolicyConfig::Auto
    }
}

// ============================================================================
// Phase 29 T10: Shadow Mode Config
// ============================================================================

/// Phase 29 T10: Shadow Mode 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowModeConfig {
    pub enabled: bool,
    pub log_all_decisions: bool,
    /// Phase 31.5 T5: Shadow Promotion Policy
    pub promotion_policy: ShadowPromotionPolicy,
}

/// Shadow Promotion Policy - determines when to promote Shadow → NewPipeline
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ShadowPromotionPolicy {
    /// Auto-promote when health gate passes
    pub auto_promote: bool,
    /// Minimum decision agreement rate to allow promotion (e.g., 0.98 = 98%)
    pub min_decision_agreement: f32,
    /// Maximum different person rate to allow promotion (e.g., 0.02 = 2%)
    pub max_different_person_rate: f32,
    /// Minimum sample count before evaluating promotion
    pub min_samples_for_evaluation: usize,
    /// Cooldown period (in evaluations) between promotion checks
    pub promotion_cooldown_evaluations: usize,
}

impl Default for ShadowPromotionPolicy {
    fn default() -> Self {
        Self {
            auto_promote: false,
            min_decision_agreement: 0.98,
            max_different_person_rate: 0.02,
            min_samples_for_evaluation: 500,
            promotion_cooldown_evaluations: 100,
        }
    }
}

impl Default for ShadowModeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            log_all_decisions: false,
            promotion_policy: ShadowPromotionPolicy::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_sane() {
        let c = Config::default();
        assert!(c.face.min_quality > 0.0 && c.face.min_quality < 1.0);
        assert!(c.face.cluster_threshold > 0.0 && c.face.cluster_threshold < 1.0);
        assert!(c.face.similarity_threshold > 0.0 && c.face.similarity_threshold < 1.0);
        assert!(!c.scanner.supported_extensions.is_empty());
        assert!(!c.roi.scales.is_empty());
    }

    /// 与 legacy ai-next 0.65 阈值对齐
    #[test]
    fn cluster_threshold_matches_ai_next() {
        let c = Config::default();
        assert!((c.face.cluster_threshold - 0.65).abs() < 1e-6);
    }

    /// 与 legacy ai-next 0.50 阈值对齐
    #[test]
    fn similarity_threshold_matches_ai_next() {
        let c = Config::default();
        assert!((c.face.similarity_threshold - 0.50).abs() < 1e-6);
    }

    /// Phase 1.2 新增：face 粗筛 fetch_k 默认值（与之前硬编码 `fetch_k = top_k * 4` 对齐）。
    #[test]
    fn face_phase1_2_fetch_defaults() {
        let c = Config::default();
        assert_eq!(c.face.coarse_fetch_multiplier, 4);
        assert_eq!(c.face.coarse_fetch_min, 4);
        assert!((c.face.fine_match_min_embedding - 0.0).abs() < 1e-6);
    }

    /// 与 legacy object_search fusion 权重对齐
    #[test]
    fn object_fusion_weights_match_ai_next() {
        let c = Config::default();
        assert!((c.object.fusion_alpha - 0.3).abs() < 1e-6);
        assert!((c.object.fusion_beta - 0.5).abs() < 1e-6);
        assert!((c.object.fusion_gamma - 0.2).abs() < 1e-6);
        assert!((c.object.min_bbox_overlap - 0.2).abs() < 1e-6);
        assert!((c.object.min_score - 0.5).abs() < 1e-6);
    }

    /// Phase 3 新增：粗筛 fetch_k / roi_type 调权 默认值（与 ai-next 对齐）
    #[test]
    fn object_phase3_defaults() {
        let c = Config::default();
        // HNSW 粗筛：与 ai-next `fetch_k=max(top_k*20, 200)` 对齐
        assert_eq!(c.object.coarse_fetch_multiplier, 20);
        assert_eq!(c.object.coarse_fetch_min, 200);
        // ROI 调权：FullImage general 信号 < SlidingWindow specific 信号
        assert!(c.object.full_image_weight < c.object.sliding_window_weight);
        assert!((c.object.full_image_weight - 0.85).abs() < 1e-6);
        assert!((c.object.sliding_window_weight - 1.0).abs() < 1e-6);
    }

    /// Phase 5 新增：LightGlue 最小匹配数（与 patch 场景模式 15 对齐，对象模式更宽松 = 3）
    #[test]
    fn object_phase5_default() {
        let c = Config::default();
        assert_eq!(c.object.lightglue_min_matches, 3);
    }

    /// Phase 7 新增：粗筛 embedding 阈值 + 提升 hnsw_top_k 默认值（让更多 candidate 经过 LightGlue）
    #[test]
    fn object_phase7_defaults() {
        let c = Config::default();
        assert!((c.object.fine_match_min_embedding - 0.3).abs() < 1e-6);
        assert_eq!(c.object.hnsw_top_k, 50);
    }

    /// 与 legacy patch_search 默认值对齐
    #[test]
    fn patch_config_matches_ai_next() {
        let c = Config::default();
        assert_eq!(c.patch.vlad_k, 64);
        assert_eq!(c.patch.max_keypoints_per_patch, 256);
        assert_eq!(c.patch.min_lightglue_matches_scene, 15);
        assert_eq!(c.patch.min_lightglue_matches_object, 3);
        assert!((c.patch.ransac_threshold - 4.0).abs() < 1e-6);
        assert!((c.patch.min_bbox_overlap - 0.3).abs() < 1e-6);
    }

    /// 与 legacy image_filter 阈值对齐
    #[test]
    fn scanner_config_matches_ai_next() {
        let c = Config::default();
        assert_eq!(c.scanner.min_width_height, 256);
        assert!((c.scanner.min_bpp - 0.05).abs() < 1e-6);
        assert!((c.scanner.max_aspect_ratio - 5.0).abs() < 1e-6);
        assert!((c.scanner.min_color_entropy - 3.5).abs() < 1e-6);
    }
}
