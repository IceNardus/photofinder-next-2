//! 人脸相关 trait。

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use pf_core::{BBox, Embedding, FaceKeypoints, ModelVersion};

use crate::error::AIError;
use crate::image_data::ImageData;

/// 人脸关键点（5 点：双眼 / 鼻 / 双嘴角）。
pub type Keypoints = FaceKeypoints;

/// Detector 来源 — cascade 启用时追踪是哪台 detector(s) 检出了这张脸。
///
/// `Secondary` = "10G-only recovered" — SCRFD 500M 漏检、cascade 救回来的人脸。
/// `Both` = 两台都检出、IoU dedup 后保留高分。
/// 关闭 cascade 时所有 face 均为 `Primary`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectorOrigin {
    /// 仅 SCRFD 500M 检出。
    Primary,
    /// 仅 SCRFD 10G 检出 (cascade 救回)。
    Secondary,
    /// 两台都检出 (dedup'd，保留高分)。
    Both,
}

impl DetectorOrigin {
    /// 序列化为 DB 存储格式。
    pub fn as_str(&self) -> &'static str {
        match self {
            DetectorOrigin::Primary => "Primary",
            DetectorOrigin::Secondary => "Secondary",
            DetectorOrigin::Both => "Both",
        }
    }

    /// 从 DB 字符串反序列化。
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "Primary" => Some(Self::Primary),
            "Secondary" => Some(Self::Secondary),
            "Both" => Some(Self::Both),
            _ => None,
        }
    }
}

/// 单次检测结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceDetection {
    /// 边界框
    pub bbox: BBox,
    /// 检测置信度
    pub score: f32,
    /// 关键点
    pub keypoints: Keypoints,
}

/// 对齐后的人脸（112×112 RGB，已中心化）。
#[derive(Debug, Clone)]
pub struct AlignedFace {
    /// 对齐后图像
    pub image: ImageData,
    /// 原始 bbox（用于映射回原图）
    pub original_bbox: BBox,
}

/// 检测 + 对齐 + embedding 后的最终特征。
#[derive(Debug, Clone)]
pub struct FaceFeature {
    /// 检测
    pub detection: FaceDetection,
    /// 512-d embedding
    pub embedding: Embedding,
    /// 5-KPS 估计的头部姿态（yaw / pitch / roll，度；退化时 None）
    pub yaw_pitch_roll: Option<(f32, f32, f32)>,
    /// blur score（Laplacian 方差归一化）
    pub blur_score: f32,
    /// pose score（由 yaw/pitch/roll 推导）
    pub pose_score: f32,
    /// face area score（归一化最小边长）
    pub face_area_score: f32,
}

/// 质量过滤器（阈值从 `pf_config::FaceConfig` 拿）。
#[derive(Debug, Clone)]
pub struct QualityFilter {
    /// 最小 detector score
    pub min_detector_score: f32,
    /// 最小 face width
    pub min_face_size: u32,
    /// 最小 quality
    pub min_quality: f32,
    /// 最大 yaw（度）
    pub max_yaw: f32,
}

impl QualityFilter {
    /// 用 pf_config 默认值构造。
    pub fn from_config(
        min_detector_score: f32,
        min_face_size: u32,
        min_quality: f32,
        max_yaw: f32,
    ) -> Self {
        Self {
            min_detector_score,
            min_face_size,
            min_quality,
            max_yaw,
        }
    }

    /// 是否通过过滤。
    pub fn passes(&self, det: &FaceDetection, quality: f32) -> bool {
        det.score >= self.min_detector_score
            && det.bbox.w >= self.min_face_size as f32
            && quality >= self.min_quality
    }
}

/// 人脸检测器。
#[async_trait]
pub trait FaceDetector: Send + Sync {
    /// 检测。
    async fn detect(&self, image: &ImageData) -> Result<Vec<FaceDetection>, AIError>;

    /// 输入尺寸（部分 detector 是固定的，如 SCRFD 640×640）。
    fn input_size(&self) -> (u32, u32);

    /// 模型版本。
    fn model_version(&self) -> ModelVersion;
}

/// 人脸对齐器（基于 5 个关键点仿射变换到 112×112）。
pub trait FaceAligner: Send + Sync {
    /// 对齐。
    fn align(
        &self,
        image: &ImageData,
        keypoints: &Keypoints,
    ) -> Result<AlignedFace, AIError>;

    /// 输出尺寸。
    fn output_size(&self) -> (u32, u32) {
        (112, 112)
    }
}

/// 人脸 embedder（ArcFace 等）。
#[async_trait]
pub trait FaceEmbedder: Send + Sync {
    /// 推理。
    async fn embed(&self, aligned: &AlignedFace) -> Result<Embedding, AIError>;

    /// 向量维度。
    fn dim(&self) -> usize;

    /// 模型版本。
    fn model_version(&self) -> ModelVersion;
}

/// 人脸 pipeline：detect → align → embed → quality filter。
pub struct FacePipeline {
    detector: Arc<dyn FaceDetector>,
    aligner: Arc<dyn FaceAligner>,
    embedder: Arc<dyn FaceEmbedder>,
    quality_filter: QualityFilter,
}

impl FacePipeline {
    /// 构造。
    pub fn new(
        detector: Arc<dyn FaceDetector>,
        aligner: Arc<dyn FaceAligner>,
        embedder: Arc<dyn FaceEmbedder>,
        quality_filter: QualityFilter,
    ) -> Self {
        Self {
            detector,
            aligner,
            embedder,
            quality_filter,
        }
    }

    /// 处理一张图片，返回所有通过的 face feature。
    /// 与 ai-next 完全一致：align → embed → quality assess → filter
    pub async fn process(&self, image: &ImageData) -> Result<Vec<FaceFeature>, AIError> {
        use crate::quality;
        let detections = self.detector.detect(image).await?;
        debug!("SCRFD detected {} faces", detections.len());
        let mut features = Vec::new();
        for det in detections {
            let yaw_pitch_roll = crate::face::pose::estimate_yaw_pitch_roll(&det.keypoints);
            // 1. 先 crop 到 bbox(扩展 10% margin),再 align 到 112x112。
            //    直接对原图 align 在大图(>2000px)上会让 Umeyama 残差很大,
            //    因为 5 kp 在原图尺度上放大 100x,而浮点精度 + 参考点比例 100x
            //    的不匹配导致采样偏移到背景。crop 后图小 1 个数量级,
            //    算法输出即人脸,中心像素 R>G>B(肤色)而非绿色 blob。
            //
            //    BUG #4 修复:crop 失败不再 panic,改用 `continue` 跳过该 frame
            //    (bbox 紧贴图像右边缘 / 下边缘时 `cw/ch` 可能下溢到 0 或负数)。
            let (align_input, shifted_kps) = match crop_to_bbox_with_margin(image, &det) {
                Ok(v) => v,
                Err(e) => {
                    debug!(error = %e, "skip frame: invalid crop bbox");
                    continue;
                }
            };
            // 2. Align face to 112x112
            let aligned = self.aligner.align(&align_input, &shifted_kps)?;
            // 3. ArcFace embed (先生成 embedding，再评估质量)
            let embedding = match self.embedder.embed(&aligned).await {
                Ok(e) => e,
                Err(e) => {
                    debug!("embedding failed: {}", e);
                    continue;
                }
            };
            // 4. Calculate quality metrics (对齐后人脸图计算 blur)
            let gray = aligned.image.to_luma8();
            let blur = quality::blur_score_from_aligned(&gray);
            let q = quality::assess(&det, blur, yaw_pitch_roll);
            // 5. Filter by quality (与 ai-next 阈值一致)
            if !q.passes() {
                debug!("face filtered by quality: detector={}, face_area={}, eye_dist={}, pose={}, quality={}",
                    q.detector_score, q.face_area_score, q.eye_distance, q.pose_score, q.quality);
                continue;
            }
            if det.score < self.quality_filter.min_detector_score {
                debug!("face filtered by detector score: {}", det.score);
                continue;
            }
            features.push(FaceFeature {
                detection: det,
                embedding,
                yaw_pitch_roll,
                blur_score: q.blur_score,
                pose_score: q.pose_score,
                face_area_score: q.face_area_score,
            });
        }
        Ok(features)
    }
}

/// 把 image 按 bbox crop 出来(加 10% margin),并把 kp 平移到 cropped 坐标。
/// 在 cropped 图上 align 让 Umeyama 的数值尺度合理(原图上 kp 数值是
/// 几百到几千,cropped 后是几十到几百,精度与 ref 匹配)。
///
/// BUG #4 修复:crop bbox 越界 / 宽高下溢时返回 `Err`,由 caller 选择跳过该帧,
/// 而非 `expect("crop")` 让进程 panic。
fn crop_to_bbox_with_margin(
    image: &ImageData,
    det: &FaceDetection,
) -> Result<(ImageData, FaceKeypoints), AIError> {
    let b = det.bbox;
    let w = b.w.max(1.0);
    let h = b.h.max(1.0);
    let mx = w * 0.1;
    let my = h * 0.1;
    let x = (b.x - mx).max(0.0);
    let y = (b.y - my).max(0.0);
    let cw = (w + 2.0 * mx).min(image.width() as f32 - x);
    let ch = (h + 2.0 * my).min(image.height() as f32 - y);
    if cw < 1.0 || ch < 1.0 {
        return Err(AIError::Preprocess(format!(
            "crop bbox too small: cw={cw:.1} ch={ch:.1} (image {}x{})",
            image.width(),
            image.height()
        )));
    }
    let crop_bbox = BBox::new(x, y, cw, ch);
    let cropped = image.crop(crop_bbox).map_err(|e| AIError::Preprocess(format!("crop: {e}")))?;
    let shifted = FaceKeypoints {
        left_eye: (det.keypoints.left_eye.0 - x, det.keypoints.left_eye.1 - y),
        right_eye: (det.keypoints.right_eye.0 - x, det.keypoints.right_eye.1 - y),
        nose: (det.keypoints.nose.0 - x, det.keypoints.nose.1 - y),
        left_mouth: (det.keypoints.left_mouth.0 - x, det.keypoints.left_mouth.1 - y),
        right_mouth: (det.keypoints.right_mouth.0 - x, det.keypoints.right_mouth.1 - y),
    };
    Ok((cropped, shifted))
}