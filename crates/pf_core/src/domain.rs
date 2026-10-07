//! 领域类型。
//!
//! 整个 codebase 只允许有这几种类型。旧的 `Rect` / `FaceRect` / `RawFace` 等变体已删除。

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::bbox::BBox;

/// 当前人脸 embedding 模型的 model_name（写入 `face_embeddings.model_name`）。
///
/// 查询（`get_embedding` / `list_unassigned_embeddings`）与写入（ArcFaceEmbedder
/// 的 `ModelVersion`）都从它派生,避免 model_name 漂移。历史教训：查询曾写死
/// `"arcface"`,而库里实际存 `"arcface-w600k-r50"`,导致聚类空跑、rebuild 误判。
pub const FACE_MODEL_NAME: &str = "arcface-w600k-r50";

/// 模型版本标识（例如 `"arcface-w600k-r50@v1"`）。
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelVersion(pub String);

impl ModelVersion {
    /// 构造一个新的 ModelVersion。
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// 字符串引用。
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// 模型名称(`"name@version"` 形式中 `@` 之前的部分;若没有 `@` 则返回全串)。
    ///
    /// 例如 `"arcface-w600k-r50@v1.0.0".name() == "arcface-w600k-r50"`。
    pub fn name(&self) -> &str {
        match self.0.find('@') {
            Some(idx) => &self.0[..idx],
            None => &self.0,
        }
    }

    /// 模型版本(`"name@version"` 形式中 `@` 之后的部分;若没有 `@` 则返回 `""`)。
    ///
    /// 例如 `"arcface-w600k-r50@v1.0.0".version() == "v1.0.0"`。
    pub fn version(&self) -> &str {
        match self.0.find('@') {
            Some(idx) => &self.0[idx + 1..],
            None => "",
        }
    }
}

impl std::fmt::Display for ModelVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// 一张图片的元数据（不含像素）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Image {
    /// DB 主键
    pub id: i64,
    /// 文件系统路径（Desktop 用，移动端为 None）
    pub path: String,
    /// BLAKE3 hash（用于去重）
    pub hash: String,
    /// 文件大小（bytes）
    pub size: u64,
    /// 像素宽
    pub width: u32,
    /// 像素高
    pub height: u32,
    /// 拍摄时间（来自 EXIF，可能为 None）
    pub captured_at: Option<DateTime<Utc>>,
    /// 缩略图路径
    pub thumbnail_path: Option<String>,
}

/// 人脸关键点（5 点）。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FaceKeypoints {
    /// 左眼中心
    pub left_eye: (f32, f32),
    /// 右眼中心
    pub right_eye: (f32, f32),
    /// 鼻尖
    pub nose: (f32, f32),
    /// 左嘴角
    pub left_mouth: (f32, f32),
    /// 右嘴角
    pub right_mouth: (f32, f32),
}

/// 一张人脸。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Face {
    /// DB 主键
    pub id: i64,
    /// 所属图片
    pub image_id: i64,
    /// 所属 person（聚类后填）
    pub person_id: Option<i64>,
    /// 边界框
    pub bbox: BBox,
    /// 检测器置信度
    pub detector_score: f32,
    /// 质量分数（用于过滤模糊 / 遮挡）
    pub quality: f32,
    /// Yaw / Pitch / Roll（用于筛选正脸）
    pub yaw_pitch_roll: Option<(f32, f32, f32)>,
    /// 关键点（可选）
    pub keypoints: Option<FaceKeypoints>,
    /// 来自哪个模型版本
    pub model_version: ModelVersion,
}

/// 一个 person（同一身份的所有 face 归并）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Person {
    /// DB 主键
    pub id: i64,
    /// 包含的 face 数
    pub face_count: u32,
    /// 用户命名（可能为 None）
    pub name: Option<String>,
    /// 创建时间
    pub created_at: DateTime<Utc>,
}

/// 一个检测到的对象。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Object {
    /// DB 主键
    pub id: i64,
    /// 所属图片
    pub image_id: i64,
    /// 类别 id（COCO 80 类）
    pub class_id: i32,
    /// 类别名
    pub class_name: String,
    /// 置信度
    pub confidence: f32,
    /// 边界框
    pub bbox: BBox,
    /// 模型版本
    pub model_version: ModelVersion,
}

/// 搜索结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// 命中的图片 id
    pub image_id: i64,
    /// 命中的具体目标（face_id / object_id / patch_id）
    pub target_id: i64,
    /// 相似度分数（越大越相似，cosine 0..1）
    pub score: f32,
    /// 排名（1-based）
    pub rank: usize,
}

/// 对象搜索结果（包含融合评分的详细信息）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectSearchResult {
    /// 命中的图片 id
    pub image_id: i64,
    /// 目标 id（与 image_id 相同）
    pub target_id: i64,
    /// 融合置信度（α*embedding + β*inlier + γ*bbox）
    pub confidence: f32,
    /// 排名（1-based）
    pub rank: usize,
    /// 向量检索分数
    pub embedding_score: f32,
    /// LightGlue 内点率
    pub inlier_ratio: f32,
    /// LightGlue 内点数量
    pub inlier_count: usize,
    /// LightGlue 匹配数量
    pub match_count: usize,
    /// 查询 ROI 与候选区域的 bbox 重叠率（IoU）
    pub bbox_overlap: f32,
    /// 命中的 ROI 区域（xywh，在候选图片像素坐标系，iOS 用于画矩形叠加）
    pub matched_bbox: BBox,
}

/// Embedding 向量。
///
/// 注意：`pf_vector` 看不到 `Embedding`，只看到 `&[f32]`。这层包装只在
/// `pf_core` / `pf_ai` / `pf_application` 之间使用。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// 实际向量
    pub values: Vec<f32>,
    /// 维度（冗余存储，避免每次 len(values)）
    pub dim: usize,
    /// 来源模型版本（用于版本切换时区分）
    pub model: ModelVersion,
}

impl Embedding {
    /// 构造一个新的 embedding。
    pub fn new(values: Vec<f32>, model: ModelVersion) -> Self {
        let dim = values.len();
        Self { values, dim, model }
    }

    /// `&[f32]` 视图（传给 vector index）。
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// L2 范数。
    pub fn norm(&self) -> f32 {
        self.values.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// 归一化（in place）。
    pub fn normalize(&mut self) {
        let n = self.norm();
        if n > 0.0 {
            for v in &mut self.values {
                *v /= n;
            }
        }
    }
}

/// 通用检测（不区分 face / object，由 pf_ai 子模块提供更具体的结构）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    /// 边界框
    pub bbox: BBox,
    /// 置信度
    pub score: f32,
    /// 关键点（可选）
    pub keypoints: Option<Vec<(f32, f32)>>,
}

/// 图片元数据 / 属性 / 标签（可扩展：EXIF、拍摄地点、相机型号等）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metadata {
    /// 相机型号
    pub camera: Option<String>,
    /// GPS 纬度
    pub gps_lat: Option<f64>,
    /// GPS 经度
    pub gps_lon: Option<f64>,
    /// 任何额外 K-V
    pub extra: std::collections::HashMap<String, String>,
}

/// 图片过滤器（用于查询时筛选）。
#[derive(Debug, Clone, Default)]
pub struct ImageFilter {
    /// 仅返回 scan_status = indexed
    pub only_indexed: bool,
    /// 时间范围起点（含）
    pub captured_from: Option<DateTime<Utc>>,
    /// 时间范围终点（含）
    pub captured_to: Option<DateTime<Utc>>,
    /// 至少包含 face 数（与 person search 配合）
    pub min_face_count: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_dim_matches_values() {
        let e = Embedding::new(vec![0.1, 0.2, 0.3], ModelVersion::new("test@v1"));
        assert_eq!(e.dim, 3);
        assert_eq!(e.as_slice().len(), 3);
    }

    #[test]
    fn embedding_norm_and_normalize() {
        let mut e = Embedding::new(vec![3.0, 4.0], ModelVersion::new("test@v1"));
        assert!((e.norm() - 5.0).abs() < 1e-6);
        e.normalize();
        assert!((e.norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bbox_roundtrip_via_xyxy() {
        let b = BBox::new(1.0, 2.0, 3.0, 4.0);
        let arr = b.xyxy();
        let back = BBox::from_xyxy(arr[0], arr[1], arr[2], arr[3]).unwrap();
        assert_eq!(b, back);
    }
}