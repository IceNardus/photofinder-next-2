//! SCRFD face detector。
//!
//! **算法参数与 ai-next/src-tauri/src/ai/face/detector.rs **逐位对齐**：
//!
//! | 常量 | 值 |
//! |---|---|
//! | `INPUT_SIZE` | 640 |
//! | `STRIDES` | `[8, 16, 32]` |
//! | `MIN_CONFIDENCE` (probability) | `0.30` |
//! | `MIN_RAW_SCORE` (raw) | `0.05` |
//! | `MIN_FACE_W` (orig pixels) | `24` |
//! | `MIN_FACE_H` (orig pixels) | `24` |
//! | `NMS_IOU_THRESHOLD` | `0.4` |
//! | 通道顺序 | BGR |
//! | 归一化 | `(x - 127.5) / 128.0` |
//! | Letterbox | 保持比例 + 黑边 |
//!
//! 输出布局（SCRFD-500M-BNKPS）：
//! - outputs[0..3] → stride 8/16/32 的 score
//! - outputs[3..6] → bbox
//! - outputs[6..9] → 5 个关键点

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use image::{imageops::FilterType, GenericImageView, Rgb, RgbImage};
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;
use tracing::info;

use pf_core::{BBox, FaceKeypoints, ModelVersion};

use crate::error::AIError;
use crate::face::traits::{FaceDetection, FaceDetector};
use crate::image_data::ImageData;

/// 推理输入尺寸。
pub const INPUT_SIZE: i32 = 640;
/// 三个 FPN 步长。
pub const STRIDES: [f32; 3] = [8.0, 16.0, 32.0];
/// probability 阈值（sigmoid 后）。
pub const MIN_CONFIDENCE: f32 = 0.30;
/// 极低 raw 阈值（未 sigmoid，直接丢弃）。
pub const MIN_RAW_SCORE: f32 = 0.05;
/// 最小人脸宽（letterbox 画布像素）。
pub const MIN_FACE_W: f32 = 16.0;
/// 最小人脸高（letterbox 画布像素）。
pub const MIN_FACE_H: f32 = 16.0;
/// IoU 阈值（enhanced soft-NMS）。
pub const NMS_IOU_THRESHOLD: f32 = 0.4;
/// NMS 后的最小保留分。
pub const NMS_KEEP_MIN: f32 = 0.10;

/// 内部人脸框（letterbox 坐标）。
#[derive(Debug, Clone)]
pub(crate) struct FaceBBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

/// 5 个关键点（letterbox 坐标）。
#[derive(Debug, Clone)]
pub(crate) struct FiveKeypoints {
    pub left_eye: (f32, f32),
    pub right_eye: (f32, f32),
    pub nose: (f32, f32),
    pub left_mouth: (f32, f32),
    pub right_mouth: (f32, f32),
}

/// 加载好的 SCRFD 模型。
pub struct ScrfdDetector {
    // BUG #5 修复:`Arc<Mutex<...>>` 让 detect 可以把 session clone 到 spawn_blocking 里,
    // 否则 ort 的同步 session.run() 会占住 tokio runtime 线程 50-500ms,
    // 在多线程 runtime 下阻塞 IPC / 其他 executor 的调度。
    session: Arc<Mutex<Option<Session>>>,
    version: ModelVersion,
}

impl ScrfdDetector {
    /// 从模型文件加载。
    pub fn load(model_path: &Path) -> Result<Arc<Self>, AIError> {
        info!("Loading SCRFD from: {}", model_path.display());
        let session = Session::builder()
            .map_err(|e| AIError::ModelLoad(format!("session builder: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| AIError::ModelLoad(format!("commit_from_file: {e}")))?;
        Ok(Arc::new(Self {
            session: Arc::new(Mutex::new(Some(session))),
            version: ModelVersion::new("scrfd-500m-bnkps@v1.0.0"),
        }))
    }

    /// 测试用：从已有 Session 构造（暴露给 unit test）。
    #[cfg(test)]
    pub(crate) fn from_session(session: Session) -> Arc<Self> {
        Arc::new(Self {
            session: Arc::new(Mutex::new(Some(session))),
            version: ModelVersion::new("scrfd-500m-bnkps@v1.0.0"),
        })
    }

    /// letterbox resize → BGR CHW 浮点 tensor。
    fn preprocess(image: &image::DynamicImage) -> (Vec<f32>, f32, f32, u32, u32) {
        let (orig_w, orig_h) = image.dimensions();
        let scale =
            (INPUT_SIZE as f32 / orig_w as f32).min(INPUT_SIZE as f32 / orig_h as f32);
        let new_w = ((orig_w as f32) * scale) as u32;
        let new_h = ((orig_h as f32) * scale) as u32;

        let resized = image.resize_exact(new_w, new_h, FilterType::Triangle);
        let mut canvas = RgbImage::from_pixel(INPUT_SIZE as u32, INPUT_SIZE as u32, Rgb([0, 0, 0]));
        let pad_x = ((INPUT_SIZE as u32 - new_w) / 2) as f32;
        let pad_y = ((INPUT_SIZE as u32 - new_h) / 2) as f32;
        image::imageops::overlay(&mut canvas, &resized.to_rgb8(), pad_x as i64, pad_y as i64);

        let mut input_data =
            Vec::with_capacity(3 * INPUT_SIZE as usize * INPUT_SIZE as usize);
        // BGR CHW, (x - 127.5) / 128
        for y in 0..INPUT_SIZE as u32 {
            for x in 0..INPUT_SIZE as u32 {
                let pixel = canvas.get_pixel(x, y);
                input_data.push((pixel[2] as f32 - 127.5) / 128.0);
            }
        }
        for y in 0..INPUT_SIZE as u32 {
            for x in 0..INPUT_SIZE as u32 {
                let pixel = canvas.get_pixel(x, y);
                input_data.push((pixel[1] as f32 - 127.5) / 128.0);
            }
        }
        for y in 0..INPUT_SIZE as u32 {
            for x in 0..INPUT_SIZE as u32 {
                let pixel = canvas.get_pixel(x, y);
                input_data.push((pixel[0] as f32 - 127.5) / 128.0);
            }
        }
        (input_data, pad_x, pad_y, new_w, new_h)
    }

    /// Enhanced Soft-NMS（与 ai-next 完全一致）。
    pub(crate) fn soft_nms(faces: &mut Vec<(FaceBBox, FiveKeypoints)>) -> Vec<usize> {
        if faces.is_empty() {
            return Vec::new();
        }
        faces.sort_by(|a, b| b.0.score.partial_cmp(&a.0.score).unwrap());
        let mut scores_soft: Vec<f32> = faces.iter().map(|f| f.0.score).collect();
        let mut keep = Vec::new();

        for i in 0..faces.len() {
            if scores_soft[i] < NMS_KEEP_MIN {
                continue;
            }
            keep.push(i);
            for j in (i + 1)..faces.len() {
                if scores_soft[j] < NMS_KEEP_MIN {
                    continue;
                }
                let a_bbox = &faces[i].0;
                let b_bbox = &faces[j].0;
                let a_kps = &faces[i].1;
                let b_kps = &faces[j].1;

                let inter_x1 = a_bbox.x1.max(b_bbox.x1);
                let inter_y1 = a_bbox.y1.max(b_bbox.y1);
                let inter_x2 = a_bbox.x2.min(b_bbox.x2);
                let inter_y2 = a_bbox.y2.min(b_bbox.y2);
                let inter_area =
                    (inter_x2 - inter_x1).max(0.0) * (inter_y2 - inter_y1).max(0.0);
                let area_a = (a_bbox.x2 - a_bbox.x1) * (a_bbox.y2 - a_bbox.y1);
                let area_b = (b_bbox.x2 - b_bbox.x1) * (b_bbox.y2 - b_bbox.y1);
                let iou = inter_area / (area_a + area_b - inter_area).max(1e-6);

                let a_center_x = (a_kps.left_eye.0 + a_kps.right_eye.0) / 2.0;
                let a_center_y = (a_kps.left_eye.1 + a_kps.right_eye.1) / 2.0;
                let b_center_x = (b_kps.left_eye.0 + b_kps.right_eye.0) / 2.0;
                let b_center_y = (b_kps.left_eye.1 + b_kps.right_eye.1) / 2.0;
                let kps_dist =
                    ((a_center_x - b_center_x).powi(2) + (a_center_y - b_center_y).powi(2))
                        .sqrt();

                let scale_a = (a_bbox.x2 - a_bbox.x1).max(a_bbox.y2 - a_bbox.y1);
                let scale_b = (b_bbox.x2 - b_bbox.x1).max(b_bbox.y2 - b_bbox.y1);
                let scale_ratio = scale_a / scale_b.max(1.0);

                let is_duplicate = iou > 0.35
                    || (iou > 0.2 && kps_dist < 25.0)
                    || (iou > 0.15 && kps_dist < 15.0 && scale_ratio < 1.3);

                if is_duplicate {
                    scores_soft[j] *= (1.0 - iou * 1.8).max(0.03);
                    if kps_dist < 15.0 {
                        scores_soft[j] = scores_soft[j].min(0.05);
                    }
                } else if iou > 0.25 {
                    scores_soft[j] *= (1.0 - iou * 1.5).max(0.05);
                }
            }
        }
        keep
    }
}

#[async_trait]
impl FaceDetector for ScrfdDetector {
    async fn detect(&self, image: &ImageData) -> Result<Vec<FaceDetection>, AIError> {
        let img = image.inner().clone();
        let session = self.session.clone();

        // BUG #5 修复:ort 的 session.run 是同步阻塞调用,直接放在 async fn 内会占住
        // tokio runtime 线程 50-500ms,在多线程 runtime 下阻塞 IPC / 其他 executor。
        // 整个推理 + 后处理挪到 spawn_blocking 里,让出 async 调度权。
        tokio::task::spawn_blocking(move || Self::detect_sync(session, img))
            .await
            .map_err(|e| AIError::Inference(format!("join detect: {e}")))?
    }

    fn input_size(&self) -> (u32, u32) {
        (INPUT_SIZE as u32, INPUT_SIZE as u32)
    }

    fn model_version(&self) -> ModelVersion {
        self.version.clone()
    }
}

impl ScrfdDetector {
    /// 同步版推理:在 spawn_blocking 里执行。独占 session 锁期间做完整 detection。
    fn detect_sync(
        session: Arc<Mutex<Option<Session>>>,
        img: image::DynamicImage,
    ) -> Result<Vec<FaceDetection>, AIError> {
        let (orig_w, orig_h) = img.dimensions();
        let (input_data, pad_x, pad_y, new_w, new_h) = Self::preprocess(&img);

        let shape = [1_i64, 3, INPUT_SIZE as i64, INPUT_SIZE as i64];
        let input = Tensor::from_array((shape, input_data))
            .map_err(|e| AIError::ModelLoad(format!("tensor: {e}")))?;

        let mut session_guard = session.lock();
        let session_ref = session_guard
            .as_mut()
            .ok_or_else(|| AIError::ModelLoad("SCRFD session not loaded".into()))?;
        let outputs = session_ref
            .run(ort::inputs![input])
            .map_err(|e| AIError::ModelLoad(format!("run: {e}")))?;

        let mut raw: Vec<(FaceBBox, FiveKeypoints)> = Vec::new();

        for scale_idx in 0..3 {
            let score_data = outputs[scale_idx]
                .try_extract_tensor::<f32>()
                .map_err(|e| AIError::ModelLoad(format!("score[{scale_idx}]: {e}")))?
                .1;
            let bbox_data = outputs[scale_idx + 3]
                .try_extract_tensor::<f32>()
                .map_err(|e| AIError::ModelLoad(format!("bbox[{scale_idx}]: {e}")))?
                .1;
            let kps_data = outputs[scale_idx + 6]
                .try_extract_tensor::<f32>()
                .map_err(|e| AIError::ModelLoad(format!("kps[{scale_idx}]: {e}")))?
                .1;

            let stride = STRIDES[scale_idx];
            let grid_w = INPUT_SIZE as usize / stride as usize;

            for i in 0..score_data.len() {
                let raw_score = score_data[i];
                if raw_score < MIN_RAW_SCORE {
                    continue;
                }
                let prob = raw_score; // 模型已输出概率，不再次 sigmoid
                if prob < MIN_CONFIDENCE {
                    continue;
                }

                // dual anchor
                let pos_idx = i / 2;
                let row = pos_idx / grid_w;
                let col = pos_idx % grid_w;
                let anchor_cx = (col as f32 + 0.5) * stride;
                let anchor_cy = (row as f32 + 0.5) * stride;

                let bbox_idx = i * 4;
                let l = bbox_data[bbox_idx] * stride;
                let t = bbox_data[bbox_idx + 1] * stride;
                let r = bbox_data[bbox_idx + 2] * stride;
                let b = bbox_data[bbox_idx + 3] * stride;

                let img_x1 = (anchor_cx - l).max(0.0).min(INPUT_SIZE as f32);
                let img_y1 = (anchor_cy - t).max(0.0).min(INPUT_SIZE as f32);
                let img_x2 = (anchor_cx + r).max(0.0).min(INPUT_SIZE as f32);
                let img_y2 = (anchor_cy + b).max(0.0).min(INPUT_SIZE as f32);

                let face_w = img_x2 - img_x1;
                let face_h = img_y2 - img_y1;
                if face_w < MIN_FACE_W || face_h < MIN_FACE_H {
                    continue;
                }

                // 检查是否在 letterbox 内（非 padding）
                let is_in_letterbox = img_x1 >= pad_x
                    && img_y1 >= pad_y
                    && img_x2 <= pad_x + new_w as f32
                    && img_y2 <= pad_y + new_h as f32;
                if !is_in_letterbox {
                    continue;
                }

                // 映射到原图
                let img_x1_orig =
                    ((img_x1 - pad_x) / new_w as f32 * orig_w as f32).max(0.0).min(orig_w as f32);
                let img_y1_orig = ((img_y1 - pad_y) / new_h as f32 * orig_h as f32)
                    .max(0.0)
                    .min(orig_h as f32);
                let img_x2_orig =
                    ((img_x2 - pad_x) / new_w as f32 * orig_w as f32).max(0.0).min(orig_w as f32);
                let img_y2_orig = ((img_y2 - pad_y) / new_h as f32 * orig_h as f32)
                    .max(0.0)
                    .min(orig_h as f32);

                let kps_offset = i * 10;
                let kps_raw = &kps_data[kps_offset..kps_offset + 10];
                let kps_scale_x = orig_w as f32 / new_w as f32;
                let kps_scale_y = orig_h as f32 / new_h as f32;
                // KPS offset 也需按 kps_scale 缩放 — 否则 5 个关键点全挤在 anchor 周围 ±stride 像素内
                // （bbox 正确是因为 l/t/r/b*stride 后整段被 (img - pad) / new_w * orig_w 缩放过一次）
                let kps = FiveKeypoints {
                    left_eye: (
                        (anchor_cx - pad_x) * kps_scale_x + kps_raw[0] * stride * kps_scale_x,
                        (anchor_cy - pad_y) * kps_scale_y + kps_raw[1] * stride * kps_scale_y,
                    ),
                    right_eye: (
                        (anchor_cx - pad_x) * kps_scale_x + kps_raw[2] * stride * kps_scale_x,
                        (anchor_cy - pad_y) * kps_scale_y + kps_raw[3] * stride * kps_scale_y,
                    ),
                    nose: (
                        (anchor_cx - pad_x) * kps_scale_x + kps_raw[4] * stride * kps_scale_x,
                        (anchor_cy - pad_y) * kps_scale_y + kps_raw[5] * stride * kps_scale_y,
                    ),
                    left_mouth: (
                        (anchor_cx - pad_x) * kps_scale_x + kps_raw[6] * stride * kps_scale_x,
                        (anchor_cy - pad_y) * kps_scale_y + kps_raw[7] * stride * kps_scale_y,
                    ),
                    right_mouth: (
                        (anchor_cx - pad_x) * kps_scale_x + kps_raw[8] * stride * kps_scale_x,
                        (anchor_cy - pad_y) * kps_scale_y + kps_raw[9] * stride * kps_scale_y,
                    ),
                };

                // KPS inside bbox? 若否降低 score（与 ai-next 一致）
                let in_bbox = kps.left_eye.0 >= img_x1_orig - 50.0
                    && kps.left_eye.0 <= img_x2_orig + 50.0
                    && kps.left_eye.1 >= img_y1_orig - 50.0
                    && kps.left_eye.1 <= img_y2_orig + 50.0
                    && kps.right_eye.0 >= img_x1_orig - 50.0
                    && kps.right_eye.0 <= img_x2_orig + 50.0
                    && kps.right_eye.1 >= img_y1_orig - 50.0
                    && kps.right_eye.1 <= img_y2_orig + 50.0;
                let adjusted_score = if in_bbox { prob } else { prob * 0.7 };

                raw.push((
                    FaceBBox {
                        x1: img_x1_orig,
                        y1: img_y1_orig,
                        x2: img_x2_orig,
                        y2: img_y2_orig,
                        score: adjusted_score,
                    },
                    kps,
                ));
            }
        }

        let keep = Self::soft_nms(&mut raw);

        let mut detections = Vec::new();
        for &i in &keep {
            let (bbox, kps) = &raw[i];
            let x = bbox.x1;
            let y = bbox.y1;
            let w = bbox.x2 - bbox.x1;
            let h = bbox.y2 - bbox.y1;
            let keypoints = FaceKeypoints {
                left_eye: kps.left_eye,
                right_eye: kps.right_eye,
                nose: kps.nose,
                left_mouth: kps.left_mouth,
                right_mouth: kps.right_mouth,
            };
            detections.push(FaceDetection {
                bbox: BBox::new(x, y, w, h),
                score: bbox.score,
                keypoints,
            });
        }
        Ok(detections)
    }

    fn input_size(&self) -> (u32, u32) {
        (INPUT_SIZE as u32, INPUT_SIZE as u32)
    }

    fn model_version(&self) -> ModelVersion {
        self.version.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fb(x1: f32, y1: f32, x2: f32, y2: f32, s: f32) -> FaceBBox {
        FaceBBox {
            x1,
            y1,
            x2,
            y2,
            score: s,
        }
    }

    fn kps(cx: f32, cy: f32) -> FiveKeypoints {
        FiveKeypoints {
            left_eye: (cx - 5.0, cy - 5.0),
            right_eye: (cx + 5.0, cy - 5.0),
            nose: (cx, cy),
            left_mouth: (cx - 3.0, cy + 5.0),
            right_mouth: (cx + 3.0, cy + 5.0),
        }
    }

    #[test]
    fn constants_match_ai_next() {
        assert_eq!(INPUT_SIZE, 640);
        assert_eq!(STRIDES, [8.0, 16.0, 32.0]);
        assert!((MIN_CONFIDENCE - 0.30).abs() < 1e-6);
        assert!((MIN_RAW_SCORE - 0.05).abs() < 1e-6);
        assert!((MIN_FACE_W - 16.0).abs() < 1e-6);
        assert!((MIN_FACE_H - 16.0).abs() < 1e-6);
        assert!((NMS_IOU_THRESHOLD - 0.4).abs() < 1e-6);
    }

    #[test]
    fn nms_keeps_non_overlapping_boxes() {
        // 两个不重叠的 box + KPS 较远 → 互不影响
        let mut faces = vec![
            (fb(0.0, 0.0, 50.0, 50.0, 0.9), kps(25.0, 25.0)),
            (fb(500.0, 500.0, 550.0, 550.0, 0.7), kps(525.0, 525.0)),
        ];
        let keep = ScrfdDetector::soft_nms(&mut faces);
        assert_eq!(keep, vec![0, 1]);
        assert!((faces[0].0.score - 0.9).abs() < 1e-6);
        assert!((faces[1].0.score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn nms_suppresses_near_duplicate_with_close_kps() {
        // 两个 box 高重叠 + KPS 很近 → 直接压到 0.05（被丢弃）
        let mut faces = vec![
            (fb(0.0, 0.0, 100.0, 100.0, 0.9), kps(50.0, 50.0)),
            (fb(2.0, 2.0, 102.0, 102.0, 0.8), kps(51.0, 51.0)), // KPS dist ≈ 1.41 < 15
        ];
        let keep = ScrfdDetector::soft_nms(&mut faces);
        assert_eq!(keep, vec![0]);
    }

    #[test]
    fn nms_drops_low_score_box() {
        // 低分应该被丢弃
        let mut faces = vec![
            (fb(0.0, 0.0, 100.0, 100.0, 0.9), kps(50.0, 50.0)),
            (fb(50.0, 50.0, 80.0, 80.0, 0.05), kps(65.0, 65.0)),
        ];
        let keep = ScrfdDetector::soft_nms(&mut faces);
        assert_eq!(keep, vec![0]);
    }

    #[test]
    fn nms_keeps_separate_boxes() {
        let mut faces = vec![
            (fb(0.0, 0.0, 50.0, 50.0, 0.9), kps(25.0, 25.0)),
            (fb(200.0, 200.0, 250.0, 250.0, 0.8), kps(225.0, 225.0)),
        ];
        let keep = ScrfdDetector::soft_nms(&mut faces);
        assert_eq!(keep, vec![0, 1]);
        // 两个 box 互不影响 score
        assert!((faces[0].0.score - 0.9).abs() < 1e-6);
        assert!((faces[1].0.score - 0.8).abs() < 1e-6);
    }

    #[test]
    fn nms_empty_input() {
        let mut faces: Vec<(FaceBBox, FiveKeypoints)> = Vec::new();
        let keep = ScrfdDetector::soft_nms(&mut faces);
        assert!(keep.is_empty());
    }

    #[test]
    fn preprocess_bgr_chw_layout() {
        // 800x600（横长方形）→ letterbox 后四周有 padding
        use image::{ImageBuffer, Rgb};
        let buf: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(800, 600, |_, _| Rgb([255, 0, 0]));
        let img = image::DynamicImage::from(buf);
        let (data, pad_x, pad_y, new_w, new_h) = ScrfdDetector::preprocess(&img);
        // 总元素 = 3 * 640 * 640
        assert_eq!(data.len(), 3 * 640 * 640);
        let n = (640 * 640) as usize;
        // 第一个像素 (0,0) 在 padding 区 → R=G=B=0 → 三个通道都是 -0.996
        assert!(data[0] < -0.99 && data[0] > -1.001);
        assert!(data[n] < -0.99);
        assert!(data[2 * n] < -0.99);
        // letterbox 后 image 区域为 (0,80)..(639,559)
        // 读 image 中心 (320, 320) 应该是红色（R 通道 ≈ 0.996）
        let center = (320 * 640 + 320) as usize;
        assert!(data[2 * n + center] > 0.99, "R channel at center");
        assert!(data[n + center] < -0.99, "G channel at center");
        assert!(data[center] < -0.99, "B channel at center");
        // 800x600 → letterbox 缩放到 640x480（保持比例），左右无 padding（new_w=640），上下各 80px 黑边
        assert_eq!(pad_x, 0.0);
        assert!(pad_y > 0.0);
        assert_eq!(new_w, 640); // 800 * 640/800 = 640
        assert_eq!(new_h, 480); // 600 * 640/800 = 480
    }

    #[test]
    fn preprocess_both_axes_have_padding_for_tall_image() {
        // 600x800（竖长方形）→ 上下有 padding
        use image::{ImageBuffer, Rgb};
        let buf: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(600, 800, |_, _| Rgb([255, 0, 0]));
        let img = image::DynamicImage::from(buf);
        let (_, pad_x, pad_y, new_w, new_h) = ScrfdDetector::preprocess(&img);
        // scale = min(640/600, 640/800) = min(1.067, 0.8) = 0.8
        // new_w = 600*0.8 = 480, new_h = 800*0.8 = 640
        assert!(pad_x > 0.0 && pad_y == 0.0);
        assert_eq!(new_w, 480);
        assert_eq!(new_h, 640);
    }
}