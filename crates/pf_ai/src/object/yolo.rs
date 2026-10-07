//! YOLOv8n 对象检测器（ONNX 实现）。
//!
//! **算法与 ai-next/src-tauri/src/category_search/detector.rs 逐位对齐**：
//!
//! | 项 | 值 |
//! |---|---|
//! | 输入尺寸 | 640×640 RGB |
//! | 输出格式 | [1, 84, 8400] — 8400 boxes × (4 coords + 80 class scores) |
//! | 坐标系 | center_x, center_y, w, h（需转 xyxy） |
//! | NMS IoU threshold | 0.4 |
//! | 置信度阈值 | 0.25 |
//! | 最小框大小 | 20px |

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use image::imageops::FilterType;
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;
use tracing::info;

use pf_core::{BBox, ModelVersion};

use crate::error::AIError;
use crate::image_data::ImageData;
use crate::object::traits::{ObjectDetection, ObjectDetector};

/// YOLO 输入分辨率。
const INPUT_SIZE: u32 = 640;
/// COCO 类别数。
const NUM_CLASSES: usize = 80;
/// NMS IoU 阈值。
const NMS_IOU_THRESHOLD: f32 = 0.4;
/// 置信度阈值。
const CONFIDENCE_THRESHOLD: f32 = 0.25;
/// 最小检测框（像素）。
const MIN_BOX_SIZE: f32 = 20.0;

/// COCO 类别名（对应 class_id 0..79）。
const COCO_CLASSES: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
];

/// YOLOv8n 检测器。
pub struct YoloV8Detector {
    session: Mutex<Option<Session>>,
    version: ModelVersion,
    input_name: String,
}

impl YoloV8Detector {
    /// 从 ONNX 文件加载。
    pub fn load(model_path: &Path) -> Result<Arc<Self>, AIError> {
        info!("Loading YOLOv8n from: {}", model_path.display());
        let session = Session::builder()
            .map_err(|e| AIError::ModelLoad(format!("session builder: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| AIError::ModelLoad(format!("commit_from_file: {e}")))?;

        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "images".to_string());
        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        info!(
            "YOLOv8n loaded — input: `{input_name}`, outputs: {output_names:?}"
        );

        Ok(Arc::new(Self {
            session: Mutex::new(Some(session)),
            version: ModelVersion::new("yolov8n@v1.0.0"),
            input_name,
        }))
    }

    /// 推理（同步）。
    fn detect_internal(&self, rgb: &image::RgbImage) -> Result<Vec<ObjectDetection>, AIError> {
        let mut guard = self.session.lock();
        let session = guard
            .as_mut()
            .ok_or_else(|| AIError::ModelLoad("YOLO session not loaded".into()))?;

        // ---- 预处理 ----
        let resized =
            image::imageops::resize(rgb, INPUT_SIZE, INPUT_SIZE, FilterType::Triangle);

        // BGR → RGB，CHW 归一化 [0,1]
        let mut input = vec![0.0_f32; 3 * INPUT_SIZE as usize * INPUT_SIZE as usize];
        for (i, pixel) in resized.pixels().enumerate() {
            let idx = i % (INPUT_SIZE as usize * INPUT_SIZE as usize);
            input[idx] = pixel[2] as f32 / 255.0;
            input[idx + (INPUT_SIZE as usize * INPUT_SIZE as usize)] = pixel[1] as f32 / 255.0;
            input[idx + 2 * (INPUT_SIZE as usize * INPUT_SIZE as usize)] =
                pixel[0] as f32 / 255.0;
        }

        let tensor = Tensor::from_array(
            ([1_i64, 3, INPUT_SIZE as i64, INPUT_SIZE as i64], input),
        )
        .map_err(|e| AIError::Inference(format!("tensor: {e}")))?;

        let outputs = session
            .run(ort::inputs![self.input_name.clone() => tensor])
            .map_err(|e| AIError::Inference(format!("run: {e}")))?;

        let data = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| AIError::Inference(format!("extract: {e}")))?
            .1;

        // data 布局：[1, 84, 8400] → flat 为 84 × 8400 = 705600
        // box i: data[i] = cx, data[8400+i] = cy, data[16800+i] = w, data[25200+i] = h
        // class j: data[(4+j)*8400 + i]
        let num_boxes = 8400usize;
        let img_w = rgb.width() as f32;
        let img_h = rgb.height() as f32;
        let scale_x = img_w / INPUT_SIZE as f32;
        let scale_y = img_h / INPUT_SIZE as f32;

        let mut raw: Vec<(usize, f32, BBox, i32)> = Vec::new();

        for i in 0..num_boxes {
            // 找最高 class score（跳过 class 0 = __background__）
            let mut max_score = 0.0_f32;
            let mut max_class = 0_i32;
            for j in 1..NUM_CLASSES {
                let score = data[(4 + j) * num_boxes + i];
                if score > max_score {
                    max_score = score;
                    max_class = j as i32;
                }
            }

            if max_score < CONFIDENCE_THRESHOLD {
                continue;
            }

            let cx = data[i];
            let cy = data[num_boxes + i];
            let w = data[2 * num_boxes + i];
            let h = data[3 * num_boxes + i];

            let x1 = ((cx - w * 0.5) * scale_x).max(0.0).min(img_w);
            let y1 = ((cy - h * 0.5) * scale_y).max(0.0).min(img_h);
            let x2 = ((cx + w * 0.5) * scale_x).max(0.0).min(img_w);
            let y2 = ((cy + h * 0.5) * scale_y).max(0.0).min(img_h);

            if x2 - x1 < MIN_BOX_SIZE || y2 - y1 < MIN_BOX_SIZE {
                continue;
            }

            // BBox { x, y, w, h }
            let bbox = BBox::new(x1, y1, x2 - x1, y2 - y1);
            raw.push((i, max_score, bbox, max_class));
        }

        // NMS（按 confidence 降序）
        raw.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut kept: Vec<ObjectDetection> = Vec::new();
        let mut suppressed = vec![false; raw.len()];

        for i in 0..raw.len() {
            if suppressed[i] {
                continue;
            }
            let (_orig_idx, conf, bbox, class_id) = raw[i];
            let class_name = COCO_CLASSES
                .get(class_id as usize)
                .copied()
                .unwrap_or("unknown")
                .to_string();
            kept.push(ObjectDetection {
                class_id,
                class_name,
                confidence: conf,
                bbox,
            });

            for j in (i + 1)..raw.len() {
                if suppressed[j] {
                    continue;
                }
                let other_bbox = &raw[j].2;
                let iou = bbox.iou(other_bbox);
                if iou > NMS_IOU_THRESHOLD {
                    suppressed[j] = true;
                }
            }
        }

        info!(
            "YOLOv8n: raw={}, after_nms={}",
            raw.len(),
            kept.len()
        );
        Ok(kept)
    }
}

#[async_trait]
impl ObjectDetector for YoloV8Detector {
    async fn detect(&self, image: &ImageData) -> Result<Vec<ObjectDetection>, AIError> {
        let rgb = image.as_rgb8();
        self.detect_internal(&rgb)
    }

    fn class_names(&self) -> &[String] {
        // Convert static str array to owned String Vec (lazy, cached)
        static NAMES: once_cell::sync::Lazy<Vec<String>> =
            once_cell::sync::Lazy::new(|| COCO_CLASSES.iter().map(|s| (*s).to_string()).collect());
        &NAMES
    }

    fn model_version(&self) -> ModelVersion {
        self.version.clone()
    }
}
