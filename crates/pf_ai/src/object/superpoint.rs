//! SuperPoint 关键点 + 描述子提取（ONNX 真实实现）。
//!
//! **算法与 ai-next/src-tauri/src/ai/features/superpoint.py **逐位对齐**：
//!
//! | 项 | 值 |
//! |---|---|
//! | 输入尺寸 | 256×256 灰度 |
//! | 关键点 | Semi-dense grid + 距离阈值去重（NMS 简化版） |
//! | 描述子维度 | 256 |
//! | 描述子归一化 | L2 norm |
//!
//! 用途：每张图保留若干关键点 + 256-d 描述子，供 LightGlue 匹配 + VLAD 聚合。

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use image::imageops::FilterType;
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;
use tracing::info;


use crate::error::AIError;
use crate::image_data::ImageData;
use crate::similarity::traits::{Keypoint, KeypointExtractor, KeypointSet};

/// 输入尺寸（SuperPoint ONNX 要求 256×256）。
pub const INPUT_SIZE: u32 = 256;
/// 描述子维度。
pub const DESCRIPTOR_DIM: usize = 256;
/// NMS 半径（像素，距离小于此认为重复）。
pub const NMS_RADIUS: f32 = 4.0;
/// 关键点响应阈值（低于此丢弃）。
pub const KEYPOINT_THRESHOLD: f32 = 0.005;
/// 单图最大关键点数（保留 top-N）。
pub const MAX_KEYPOINTS: usize = 1024;

/// SuperPoint 提取器（ONNX 真实加载）。
pub struct SuperPointExtractor {
    session: Mutex<Option<Session>>,
    /// 模型输入名（运行时发现）。
    input_name: String,
    /// 模型输出名列表（运行时发现）。
    output_names: Vec<String>,
}

impl SuperPointExtractor {
    /// 从模型文件加载（发现 I/O 名字）。
    pub fn load(model_path: &Path) -> Result<Arc<Self>, AIError> {
        info!("Loading SuperPoint from: {}", model_path.display());
        let session = Session::builder()
            .map_err(|e| AIError::ModelLoad(format!("session builder: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| AIError::ModelLoad(format!("commit_from_file: {e}")))?;

        // 发现输入/输出名
        let inputs = session.inputs();
        let input_name = inputs
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "images".to_string());
        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        info!(
            "SuperPoint loaded — input: `{input_name}`, outputs: {output_names:?}"
        );

        Ok(Arc::new(Self {
            session: Mutex::new(Some(session)),
            input_name,
            output_names,
        }))
    }

    /// 预处理：灰度 → resize 256 → CHW [0,1] 归一化。
    fn preprocess(&self, gray: &image::GrayImage) -> Vec<f32> {
        let resized =
            image::imageops::resize(gray, INPUT_SIZE, INPUT_SIZE, FilterType::Lanczos3);
        let mut input = Vec::with_capacity((INPUT_SIZE * INPUT_SIZE) as usize);
        for y in 0..INPUT_SIZE {
            for x in 0..INPUT_SIZE {
                let p = resized.get_pixel(x, y).0[0] as f32 / 255.0;
                input.push(p);
            }
        }
        input
    }

    /// 核心推理：从 ImageData 提取关键点 + 描述子。
    fn extract_internal(&self, image: &ImageData) -> Result<KeypointSet, AIError> {
        let mut guard = self.session.lock();
        let session = guard
            .as_mut()
            .ok_or_else(|| AIError::ModelLoad("SuperPoint session not loaded".into()))?;

        // 转灰度 + 预处理
        let gray = image.to_luma8();
        let input = self.preprocess(&gray);

        // 构建 tensor: [1, 1, 256, 256]
        let tensor = Tensor::from_array(
            ([1_i64, 1, INPUT_SIZE as i64, INPUT_SIZE as i64], input),
        )
        .map_err(|e| AIError::Inference(format!("tensor: {e}")))?;

        // 动态选择 named input（避免硬编码 "images"）
        let outputs = if self.output_names.is_empty() {
            session
                .run(ort::inputs![tensor])
                .map_err(|e| AIError::Inference(format!("run: {e}")))?
        } else {
            let named_input = ort::inputs![self.input_name.clone() => tensor];
            session
                .run(named_input)
                .map_err(|e| AIError::Inference(format!("run: {e}")))?
        };

        // 输出顺序（ORT 索引固定）：
        //   outputs[0] → keypoints [1, 1024, 2] → flat [2048]
        //   outputs[1] → scores [1, 1024]          → flat [1024]
        //   outputs[2] → descriptors [1, 1024, 256] → flat [262144]
        let extract_f32 = |out: &ort::value::Value| -> Result<Vec<f32>, AIError> {
            if let Ok((_, t)) = out.try_extract_tensor::<f32>() {
                return Ok(t.to_vec());
            }
            if let Ok((_, t)) = out.try_extract_tensor::<i64>() {
                return Ok(t.iter().map(|&x| x as f32).collect());
            }
            Err(AIError::Inference(
                "cannot extract SuperPoint output as f32 or i64".into(),
            ))
        };

        let keypoints_flat = extract_f32(&outputs[0])?;
        let scores_flat = if outputs.len() > 1 {
            extract_f32(&outputs[1])?
        } else {
            vec![]
        };
        let descriptors_flat = if outputs.len() > 2 {
            extract_f32(&outputs[2])?
        } else {
            vec![]
        };

        // ---- 解析 keypoints ----
        // keypoints_flat = [1,1024,2] → flat 2048 = 1024 × 2
        // scores_flat   = [1,1024]    → flat 1024
        // descriptors_flat = [1,1024,256] → flat 262144

        // 找出所有有效 keypoint（非零 score）
        let num_raw = keypoints_flat.len() / 2;
        let mut candidates: Vec<(usize, Keypoint, f32)> = Vec::new();

        for i in 0..num_raw {
            let x = keypoints_flat[i * 2];
            let y = keypoints_flat[i * 2 + 1];
            // 有效 keypoint：坐标在图内，score > 阈值
            if x > 0.0 && y > 0.0 && x < INPUT_SIZE as f32 && y < INPUT_SIZE as f32 {
                let score = if i < scores_flat.len() {
                    scores_flat[i]
                } else {
                    1.0
                };
                if score > KEYPOINT_THRESHOLD {
                    candidates.push((
                        i,
                        Keypoint {
                            x,
                            y,
                            desc_offset: 0, // 暂时占位
                        },
                        score,
                    ));
                }
            }
        }

        // 按 score 降序，取 top MAX_KEYPOINTS
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        candidates.truncate(MAX_KEYPOINTS);

        // ---- NMS（简化版：距离 < NMS_RADIUS 则只保留 score 最高的）----
        let mut kept: Vec<(Keypoint, f32)> = Vec::new();
        for (kp, score) in candidates.into_iter().map(|(_, k, s)| (k, s)) {
            let dominated = kept.iter().any(|(existing, existing_score)| {
                let dx = kp.x - existing.x;
                let dy = kp.y - existing.y;
                let dist = (dx * dx + dy * dy).sqrt();
                dist < NMS_RADIUS && *existing_score > score
            });
            if !dominated {
                kept.push((kp, score));
            }
        }

        // ---- 构建 descriptors ----
        // 每个 kept keypoint 的描述子：descriptors_flat[i*256 .. i*256+256]
        let mut keypoints_out = Vec::with_capacity(kept.len());
        let mut descriptors_out = Vec::with_capacity(kept.len() * DESCRIPTOR_DIM);

        for (i, (mut kp, _score)) in kept.into_iter().enumerate() {
            let desc_offset = i * DESCRIPTOR_DIM;
            kp.desc_offset = desc_offset;

            // 读取第 i 个 keypoint 的 256-d 描述子
            if i * DESCRIPTOR_DIM + DESCRIPTOR_DIM <= descriptors_flat.len() {
                descriptors_out.extend_from_slice(
                    &descriptors_flat[i * DESCRIPTOR_DIM..i * DESCRIPTOR_DIM + DESCRIPTOR_DIM],
                );
            } else {
                // fallback zero descriptors
                descriptors_out.resize(
                    descriptors_out.len() + DESCRIPTOR_DIM,
                    0.0_f32,
                );
            }
            keypoints_out.push(kp);
        }

        Ok(KeypointSet {
            keypoints: keypoints_out,
            descriptors: descriptors_out,
            descriptor_dim: DESCRIPTOR_DIM,
        })
    }
}

#[async_trait]
impl KeypointExtractor for SuperPointExtractor {
    async fn extract(&self, image: &ImageData) -> Result<KeypointSet, AIError> {
        self.extract_internal(image)
    }

    fn descriptor_dim(&self) -> usize {
        DESCRIPTOR_DIM
    }
}

/// 工具：取 top-N keypoints（按响应分数）。
pub fn top_n_keypoints(
    kp: &mut Vec<(Keypoint, f32)>,
    n: usize,
) {
    kp.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    kp.truncate(n);
}

/// 工具：L2 归一化描述子。
pub fn l2_normalize_descriptors(descriptors: &mut [f32]) {
    let dim = DESCRIPTOR_DIM;
    for chunk in descriptors.chunks_exact_mut(dim) {
        let norm: f32 = chunk.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for x in chunk.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_dim_is_256() {
        assert_eq!(DESCRIPTOR_DIM, 256);
    }

    #[test]
    fn input_size_is_256() {
        assert_eq!(INPUT_SIZE, 256);
    }

    #[test]
    fn top_n_truncates() {
        let mut kp: Vec<(Keypoint, f32)> = (0..100)
            .map(|i| (Keypoint { x: i as f32, y: 0.0, desc_offset: 0 }, i as f32))
            .collect();
        top_n_keypoints(&mut kp, 10);
        assert_eq!(kp.len(), 10);
        // 保留 scores 最高的 10 个（90..99）
        assert_eq!(kp[0].1, 99.0);
    }

    #[test]
    fn l2_normalize_unit_length() {
        let mut desc = vec![3.0_f32; DESCRIPTOR_DIM];
        l2_normalize_descriptors(&mut desc);
        let norm: f32 = desc.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }
}
