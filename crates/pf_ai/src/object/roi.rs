//! ROI (Region of Interest) 提取 — per-scale multi-scale sliding window。
//!
//! **Phase 1 重构**（Crop → Same Object Retrieval）：
//!
//! 旧逻辑：
//! ```text
//! for each scale:
//!     generate ROIs
//! 合并 → 按面积排序 → max_rois truncate
//! ```
//! 问题：大 ROI 天然面积更大，挤掉小 ROI → 小物体没有 embedding → Query Crop 召回不到。
//!
//! 新逻辑：
//! ```text
//! 1. full_image 始终在 ROI[0]
//! 2. for each scale:
//!     generate ROIs
//!     spatial NMS (IoU > iou_threshold → drop)
//!     truncate 到 quota_per_scale[scale]
//! 3. 拼接：full_image + sum(per-scale ROIs)
//! ```
//!
//! 关键参数（来自 `RoiConfig::default()`）：
//! - `scales = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]`
//! - `quota_per_scale = [20, 20, 20, 15, 10, 5]`
//! - `stride_ratio = 0.5`
//! - `min_roi_size = 32`（像素）
//! - `iou_threshold = 0.7`（同 scale 内 spatial dedup）
//!
//! 不依赖 YOLO / 选择性搜索 / 复杂 detector。Category-independent。

use image::RgbImage;
use pf_config::RoiConfig;

use pf_core::BBox;

/// 一个 ROI 区域。
#[derive(Debug, Clone)]
pub struct Roi {
    /// 边界框 [x, y, w, h]
    pub bbox: BBox,
    /// 缩放比例（用于调权重）
    pub scale: f32,
    /// 面积（用于排序）
    pub area: f32,
}

/// Per-scale + spatial-coverage ROI 提取。
///
/// 返回的 ROIs 数量上限：`1 (full_image) + sum(quota_per_scale)`，
/// 实际可能更少（图像太小或某个 scale 被 min_roi_size 过滤）。
pub fn extract_rois(img: &RgbImage, config: &RoiConfig) -> Vec<Roi> {
    let (w, h) = (img.width() as f32, img.height() as f32);
    if w <= 0.0 || h <= 0.0 {
        return Vec::new();
    }

    let mut final_rois = Vec::with_capacity(config.max_rois_per_image as usize);

    // [1] Full image 始终在第一位
    final_rois.push(Roi {
        bbox: BBox::new(0.0, 0.0, w, h),
        scale: 1.0,
        area: w * h,
    });

    // [2] 每个 scale 独立处理：生成 → spatial NMS → quota 截断
    for (scale_idx, &scale) in config.scales.iter().enumerate() {
        // scale=1.0 已被 full_image 覆盖，跳过避免重复
        if (scale - 1.0).abs() < 0.001 {
            continue;
        }

        let roi_w = w * scale;
        let roi_h = h * scale;

        // 太小就跳过这个 scale（保留其他 scale 的 quota）
        if roi_w < config.min_roi_size as f32 || roi_h < config.min_roi_size as f32 {
            continue;
        }

        let stride = (roi_w.min(roi_h) * config.stride_ratio).max(1.0);
        let mut rois_at_scale = Vec::new();

        let mut y = 0.0_f32;
        while y + roi_h <= h {
            let mut x = 0.0_f32;
            while x + roi_w <= w {
                rois_at_scale.push(Roi {
                    bbox: BBox::new(x, y, roi_w, roi_h),
                    scale,
                    area: roi_w * roi_h,
                });
                x += stride;
            }
            y += stride;
        }

        // Spatial dedup（NMS）within scale
        let before_dedup = rois_at_scale.len();
        rois_at_scale = spatial_nms(rois_at_scale, config.iou_threshold);

        // Per-scale quota
        let quota = config
            .quota_per_scale
            .get(scale_idx)
            .copied()
            .unwrap_or(20) as usize;
        if rois_at_scale.len() > quota {
            rois_at_scale.truncate(quota);
        }

        tracing::debug!(
            scale,
            scale_idx,
            generated = before_dedup,
            after_nms = rois_at_scale.len(),
            quota,
            "roi generation at scale"
        );

        final_rois.extend(rois_at_scale);
    }

    final_rois
}

/// Greedy NMS：面积大的优先；后续 ROI 与已保留的 IoU > threshold 则丢弃。
///
/// 这里按 3×3 网格位置对 ROI 进行优先级排序：
/// 网格中心 (TL→TC→TR→ML→MC→MR→BL→BC→BR) 按顺序轮转，
/// 让空间覆盖优先于面积（同一格内再按面积）。
fn spatial_nms(mut rois: Vec<Roi>, iou_threshold: f32) -> Vec<Roi> {
    if rois.is_empty() {
        return rois;
    }

    let img_w = rois
        .iter()
        .map(|r| r.bbox.x + r.bbox.w)
        .fold(0.0_f32, f32::max);
    let img_h = rois
        .iter()
        .map(|r| r.bbox.y + r.bbox.h)
        .fold(0.0_f32, f32::max);

    // 计算每个 ROI 所在的 3×3 网格 cell（0..8）
    let cell_of = |r: &Roi| -> u8 {
        let cx = r.bbox.x + r.bbox.w * 0.5;
        let cy = r.bbox.y + r.bbox.h * 0.5;
        let col = ((cx / img_w.max(1.0)) * 3.0).floor().clamp(0.0, 2.0) as u8;
        let row = ((cy / img_h.max(1.0)) * 3.0).floor().clamp(0.0, 2.0) as u8;
        row * 3 + col
    };

    // 排序：(cell, -area)
    rois.sort_by(|a, b| {
        let ca = cell_of(a);
        let cb = cell_of(b);
        ca.cmp(&cb)
            .then_with(|| b.area.partial_cmp(&a.area).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut kept: Vec<Roi> = Vec::new();
    for r in rois {
        let dominated = kept
            .iter()
            .any(|k| k.bbox.iou(&r.bbox) > iou_threshold);
        if !dominated {
            kept.push(r);
        }
    }
    kept
}

/// 根据 ROI 列表执行给定函数（带 early-return `global_max`）。
pub fn map_rois<F>(img: &RgbImage, rois: &[Roi], mut f: F) -> Vec<(Roi, RgbImage)>
where
    F: FnMut(&Roi, &RgbImage) -> RgbImage,
{
    rois.iter()
        .map(|r| {
            let crop = crop_roi(img, r);
            (r.clone(), f(r, &crop))
        })
        .collect()
}

/// 从图中裁剪 ROI 区域。
pub fn crop_roi(img: &RgbImage, roi: &Roi) -> RgbImage {
    let b = roi.bbox;
    let x = b.x.max(0.0) as u32;
    let y = b.y.max(0.0) as u32;
    let w = b.w.max(1.0) as u32;
    let h = b.h.max(1.0) as u32;
    let view = image::imageops::crop_imm(
        img,
        x,
        y,
        w.min(img.width().saturating_sub(x)),
        h.min(img.height().saturating_sub(y)),
    );
    view.to_image()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    fn blank(w: u32, h: u32) -> RgbImage {
        RgbImage::from_pixel(w, h, Rgb([128, 128, 128]))
    }

    #[test]
    fn full_image_always_first() {
        let cfg = RoiConfig::default();
        let img = blank(1024, 768);
        let rois = extract_rois(&img, &cfg);
        assert!(!rois.is_empty());
        let r0 = &rois[0];
        assert_eq!(r0.bbox.x, 0.0);
        assert_eq!(r0.bbox.y, 0.0);
        assert_eq!(r0.bbox.w, 1024.0);
        assert_eq!(r0.bbox.h, 768.0);
        assert_eq!(r0.scale, 1.0);
    }

    #[test]
    fn per_scale_quota_respected() {
        let cfg = RoiConfig::default();
        // 大图 → 每个 scale 都应该有 ROI
        let img = blank(4000, 3000);
        let rois = extract_rois(&img, &cfg);

        // 数每个 scale 的数量
        let mut counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for r in &rois {
            // 把 f32 scale 量化成 u32 key
            let key = (r.scale * 100.0) as u32;
            *counts.entry(key).or_insert(0) += 1;
        }

        // full_image (scale=1.0) → 1
        assert_eq!(counts.get(&100), Some(&1));
        // 每个 scale ≤ quota
        for (scale_key, count) in &counts {
            if *scale_key == 100 {
                continue;
            }
            let scale_idx = cfg
                .scales
                .iter()
                .position(|s| ((*s * 100.0) as u32) == *scale_key)
                .expect("scale not in config");
            let quota = cfg.quota_per_scale[scale_idx] as usize;
            assert!(
                *count <= quota,
                "scale {scale_key} ({}) has {count} ROIs > quota {quota}",
                scale_idx
            );
        }
    }

    #[test]
    fn small_scale_preserved_not_overwritten() {
        // 关键测试：小 scale ROI 不能被全局截断挤掉
        let cfg = RoiConfig::default();
        let img = blank(4000, 3000);
        let rois = extract_rois(&img, &cfg);

        // scale=0.10 应该存在（400x300 在 4000x3000 上）
        let small_scale_present = rois.iter().any(|r| (r.scale - 0.10).abs() < 0.01);
        assert!(
            small_scale_present,
            "scale 0.10 ROI missing — small objects would be lost!"
        );
    }

    #[test]
    fn spatial_nms_dedups_overlapping_rois() {
        // 构造 4 个高重叠的 ROI，IoU > 0.7 → 留 1 个
        let rois = vec![
            Roi {
                bbox: BBox::new(0.0, 0.0, 100.0, 100.0),
                scale: 0.1,
                area: 10_000.0,
            },
            Roi {
                bbox: BBox::new(5.0, 5.0, 100.0, 100.0),
                scale: 0.1,
                area: 10_000.0,
            },
            Roi {
                bbox: BBox::new(50.0, 50.0, 100.0, 100.0),
                scale: 0.1,
                area: 10_000.0,
            },
            Roi {
                bbox: BBox::new(500.0, 500.0, 100.0, 100.0),
                scale: 0.1,
                area: 10_000.0,
            },
        ];
        let kept = spatial_nms(rois, 0.7);
        // 第一组 (0,0)/(5,5) IoU 接近 1 → 只留 1
        // (50,50) 与 (0,0) IoU = (50*50) / (10000 + 10000 - 2500) = 2500/17500 ≈ 0.14 → 留
        // (500,500) 与前两个 IoU = 0 → 留
        assert_eq!(kept.len(), 3, "expected 3 after NMS, got {}", kept.len());
    }

    #[test]
    fn spatial_coverage_prioritized_over_area() {
        // 同一 cell 内的大 ROI 优先，但跨 cell 的小 ROI 不会被大 cell 的更大 ROI 挤掉
        let mut rois = vec![
            // TL cell, 大面积
            Roi {
                bbox: BBox::new(0.0, 0.0, 500.0, 500.0),
                scale: 0.1,
                area: 250_000.0,
            },
            // BR cell, 较小面积
            Roi {
                bbox: BBox::new(800.0, 800.0, 100.0, 100.0),
                scale: 0.05,
                area: 10_000.0,
            },
        ];
        rois = spatial_nms(rois, 0.7);
        assert_eq!(rois.len(), 2, "different cells should both be kept");
    }

    #[test]
    fn total_count_within_bound() {
        let cfg = RoiConfig::default();
        let img = blank(4000, 3000);
        let rois = extract_rois(&img, &cfg);
        let expected_max = cfg.max_rois_per_image as usize;
        assert!(
            rois.len() <= expected_max,
            "got {} ROIs > max_rois_per_image {}",
            rois.len(),
            expected_max
        );
    }

    #[test]
    fn small_image_does_not_panic() {
        let cfg = RoiConfig::default();
        let img = blank(100, 100);
        let rois = extract_rois(&img, &cfg);
        // 至少 full_image
        assert!(!rois.is_empty());
    }

    #[test]
    fn crop_roi_returns_correct_size() {
        let img = blank(100, 100);
        let r = Roi {
            bbox: BBox::new(10.0, 20.0, 30.0, 40.0),
            scale: 1.0,
            area: 30.0 * 40.0,
        };
        let crop = crop_roi(&img, &r);
        assert_eq!(crop.width(), 30);
        assert_eq!(crop.height(), 40);
    }
}