//! BBox — 统一的几何边界框类型。
//!
//! **唯一允许的 BBox 类型**。整个 codebase 禁止再定义 `Rect` / `FaceRect` 等变体。
//!
//! 坐标系：左上角原点 + 右下角正方向（xywh）。

use serde::{Deserialize, Serialize};

use crate::error::CoreError;

/// 边界框（左上角 + 宽高，xywh）。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    /// 左上角 x
    pub x: f32,
    /// 左上角 y
    pub y: f32,
    /// 宽度
    pub w: f32,
    /// 高度
    pub h: f32,
}

impl BBox {
    /// 构造一个新的 BBox，不做校验。
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    /// 从 (x1, y1, x2, y2) 构造（左上和右下角）。
    pub fn from_xyxy(x1: f32, y1: f32, x2: f32, y2: f32) -> Result<Self, CoreError> {
        if x2 < x1 || y2 < y1 {
            return Err(CoreError::InvalidBBox(format!(
                "xyxy order invalid: ({x1},{y1}) -> ({x2},{y2})"
            )));
        }
        Ok(Self {
            x: x1,
            y: y1,
            w: x2 - x1,
            h: y2 - y1,
        })
    }

    /// 转成 (x1, y1, x2, y2) 数组。
    pub fn xyxy(&self) -> [f32; 4] {
        [self.x, self.y, self.x + self.w, self.y + self.h]
    }

    /// 面积。
    #[inline]
    pub fn area(&self) -> f32 {
        self.w * self.h
    }

    /// 是否有效（面积 > 0 且各分量有限）。
    pub fn is_valid(&self) -> bool {
        self.w > 0.0 && self.h > 0.0 && self.x.is_finite() && self.y.is_finite()
    }

    /// 与另一个 BBox 的 IoU（Intersection over Union）。
    pub fn iou(&self, other: &BBox) -> f32 {
        match self.intersection(other) {
            Some(i) => {
                let inter = i.area();
                let union = self.area() + other.area() - inter;
                if union <= 0.0 {
                    0.0
                } else {
                    inter / union
                }
            }
            None => 0.0,
        }
    }

    /// 与另一个 BBox 的交集（若存在）。
    pub fn intersection(&self, other: &BBox) -> Option<BBox> {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.w).min(other.x + other.w);
        let y2 = (self.y + self.h).min(other.y + other.h);

        if x2 > x1 && y2 > y1 {
            Some(BBox {
                x: x1,
                y: y1,
                w: x2 - x1,
                h: y2 - y1,
            })
        } else {
            None
        }
    }

    /// 按中心点缩放到目标框（保留宽高比例）。用于 align 后裁剪。
    pub fn scaled(&self, sx: f32, sy: f32) -> BBox {
        let cx = self.x + self.w * 0.5;
        let cy = self.y + self.h * 0.5;
        let nw = self.w * sx;
        let nh = self.h * sy;
        BBox {
            x: cx - nw * 0.5,
            y: cy - nh * 0.5,
            w: nw,
            h: nh,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn xyxy_roundtrip() {
        let b = BBox::from_xyxy(10.0, 20.0, 110.0, 220.0).unwrap();
        assert_eq!(b.x, 10.0);
        assert_eq!(b.y, 20.0);
        assert_eq!(b.w, 100.0);
        assert_eq!(b.h, 200.0);
        let xy = b.xyxy();
        assert_eq!(xy, [10.0, 20.0, 110.0, 220.0]);
    }

    #[test]
    fn invalid_xyxy_rejected() {
        assert!(BBox::from_xyxy(10.0, 20.0, 5.0, 15.0).is_err());
    }

    #[test]
    fn area_positive() {
        let b = BBox::new(0.0, 0.0, 10.0, 20.0);
        assert!(approx(b.area(), 200.0));
    }

    #[test]
    fn iou_full_overlap() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        assert!(approx(a.iou(&a), 1.0));
    }

    #[test]
    fn iou_no_overlap() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BBox::new(20.0, 20.0, 5.0, 5.0);
        assert!(approx(a.iou(&b), 0.0));
    }

    #[test]
    fn iou_partial() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BBox::new(5.0, 5.0, 10.0, 10.0);
        // intersection = 5x5 = 25; union = 100 + 100 - 25 = 175
        assert!(approx(a.iou(&b), 25.0 / 175.0));
    }

    #[test]
    fn intersection_none() {
        let a = BBox::new(0.0, 0.0, 5.0, 5.0);
        let b = BBox::new(10.0, 10.0, 5.0, 5.0);
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn intersection_some() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BBox::new(5.0, 5.0, 10.0, 10.0);
        let i = a.intersection(&b).unwrap();
        assert_eq!(i.x, 5.0);
        assert_eq!(i.y, 5.0);
        assert_eq!(i.w, 5.0);
        assert_eq!(i.h, 5.0);
    }

    #[test]
    fn scaled_keeps_center() {
        let b = BBox::new(10.0, 20.0, 30.0, 40.0);
        let s = b.scaled(2.0, 2.0);
        let cx_orig = 10.0 + 30.0 * 0.5;
        let cy_orig = 20.0 + 40.0 * 0.5;
        let cx_new = s.x + s.w * 0.5;
        let cy_new = s.y + s.h * 0.5;
        assert!(approx(cx_orig, cx_new));
        assert!(approx(cy_orig, cy_new));
        assert!(approx(s.w, 60.0));
        assert!(approx(s.h, 80.0));
    }
}