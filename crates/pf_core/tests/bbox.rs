//! BBox 集成测试（在 `tests/` 目录，作为 crate 外部视角）。

use pf_core::BBox;

#[test]
fn full_lifecycle() {
    // xyxy → xywh → xyxy
    let xyxy = [10.0_f32, 20.0, 110.0, 220.0];
    let b = BBox::from_xyxy(xyxy[0], xyxy[1], xyxy[2], xyxy[3]).unwrap();
    let back = b.xyxy();
    assert_eq!(back, xyxy);
    assert!(b.is_valid());

    // 缩放保持中心
    let bigger = b.scaled(2.0, 2.0);
    let orig_cx = b.x + b.w * 0.5;
    let orig_cy = b.y + b.h * 0.5;
    let big_cx = bigger.x + bigger.w * 0.5;
    let big_cy = bigger.y + bigger.h * 0.5;
    assert!((orig_cx - big_cx).abs() < 1e-4);
    assert!((orig_cy - big_cy).abs() < 1e-4);
}

#[test]
fn iou_edge_cases() {
    // 完全相同
    let a = BBox::new(0.0, 0.0, 10.0, 10.0);
    assert!((a.iou(&a) - 1.0).abs() < 1e-6);

    // 部分重叠
    let b = BBox::new(5.0, 5.0, 10.0, 10.0);
    // intersection 25, union 175, iou = 25/175
    let expected = 25.0 / 175.0;
    assert!((a.iou(&b) - expected).abs() < 1e-6);

    // 完全分离
    let c = BBox::new(100.0, 100.0, 10.0, 10.0);
    assert!(a.iou(&c).abs() < 1e-6);

    // 一个完全包含另一个
    let big = BBox::new(0.0, 0.0, 100.0, 100.0);
    let small = BBox::new(10.0, 10.0, 5.0, 5.0);
    // iou = small / big = 25 / 10000
    assert!((big.iou(&small) - 25.0 / 10000.0).abs() < 1e-6);
}

#[test]
fn intersection_boundary() {
    // 边界相交（不算相交，因为 x2 > x1 严格）
    let a = BBox::new(0.0, 0.0, 10.0, 10.0);
    let b = BBox::new(10.0, 0.0, 10.0, 10.0);
    assert!(a.intersection(&b).is_none());

    // 一个像素重叠
    let c = BBox::new(9.0, 9.0, 2.0, 2.0);
    let i = a.intersection(&c).unwrap();
    assert_eq!(i.x, 9.0);
    assert_eq!(i.y, 9.0);
    assert_eq!(i.w, 1.0);
    assert_eq!(i.h, 1.0);
}