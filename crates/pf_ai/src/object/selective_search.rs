//! Selective Search for Object Recognition (Ported from ai-next)
//!
//! Based on paper: "Selective Search for Object Recognition" (Uijlings et al., 2012)
//! Modified for efficiency in Rust.
//!
//! Algorithm:
//! 1. Felzenszwalb-Huttenlocher graph segmentation
//! 2. Initial regions from segments
//! 3. Hierarchical grouping with multi-scale boxes
//! 4. Filter by size, sort, truncate to max_regions

use image::RgbImage;
use std::collections::HashMap;

/// A region found by selective search.
#[derive(Debug, Clone)]
pub struct SelectiveRoi {
    /// Bounding box [x, y, w, h] in pixels.
    pub bbox: [f32; 4],
    /// Region type label.
    pub region_type: &'static str,
    /// Region size in pixels.
    pub size: u32,
}

/// Perform selective search on an RGB image.
///
/// `min_side`: 最小边长（像素），与 ai-next `min_roi_size=64` 对齐。
pub fn selective_search(img: &RgbImage, max_regions: usize, min_side: u32) -> Vec<SelectiveRoi> {
    let (width, height) = img.dimensions();

    // Step 1: Felzenszwalb-Huttenlocher segmentation
    let segments = felzenszwalb_segmentation(img);

    // Step 2: Get initial regions from segmentation
    let mut region_map: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    for y in 0..height {
        for x in 0..width {
            let seg_id = segments[(y * width + x) as usize];
            region_map.entry(seg_id).or_insert_with(Vec::new).push((x, y));
        }
    }

    // Step 3: Create regions from segments
    let mut regions = Vec::new();
    for (_seg_id, pixels) in region_map {
        if pixels.is_empty() {
            continue;
        }
        let min_x = pixels.iter().map(|p| p.0).min().unwrap();
        let max_x = pixels.iter().map(|p| p.0).max().unwrap();
        let min_y = pixels.iter().map(|p| p.1).min().unwrap();
        let max_y = pixels.iter().map(|p| p.1).max().unwrap();

        let bbox = [
            min_x as f32,
            min_y as f32,
            (max_x - min_x + 1) as f32,
            (max_y - min_y + 1) as f32,
        ];

        regions.push(SelectiveRoi {
            bbox,
            region_type: "segment",
            size: pixels.len() as u32,
        });
    }

    // Step 4: Hierarchical grouping (simplified)
    let merged = hierarchical_grouping(&mut regions, width, height);

    // Step 5: Filter by min side length + min area（ai-next 对齐）
    let mut final_regions: Vec<SelectiveRoi> = merged
        .into_iter()
        .filter(|r| {
            let w = r.bbox[2] as u32;
            let h = r.bbox[3] as u32;
            w >= min_side && h >= min_side && r.size >= 500
        })
        .collect();

    final_regions.sort_by(|a, b| b.size.cmp(&a.size));
    final_regions.truncate(max_regions.saturating_sub(1)); // leave room for full_image

    // Always add full image as the last ROI
    final_regions.push(SelectiveRoi {
        bbox: [0.0, 0.0, width as f32, height as f32],
        region_type: "full_image",
        size: width * height,
    });

    tracing::debug!(
        roi_count = final_regions.len(),
        width,
        height,
        "selective_search done"
    );

    final_regions
}

/// Felzenszwalb-Huttenlocher segmentation.
fn felzenszwalb_segmentation(img: &RgbImage) -> Vec<u32> {
    let (width, height) = img.dimensions();
    let num_pixels = (width * height) as usize;

    // Initialize disjoint set (union-find)
    let mut parent: Vec<u32> = (0..num_pixels as u32).collect();
    let mut rank = vec![0u32; num_pixels];

    // Build gradient image
    let gradient = compute_gradient(img);

    // Initial threshold k=30
    let mut thresholds = vec![30.0f32; num_pixels];

    // Sort edges by gradient difference
    let mut edges: Vec<(u32, u32, f32)> = Vec::new();
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let idx = (y * width + x) as usize;

            // Horizontal edge
            let idx2 = (y * width + x + 1) as usize;
            let diff_h = (gradient[idx] - gradient[idx2]).abs();
            edges.push((idx as u32, idx2 as u32, diff_h));

            // Vertical edge
            let idx3 = ((y + 1) * width + x) as usize;
            let diff_v = (gradient[idx] - gradient[idx3]).abs();
            edges.push((idx as u32, idx3 as u32, diff_v));
        }
    }

    edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    fn find(parent: &[u32], x: u32) -> u32 {
        let xi = x as usize;
        if parent[xi] != x {
            find(parent, parent[xi])
        } else {
            x
        }
    }

    fn union(parent: &mut [u32], rank: &mut [u32], x: u32, y: u32) {
        let px = find(parent, x);
        let py = find(parent, y);
        if px == py {
            return;
        }
        let pxi = px as usize;
        let pyi = py as usize;
        if rank[pxi] < rank[pyi] {
            parent[pxi] = py;
        } else if rank[pxi] > rank[pyi] {
            parent[pyi] = px;
        } else {
            parent[pyi] = px;
            rank[pxi] += 1;
        }
    }

    // Process edges in sorted order
    for (a, b, diff) in edges {
        let pa = find(&parent, a);
        let pb = find(&parent, b);
        if pa != pb {
            let pa_usize = pa as usize;
            let pb_usize = pb as usize;
            let threshold = thresholds[pa_usize].min(thresholds[pb_usize]) + diff;
            if diff <= threshold {
                union(&mut parent, &mut rank, pa, pb);
                let new_parent = find(&parent, pa);
                thresholds[new_parent as usize] = threshold;
            }
        }
    }

    // Renumber segments to consecutive ids
    let mut segment_map = HashMap::new();
    let mut next_id = 0u32;
    let mut result = vec![0u32; num_pixels];

    for i in 0..num_pixels {
        let p = find(&parent, i as u32);
        let seg_id = *segment_map.entry(p).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        result[i] = seg_id;
    }

    result
}

/// Compute gradient magnitude using Sobel operator.
fn compute_gradient(img: &RgbImage) -> Vec<f32> {
    let (width, height) = img.dimensions();
    let mut gradient = vec![0.0f32; (width * height) as usize];

    let lum = |p: &image::Rgb<u8>| -> f32 {
        0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32
    };

    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let idx = (y * width + x) as usize;

            // Sobel kernels
            let p00 = img.get_pixel(x - 1, y - 1);
            let p10 = img.get_pixel(x, y - 1);
            let p20 = img.get_pixel(x + 1, y - 1);
            let p01 = img.get_pixel(x - 1, y);
            let p21 = img.get_pixel(x + 1, y);
            let p02 = img.get_pixel(x - 1, y + 1);
            let p12 = img.get_pixel(x, y + 1);
            let p22 = img.get_pixel(x + 1, y + 1);

            let gx = -lum(p00) - 2.0 * lum(p01) - lum(p02) + lum(p20) + 2.0 * lum(p21) + lum(p22);
            let gy = -lum(p00) - 2.0 * lum(p10) - lum(p20) + lum(p02) + 2.0 * lum(p12) + lum(p22);

            gradient[idx] = (gx * gx + gy * gy).sqrt();
        }
    }

    gradient
}

/// Simplified hierarchical grouping: filter by size ratio + add multi-scale boxes.
fn hierarchical_grouping(
    regions: &mut Vec<SelectiveRoi>,
    width: u32,
    height: u32,
) -> Vec<SelectiveRoi> {
    let total_pixels = width * height;

    // Keep regions between 0.1% and 95% of image size
    regions.retain(|r| {
        let size_ratio = r.size as f32 / total_pixels as f32;
        size_ratio >= 0.001 && size_ratio <= 0.95
    });

    // Add multi-scale boxes centered on image
    let scales = [0.2, 0.4, 0.6, 0.8];
    for &scale in &scales {
        let w = (width as f32 * scale) as u32;
        let h = (height as f32 * scale) as u32;
        let x = (width.saturating_sub(w)) / 2;
        let y = (height.saturating_sub(h)) / 2;

        regions.push(SelectiveRoi {
            bbox: [x as f32, y as f32, w as f32, h as f32],
            region_type: "multi_scale",
            size: w * h,
        });
    }

    regions.sort_by(|a, b| b.size.cmp(&a.size));

    // Deduplicate by proximity (within 20px in any dimension)
    let mut seen: Vec<SelectiveRoi> = Vec::new();
    for r in regions.drain(..) {
        let dominated = seen.iter().any(|s| {
            (s.bbox[0] - r.bbox[0]).abs() < 20.0
                && (s.bbox[1] - r.bbox[1]).abs() < 20.0
                && (s.bbox[2] - r.bbox[2]).abs() < 20.0
                && (s.bbox[3] - r.bbox[3]).abs() < 20.0
        });
        if !dominated {
            seen.push(r);
        }
    }

    seen
}
