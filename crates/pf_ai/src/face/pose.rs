//! 5 关键点 → 头部姿态估计（yaw / pitch / roll）。
//!
//! **算法：缩放正交投影（POS，Pose from Orthography and Scaling）。**
//! 用一个固定的通用 3D 人脸模型，最小二乘拟合缩放正交投影矩阵
//! `x_i = s·(r11 X_i + r12 Y_i + r13 Z_i) + tx`、`y_i = s·(r21 X_i + r22 Y_i + r23 Z_i) + ty`，
//! 对旋转矩阵做 Gram-Schmidt 正交化后分解为欧拉角（度）。
//!
//! 参考：Dementhon & Davis, "Model-Based Object Pose in 25 Lines of Code" (1995)；
//! insightface 5-KPS weak-perspective 变体的无依赖实现。
//!
//! 用途：
//! - item 19：把 yaw/pitch/roll 写入 `faces` 表
//! - item 20：v3 质量公式的 `pose_score = 1 - (yaw_dev + pitch_dev + roll_dev) / 3`
//! - item 21：prototype 按 yaw 分桶（frontal / left / right）

use pf_core::FaceKeypoints;

const RAD_TO_DEG: f32 = 180.0 / std::f32::consts::PI;

/// 退化阈值：眼距（像素）小于该值时关键点不可靠，返回 `None`。
const MIN_EYE_DIST: f32 = 1.0;

/// 通用 3D 人脸模型（5 点）。单位任意，比例取标准人脸：
/// 眼距 ~2.0、嘴宽 ~1.5、眼线到嘴角 ~2.5（眼距:眼到嘴 = 2:2.5 ≈ 真实 60:75mm）。
/// 坐标约定：X 右、Y 下、Z 朝向观察者（正 Z 更近）。
/// 鼻子在 Z=0（最前），眼睛 / 嘴角在 Z<0（面后）→ 提供 yaw/pitch 可观测的深度差。
const MODEL_3D: [[f32; 3]; 5] = [
    [-1.0, 0.0, -0.6],   // 左眼
    [1.0, 0.0, -0.6],    // 右眼
    [0.0, 1.5, 0.0],     // 鼻尖（眼线下方 1.5 = 0.6 × 眼到嘴距离）
    [-0.75, 2.5, -0.5],  // 左嘴角
    [0.75, 2.5, -0.5],   // 右嘴角
];

/// 从 5 个关键点估计头部姿态。
///
/// 返回 `(yaw, pitch, roll)`，单位度。
/// - 正 yaw = 头转向主体左侧（观察者看到其右颊）；负 yaw = 转右侧。
///   该符号约定与 `prototype_selector::classify_face_to_prototype_type`
///   （yaw < -30 → LeftProfile）一致。
/// - 正 pitch = 低头；正 roll = 观察者视角逆时针。
///
/// 退化输入（眼距过小 / 坐标非有限）返回 `None`。
pub fn estimate_yaw_pitch_roll(kps: &FaceKeypoints) -> Option<(f32, f32, f32)> {
    let pts = [
        (kps.left_eye.0, kps.left_eye.1),
        (kps.right_eye.0, kps.right_eye.1),
        (kps.nose.0, kps.nose.1),
        (kps.left_mouth.0, kps.left_mouth.1),
        (kps.right_mouth.0, kps.right_mouth.1),
    ];

    if pts.iter().any(|(x, y)| !x.is_finite() || !y.is_finite()) {
        return None;
    }
    let dx = pts[1].0 - pts[0].0;
    let dy = pts[1].1 - pts[0].1;
    if (dx * dx + dy * dy).sqrt() < MIN_EYE_DIST {
        return None;
    }

    let r = solve_scaled_ortho(&MODEL_3D, &pts)?;
    Some(decompose_euler(&r))
}

/// 缩放正交投影最小二乘：解 8 维线性系统，返回行优先旋转矩阵 `[r1; r2; r3]`（3×3）。
fn solve_scaled_ortho(model: &[[f32; 3]; 5], pts: &[(f32, f32); 5]) -> Option<[f32; 9]> {
    // u = [s·r11, s·r12, s·r13, tx, s·r21, s·r22, s·r23, ty]
    let mut ata = [[0.0f32; 8]; 8];
    let mut atb = [0.0f32; 8];
    for i in 0..5 {
        let (mx, my, mz) = (model[i][0], model[i][1], model[i][2]);
        let (px, py) = pts[i];
        let row_x = [mx, my, mz, 1.0, 0.0, 0.0, 0.0, 0.0];
        let row_y = [0.0, 0.0, 0.0, 0.0, mx, my, mz, 1.0];
        for r in 0..8 {
            for c in 0..8 {
                ata[r][c] += row_x[r] * row_x[c];
                ata[r][c] += row_y[r] * row_y[c];
            }
            atb[r] += row_x[r] * px;
            atb[r] += row_y[r] * py;
        }
    }
    let u = solve_8x8(&ata, &atb)?;

    let r1_raw = [u[0], u[1], u[2]];
    let r2_raw = [u[4], u[5], u[6]];
    let s = (norm3(r1_raw) + norm3(r2_raw)) * 0.5;
    if s < 1e-6 {
        return None;
    }

    // Gram-Schmidt 正交化（弱透视下两行不保证精确正交）
    let r1 = normalize3([r1_raw[0] / s, r1_raw[1] / s, r1_raw[2] / s]);
    let mut r2 = [r2_raw[0] / s, r2_raw[1] / s, r2_raw[2] / s];
    let dot = dot3(r2, r1);
    r2 = [r2[0] - dot * r1[0], r2[1] - dot * r1[1], r2[2] - dot * r1[2]];
    r2 = normalize3(r2);
    let r3 = cross3(r1, r2);

    Some([
        r1[0], r1[1], r1[2], r2[0], r2[1], r2[2], r3[0], r3[1], r3[2],
    ])
}

/// 分解旋转矩阵（约定 `R = Rz(roll)·Ry(yaw)·Rx(pitch)`）为欧拉角（度）。
fn decompose_euler(r: &[f32; 9]) -> (f32, f32, f32) {
    let roll = r[3].atan2(r[0]); // r2.x atan2 r1.x
    let yaw = (-r[6]).asin(); // -r3.x
    let pitch = r[7].atan2(r[8]); // r3.y atan2 r3.z
    (
        yaw * RAD_TO_DEG,
        pitch * RAD_TO_DEG,
        roll * RAD_TO_DEG,
    )
}

/// 解 8×8 线性方程组 `A u = b`（高斯消元 + 部分主元）。奇异返回 `None`。
fn solve_8x8(a: &[[f32; 8]; 8], b: &[f32; 8]) -> Option<[f32; 8]> {
    let mut m = [[0.0f32; 9]; 8];
    for i in 0..8 {
        for j in 0..8 {
            m[i][j] = a[i][j];
        }
        m[i][8] = b[i];
    }
    for col in 0..8 {
        let mut piv = col;
        for r in (col + 1)..8 {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        if m[piv][col].abs() < 1e-9 {
            return None;
        }
        m.swap(col, piv);
        let div = m[col][col];
        for j in col..9 {
            m[col][j] /= div;
        }
        for r in 0..8 {
            if r == col {
                continue;
            }
            let f = m[r][col];
            if f.abs() < 1e-12 {
                continue;
            }
            for j in col..9 {
                m[r][j] -= f * m[col][j];
            }
        }
    }
    let mut u = [0.0f32; 8];
    for i in 0..8 {
        u[i] = m[i][8];
    }
    Some(u)
}

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let n = norm3(v).max(1e-9);
    [v[0] / n, v[1] / n, v[2] / n]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kps(pts: [(f32, f32); 5]) -> FaceKeypoints {
        FaceKeypoints {
            left_eye: pts[0],
            right_eye: pts[1],
            nose: pts[2],
            left_mouth: pts[3],
            right_mouth: pts[4],
        }
    }

    fn rot_mat(yaw: f32, pitch: f32, roll: f32) -> [f32; 9] {
        let (sy, cy) = yaw.to_radians().sin_cos();
        let (sx, cx) = pitch.to_radians().sin_cos();
        let (sz, cz) = roll.to_radians().sin_cos();
        // R = Rz(roll)·Ry(yaw)·Rx(pitch)
        let rz = [cz, -sz, 0.0, sz, cz, 0.0, 0.0, 0.0, 1.0];
        let ry = [cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy];
        let rx = [1.0, 0.0, 0.0, 0.0, cx, -sx, 0.0, sx, cx];
        mat_mul3(&mat_mul3(&rz, &ry), &rx)
    }

    fn mat_mul3(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
        let mut out = [0.0f32; 9];
        for i in 0..3 {
            for j in 0..3 {
                out[i * 3 + j] = (0..3)
                    .map(|k| a[i * 3 + k] * b[k * 3 + j])
                    .sum();
            }
        }
        out
    }

    fn rot_vec(r: &[f32; 9], v: [f32; 3]) -> [f32; 3] {
        [
            r[0] * v[0] + r[1] * v[1] + r[2] * v[2],
            r[3] * v[0] + r[4] * v[1] + r[5] * v[2],
            r[6] * v[0] + r[7] * v[1] + r[8] * v[2],
        ]
    }

    /// 用已知旋转 + 缩放 + 平移投影模型，得到 2D 关键点（自洽 → 应精确恢复）。
    fn project(yaw: f32, pitch: f32, roll: f32) -> [(f32, f32); 5] {
        let r = rot_mat(yaw, pitch, roll);
        let s = 20.0;
        let (tx, ty) = (100.0, 80.0);
        let mut out = [(0.0f32, 0.0f32); 5];
        for i in 0..5 {
            let p = rot_vec(&r, MODEL_3D[i]);
            out[i] = (s * p[0] + tx, s * p[1] + ty);
        }
        out
    }

    #[test]
    fn frontal_identity_recovers_zero() {
        let (y, p, r) = estimate_yaw_pitch_roll(&kps(project(0.0, 0.0, 0.0))).unwrap();
        assert!(y.abs() < 1e-2, "yaw={y}");
        assert!(p.abs() < 1e-2, "pitch={p}");
        assert!(r.abs() < 1e-2, "roll={r}");
    }

    #[test]
    fn known_yaw_recovers() {
        for deg in [-45.0, -30.0, -15.0, 15.0, 30.0, 45.0] {
            let (y, _p, _r) = estimate_yaw_pitch_roll(&kps(project(deg, 0.0, 0.0))).unwrap();
            assert!((y - deg).abs() < 0.5, "yaw expected {deg}, got {y}");
        }
    }

    #[test]
    fn known_pitch_recovers() {
        for deg in [-20.0, -10.0, 10.0, 20.0] {
            let (_y, p, _r) = estimate_yaw_pitch_roll(&kps(project(0.0, deg, 0.0))).unwrap();
            assert!((p - deg).abs() < 0.5, "pitch expected {deg}, got {p}");
        }
    }

    #[test]
    fn known_roll_recovers() {
        for deg in [-25.0, -10.0, 10.0, 25.0] {
            let (_y, _p, r) = estimate_yaw_pitch_roll(&kps(project(0.0, 0.0, deg))).unwrap();
            assert!((r - deg).abs() < 0.5, "roll expected {deg}, got {r}");
        }
    }

    #[test]
    fn combined_rotation_recovers() {
        for (y, p, r) in [(15.0, 10.0, 5.0), (-20.0, -8.0, 12.0), (30.0, -15.0, -10.0)] {
            let (yy, pp, rr) = estimate_yaw_pitch_roll(&kps(project(y, p, r))).unwrap();
            assert!((yy - y).abs() < 1.0, "yaw {y} vs {yy}");
            assert!((pp - p).abs() < 1.0, "pitch {p} vs {pp}");
            assert!((rr - r).abs() < 1.0, "roll {r} vs {rr}");
        }
    }

    #[test]
    fn degenerate_zero_eye_distance_is_none() {
        let k = kps([
            (50.0, 50.0),
            (50.0, 50.0), // 眼距 = 0
            (50.0, 80.0),
            (40.0, 100.0),
            (60.0, 100.0),
        ]);
        assert!(estimate_yaw_pitch_roll(&k).is_none());
    }

    #[test]
    fn non_finite_input_is_none() {
        let k = kps([
            (f32::NAN, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (0.0, 2.0),
            (0.0, 3.0),
        ]);
        assert!(estimate_yaw_pitch_roll(&k).is_none());
    }
}
