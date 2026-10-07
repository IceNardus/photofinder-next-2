//! #127 — LFW 数据集定量评估（Recall@K / P-R / false merge-split / TTA）。
//!
//! 用法（先解压 LFW，默认路径 `~/Library/Caches/PhotoFinder/lfw-eval/lfw-deepfunneled/lfw-deepfunneled`）：
//! ```bash
//! cargo test --release -p pf_application --test lfw_eval -- --ignored --nocapture
//! ```
//!
//! 四个指标（全部 eprintln 打印 + 结构 sanity assert）：
//! 1. Recall@K + MRR —— 人脸检索命中同一身份的比例
//! 2. P-R 阈值扫描 —— 全对 cosine，τ∈0.30..0.90；标注生产阈值（搜索 0.50 / 聚类 0.65）
//! 3. False merge / false split —— 生产聚类 `PersonService::cluster_all` vs ground truth（B-cubed + 计数）
//! 4. TTA 决策 —— 6/2/1 变体（flip+旋转 / 仅 flip / 无）per-face 延迟 + 1NN 准确率 + max-F1
//!
//! 生产代码零改动：只测量现状。`crop_to_bbox_with_margin` 从 `pf_ai/src/face/mod.rs`
//! 复制（含 BUG#4 边界保护），避免改生产模块。

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use pf_ai::face::pose::estimate_yaw_pitch_roll;
use pf_ai::quality::{assess, blur_score_from_aligned, combined_quality};
use pf_ai::{
    AlignedFace, ArcFaceEmbedder, FaceAligner, FaceDetection, FaceDetector, FaceEmbedder,
    ImageData, QualityFilter, ScrfdDetector, SimpleAligner,
};
use pf_application::{ClusterPolicy, PersonService, PrototypeService};
use pf_core::{BBox, FaceKeypoints, FACE_MODEL_NAME};
use pf_database::{builtin_migrations, Database, FaceStatus, NewFace, NewImage};
use pf_vector::{HnswIndex, VectorIndex};

const DIM: usize = 512;
const SUBSET_PEOPLE: usize = 30;
const IMGS_PER_PERSON: usize = 8;

const DEFAULT_SCRFD: &str =
    "/Users/mac/ai-project/photofinder-next-2/models/scrfd_500m_bnkps.onnx";
const DEFAULT_ARC: &str = "/Users/mac/ai-project/photofinder-next-2/models/w600k_r50.onnx";
const DEFAULT_LFW_ROOT: &str =
    "/Users/mac/Library/Caches/PhotoFinder/lfw-eval/lfw-deepfunneled/lfw-deepfunneled";

/// 一张提取到的人脸（主 face = 该图质量最高者）。
struct FaceEntry {
    identity: String,
    path: PathBuf,
    /// 缓存对齐后的 112×112 人脸（供 TTA 对比，不重复 detect/align）
    aligned: AlignedFace,
    /// 6-TTA（生产默认）embedding
    emb6: Vec<f32>,
    quality: f32,
    pose: f32,
    blur: f32,
    area: f32,
    /// HNSW / face_embeddings 的 vector_id（建库时分配）
    vector_id: i64,
    /// DB face id（插入后回填）
    face_id: i64,
    /// 聚类结果（cluster_all 后回填）
    person_id: Option<i64>,
}

/// 从 LFW_ROOT 选子集：≥8 张/人的目录，按图数降序取前 30 人，每人取前 8 张。
fn select_subset(root: &Path) -> Vec<(String, Vec<PathBuf>)> {
    let mut people: Vec<(String, usize, Vec<PathBuf>)> = Vec::new();
    for entry in fs::read_dir(root).expect("LFW_ROOT readable") {
        let entry = entry.expect("entry");
        if !entry.path().is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let mut jpgs: Vec<PathBuf> = fs::read_dir(entry.path())
            .expect("person dir readable")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|e| e == "jpg").unwrap_or(false))
            .collect();
        jpgs.sort();
        if jpgs.len() >= IMGS_PER_PERSON {
            people.push((name, jpgs.len(), jpgs));
        }
    }
    people.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    people.truncate(SUBSET_PEOPLE);
    people
        .into_iter()
        .map(|(name, _, mut jpgs)| {
            jpgs.truncate(IMGS_PER_PERSON);
            (name, jpgs)
        })
        .collect()
}

/// `pf_ai/src/face/mod.rs` 私有函数复制（含 BUG#4 边界保护），不修改生产模块。
fn crop_to_bbox_with_margin(
    image: &ImageData,
    det: &FaceDetection,
) -> Result<(ImageData, FaceKeypoints), ()> {
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
        return Err(());
    }
    let crop_bbox = BBox::new(x, y, cw, ch);
    let cropped = image.crop(crop_bbox).map_err(|_| ())?;
    let shifted = FaceKeypoints {
        left_eye: (det.keypoints.left_eye.0 - x, det.keypoints.left_eye.1 - y),
        right_eye: (det.keypoints.right_eye.0 - x, det.keypoints.right_eye.1 - y),
        nose: (det.keypoints.nose.0 - x, det.keypoints.nose.1 - y),
        left_mouth: (det.keypoints.left_mouth.0 - x, det.keypoints.left_mouth.1 - y),
        right_mouth: (det.keypoints.right_mouth.0 - x, det.keypoints.right_mouth.1 - y),
    };
    Ok((cropped, shifted))
}

/// 单图提取主 face：复刻 `FacePipeline::process` 的 detect→align→质量过滤→embed（6-TTA），
/// 但缓存 AlignedFace。返回 (aligned, emb, quality, pose, blur, area)。
fn extract_best_face(
    rt: &tokio::runtime::Runtime,
    scrfd: &ScrfdDetector,
    aligner: &SimpleAligner,
    e6: &ArcFaceEmbedder,
    qf: &QualityFilter,
    image: &ImageData,
) -> Option<(AlignedFace, Vec<f32>, f32, f32, f32, f32)> {
    let dets = rt.block_on(scrfd.detect(image)).ok()?;
    let mut best: Option<(AlignedFace, Vec<f32>, f32, f32, f32, f32)> = None;
    for det in dets {
        let ypr = estimate_yaw_pitch_roll(&det.keypoints);
        let (ci, sk) = match crop_to_bbox_with_margin(image, &det) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let aligned = match aligner.align(&ci, &sk) {
            Ok(a) => a,
            Err(_) => continue,
        };
        let gray = aligned.image.to_luma8();
        let blur = blur_score_from_aligned(&gray);
        let q = assess(&det, blur, ypr);
        if !q.passes() {
            continue;
        }
        if det.score < qf.min_detector_score {
            continue;
        }
        let quality = combined_quality(det.score, q.face_area_score, q.blur_score, q.pose_score);
        let emb = rt.block_on(e6.embed(&aligned)).ok()?.values;
        let candidate = (aligned, emb, quality, q.pose_score, blur, q.face_area_score);
        if best.as_ref().map(|b| quality > b.2).unwrap_or(true) {
            best = Some(candidate);
        }
    }
    best
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Recall@K + MRR@10（在 HNSW 上）。排除 query 自身（fetch k+1 保证排除后仍有 k 个结果）。
fn recall_and_mrr(index: &dyn VectorIndex, faces: &[FaceEntry]) {
    let vid_identity: HashMap<i64, &str> =
        faces.iter().map(|f| (f.vector_id, f.identity.as_str())).collect();
    for k in [1usize, 5, 10, 20] {
        let mut hits = 0usize;
        for f in faces {
            let res = index.search(&f.emb6, k + 1).unwrap_or_default();
            let hit = res
                .iter()
                .filter(|h| h.id != f.vector_id)
                .take(k)
                .any(|h| {
                    vid_identity
                        .get(&h.id)
                        .map(|id| *id == f.identity)
                        .unwrap_or(false)
                });
            if hit {
                hits += 1;
            }
        }
        eprintln!(
            "  Recall@{k:<2} = {:.4}  ({}/{} faces)",
            hits as f32 / faces.len() as f32,
            hits,
            faces.len()
        );
    }
    let mut rr = 0.0f32;
    for f in faces {
        let res = index.search(&f.emb6, 11).unwrap_or_default();
        let mut rank = 0usize;
        for h in res.iter().filter(|h| h.id != f.vector_id) {
            rank += 1;
            if vid_identity
                .get(&h.id)
                .map(|id| *id == f.identity)
                .unwrap_or(false)
            {
                rr += 1.0 / rank as f32;
                break;
            }
        }
    }
    eprintln!("  MRR@10 = {:.4}", rr / faces.len() as f32);
}

/// 同身份 vs 不同身份 cosine 分布（mean / p50 / p90）—— 给阈值校准提供量化依据。
fn cosine_distribution(embs: &[&[f32]], ids: &[&str], label: &str) {
    let n = embs.len();
    let mut same: Vec<f32> = Vec::new();
    let mut diff: Vec<f32> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let s = dot(embs[i], embs[j]);
            if ids[i] == ids[j] {
                same.push(s);
            } else {
                diff.push(s);
            }
        }
    }
    same.sort_by(|a, b| a.partial_cmp(b).unwrap());
    diff.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |v: &Vec<f32>, q: f32| v[(q * (v.len().max(2) - 1) as f32) as usize];
    let mean = |v: &Vec<f32>| v.iter().sum::<f32>() / v.len() as f32;
    eprintln!(
        "  {label}: same(n={}) mean={:.4} p50={:.4} p90={:.4} | diff(n={}) mean={:.4} p50={:.4} p90={:.4}",
        same.len(),
        mean(&same),
        p(&same, 0.5),
        p(&same, 0.9),
        diff.len(),
        mean(&diff),
        p(&diff, 0.5),
        p(&diff, 0.9)
    );
}

/// 全对 cosine P-R 扫描。`embs[i]` 与 `ids[i]` 一一对应。
/// 打印每 5 档 + 0.50/0.65 两处，返回 (max_f1_tau, max_f1)。
fn pr_sweep(embs: &[&[f32]], ids: &[&str], label: &str) -> (f32, f32) {
    let n = embs.len();
    let mut pairs: Vec<(f32, bool)> = Vec::with_capacity(n * n / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((dot(embs[i], embs[j]), ids[i] == ids[j]));
        }
    }
    let total = pairs.len();
    let mut best = (0.0f32, 0.0f32);
    eprintln!("  --- P-R sweep: {label} ---");
    eprintln!("    tau   prec   rec    f1    acc");
    for t100 in (30..=90).step_by(1) {
        let tau = t100 as f32 / 100.0;
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fnc = 0usize;
        let mut tn = 0usize;
        for &(s, same) in &pairs {
            if same {
                if s >= tau {
                    tp += 1;
                } else {
                    fnc += 1;
                }
            } else if s >= tau {
                fp += 1;
            } else {
                tn += 1;
            }
        }
        let prec = if tp + fp == 0 {
            0.0
        } else {
            tp as f32 / (tp + fp) as f32
        };
        let rec = if tp + fnc == 0 {
            0.0
        } else {
            tp as f32 / (tp + fnc) as f32
        };
        let f1 = if prec + rec == 0.0 {
            0.0
        } else {
            2.0 * prec * rec / (prec + rec)
        };
        let acc = (tp + tn) as f32 / total as f32;
        if t100 % 5 == 0 || t100 == 50 || t100 == 65 {
            eprintln!("    {:.2}  {:.4} {:.4} {:.4} {:.4}", tau, prec, rec, f1, acc);
        }
        if f1 > best.1 {
            best = (tau, f1);
        }
    }
    eprintln!("    >> max-F1 = {:.4} @ tau = {:.2}", best.1, best.0);
    best
}

/// 聚类 vs ground truth：B-cubed P/R/F1 + false merge / false split 计数。
fn clustering_metrics(faces: &[FaceEntry]) {
    let n = faces.len();
    let mut person_faces: HashMap<Option<i64>, Vec<usize>> = HashMap::new();
    let mut id_faces: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, f) in faces.iter().enumerate() {
        person_faces.entry(f.person_id).or_default().push(i);
        id_faces.entry(f.identity.as_str()).or_default().push(i);
    }
    eprintln!("  persons (clusters) = {}, identities = {}", person_faces.len(), id_faces.len());

    let mut bc_p = 0.0f64;
    let mut bc_r = 0.0f64;
    for f in faces {
        let cluster = person_faces.get(&f.person_id).expect("in person_faces");
        let ident = id_faces.get(f.identity.as_str()).expect("in id_faces");
        let correct = cluster.iter().filter(|&&j| faces[j].identity == f.identity).count();
        bc_p += correct as f64 / cluster.len() as f64;
        bc_r += correct as f64 / ident.len() as f64;
    }
    bc_p /= n as f64;
    bc_r /= n as f64;
    let bc_f1 = 2.0 * bc_p * bc_r / (bc_p + bc_r);
    eprintln!("  B-cubed precision = {:.4}", bc_p);
    eprintln!("  B-cubed recall    = {:.4}", bc_r);
    eprintln!("  B-cubed F1        = {:.4}", bc_f1);

    // False merge：簇含 ≥2 个身份
    let mut merge_clusters = 0usize;
    let mut merge_pairs = 0usize;
    for cluster in person_faces.values() {
        let distinct: HashSet<&str> = cluster.iter().map(|&j| faces[j].identity.as_str()).collect();
        if distinct.len() >= 2 {
            merge_clusters += 1;
            for a in 0..cluster.len() {
                for b in (a + 1)..cluster.len() {
                    if faces[cluster[a]].identity != faces[cluster[b]].identity {
                        merge_pairs += 1;
                    }
                }
            }
        }
    }
    eprintln!("  False-merge: {merge_clusters} clusters with ≥2 identities, {merge_pairs} cross-identity pairs merged");

    // False split：身份散到 ≥2 个簇
    let mut split_ids = 0usize;
    let mut split_pairs = 0usize;
    for ident in id_faces.values() {
        let distinct: HashSet<Option<i64>> = ident.iter().map(|&j| faces[j].person_id).collect();
        if distinct.len() >= 2 {
            split_ids += 1;
            for a in 0..ident.len() {
                for b in (a + 1)..ident.len() {
                    if faces[ident[a]].person_id != faces[ident[b]].person_id {
                        split_pairs += 1;
                    }
                }
            }
        }
    }
    eprintln!("  False-split: {split_ids} identities split across ≥2 clusters, {split_pairs} same-identity pairs split");
}

/// 两个向量之间的 cosine 一致度：mean abs diff 之外，直接算 mean cosine。
fn mean_embedding_cosine(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    let mut acc = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        acc += dot(x, y);
    }
    acc / a.len() as f32
}

/// TTA 6/2/1 对比：per-face 延迟 + 1NN 同身份准确率 + max-F1。
/// 先算齐三个变体，再交叉对比 1/2/6 之间 embedding 一致度 —— 确认"2-crop == 1-crop"
/// 是 flip 对已对齐正脸近似恒等，而非 harness bug。
fn tta_decision(
    rt: &tokio::runtime::Runtime,
    embedders: &[(&str, &ArcFaceEmbedder)],
    faces: &[FaceEntry],
) {
    let ids: Vec<&str> = faces.iter().map(|f| f.identity.as_str()).collect();
    let mut collected: Vec<(&str, Vec<Vec<f32>>)> = Vec::new();
    for (name, e) in embedders {
        let mut embs: Vec<Vec<f32>> = Vec::with_capacity(faces.len());
        let t0 = Instant::now();
        for f in faces {
            match rt.block_on(e.embed(&f.aligned)) {
                Ok(emb) => embs.push(emb.values),
                Err(err) => {
                    eprintln!("    {name}: embed failed: {err}");
                    return;
                }
            }
        }
        let elapsed = t0.elapsed();
        let per = elapsed / faces.len() as u32;
        collected.push((name, embs.clone()));

        let embs_ref: Vec<&[f32]> = embs.iter().map(|v| v.as_slice()).collect();
        let mut nn_hits = 0usize;
        for i in 0..faces.len() {
            let mut best_j = None;
            let mut best_s = -1.0f32;
            for j in 0..faces.len() {
                if i == j {
                    continue;
                }
                let s = dot(&embs[i], &embs[j]);
                if s > best_s {
                    best_s = s;
                    best_j = Some(j);
                }
            }
            if let Some(j) = best_j {
                if ids[j] == ids[i] {
                    nn_hits += 1;
                }
            }
        }
        let (tau, f1) = pr_sweep(&embs_ref, &ids, name);
        eprintln!(
            "    {name}: per_face={per:?}  1NN-acc={:.4}  max-F1={f1:.4} @tau={tau:.2}",
            nn_hits as f32 / faces.len() as f32
        );
    }

    let cos = |na: &str, nb: &str| -> f32 {
        let a = collected.iter().find(|(n, _)| *n == na).map(|(_, e)| e).unwrap();
        let b = collected.iter().find(|(n, _)| *n == nb).map(|(_, e)| e).unwrap();
        mean_embedding_cosine(a, b)
    };
    eprintln!(
        "  cross-variant embedding agreement (mean cosine on same faces):\n    \
         1-crop vs 2-crop = {:.4}\n    1-crop vs 6-crop = {:.4}",
        cos("1-crop", "2-crop"),
        cos("1-crop", "6-crop")
    );
    eprintln!(
        "    (>0.999 = flip/旋转对已对齐正脸近似恒等，TTA 不改变识别结果)"
    );
}

#[test]
#[ignore] // 需要模型 + LFW 数据
fn lfw_eval() {
    let lfw_root = std::env::var("LFW_ROOT").unwrap_or_else(|_| DEFAULT_LFW_ROOT.to_string());
    let scrfd_path = std::env::var("SCRFD_MODEL").unwrap_or_else(|_| DEFAULT_SCRFD.to_string());
    let arc_path = std::env::var("FACE_MODEL").unwrap_or_else(|_| DEFAULT_ARC.to_string());
    eprintln!(
        "=== LFW eval: root={lfw_root}\n    scrfd={scrfd_path}\n    arcface={arc_path}"
    );

    let subset = select_subset(Path::new(&lfw_root));
    eprintln!(
        "subset: {} people, {} images ({}×{})",
        subset.len(),
        subset.iter().map(|(_, p)| p.len()).sum::<usize>(),
        SUBSET_PEOPLE,
        IMGS_PER_PERSON
    );
    assert!(subset.len() >= 20, "too few eligible people: {}", subset.len());

    // 模型加载（6-TTA 参照 embedder + 2/1 变体）。
    // 注：生产默认已改为 1-crop（见 ArcFaceEmbedder::load），这里显式开启 flip+rotation
    // 保留"6-crop"作为 TTA 对比的上界参照。
    let scrfd = ScrfdDetector::load(Path::new(&scrfd_path)).expect("load scrfd");
    let aligner = SimpleAligner::new();
    let e6 = Arc::new(
        Arc::try_unwrap(ArcFaceEmbedder::load(Path::new(&arc_path)).expect("load arcface"))
            .ok()
            .expect("unwrap arc 6")
            .with_flip(true)
            .with_rotation_tta(true),
    );
    // 2-crop = flip only。默认已改为 1-crop,必须显式 with_flip(true) 才能得到真 2-crop。
    let e2 = Arc::new(
        Arc::try_unwrap(ArcFaceEmbedder::load(Path::new(&arc_path)).expect("load arcface 2"))
            .ok()
            .expect("unwrap arc 2")
            .with_flip(true)
            .with_rotation_tta(false),
    );
    let e1 = Arc::new(
        Arc::try_unwrap(ArcFaceEmbedder::load(Path::new(&arc_path)).expect("load arcface 1"))
            .ok()
            .expect("unwrap arc 1")
            .with_flip(false)
            .with_rotation_tta(false),
    );
    let qf = QualityFilter::from_config(0.5, 20, 0.5, 90.0);
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");

    // 提取人脸
    let mut faces: Vec<FaceEntry> = Vec::new();
    let mut images_processed = 0usize;
    for (identity, paths) in &subset {
        for p in paths {
            images_processed += 1;
            let bytes = match fs::read(p) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("  read fail {p:?}: {e}");
                    continue;
                }
            };
            let img = match ImageData::from_bytes(&bytes) {
                Ok(i) => i,
                Err(e) => {
                    eprintln!("  decode fail {p:?}: {e}");
                    continue;
                }
            };
            if let Some((aligned, emb6, quality, pose, blur, area)) =
                extract_best_face(&rt, &scrfd, &aligner, &e6, &qf, &img)
            {
                faces.push(FaceEntry {
                    identity: identity.clone(),
                    path: p.clone(),
                    aligned,
                    emb6,
                    quality,
                    pose,
                    blur,
                    area,
                    vector_id: 0,
                    face_id: 0,
                    person_id: None,
                });
            }
        }
    }
    let detection_rate = faces.len() as f32 / images_processed as f32;
    eprintln!("\n=== Extraction: {} faces from {} images (rate {:.3}) ===", faces.len(), images_processed, detection_rate);
    assert!(faces.len() >= 200, "too few faces extracted: {}", faces.len());
    assert!(detection_rate >= 0.8, "detection rate too low: {detection_rate}");
    for f in &faces {
        assert!(f.emb6.iter().all(|x| x.is_finite()), "non-finite embedding");
    }

    // 建 DB + HNSW
    let db = Arc::new(Database::open_in_memory(builtin_migrations()).expect("in-memory db"));
    let dir = std::env::temp_dir().join(format!(
        "pf-lfw-eval-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    fs::create_dir_all(&dir).expect("tempdir");
    let index: Arc<dyn VectorIndex> = Arc::new(HnswIndex::new(DIM, &dir, "lfw"));

    let mut vid = 0i64;
    for f in &mut faces {
        vid += 1;
        f.vector_id = vid;
        let emb = pf_application::l2_normalize(&f.emb6);
        let face_id = db
            .transaction(|tx| {
                let image_id = tx.images().insert(&NewImage {
                    path: f.path.to_string_lossy().into_owned(),
                    hash: format!("lfw-{vid}"),
                    size: 0,
                    modified_time: 0,
                    width: 250,
                    height: 250,
                    captured_at: None,
                })?;
                let fid = tx.faces().insert(&NewFace {
                    image_id,
                    bbox: BBox::new(0.0, 0.0, 100.0, 100.0),
                    detector_score: 0.9,
                    detector_model: "scrfd-500m-bnkps".into(),
                    keypoints_json: None,
                    yaw_pitch_roll: None,
                    quality: f.quality,
                    blur_score: Some(f.blur),
                    pose_score: Some(f.pose),
                    face_area_score: Some(f.area),
                    alignment_version: Some("v1".into()),
                    embedding_model: Some(FACE_MODEL_NAME.into()),
                    model_version: "arcface@v1".into(),
                    vector_id: Some(f.vector_id),
                    hnsw_handle: Some(f.vector_id),
                    status: FaceStatus::Indexed,
                    index_generation: 0,
                    cluster_score: None,
                    cluster_method: None,
                })?;
                tx.faces().insert_embedding(
                    fid,
                    FACE_MODEL_NAME,
                    "v1",
                    DIM as i64,
                    &emb,
                    Some(f.vector_id),
                    true,
                )?;
                Ok::<_, pf_database::DatabaseError>(fid)
            })
            .expect("insert face");
        f.face_id = face_id;
        index.insert(f.vector_id, &emb).expect("index insert");
    }
    eprintln!("indexed {} faces in HNSW", faces.len());

    // 1. Recall@K + MRR
    eprintln!("\n=== [1] Recall@K / MRR (6-TTA embeddings, HNSW) ===");
    recall_and_mrr(index.as_ref(), &faces);

    // 2. P-R 扫描（6-TTA，全对 cosine）
    eprintln!("\n=== [2] P-R threshold sweep (6-TTA, {} faces, {} pairs) ===", faces.len(), faces.len() * (faces.len() - 1) / 2);
    let embs_ref: Vec<&[f32]> = faces.iter().map(|f| f.emb6.as_slice()).collect();
    let ids: Vec<&str> = faces.iter().map(|f| f.identity.as_str()).collect();
    pr_sweep(&embs_ref, &ids, "6-TTA");
    eprintln!("\n  cosine distribution (threshold calibration context):");
    cosine_distribution(&embs_ref, &ids, "6-TTA");

    // 3. 聚类（生产路径）→ false merge / false split
    eprintln!("\n=== [3] Clustering (PersonService::cluster_all, production path) ===");
    let person = PersonService::new(db.clone(), index.clone(), ClusterPolicy::default())
        .with_prototype_service(Arc::new(PrototypeService::new(db.clone())));
    let summary = rt.block_on(person.cluster_all()).expect("cluster_all");
    eprintln!("  cluster_all summary: assigned={} created={}", summary.assigned, summary.created);
    assert!(summary.created >= 2, "clustering collapsed to <2 persons");

    let rows = db
        .transaction(|tx| tx.faces().list_all())
        .expect("list faces");
    let person_by_face: HashMap<i64, Option<i64>> =
        rows.iter().map(|r| (r.id, r.person_id)).collect();
    for f in &mut faces {
        f.person_id = person_by_face.get(&f.face_id).copied().flatten();
    }
    clustering_metrics(&faces);

    // 4. TTA 决策
    eprintln!("\n=== [4] TTA 6/2/1 decision (cached aligned faces) ===");
    tta_decision(&rt, &[("6-crop", &e6), ("2-crop", &e2), ("1-crop", &e1)], &faces);

    eprintln!("\n=== DONE ===");
}
