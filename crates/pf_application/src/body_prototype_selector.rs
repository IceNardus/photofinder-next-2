//! Body prototype selection — 从一个 person 的 bodies 中挑选代表性 body 作为 prototype.
//!
//! 设计（V2）：
//! - crop_type 分桶（HeadBody / UpperBody）
//! - 桶内 greedy single-linkage 去重（cosine ≥ 0.90 视为近重复）
//! - cap 8 个 prototype
//! - 质量过滤：quality_score < ACCEPTABLE_QUALITY 跳过
//!
//! YouTu Re-ID 模型不输出 pose 信息，所以不能像 face prototype 那样用 yaw 分桶。

use std::collections::HashMap;

use pf_ai::body::BodyCropStrategy;
use pf_ai::quality::ACCEPTABLE_QUALITY;
use pf_database::{BodyRow, NewPersonBodyPrototype};

use crate::cluster_v2::cosine_similarity;

/// 桶内去重阈值：同一 crop_type 桶内两个 body 的 embedding cosine ≥ 该值时
/// 视为近重复，只保留 quality 更高者。
const DEDUP_COSINE_THRESHOLD: f32 = 0.90;

/// 每个 person 的 body prototype 数量上限（cap 8，与 face prototype 一致）。
const MAX_BODY_PROTOTYPES: usize = 8;

/// Body prototype 候选 — selector 输出。
#[derive(Debug, Clone)]
pub struct BodyPrototypeCandidate {
    pub body_id: i64,
    pub embedding: Vec<f32>,
    pub crop_type: String,
    pub quality_score: f32,
    pub weight: f32,
    pub model_version: String,
    /// 是否为 Core prototype（通过 LOO 验证）
    pub is_core: bool,
}

/// 将 body crop_type 字符串转换为 BodyCropStrategy。
pub fn crop_type_to_strategy(crop_type: &str) -> BodyCropStrategy {
    match crop_type {
        "head_body" => BodyCropStrategy::HeadBody,
        "upper_body" => BodyCropStrategy::UpperBody,
        "bbox_tight" => BodyCropStrategy::BBoxTight,
        "bbox_p20" => BodyCropStrategy::BBoxP20,
        _ => BodyCropStrategy::HeadBody,
    }
}

/// 将 BodyCropStrategy 转换为字符串。
pub fn strategy_to_crop_type(strategy: BodyCropStrategy) -> &'static str {
    match strategy {
        BodyCropStrategy::HeadBody => "head_body",
        BodyCropStrategy::UpperBody => "upper_body",
        BodyCropStrategy::BBoxTight => "bbox_tight",
        BodyCropStrategy::BBoxP20 => "bbox_p20",
    }
}

/// 给定 person 的所有 bodies + 它们的 embedding，产出 body prototype 候选列表。
///
/// 规则（V2）：
/// 1. 过滤 quality < ACCEPTABLE_QUALITY 且必须有 embedding 的 body
/// 2. crop_type 分桶（HeadBody / UpperBody）
/// 3. 桶内 greedy single-linkage 聚类去重（cosine ≥ DEDUP_COSINE_THRESHOLD 视为近重复）
/// 4. cap 8：按 HeadBody > UpperBody 优先级填充
///
/// `embeddings` key = body_id, value = (Vec<f32>, String) = (embedding, model_version)。
pub fn select_body_prototypes_for_person(
    bodies: &[BodyRow],
    embeddings: &HashMap<i64, (Vec<f32>, String)>,
) -> Vec<BodyPrototypeCandidate> {
    // 1. 过滤
    let eligible: Vec<&BodyRow> = bodies
        .iter()
        .filter(|b| b.quality_score >= ACCEPTABLE_QUALITY)
        .filter(|b| embeddings.contains_key(&b.id))
        .collect();
    if eligible.is_empty() {
        return Vec::new();
    }

    let mk_candidate = |body: &BodyRow, ct: BodyCropStrategy, is_core: bool| -> BodyPrototypeCandidate {
        let (vec, model_version) = embeddings.get(&body.id).expect("pre-filtered").clone();
        BodyPrototypeCandidate {
            body_id: body.id,
            embedding: vec,
            crop_type: strategy_to_crop_type(ct).to_string(),
            quality_score: body.quality_score,
            weight: default_weight_for(ct),
            model_version,
            is_core,
        }
    };

    // 2. crop_type 分桶
    let mut head_body: Vec<&BodyRow> = Vec::new();
    let mut upper_body: Vec<&BodyRow> = Vec::new();
    for body in &eligible {
        let ct = crop_type_to_strategy(&body.crop_type);
        match ct {
            BodyCropStrategy::HeadBody => head_body.push(body),
            BodyCropStrategy::UpperBody => upper_body.push(body),
            _ => head_body.push(body),
        }
    }

    // 3+4. 桶内去重聚类 + cap 8
    let mut out: Vec<BodyPrototypeCandidate> = Vec::with_capacity(MAX_BODY_PROTOTYPES);
    for (bucket, ct) in [
        (head_body.as_slice(), BodyCropStrategy::HeadBody),
        (upper_body.as_slice(), BodyCropStrategy::UpperBody),
    ] {
        for rep in dedup_cluster_reps(bucket, embeddings) {
            if out.len() >= MAX_BODY_PROTOTYPES {
                break;
            }
            out.push(mk_candidate(rep, ct, false)); // 初始为 Expansion，等待 LOO 验证
        }
    }

    out
}

/// LOO (Leave-One-Out) 验证 — 检查移除某个样本后 prototype 是否仍然有效.
///
/// 返回 true 如果该样本应该进入 Core prototype（LOO 后仍然 match）。
///
/// `loo_threshold`: LOO 后 query 仍能 match prototype 的最低分数要求。
pub fn validate_loo_candidate(
    candidate_emb: &[f32],
    gallery_embs: &[Vec<f32>],
    loo_threshold: f32,
) -> bool {
    if gallery_embs.len() < 2 {
        // 单样本无法 LOO，默认进入 Core
        return true;
    }

    // 计算 LOO prototype（移除 candidate 后的均值）
    let mut loo_prototype = vec![0.0f32; candidate_emb.len()];
    for emb in gallery_embs.iter().filter(|e| !emb_same(e, candidate_emb)) {
        for (i, v) in emb.iter().enumerate() {
            loo_prototype[i] += v;
        }
    }
    let count = gallery_embs.len() - 1;
    for v in &mut loo_prototype {
        *v /= count as f32;
    }

    // L2 normalize
    let norm = loo_prototype.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    for v in &mut loo_prototype {
        *v /= norm;
    }

    // 验证 candidate_emb 与 LOO prototype 的相似度
    cosine_similarity(candidate_emb, &loo_prototype) >= loo_threshold
}

fn emb_same(a: &[f32], b: &[f32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-6)
}

fn default_weight_for(ct: BodyCropStrategy) -> f32 {
    match ct {
        BodyCropStrategy::HeadBody => 1.0,
        BodyCropStrategy::UpperBody => 0.8,
        BodyCropStrategy::BBoxTight => 0.9,
        BodyCropStrategy::BBoxP20 => 0.85,
    }
}

fn dedup_cluster_reps<'a>(
    bucket: &'a [&BodyRow],
    embeddings: &HashMap<i64, (Vec<f32>, String)>,
) -> Vec<&'a BodyRow> {
    let mut sorted: Vec<&&BodyRow> = bucket.iter().collect();
    sorted.sort_by(|a, b| b.quality_score.partial_cmp(&a.quality_score).unwrap_or(std::cmp::Ordering::Equal));
    let mut reps: Vec<&BodyRow> = Vec::new();
    'outer: for body in &sorted {
        let emb = &embeddings.get(&body.id).expect("pre-filtered").0;
        for rep in &reps {
            let rep_emb = &embeddings.get(&rep.id).expect("pre-filtered").0;
            if cosine_similarity(emb, rep_emb) >= DEDUP_COSINE_THRESHOLD {
                continue 'outer;
            }
        }
        reps.push(body);
    }
    reps
}
