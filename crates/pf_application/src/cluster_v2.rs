//! Phase 4 — Cluster V2 (prototype 验证)。
//!
//! 设计动机:
//! - 旧 `assign_or_create` 用 k-NN 投票 + top single score 判断"是不是这个人"
//! - 弱点:top_score 是相对最近邻 face 的相似度,不是相对 person 本身的
//! - Phase 3 给每个 person 产出 prototype(frontal/left_profile/right_profile/
//!   high_quality/general),Phase 4 改用 prototype 做"验证":给一个 query embedding,
//!   对每个候选 person,取其所有 prototype,挑其中 cosine 最大的一个作为分数
//! - 这样能区分"两张脸碰巧相似但属于不同人"和"两张脸属于同一人"
//!
//! 核心函数:
//! - `best_prototype_match`: 纯函数 — 给 query + candidates (pid → prototypes),
//!   返回 (best_pid, max_cosine)。无 prototype 的 candidate 跳过(由 caller 兜底)。
//! - `cosine_similarity`: 单位向量的 cosine(从 person.rs 抽出来共享)
//!
//! 集成位置:`PersonService::assign_or_create` 在 k-NN 投票后,取 top-N candidate,
//! 调 `prototype_service.list_for_person(pid)` 拿 prototype embeddings,
//! 走 `best_prototype_match` 决定最终人选。

/// (pid → 该 person 的 prototype embeddings)。
///
/// `Vec<f32>` 是单个 prototype 的 ArcFace embedding(已归一化)。
pub type CandidatePrototypes<'a> = &'a [(i64, Vec<Vec<f32>>)];

/// 给定 query embedding + 候选 persons 及其 prototype 集合,
/// 返回 cosine 最高的 (pid, score)。
///
/// 规则:
/// - 跳过 prototypes 为空的 candidate(由 caller 兜底到 top-neighbor score)
/// - 多个 candidate 取 max cosine across own prototypes(prototype 间不互相比较)
/// - 输入 embedding 应已 L2 归一化(prototype_service.list_for_person 返回的也是归一化的)
pub fn best_prototype_match(
    query: &[f32],
    candidates: CandidatePrototypes<'_>,
) -> Option<(i64, f32)> {
    candidates
        .iter()
        .filter(|(_, protos)| !protos.is_empty())
        .map(|(pid, protos)| {
            let max_sim = protos
                .iter()
                .map(|v| cosine_similarity(query, v))
                .fold(f32::MIN, f32::max);
            (*pid, max_sim)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}

/// 单位向量的 cosine 相似度(应已归一化;未归一化也工作,只是结果不是真 cosine)。
///
/// 与 person.rs::cosine_similarity 一致 — 复制是因为 pf_application 内禁止
/// `mod person` 反向引用,且该函数太短不值得做共享 utility 模块。
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-6);
    (dot / denom).clamp(0.0, 1.0)
}

/// 把任意向量 L2 归一化,返回新向量(不修改原向量)。
///
/// prototype verification 假定输入已归一化;此处提供 helper 便于测试和上层调用。
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    v.iter().map(|x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(values: &[f32]) -> Vec<f32> {
        l2_normalize(values)
    }

    #[test]
    fn best_match_picks_highest_cosine() {
        // query 与 P1 的 prototype cosine=1.0,与 P2 的 cosine≈0.5
        let q = unit_vec(&[1.0, 0.0, 0.0]);
        let p1_proto = unit_vec(&[1.0, 0.0, 0.0]);
        let p2_proto = unit_vec(&[1.0, 1.0, 0.0]); // 60° 角
        let candidates: Vec<(i64, Vec<Vec<f32>>)> = vec![
            (1, vec![p1_proto]),
            (2, vec![p2_proto]),
        ];
        let (pid, score) = best_prototype_match(&q, &candidates).unwrap();
        assert_eq!(pid, 1);
        assert!((score - 1.0).abs() < 1e-4);
    }

    #[test]
    fn best_match_takes_max_across_multiple_prototypes_of_same_person() {
        // P1 有两个 prototype:frontal (cosine=0.5) + left_profile (cosine=1.0)
        // 应取 max,即 left_profile
        let q = unit_vec(&[0.0, 1.0, 0.0]);
        let p1_frontal = unit_vec(&[1.0, 0.0, 0.0]);     // 90° from q
        let p1_profile = unit_vec(&[0.0, 1.0, 0.0]);     // 0° from q
        let candidates: Vec<(i64, Vec<Vec<f32>>)> = vec![(1, vec![p1_frontal, p1_profile])];
        let (pid, score) = best_prototype_match(&q, &candidates).unwrap();
        assert_eq!(pid, 1);
        assert!((score - 1.0).abs() < 1e-4, "score={score}");
    }

    #[test]
    fn best_match_skips_empty_candidates() {
        // P1 没 prototype,P2 有 → 只看 P2
        let q = unit_vec(&[1.0, 0.0]);
        let p2_proto = unit_vec(&[1.0, 0.0]);
        let candidates: Vec<(i64, Vec<Vec<f32>>)> = vec![
            (1, vec![]),
            (2, vec![p2_proto]),
        ];
        let (pid, score) = best_prototype_match(&q, &candidates).unwrap();
        assert_eq!(pid, 2);
        assert!(score > 0.99);
    }

    #[test]
    fn best_match_returns_none_when_all_empty() {
        let q = unit_vec(&[1.0, 0.0]);
        let candidates: Vec<(i64, Vec<Vec<f32>>)> = vec![(1, vec![]), (2, vec![])];
        assert!(best_prototype_match(&q, &candidates).is_none());
    }

    #[test]
    fn best_match_returns_none_for_empty_candidates() {
        let q = unit_vec(&[1.0, 0.0]);
        let candidates: Vec<(i64, Vec<Vec<f32>>)> = vec![];
        assert!(best_prototype_match(&q, &candidates).is_none());
    }

    #[test]
    fn cosine_similarity_identical() {
        let a = unit_vec(&[1.0, 2.0, 3.0]);
        let b = unit_vec(&[1.0, 2.0, 3.0]);
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = unit_vec(&[1.0, 0.0]);
        let b = unit_vec(&[0.0, 1.0]);
        assert!(cosine_similarity(&a, &b).abs() < 1e-4);
    }

    #[test]
    fn l2_normalize_produces_unit_norm() {
        let v = vec![3.0, 4.0];
        let n = l2_normalize(&v);
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }
}