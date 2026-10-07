//! 聚类抽象。
//!
//! pf_core 只定义 trait 和数据，不实现具体算法。HDBSCAN / BFS / KMeans 等具体实现
//! 由 `pf_ai::clustering::*` 或 `pf_application` 提供。

use crate::domain::Embedding;

/// 聚类器：把一个新的 embedding 分配到已有的 cluster，或者创建新 cluster。
///
/// 实现方负责：
/// - 状态管理（cluster centers / DBSCAN 邻域等）
/// - 距离度量（cosine / euclidean）
/// - 持久化（如果需要）
pub trait Clusterer: Send + Sync {
    /// 把 embedding 分配到现有 cluster，或建议创建新 cluster。
    fn assign_or_create(&self, embedding: &Embedding) -> PersonAssignment;
}

/// 聚类分配结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersonAssignment {
    /// 分配到已存在的 person（id = person_id）
    Existing(i64),
    /// 应创建新 person
    New {
        /// 建议的新 person id（由调用方落库后得到真实 id）
        person_id: i64,
    },
}