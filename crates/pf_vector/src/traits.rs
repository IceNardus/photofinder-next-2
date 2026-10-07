//! 向量索引 trait。

use serde::{Deserialize, Serialize};

use crate::error::VectorError;

/// 单条命中结果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHit {
    /// 向量 id（由调用方映射到 image_id / face_id / object_id / patch_id）
    pub id: i64,
    /// 相似度（越大越相似，cosine 范围 0..1）
    pub score: f32,
}

/// 距离度量。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    /// 余弦相似度（输入向量通常先归一化）
    Cosine,
    /// 欧氏距离（越小越相似；接口统一返回「越大越相似」的 score）
    Euclidean,
    /// 内积（向量已归一化时等价 cosine）
    InnerProduct,
}

/// 向量索引抽象。
///
/// 实现要求：
/// - `Send + Sync`
/// - 内部可变性（实现者内部 `Mutex` / `RwLock`），通过 `&self` 即可读写
/// - `insert` 后立即可 `search`
/// - `save` 后必须可用 `new + load` 重建出等价状态
pub trait VectorIndex: Send + Sync {
    /// 向量维度。
    fn dim(&self) -> usize;

    /// 当前已插入数量。
    fn len(&self) -> usize;

    /// 是否为空。
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 插入单条向量。
    fn insert(&self, id: i64, vector: &[f32]) -> Result<(), VectorError>;

    /// 批量插入（实现可优化以减少锁竞争）。
    fn insert_batch(&self, items: &[(i64, &[f32])]) -> Result<(), VectorError> {
        for (id, v) in items {
            self.insert(*id, v)?;
        }
        Ok(())
    }

    /// 搜索 top_k。
    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchHit>, VectorError>;

    /// 删除（可选；返回是否真的删除了）。
    fn remove(&self, _id: i64) -> Result<bool, VectorError> {
        Err(VectorError::Index("remove not supported by this impl".into()))
    }

    /// 清空索引(plan §11 Phase 2 rebuild_face_index 用)。
    ///
    /// HnswIndex 实现:重新构造空的 Hnsw 结构 + 清空 ids。
    /// 没有"清除后保留已插入元素"语义——调用方应保证已 save 持久化或准备好重建。
    fn clear(&self) -> Result<(), VectorError> {
        // 默认实现 no-op(便于不关心 clear 的实现 back-compat)
        Ok(())
    }

    /// 持久化到磁盘。
    fn save(&self) -> Result<(), VectorError>;

    /// 从磁盘加载（会清空当前状态）。
    fn load(&self) -> Result<(), VectorError>;

    /// 距离度量。
    fn metric(&self) -> Metric;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_hit_equality() {
        let a = SearchHit { id: 1, score: 0.9 };
        let b = SearchHit { id: 1, score: 0.9 };
        assert_eq!(a, b);
    }
}