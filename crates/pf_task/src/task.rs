//! 任务定义（数据）。

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::scheduler::TaskKindDiscriminant;

/// 任务种类。
///
/// 注意：领域语义在 `pf_application` 里通过 `TaskExecutor` 实现，这里只做数据载体。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TaskKind {
    /// 扫描一个文件夹
    Scan {
        /// 文件夹 ID（由调用方映射）
        folder_id: i64,
    },
    /// 对单张图片做人脸索引
    IndexFace {
        /// 图片 id
        image_id: i64,
    },
    /// 对单张图片做对象索引
    IndexObject {
        /// 图片 id
        image_id: i64,
    },
    /// 对单张图片做完整索引（face + object 顺序执行，批量化 DB 写入）。
    ///
    /// 替代 `IndexFace + IndexObject` 配对：从根本上消除同一图片的并发
    /// face/object 写竞争。Phase 8+ 推荐用此任务。
    IndexImage {
        /// 图片 id
        image_id: i64,
    },
    /// 对单张图片做 patch 索引（Phase 3）
    IndexPatch {
        /// 图片 id
        image_id: i64,
    },
    /// 重建某种索引（全部）
    BuildIndex {
        /// 哪种索引
        kind: IndexKind,
    },
    /// 人物聚类
    ClusterFaces,
}

/// 索引种类。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexKind {
    /// 人脸索引
    Face,
    /// 对象索引
    Object,
    /// Patch 索引
    Patch,
}

/// 优先级。值越大优先级越高。
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub enum Priority {
    /// 低
    Low = 0,
    /// 正常
    #[default]
    Normal = 1,
    /// 高
    High = 2,
    /// 紧急
    Urgent = 3,
}

/// 任务状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// 等待执行
    Pending,
    /// 正在执行
    Running,
    /// 已完成
    Completed,
    /// 已失败（可重试）
    Failed,
    /// 已取消
    Cancelled,
}

/// 一条任务。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// DB 主键
    pub id: i64,
    /// 任务类型 + 参数
    pub kind: TaskKind,
    /// 优先级
    pub priority: Priority,
    /// 当前状态
    pub status: TaskStatus,
    /// 已重试次数
    pub retry_count: u32,
    /// 最大重试次数
    pub max_retries: u32,
    /// 错误消息（失败时填）
    pub error: Option<String>,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 开始时间
    pub started_at: Option<DateTime<Utc>>,
    /// 完成时间
    pub completed_at: Option<DateTime<Utc>>,
}

/// 任务过滤器（用于 `TaskScheduler::list`）。
#[derive(Debug, Clone, Default)]
pub struct TaskFilter {
    /// 只看某种状态
    pub status: Option<TaskStatus>,
    /// 只看某种类型
    pub kind: Option<TaskKindDiscriminant>,
    /// 最多返回多少条
    pub limit: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_ordering() {
        assert!(Priority::Urgent > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn task_kind_equality() {
        let a = TaskKind::IndexFace { image_id: 1 };
        let b = TaskKind::IndexFace { image_id: 1 };
        assert_eq!(a, b);
    }
}