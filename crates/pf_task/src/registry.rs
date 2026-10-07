//! `TaskExecutor` 注册表。
//!
//! `pf_task` 不知道具体业务（不 import `pf_ai` / `pf_application`）。
//! `pf_application` 实现 `TaskExecutor` trait 并通过本注册表提供给 `SqliteTaskScheduler`。
//!
//! 设计：线程安全的 `HashMap<TaskKindDiscriminant, Arc<dyn TaskExecutor>>`。

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::executor::TaskExecutor;
use crate::scheduler::TaskKindDiscriminant;

/// `TaskExecutor` 注册表。
#[derive(Default)]
pub struct ExecutorRegistry {
    inner: RwLock<HashMap<TaskKindDiscriminant, Arc<dyn TaskExecutor>>>,
}

impl ExecutorRegistry {
    /// 构造空注册表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 注册一个 executor（按 `kind()` 索引）。
    pub fn register(&self, exec: Arc<dyn TaskExecutor>) {
        let kind = exec.kind();
        self.inner.write().insert(kind, exec);
    }

    /// 取出 executor。
    pub fn get(&self, kind: TaskKindDiscriminant) -> Option<Arc<dyn TaskExecutor>> {
        self.inner.read().get(&kind).cloned()
    }

    /// 当前已注册数量。
    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }

    /// 列出所有已注册判别式。
    pub fn kinds(&self) -> Vec<TaskKindDiscriminant> {
        self.inner.read().keys().copied().collect()
    }
}

impl std::fmt::Debug for ExecutorRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let g = self.inner.read();
        f.debug_struct("ExecutorRegistry")
            .field("count", &g.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::{TaskExecutor, TaskOutcome};
    use crate::progress::TaskContext;
    use crate::TaskError;

    struct DummyExec {
        kind: TaskKindDiscriminant,
    }

    #[async_trait::async_trait]
    impl TaskExecutor for DummyExec {
        fn kind(&self) -> TaskKindDiscriminant {
            self.kind
        }
        fn name(&self) -> &'static str {
            "dummy"
        }
        async fn execute(&self, _ctx: &TaskContext) -> Result<TaskOutcome, TaskError> {
            Ok(TaskOutcome::Success(None))
        }
    }

    #[test]
    fn register_and_get() {
        let reg = ExecutorRegistry::new();
        assert!(reg.is_empty());
        reg.register(Arc::new(DummyExec {
            kind: TaskKindDiscriminant::Scan,
        }));
        reg.register(Arc::new(DummyExec {
            kind: TaskKindDiscriminant::IndexFace,
        }));
        assert_eq!(reg.len(), 2);
        assert!(reg.get(TaskKindDiscriminant::Scan).is_some());
        assert!(reg.get(TaskKindDiscriminant::IndexFace).is_some());
        assert!(reg.get(TaskKindDiscriminant::ClusterFaces).is_none());
        let mut kinds = reg.kinds();
        kinds.sort_by_key(|k| *k as u8);
        assert_eq!(kinds.len(), 2);
    }
}
