//! ONNX Runtime 会话配置选项。

/// ONNX Runtime 会话构建选项。
///
/// 用于配置推理线程数、内存模式等参数。
#[derive(Debug, Clone, Default)]
pub struct OrtSessionOptions {
    /// 运算符内的线程数（intra-op parallelism）。
    /// 控制单个运算符内的并行化程度。
    pub intra_threads: Option<usize>,
    /// 运算符间的线程数（inter-op parallelism）。
    /// 控制不同运算符间的并行化程度。
    pub inter_threads: Option<usize>,
    /// 是否启用内存分配模式。
    /// 启用后可能减少推理延迟但增加内存使用。
    pub memory_pattern: bool,
}

impl OrtSessionOptions {
    /// 创建默认配置。
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置 intra-op 线程数。
    pub fn with_intra_threads(mut self, n: usize) -> Self {
        self.intra_threads = Some(n);
        self
    }

    /// 设置 inter-op 线程数。
    pub fn with_inter_threads(mut self, n: usize) -> Self {
        self.inter_threads = Some(n);
        self
    }

    /// 设置内存模式。
    pub fn with_memory_pattern(mut self, enable: bool) -> Self {
        self.memory_pattern = enable;
        self
    }

    /// 从环境变量读取配置。
    ///
    /// 环境变量：
    /// - `PF_ORT_INTRA_THREADS` - intra-op 线程数
    /// - `PF_ORT_INTER_THREADS` - inter-op 线程数
    pub fn from_env() -> Self {
        let mut opts = Self::default();
        if let Ok(v) = std::env::var("PF_ORT_INTRA_THREADS") {
            if let Ok(n) = v.parse() {
                opts.intra_threads = Some(n);
            }
        }
        if let Ok(v) = std::env::var("PF_ORT_INTER_THREADS") {
            if let Ok(n) = v.parse() {
                opts.inter_threads = Some(n);
            }
        }
        opts
    }
}
