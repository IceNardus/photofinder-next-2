//! `pf_ai` 错误类型。

use thiserror::Error;

/// `pf_ai` 的统一错误。
#[derive(Debug, Error)]
pub enum AIError {
    /// 模型加载失败
    #[error("model load: {0}")]
    ModelLoad(String),

    /// 推理失败
    #[error("inference: {0}")]
    Inference(String),

    /// 预处理失败
    #[error("preprocess: {0}")]
    Preprocess(String),

    /// 后处理失败
    #[error("postprocess: {0}")]
    Postprocess(String),

    /// 输入非法（如 NaN / 空 / 维度错）
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// IO 错误
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// 图片解码失败
    #[error("image decode: {0}")]
    ImageDecode(String),

    /// 模型未加载
    #[error("model not loaded: {0}")]
    NotLoaded(String),

    /// 向量索引错误
    #[error("vector: {0}")]
    Vector(String),
}