//! `pf_core` 错误类型。
//!
//! pf_core 是最底层，错误必须能被所有上层 crate 用 `?` 传播。

use thiserror::Error;

/// `pf_core` 的统一错误。
#[derive(Debug, Error)]
pub enum CoreError {
    /// BBox 几何计算错误（如负面积、非法坐标）
    #[error("invalid bbox: {0}")]
    InvalidBBox(String),

    /// 类型转换错误
    #[error("conversion: {0}")]
    Conversion(String),

    /// 不变量被违反（应在 debug_assertions 中触发）
    #[error("invariant violated: {0}")]
    Invariant(String),
}