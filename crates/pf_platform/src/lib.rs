//! # pf_platform — 平台抽象
//!
//! 唯一允许触碰 OS / 文件系统 / 平台权限 API 的层。
//!
//! 设计：
//! - `PhotoProvider` trait — 列出照片 / 读取字节 / 请求权限
//! - `PathResolver` trait — 模型 / 数据 / 缓存路径
//! - 每个平台（desktop / android / ios）独立 impl
//!
//! pf_platform 不依赖任何业务 crate（pf_ai / pf_database / pf_vector / pf_application）。
//!
//! ## 平台分发
//!
//! | 平台   | PathResolver            | PhotoProvider              | model_path 策略      |
//! |--------|-------------------------|----------------------------|----------------------|
//! | desktop| `DesktopPathResolver`   | `FileSystemPhotoProvider`  | exe 旁 / bundle / CWD |
//! | android| `AndroidPathResolver`   | `MediaStorePhotoProvider`  | app files dir        |
//! | ios    | `IosPathResolver`       | `PhotoKitPhotoProvider`    | app bundle resources |
//!
//! 平台 impl 通过 `#[cfg(target_os = "...")]` 分发：
//! - `dirs` crate 仅 desktop 依赖
//! - Android impl 用 `std::env`（APP 数据目录通过 `Context.getFilesDir()` JNI 注入）
//! - iOS impl 用 `std::env`（bundle 路径通过 `Bundle.main.path(forResource:)` FFI 注入）

#![deny(unsafe_code)]
#![warn(missing_docs)]

// Desktop-only 模块（FileSystemPhotoProvider 用 walkdir 扫文件系统；移动端走 MediaStore / PhotoKit）
#[cfg(not(any(target_os = "android", target_os = "ios")))]
pub mod desktop;

#[cfg(not(any(target_os = "android", target_os = "ios")))]
pub mod filesystem_provider;

pub mod error;
pub mod path_resolver;
pub mod types;

// 移动端平台模块
#[cfg(target_os = "android")]
pub mod android;

// iOS 模块 — 类型 + provider 永远编译(便于 ios-bridge 在 host 上跑 tests),
// 内部的 `imp` 子模块 unsafe extern "C" 仍只在 iOS target 编译。
pub mod ios;

pub use error::PlatformError;
pub use path_resolver::PathResolver;
pub use types::{
    PageRequest, PermissionStatus, PhotoEntry, PhotoFolder, PhotoId, PhotoMetadata, PhotoProvider,
};

// Desktop-only re-exports
#[cfg(not(any(target_os = "android", target_os = "ios")))]
pub use desktop::DesktopPathResolver;
#[cfg(not(any(target_os = "android", target_os = "ios")))]
pub use filesystem_provider::FileSystemPhotoProvider;

// Android 平台再导出
#[cfg(target_os = "android")]
pub use android::{AndroidPathResolver, MediaStorePhotoProvider};

// iOS 平台再导出
pub use ios::{IosNativeMediaBridge, IosPathResolver, NativeMediaBridge, NoopNativeMediaBridge, PhotoKitPhotoProvider};
