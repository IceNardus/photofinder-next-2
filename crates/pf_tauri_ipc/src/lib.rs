//! # pf_tauri_ipc — Tauri IPC 共享层
//!
//! 桌面端和移动端共用同一套 Tauri commands / DTO / state。
//! 业务逻辑全部在 `pf_application` 里，IPC 层只做参数解析 + DTO 转换。
//!
//! ## 平台分发
//!
//! - `state` — `AppState` 包 `pf_application::Bootstrapped`（已经 platform-agnostic）
//! - `commands` — 所有 `#[tauri::command]` 函数，平台无关
//! - `ipc_types` — DTO（snake_case 字段，serde Serialize/Deserialize）
//! - `events` — Tauri event channel 名称
//!
//! 桌面端和移动端各自：
//! 1. 调用 `pf_application::assemble(...)` 装配（带各自平台的 PathResolver / PhotoProvider）
//! 2. 把 `Bootstrapped` 包成 `AppState` 塞进 Tauri
//! 3. `invoke_handler` 注册本 crate 的所有 commands

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod commands;
pub mod events;
pub mod ipc_types;
pub mod state;

pub use state::AppState;
