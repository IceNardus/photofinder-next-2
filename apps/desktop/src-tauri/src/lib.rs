//! # PhotoFinder Next 2 — Desktop Shell (Tauri 2)
//!
//! 这是整个项目的**桌面端入口**。架构约束：
//!
//! - 唯一允许依赖 `tauri::*` 的层（除 `pf_tauri_ipc` 共享 commands 外）。
//! - 业务逻辑全部走 `pf_application::*`（不直接访问 `pf_ai` / `pf_vector` 等）。
//! - 启动流程：
//!   1. 加载 Config（TOML 或 default）
//!   2. 加载 ModelRegistry（manifest.toml）
//!   3. 加载 AI 模型（ArcFace + SCRFD）
//!   4. 打开 SQLite 数据库
//!   5. 加载或创建 HNSW 索引
//!   6. 装配 `pf_application::Bootstrapped`
//!   7. 启动后台 workers
//!   8. 把服务句柄塞进 Tauri `State`，注册 `#[tauri::command]`
//!
//! ## 与 mobile shell 的关系
//!
//! `commands` / `state` / `ipc_types` / `events` 都在 `pf_tauri_ipc` 里。
//! Desktop / mobile 两个 shell 都依赖这个 crate，复用同一套 IPC 协议。
//! 两者差异仅在 bootstrap（desktop_setup vs mobile_setup）和 Tauri config。

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod desktop_setup;

use std::sync::Arc;

use pf_tauri_ipc::AppState;
use tracing::{error, info};

use crate::desktop_setup::bootstrap;

/// 应用入口（`main.rs` 调用）。
pub fn run() {
    if let Err(e) = desktop_setup::init_logging() {
        eprintln!("failed to init logging: {e}");
    }

    info!("PhotoFinder Next 2 starting…");

    let boot = match bootstrap() {
        Ok(b) => b,
        Err(e) => {
            error!(error = %e, "bootstrap failed");
            std::process::exit(1);
        }
    };

    let app_state = Arc::new(AppState::from(boot));

    let result = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(app_state.clone())
        .setup(|_app| {
            info!("Tauri app ready");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            pf_tauri_ipc::commands::scan_folder,
            pf_tauri_ipc::commands::stop_scan,
            pf_tauri_ipc::commands::clear_database,
            pf_tauri_ipc::commands::search_by_face_image,
            pf_tauri_ipc::commands::search_by_face_embedding,
            pf_tauri_ipc::commands::list_objects_by_class,
            pf_tauri_ipc::commands::list_persons,
            pf_tauri_ipc::commands::rename_person,
            pf_tauri_ipc::commands::faces_of_person,
            pf_tauri_ipc::commands::cluster_faces,
            pf_tauri_ipc::commands::rebuild_person_clusters,
            pf_tauri_ipc::commands::search_by_person,
            pf_tauri_ipc::commands::list_library,
            pf_tauri_ipc::commands::get_statistics,
            pf_tauri_ipc::commands::get_thumbnail,
            pf_tauri_ipc::commands::list_tasks,
            pf_tauri_ipc::commands::cancel_task,
            pf_tauri_ipc::commands::get_models_status,
            pf_tauri_ipc::commands::index_pending,
            pf_tauri_ipc::commands::get_data_dir,
            pf_tauri_ipc::commands::get_app_info,
            // Phase 4 新增
            pf_tauri_ipc::commands::rebuild_thumbnails,
            pf_tauri_ipc::commands::rebuild_face_index,
            pf_tauri_ipc::commands::get_scan_status,
            pf_tauri_ipc::commands::get_processing_status,
            pf_tauri_ipc::commands::copy_files,
            pf_tauri_ipc::commands::init_object_search,
            pf_tauri_ipc::commands::search_objects,
            pf_tauri_ipc::commands::index_images_for_object,
            pf_tauri_ipc::commands::write_query_image,
            pf_tauri_ipc::commands::write_cropped_image,
            pf_tauri_ipc::commands::get_image_thumbnail,
        ])
        .run(tauri::generate_context!());

    if let Err(e) = result {
        error!(error = %e, "tauri run failed");
        std::process::exit(1);
    }
}
