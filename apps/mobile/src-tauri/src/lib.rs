//! # PhotoFinder Next 2 — Mobile Shell (Tauri 2)
//!
//! 跨平台兼容层：iOS + Android 共用同一份 Rust 代码。
//!
//! 启动流程：
//! 1) 初始化 logging
//! 2) 装配 `pf_application::Bootstrapped`（用 mobile PathResolver / PhotoProvider）
//! 3) 启动后台 workers
//! 4) 把 `Bootstrapped` 包成 `AppState` 塞进 Tauri
//! 5) 注册 `pf_tauri_ipc` 的所有 commands（与桌面端同一套）

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod mobile_setup;

use std::sync::Arc;

use pf_tauri_ipc::AppState;
use tracing::{error, info};

/// Mobile 入口。
///
/// 与桌面端 `photofinder_desktop::run()` 的差异：
/// - Bootstrap 用 `mobile_setup::bootstrap`（带 AndroidPathResolver / IosPathResolver）
/// - 不需要 dialog plugin（移动端用 native file picker 或 photo picker）
/// - 不需要 opener plugin（移动端用 deep link）
pub fn run() {
    if let Err(e) = mobile_setup::init_logging() {
        eprintln!("failed to init logging: {e}");
    }

    info!("PhotoFinder Next 2 (mobile) starting…");

    let boot = match mobile_setup_full() {
        Ok(b) => b,
        Err(e) => {
            error!(error = %e, "mobile bootstrap failed");
            std::process::exit(1);
        }
    };

    pf_application::start_workers(&boot.scheduler);

    let app_state = Arc::new(AppState::from(boot));

    let result = tauri::Builder::default()
        .manage(app_state)
        .setup(|_app| {
            info!("Tauri mobile app ready");
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
        error!(error = %e, "tauri mobile run failed");
        std::process::exit(1);
    }
}

/// 装配完整 Bootstrapped（链式调用 mobile_setup 的步骤）。
fn mobile_setup_full() -> anyhow::Result<pf_application::Bootstrapped> {
    use mobile_setup::*;

    let config = default_config();
    let path_resolver = mobile_path_resolver();
    let photo_provider = mobile_photo_provider();
    let model_manager = mobile_model_manager(path_resolver.clone());

    let database = mobile_database()?;
    let face_pipeline = try_load_face_pipeline(model_manager.clone(), config.clone());
    let face_index = mobile_face_index()?;

    bootstrap(
        config,
        load_model_registry()?,
        database,
        face_pipeline,
        face_index,
        None, // object_pipeline 暂未实现
        None, // object_embedder 暂未实现
        None, // object_index 暂未实现
        photo_provider,
        path_resolver,
        model_manager,
    )
}
