//
//  AppBootstrap.swift — Swift → Rust C ABI bridge declarations + lazy init.
//
//  Each function below is implemented by `apps/ios-bridge/src/lib.rs`
//  with `#[no_mangle] pub extern "C"`. The Swift side imports them via
//  `@_silgen_name` (no bridging header needed — Swift just needs the symbol
//  name to match).
//
//  Direction:
//    Swift → Rust:  pf_ios_init_app, pf_ios_run_scan, pf_ios_scan_in_progress,
//                   pf_ios_db_image_count  (this file)
//    Rust → Swift:  pf_ios_photo_permission_status, pf_ios_list_photo_items,
//                   pf_ios_read_bytes, etc.  (PhotoKitBridge.swift @_cdecl)
//
//  Calling `pf_ios_init_app()` is idempotent — the Rust side uses OnceLock;
//  the second call returns 4 (already initialized) and Swift treats it as OK.
//

import Foundation
import os

@_silgen_name("pf_ios_init_app")
private func _pf_ios_init_app() -> Int32

@_silgen_name("pf_ios_run_scan")
private func _pf_ios_run_scan() -> Int32

@_silgen_name("pf_ios_scan_in_progress")
private func _pf_ios_scan_in_progress() -> Int32

@_silgen_name("pf_ios_db_image_count")
private func _pf_ios_db_image_count() -> Int32

enum AppBootstrap {
    private static let logger = Logger(
        subsystem: "com.photofinder.next",
        category: "AppBootstrap"
    )
    private static var didInit = false

    /// Initialize the Rust-side Application singleton. Called from RootView's
    /// `.task` so it runs once per app launch (the Rust OnceLock makes the
    /// call itself idempotent). Blocking the calling thread for ~100ms while
    /// SQLite initializes + the processing-service thread spawns is fine —
    /// this happens before the user can interact with the UI.
    @MainActor
    static func initRustIfNeeded() {
        guard !didInit else { return }
        didInit = true

        let rc = _pf_ios_init_app()
        switch rc {
        case 0:
            logger.info("Rust Application initialized")
        case 4:
            logger.info("Rust Application already initialized (OnceLock hit)")
        default:
            logger.error("pf_ios_init_app returned \(rc) — scan will not work")
        }
    }
}