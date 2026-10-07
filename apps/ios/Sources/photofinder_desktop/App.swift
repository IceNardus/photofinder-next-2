//
//  App.swift — PhotoFinder Next iOS SwiftUI native shell.
//
//  Pure SwiftUI app; no Tauri / no WKWebView. PhotoKit access goes directly
//  through Photos.framework; the Rust backend (libphotofinder_desktop_lib.a)
//  is linked for Phase 1 to provide scan + DB persistence. Swift calls into
//  Rust via @_silgen_name C ABI declared in `AppBootstrap.swift`. The
//  pf_ios_* @_cdecl definitions in PhotoKitBridge.swift remain for the
//  reverse direction (Rust → Swift PhotoKit access).
//

import SwiftUI

@main
struct PhotoFinderNextApp: App {
    init() {
        // DIAGNOSIS: prove Swift runtime is executing.
        NSLog("PhotoKitBridge: PhotoFinderNextApp init — Swift runtime live")
        if let dir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first {
            let path = dir.appendingPathComponent("pf_ios_diag.log").path
            let line = "[DIAG PhotoFinderNextApp init] path=\(path)\n"
            try? Data(line.utf8).write(to: URL(fileURLWithPath: path))
        }
    }

    var body: some Scene {
        WindowGroup {
            RootView()
        }
    }
}