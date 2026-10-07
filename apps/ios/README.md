# PhotoFinder Next 2 — iOS App

SwiftUI native shell + Rust C ABI bridge (`apps/ios-bridge` crate)。Phase 2.5+
完成:12 个 `pf_ios_*` ABI 全部实接,人物 / 物品搜索可用,设备 + 模拟器均可
`xcodebuild` 链接成功。

## 当前状态(2026-08-14)

- ✅ 9 个 .swift 文件(2531 行)
- ✅ iOS 工程文件(project.yml / Info.plist / entitlements / LaunchScreen / Assets)
- ✅ Rust 桥 (`apps/ios-bridge`) 12 个 `pf_ios_*` extern "C" 全部实接
- ✅ `libpf_ios_bridge.a` 静态库由 `project.yml` preBuildScript 自动构建
- ✅ `pf_ios_shim.c` weak NULL stubs + 强 Swift `@_cdecl` 覆盖
- ✅ iOS device + 模拟器 `xcodebuild ... build` 均成功

## 与桌面端差异

| 桌面端 (Tauri + WebView) | iOS (本目录) |
|---|---|
| Vue + Pinia + Vite | SwiftUI 原生 |
| Tauri IPC (invoke) | C ABI (`@_silgen_name` + `@_cdecl`) |
| `apps/desktop/src-tauri/src-tauri/` Rust | `crates/pf_application::*` 服务 |
| `pf_tauri_ipc::commands::*` | `apps/ios-bridge/src/lib.rs` 中 `pf_ios_*` |

iOS 的 `pf_ios_*` 调用先进入 Rust C ABI 层,再路由到 `pf_application::Bootstrapped` 服务。
反向(Rust 调 Swift 的 PhotoKit)通过 `pf_platform::ios::IosNativeMediaBridge` 走
`unsafe extern "C"` + Swift `@_cdecl`。

## 目录结构

```
apps/ios/
├── project.yml                      # XcodeGen 配置 + Rust 构建脚本
├── LaunchScreen.storyboard
├── Assets.xcassets/                 # AppIcon + AccentColor
└── Sources/photofinder_desktop/
    ├── App.swift                    # @main 入口
    ├── AppBootstrap.swift           # Swift → Rust C ABI 桥(@_silgen_name)
    ├── PhotoKitBridge.swift         # Rust → Swift 反向桥(@_cdecl)
    ├── pf_ios_shim.c                # 8 个 weak NULL stubs(无 Swift 实现时安全 fallback)
    ├── RootView.swift               # 4-tab TabView
    ├── Theme.swift                  # Editorial Atelier 设计语言
    ├── Info.plist                   # NSPhotoLibraryUsageDescription 等
    ├── PhotoFinderNext.entitlements
    └── Views/
        ├── FacesView.swift          # 人物搜索
        ├── ScanView.swift           # 相册扫描
        ├── SearchView.swift         # 物品搜索
        └── SettingsView.swift       # 设置
```

## 构建运行

```bash
# 1. 生成 Xcode 工程
brew install xcodegen
cd apps/ios
xcodegen generate

# 2. 编译并运行到模拟器/真机(xcodebuild 或 Xcode UI)
xcodebuild -project photofinder_desktop.xcodeproj \
  -scheme photofinder_desktop_iOS \
  -sdk iphonesimulator \
  -configuration Debug build

# 设备需 code signing
xcodebuild -project photofinder_desktop.xcodeproj \
  -scheme photofinder_desktop_iOS \
  -sdk iphoneos \
  -configuration Debug build
```

`preBuildScript` 自动检测 `EFFECTIVE_PLATFORM_NAME` → 调 `cargo build --target
aarch64-apple-ios[-sim] -p pf_ios_bridge` → 复制到
`target/ios-active/$(CONFIGURATION)/<PLATFORM>/libpf_ios_bridge.a` → xcodebuild
从 `LIBRARY_SEARCH_PATHS` 找到对应架构的 `.a` 自动链接。

## 符号协议

**Swift → Rust**(12 个 ABI,Swift `@_silgen_name` 调 Rust `#[no_mangle] pub extern "C"`):

- `pf_ios_init_app` / `pf_ios_run_scan` / `pf_ios_scan_in_progress` / `pf_ios_db_image_count`
- `pf_ios_scan_progress` / `pf_ios_scan_result` / `pf_ios_clear_db`
- `pf_ios_search_person` / `pf_ios_search_object` / `pf_ios_free_string`

**Rust → Swift**(8 个 ABI,Rust `unsafe extern "C"` 调 Swift `@_cdecl`):

- `pf_ios_photo_permission_status` / `pf_ios_request_photo_permission`
- `pf_ios_list_photo_items` / `pf_ios_photo_item_by_uri`
- `pf_ios_export_to_temp_file` / `pf_ios_export_thumbnail_to_temp_file`
- `pf_ios_read_bytes` / `pf_ios_schedule_bg_task`

后 8 个在 Rust staticlib (`libpf_ios_bridge.a`) 中是 UND,需要 iOS app 提供实现。
`pf_ios_shim.c` 提供 weak NULL stubs(若 PhotoKitBridge.swift 缺失,降级到 NULL,不 crash);
strong Swift `@_cdecl` 实现覆盖 weak shim。

## 来源

从 `photofinder-next/.claude/worktrees/cross-platform-migration/apps/ios/` 复制,
原始 commit 3d59c34。Swift 代码未修改,工程文件按 `apps/ios-bridge` 实际 crate 名重写。