//
//  RootView.swift — 4-tab TabView container（Editorial Atelier）。
//
//  4 个 tab 把桌面版的扫描 / 人脸搜索 / 类别搜索 / 设置映射成 iOS 入口：
//  相册 · 面孔 · 搜索 · 设置。Tab tint = terracotta，全局 paperBg。
//

import SwiftUI

struct RootView: View {
    @State private var selection: Tab = {
        let args = ProcessInfo.processInfo.arguments
        if args.contains("--auto-pick-first") {
            return .search
        }
        if args.contains("--auto-pick") {
            return .faces
        }
        return .scan
    }()

    enum Tab: Hashable {
        case faces, search, scan, settings
    }

    var body: some View {
        TabView(selection: $selection) {
            FacesView()
                .tabItem {
                    Label("面孔", systemImage: "person.crop.rectangle.stack")
                }
                .tag(Tab.faces)

            ScanView()
                .tabItem {
                    Label("扫描", systemImage: "tray.and.arrow.down.fill")
                }
                .tag(Tab.scan)

            SearchView()
                .tabItem {
                    Label("搜索", systemImage: "magnifyingglass")
                }
                .tag(Tab.search)

            SettingsView()
                .tabItem {
                    Label("设置", systemImage: "gearshape")
                }
                .tag(Tab.settings)
        }
        .tint(Color.terracotta)
        .task { AppBootstrap.initRustIfNeeded() }
    }
}

#Preview {
    RootView()
}