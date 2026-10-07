//
//  SettingsView.swift — 编辑室风格的设置面板。
//

import SwiftUI
import Photos

struct SettingsView: View {
    @State private var photoAuthStatus: PHAuthorizationStatus = .notDetermined

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: Layout.sectionGap) {
                    masthead
                    permissionSection
                    appInfoSection
                    backendSection
                    footer
                }
                .padding(.horizontal, Layout.pagePadding)
                .padding(.bottom, 64)
            }
            .background(Color.paperBg.ignoresSafeArea())
            .toolbar(.hidden, for: .navigationBar)
            .task {
                photoAuthStatus = PHPhotoLibrary.authorizationStatus(for: .readWrite)
            }
        }
    }

    private var masthead: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("卷末 · Settings")
                .font(.monoCaption)
                .tracking(2)
                .foregroundStyle(Color.softInk)
            Text("PhotoFinder Next")
                .font(.editorialDisplay)
                .foregroundStyle(Color.ink)
            Text("iOS 预览版")
                .font(.system(.callout, design: .serif).italic())
                .foregroundStyle(Color.softInk)
                .padding(.top, 2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.top, 8)
    }

    private var permissionSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionLabel("权限 · PERMISSIONS")
            infoRow(
                icon: "photo.on.rectangle.angled",
                title: "照片访问",
                value: authLabel,
                valueColor: authColor
            )
        }
    }

    private var appInfoSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionLabel("应用 · APP")
            infoRow(icon: "tag", title: "名称", value: appName)
            infoRow(icon: "number", title: "版本", value: appVersion)
            infoRow(icon: "hammer", title: "构建号", value: appBuild)
        }
    }

    private var backendSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionLabel("后端 · BACKEND")
            infoRow(icon: "cpu", title: "AI 模型", value: "桌面版独占")
            infoRow(icon: "rectangle.stack.badge.person.crop", title: "人脸聚类", value: "桌面版独占")
            infoRow(icon: "magnifyingglass", title: "搜索 / 扫描", value: "桌面版独占")
        }
    }

    private var footer: some View {
        Text("iOS 端目前为只读 + Vision 本地预览壳。完整的扫描、索引、聚类、搜索功能请使用 macOS / Windows 桌面版本。")
            .font(.system(.callout, design: .serif).italic())
            .foregroundStyle(Color.softInk)
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .paperCard()
    }

    private func infoRow(icon: String, title: String, value: String, valueColor: Color = .softInk) -> some View {
        HStack(spacing: 14) {
            Image(systemName: icon)
                .font(.system(size: 16))
                .foregroundStyle(Color.terracotta)
                .frame(width: 22)
            Text(title)
                .font(.system(.body))
                .foregroundStyle(Color.ink)
            Spacer()
            Text(value)
                .font(.system(.callout, design: .monospaced))
                .foregroundStyle(valueColor)
        }
        .padding(16)
        .paperCard()
    }

    private func sectionLabel(_ title: String) -> some View {
        Text(title)
            .font(.monoCaption)
            .tracking(2)
            .foregroundStyle(Color.softInk)
            .padding(.horizontal, 4)
    }

    private var authLabel: String {
        switch photoAuthStatus {
        case .authorized: return "完全访问"
        case .limited: return "有限访问"
        case .denied: return "已拒绝"
        case .restricted: return "受限制"
        case .notDetermined: return "未询问"
        @unknown default: return "未知"
        }
    }

    private var authColor: Color {
        switch photoAuthStatus {
        case .authorized, .limited: return .softInk
        default: return .terracotta
        }
    }

    private var appName: String {
        Bundle.main.infoDictionary?["CFBundleDisplayName"] as? String
            ?? Bundle.main.infoDictionary?["CFBundleName"] as? String
            ?? "PhotoFinder Next"
    }

    private var appVersion: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "—"
    }

    private var appBuild: String {
        Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "—"
    }
}