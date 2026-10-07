//
//  Theme.swift — Editorial Atelier 设计语言。
//
//  cream paper + ink + terracotta 三色板，New York 衬线大标题 + SF Pro 正文
//  + SF Mono 元数据。留白慷慨，呼应 Kinfolk / Cereal 杂志感。
//

import SwiftUI

extension Color {
    /// 杂志纸面：浅色模式 = 米白纸 / 深色模式 = 深墨。
    static let paperBg = Color(
        light: .init(red: 0.961, green: 0.941, blue: 0.910),
        dark:  .init(red: 0.075, green: 0.063, blue: 0.055)
    )
    /// 主文字：深墨 / 米白。
    static let ink = Color(
        light: .init(red: 0.102, green: 0.086, blue: 0.071),
        dark:  .init(red: 0.961, green: 0.941, blue: 0.910)
    )
    /// 次文字：浅色 = 柔墨 / 深色 = 柔棕。
    static let softInk = Color(
        light: .init(red: 0.353, green: 0.318, blue: 0.290),
        dark:  .init(red: 0.659, green: 0.580, blue: 0.482)
    )
    /// 卡片底：比 paper 略亮 / 略暗。
    static let cardBg = Color(
        light: .init(red: 1.000, green: 0.992, blue: 0.980),
        dark:  .init(red: 0.137, green: 0.118, blue: 0.102)
    )
    /// 重音色：陶土红（accent，亮暗模式同色，靠对比度调节）。
    static let terracotta = Color(red: 0.784, green: 0.333, blue: 0.239)
    /// 浅陶土：用于 chip 背景、辅助高亮。
    static let terracottaSoft = Color(red: 0.929, green: 0.722, blue: 0.659)
}

extension Color {
    /// 自适应 Color（light / dark）。
    init(light: Color, dark: Color) {
        self.init(uiColor: UIColor { trait in
            trait.userInterfaceStyle == .dark ? UIColor(dark) : UIColor(light)
        })
    }
}

extension Font {
    /// 杂志大标题（New York 衬线，iOS 17+ 自动启用，旧版 fallback 到 Charter）。
    static let editorialDisplay = Font.system(.largeTitle, design: .serif).weight(.regular)
    static let editorialTitle = Font.system(.title, design: .serif).weight(.regular)
    static let editorialHeadline = Font.system(.title3, design: .serif).weight(.regular)
    /// 元数据 / 数字 / 日期一律走等宽，保持「编辑部稿笺」质感。
    static let monoCaption = Font.system(.caption, design: .monospaced)
    static let monoFootnote = Font.system(.footnote, design: .monospaced)
    /// 衬线正文，用于卡片小标题与引用。
    static let serifBody = Font.system(.body, design: .serif)
}

enum Layout {
    static let pagePadding: CGFloat = 24
    static let sectionGap: CGFloat = 28
    static let cardGap: CGFloat = 14
    static let cardCornerRadius: CGFloat = 16
    static let chipCornerRadius: CGFloat = 12
}

/// 全局可复用的视图修饰符。
extension View {
    /// 卡片表面 + 圆角 + 极淡阴影。
    func paperCard(cornerRadius: CGFloat = Layout.cardCornerRadius) -> some View {
        self
            .background(Color.cardBg)
            .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            .shadow(color: .black.opacity(0.05), radius: 8, x: 0, y: 2)
    }
}