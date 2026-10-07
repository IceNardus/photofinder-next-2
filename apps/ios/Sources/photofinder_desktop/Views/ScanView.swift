//
//  ScanView.swift — 扫描相册 → 写入本地 SQLite 数据库。
//
//  SwiftUI @_silgen_name 调用 Rust 端 ios_bridge（`apps/ios-bridge/src/lib.rs`）的 C ABI：
//    pf_ios_init_app / pf_ios_run_scan / pf_ios_scan_in_progress /
//    pf_ios_db_image_count / pf_ios_scan_progress / pf_ios_scan_result /
//    pf_ios_clear_db / pf_ios_free_string。
//
//  扫描 = 走 PhotoKit 枚举所有 PHAsset + SHA256 hash + 入库到
//  ~/Library/Application Support/PhotoFinderNext/photofinder.db。
//
//  UI:
//   · ProgressView(value:) — 从 pf_ios_scan_progress 每 200 ms 轮询一次。
//   · 「清除扫描记录」 — 调 pf_ios_clear_db 删除 images / faces /
//     scan_tasks 三张表 + 落盘 HNSW 索引，配 .alert 二次确认。
//
//  Phase 1 (2026-08-05)：只录元数据 + hash，不跑 AI 模型；
//  Phase 2 接 onnxruntime iOS 后会再加 MobileCLIP/HNSW，类别搜索上线。
//

import SwiftUI

// MARK: - C ABI: Swift → Rust

@_silgen_name("pf_ios_init_app")
fileprivate func pf_ios_init_app() -> Int32

@_silgen_name("pf_ios_run_scan")
fileprivate func pf_ios_run_scan() -> Int32

@_silgen_name("pf_ios_scan_in_progress")
fileprivate func pf_ios_scan_in_progress() -> Int32

@_silgen_name("pf_ios_db_image_count")
fileprivate func pf_ios_db_image_count() -> Int32

/// 返回 JSON 字符串指针：{"processed": u64, "total": u64, "current_file": String}
/// 失败返回 NULL。
@_silgen_name("pf_ios_scan_progress")
private func pf_ios_scan_progress() -> UnsafeMutablePointer<CChar>?

/// 返回 JSON 字符串指针：{"total_found": u64, "new_images": u64, "skipped": u64}
/// 还没扫过一次返回 NULL。
@_silgen_name("pf_ios_scan_result")
private func pf_ios_scan_result() -> UnsafeMutablePointer<CChar>?

/// 删除 images / faces / scan_tasks 三张表 + 落盘 HNSW 索引。
/// 返回 0 成功，1 未初始化，2 SQL 失败，3 向量 wipe 失败。
@_silgen_name("pf_ios_clear_db")
fileprivate func pf_ios_clear_db() -> Int32

/// 释放 Rust 端 `CString::into_raw()` 分配的字符串。
@_silgen_name("pf_ios_free_string")
private func pf_ios_free_string(_ ptr: UnsafeMutablePointer<CChar>?)

// MARK: - DTO

/// Rust 端 `ScanProgressOut` 的 Swift 镜像。
private struct ScanProgressDTO: Decodable {
    let processed: UInt64
    let total: UInt64
    let current_file: String
}

/// Rust 端 `ScanResultOut` 的 Swift 镜像。
private struct ScanResultDTO: Decodable {
    let total_found: UInt64
    let new_images: UInt64
    let skipped: UInt64
}

// MARK: - ScanView

struct ScanView: View {
    @State private var isScanning = false
    @State private var lastResult: LastScanSummary? = nil
    @State private var dbCount: Int32 = -1
    @State private var lastError: String? = nil

    /// 扫描进度（来自 pf_ios_scan_progress 轮询）。`nil` 表示尚未收到第一帧。
    @State private var progress: ScanProgress? = nil

    /// "清除扫描记录" 二次确认 alert。
    @State private var showClearConfirm = false

    struct LastScanSummary: Equatable {
        let newImages: Int
        let skipped: Int
        let totalFound: Int
        let finishedAt: Date
    }

    /// 当前进度快照。`processed == total` 视为完成；`total == 0` 是「还在算总数」
    /// 状态，进度条只显示 indeterminate 旋转条。
    struct ScanProgress: Equatable {
        let processed: Int
        let total: Int
        let currentFile: String

        var fraction: Double {
            guard total > 0 else { return 0 }
            return min(1.0, Double(processed) / Double(total))
        }
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: Layout.sectionGap) {
                    masthead
                    dbStatusCard
                    actionCard
                    if isScanning, let p = progress {
                        progressCard(p)
                    }
                    if let last = lastResult {
                        lastScanCard(last)
                    }
                    if let err = lastError {
                        errorCard(err)
                    }
                    clearCard
                    disclaimerCard
                }
                .padding(.horizontal, Layout.pagePadding)
                .padding(.bottom, 64)
            }
            .background(Color.paperBg.ignoresSafeArea())
            .toolbar(.hidden, for: .navigationBar)
            .task {
                await refreshDbCount()
            }
            .alert("清除全部扫描记录？", isPresented: $showClearConfirm) {
                Button("取消", role: .cancel) {}
                Button("清除", role: .destructive) {
                    Task { await clearDb() }
                }
            } message: {
                Text("将删除 \(dbCount) 张已入库照片的元数据、所有人脸向量和 HNSW 索引。照片本身不会被删除，但下一次搜索需要先重新扫描。")
            }
        }
    }

    private var masthead: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("把相册装进数据库 · SCAN")
                .font(.monoCaption)
                .tracking(2)
                .foregroundStyle(Color.softInk)
            Text("扫描")
                .font(.editorialDisplay)
                .foregroundStyle(Color.ink)
            Text("iOS 端录入：枚举 PHAsset → SHA256 → SQLite")
                .font(.system(.callout))
                .foregroundStyle(Color.softInk)
                .padding(.top, 4)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.top, 8)
    }

    private var dbStatusCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Text("数据库状态")
                    .font(.editorialHeadline)
                    .foregroundStyle(Color.ink)
                Spacer()
                Text(dbCount >= 0 ? "\(dbCount) 张" : "尚未初始化")
                    .font(.monoFootnote)
                    .foregroundStyle(dbCount >= 0 ? Color.terracotta : Color.softInk)
            }
            Text("路径 ~/Library/Application Support/PhotoFinderNext/photofinder.db")
                .font(.monoCaption)
                .foregroundStyle(Color.softInk)
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .paperCard()
    }

    private var actionCard: some View {
        VStack(spacing: 12) {
            Button {
                Task { await runScan() }
            } label: {
                HStack(spacing: 12) {
                    if isScanning {
                        ProgressView()
                            .tint(.white)
                            .controlSize(.small)
                    } else {
                        Image(systemName: "arrow.down.doc.fill")
                            .font(.system(size: 16, weight: .semibold))
                    }
                    Text(isScanning ? "正在扫描…" : "开始扫描")
                        .font(.system(.headline))
                }
                .foregroundStyle(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(isScanning ? Color.softInk : Color.terracotta)
                .clipShape(RoundedRectangle(cornerRadius: Layout.chipCornerRadius, style: .continuous))
            }
            .disabled(isScanning || dbCount < 0)

            Text(isScanning
                 ? "正在枚举照片库并写入数据库。下方进度条每 200 ms 刷新一次。"
                 : "第一次进 app 时 Rust 后台已经初始化；如果看到「尚未初始化」请重启 app。")
                .font(.system(.caption))
                .foregroundStyle(Color.softInk)
                .multilineTextAlignment(.center)
        }
        .padding(20)
        .frame(maxWidth: .infinity)
        .paperCard()
    }

    /// 扫描中进度条卡片。`total == 0` 时显示 indeterminate 旋转条
    /// （「正在枚举相册」阶段，没拿到总照片数）。
    @ViewBuilder
    private func progressCard(_ p: ScanProgress) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Text("扫描进度")
                    .font(.editorialHeadline)
                    .foregroundStyle(Color.ink)
                Spacer()
                if p.total > 0 {
                    Text("\(p.processed) / \(p.total)")
                        .font(.monoFootnote)
                        .foregroundStyle(Color.terracotta)
                } else {
                    Text("枚举中…")
                        .font(.monoFootnote)
                        .foregroundStyle(Color.softInk)
                }
            }

            if p.total > 0 {
                ProgressView(value: p.fraction)
                    .progressViewStyle(.linear)
                    .tint(Color.terracotta)
                Text("\(Int(p.fraction * 100))% 完成")
                    .font(.monoCaption)
                    .foregroundStyle(Color.softInk)
            } else {
                ProgressView()
                    .progressViewStyle(.linear)
                    .tint(Color.terracotta)
                Text("正在枚举照片库…")
                    .font(.monoCaption)
                    .foregroundStyle(Color.softInk)
            }

            if !p.currentFile.isEmpty {
                HStack(spacing: 6) {
                    Image(systemName: "doc.fill")
                        .font(.system(size: 11))
                        .foregroundStyle(Color.softInk)
                    Text(p.currentFile)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(Color.softInk)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .paperCard()
    }

    private func lastScanCard(_ result: LastScanSummary) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Text("最近一次")
                    .font(.editorialHeadline)
                    .foregroundStyle(Color.ink)
                Spacer()
                Image(systemName: "checkmark.seal.fill")
                    .foregroundStyle(Color.terracotta)
            }
            HStack(spacing: 18) {
                stat(label: "总数", value: "\(result.totalFound)")
                stat(label: "新增", value: "\(result.newImages)")
                stat(label: "跳过", value: "\(result.skipped)")
                stat(label: "时间", value: Self.timeFormatter.string(from: result.finishedAt))
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .paperCard()
    }

    private func stat(label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.monoCaption)
                .tracking(1)
                .foregroundStyle(Color.softInk)
            Text(value)
                .font(.system(.title3, design: .monospaced).weight(.semibold))
                .foregroundStyle(Color.ink)
        }
    }

    private func errorCard(_ msg: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(Color.terracotta)
            VStack(alignment: .leading, spacing: 4) {
                Text("扫描失败")
                    .font(.system(.headline))
                    .foregroundStyle(Color.ink)
                Text(msg)
                    .font(.system(.callout, design: .monospaced))
                    .foregroundStyle(Color.softInk)
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .paperCard()
    }

    private var clearCard: some View {
        VStack(spacing: 12) {
            Button {
                showClearConfirm = true
            } label: {
                HStack(spacing: 10) {
                    Image(systemName: "trash")
                        .font(.system(size: 14, weight: .semibold))
                    Text("清除扫描记录")
                        .font(.system(.headline))
                }
                .foregroundStyle(Color.terracotta)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(Color.terracottaSoft.opacity(0.18))
                .clipShape(RoundedRectangle(cornerRadius: Layout.chipCornerRadius, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: Layout.chipCornerRadius, style: .continuous)
                        .stroke(Color.terracotta.opacity(0.45), style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
                )
            }
            .buttonStyle(.plain)
            .disabled(isScanning || dbCount <= 0)

            Text("会清空数据库里的照片元数据和人脸向量，下次搜索前需要重新扫描。")
                .font(.system(.caption))
                .foregroundStyle(Color.softInk)
                .multilineTextAlignment(.center)
        }
        .padding(20)
        .frame(maxWidth: .infinity)
        .paperCard()
    }

    private var disclaimerCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("关于这个 tab")
                .font(.system(.headline))
                .foregroundStyle(Color.ink)
            Text("Phase 1：只录元数据 + SHA256 hash，类别搜索仍是占位（需要 onnxruntime iOS，Phase 2）。数据库 schema 与桌面端共用，等类别搜索上线后 iOS 端搜索结果会和桌面一致。")
                .font(.system(.callout))
                .foregroundStyle(Color.softInk)
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .paperCard()
    }

    // MARK: - Logic

    private func refreshDbCount() async {
        let n = pf_ios_db_image_count()
        await MainActor.run { self.dbCount = n }
    }

    private func runScan() async {
        await MainActor.run {
            self.isScanning = true
            self.lastError = nil
            self.progress = nil
        }

        // 后台跑 Rust scan，同时主线程每 200 ms 拉一次 progress。
        async let scanResult: Int32 = Task.detached(priority: .userInitiated) {
            pf_ios_run_scan()
        }.value

        let progressTask = Task { await pollProgress() }
        let result = await scanResult
        progressTask.cancel()

        await MainActor.run {
            self.isScanning = false
            switch result {
            case 0:
                // Rust 端 pf_ios_run_scan 已把 ScanResult 写入 IosState.last_scan_result，
                // 这里把它读出来作为「最近一次」卡片。
                if let summary = readLastScanSummary(priorDbCount: self.dbCount) {
                    self.lastResult = summary
                }
                self.dbCount = pf_ios_db_image_count()
                self.progress = nil
            case 1:
                self.lastError = "Rust 后端未初始化 — 请重启 app"
            case 2:
                self.lastError = "已有扫描在运行中"
            case 3:
                self.lastError = "扫描器内部错误（请查看 ~/Library/Application Support/PhotoFinderNext/logs/）"
            default:
                self.lastError = "未知返回码 \(result)"
            }
        }
    }

    /// 轮询 Rust 端 ScanStats，写回 @State progress。期间 isScanning == true。
    /// 用 try? await Task.sleep(...) 而不是 Timer，这样跟随 SwiftUI 生命周期。
    private func pollProgress() async {
        while !Task.isCancelled {
            if let snapshot = readScanProgress() {
                await MainActor.run { self.progress = snapshot }
            }
            try? await Task.sleep(nanoseconds: 200_000_000)
        }
    }

    private func clearDb() async {
        // 离线跑 Rust DELETE，避免阻塞主线程；DB 操作通常 100ms 量级。
        let result = await Task.detached(priority: .userInitiated) {
            pf_ios_clear_db()
        }.value
        await MainActor.run {
            switch result {
            case 0:
                self.lastResult = nil
                self.lastError = nil
                self.dbCount = pf_ios_db_image_count()
            case 1:
                self.lastError = "Rust 后端未初始化 — 请重启 app"
            case 2:
                self.lastError = "删除数据库失败（请查看 logs）"
            case 3:
                self.lastError = "数据库已清空，但部分索引文件未能删除（下次扫描会重建）"
            default:
                self.lastError = "未知返回码 \(result)"
            }
        }
    }

    /// 从 Rust 端 IosState.last_scan_result 读出最近一次的 ScanResult。
    /// 失败（NULL / 解析失败）时用 db_count 差值兜底。
    private func readLastScanSummary(priorDbCount: Int32) -> LastScanSummary? {
        guard let cptr = pf_ios_scan_result() else {
            // 兜底：不知道总数，按 dbCount 增量算。
            let now = pf_ios_db_image_count()
            let delta = max(0, Int(now) - Int(priorDbCount))
            return LastScanSummary(newImages: delta, skipped: 0, totalFound: Int(now), finishedAt: Date())
        }
        defer { pf_ios_free_string(cptr) }
        let json = String(cString: cptr)
        guard let data = json.data(using: .utf8),
              let dto = try? JSONDecoder().decode(ScanResultDTO.self, from: data) else {
            let now = pf_ios_db_image_count()
            let delta = max(0, Int(now) - Int(priorDbCount))
            return LastScanSummary(newImages: delta, skipped: 0, totalFound: Int(now), finishedAt: Date())
        }
        return LastScanSummary(
            newImages: Int(dto.new_images),
            skipped: Int(dto.skipped),
            totalFound: Int(dto.total_found),
            finishedAt: Date()
        )
    }

    private func readScanProgress() -> ScanProgress? {
        guard let cptr = pf_ios_scan_progress() else { return nil }
        defer { pf_ios_free_string(cptr) }
        let json = String(cString: cptr)
        guard let data = json.data(using: .utf8),
              let dto = try? JSONDecoder().decode(ScanProgressDTO.self, from: data) else {
            return nil
        }
        return ScanProgress(
            processed: Int(dto.processed),
            total: Int(dto.total),
            currentFile: dto.current_file
        )
    }

    private static let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f
    }()
}