//
//  FacesView.swift — 人物搜索 (按参考面孔找图库中所有同一人照片)。
//
//  和桌面 V1 一致：选一张含人脸的参考图 → Rust 端跑 SCRFD+ArcFace 拿到
//  512-d embedding → 跟本地 HNSW 索引（`data_dir/index/face.hnsw.data`）做 cosine similarity → 返回
//  Top-K 命中。算法走 `pf_application::SearchService::search_by_person`，iOS 端只是
//  picker + 缩略图渲染壳。
//
//  数据流：
//    1. PHPicker 选图（FacePickerSheet，三段式 dismiss 已稳定）
//    2. JPEG bytes 落到 ~/Library/Application Support/PhotoFinderNext/query_image_<uuid>.jpg
//    3. pf_ios_search_person(query_path, 50) → JSON [{image_id, face_id, score, bbox, thumbnail_path}]
//    4. 对每个 result 解析 thumbnail_path（iOS 上 = phasset://LOCALID），反查 PHAsset 拉缩略图
//    5. 网格化展示 + 分数
//
//  Rust 侧实现见 `apps/ios-bridge/src/lib.rs::pf_ios_search_person`。
//

import SwiftUI
import Photos
import PhotosUI
import UniformTypeIdentifiers

// MARK: - C ABI: Swift → Rust (person search)

@_silgen_name("pf_ios_search_person")
private func _pf_ios_search_person(_ path: UnsafePointer<CChar>?, _ top_k: Int32) -> UnsafeMutablePointer<CChar>?

/// 释放 Rust 端 `CString::into_raw()` 分配的字符串 —— 同一 libc 堆，
/// 直接 `libc::free`。NULL 安全。
@_silgen_name("pf_ios_free_string")
private func _pf_ios_free_string(_ ptr: UnsafeMutablePointer<CChar>?)

// MARK: - DTO

/// Rust 返回的 JSON 元素。`thumbnail_path` 在 iOS 上是 `phasset://LOCALID`
/// 字符串（见 `core/src/scanner.rs` 第 256 行 INSERT：iOS 扫描时把 platform
/// URI 同时写到 `path` 列，所以 desktop 那边叫 thumbnail_path，在 iOS 端
/// 实际就是 platform URI）。
private struct PersonSearchResultDTO: Decodable {
    let image_id: Int64
    let face_id: Int64
    let score: Float
    let bbox: [Float]?
    let thumbnail_path: String
}

/// 视图层使用的 model。Identifiable 用 `image_id` 作 key（同图只显示一行）。
struct PersonSearchResult: Identifiable, Equatable {
    let id: Int64          // == image_id
    let faceId: Int64
    let score: Float
    let bbox: [Float]?
    let phAssetLocalId: String?   // 从 phasset://LOCALID 抽出
    /// 每行自己的缩略图加载状态（懒加载，每个 cell 一个 @State）。
    @MainActor var thumbnail: UIImage?

    static func == (lhs: PersonSearchResult, rhs: PersonSearchResult) -> Bool {
        lhs.id == rhs.id && lhs.faceId == rhs.faceId && lhs.score == rhs.score
    }
}

// MARK: - FacesView

struct FacesView: View {
    @State private var queryImage: UIImage?
    @State private var isPickerPresented = false
    @State private var searchState: SearchState = .idle

    enum SearchState {
        case idle                          // 未选图
        case loading                       // 搜索进行中
        case ready([PersonSearchResult])   // 有结果
        case empty                         // 无命中
        case error(String)                 // 失败
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: Layout.sectionGap) {
                    masthead
                    pickerCard
                    searchBody
                }
                .padding(.horizontal, Layout.pagePadding)
                .padding(.bottom, 64)
            }
            .background(Color.paperBg.ignoresSafeArea())
            .toolbar(.hidden, for: .navigationBar)
            .task {
                // 烟测 hook：自动弹 picker 跑通流程。
                let args = ProcessInfo.processInfo.arguments
                if args.contains("--auto-pick") {
                    try? await Task.sleep(nanoseconds: 800_000_000)
                    isPickerPresented = true
                }
            }
            .fullScreenCover(isPresented: $isPickerPresented) {
                FacePickerSheet(isPresented: $isPickerPresented) { image in
                    self.queryImage = image
                    self.searchState = .loading
                    Task { await self.searchPerson(image: image) }
                }
            }
            .task {
                // 烟测 hook 2：跳过 PHPicker，直接拿最近一张图跑搜索。
                let args = ProcessInfo.processInfo.arguments
                if args.contains("--auto-pick-first") {
                    try? await Task.sleep(nanoseconds: 600_000_000)
                    autoPickFirstPhoto()
                }
            }
        }
    }

    private var masthead: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("AI 预览 · 桌面 InsightFace")
                .font(.monoCaption)
                .tracking(2)
                .foregroundStyle(Color.softInk)
            Text("人物搜索")
                .font(.editorialDisplay)
                .foregroundStyle(Color.ink)
            Text("选一张参考面孔 · 找图库里同一人所有照片")
                .font(.system(.callout))
                .foregroundStyle(Color.softInk)
                .padding(.top, 4)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.top, 8)
    }

    private var pickerCard: some View {
        Button {
            isPickerPresented = true
        } label: {
            HStack(spacing: 14) {
                Image(systemName: "person.crop.rectangle.badge.plus")
                    .font(.system(size: 22))
                    .foregroundStyle(Color.terracotta)
                VStack(alignment: .leading, spacing: 2) {
                    Text(queryImage == nil ? "选择一张参考照片" : "换一张")
                        .font(.system(.headline))
                        .foregroundStyle(Color.ink)
                    Text("从相册导入，系统自动找最大人脸")
                        .font(.monoCaption)
                        .foregroundStyle(Color.softInk)
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Color.softInk)
            }
            .padding(20)
            .paperCard()
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var searchBody: some View {
        switch searchState {
        case .idle:
            emptyHint
        case .loading:
            processingState
        case .ready(let results):
            resultsGrid(results)
        case .empty:
            emptyResultsHint
        case .error(let msg):
            errorHint(msg)
        }
    }

    private var processingState: some View {
        HStack(spacing: 12) {
            ProgressView().tint(Color.terracotta)
            Text("SCRFD 检测 + 向量检索…")
                .font(.system(.callout))
                .foregroundStyle(Color.softInk)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 24)
    }

    private var emptyHint: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("暂未选择参考照片")
                .font(.editorialHeadline)
                .foregroundStyle(Color.ink)
            Text("点击上方卡片从相册导入含人脸的照片。SCRFD 模型首次调用会懒加载（约 1–3 秒），所有计算在本地完成，照片不会上传到任何服务器。")
                .font(.system(.callout))
                .foregroundStyle(Color.softInk)
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .paperCard()
    }

    private var emptyResultsHint: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("无匹配结果")
                .font(.editorialHeadline)
                .foregroundStyle(Color.ink)
            Text("图库里没找到与这张参考面孔相似度 ≥ 50% 的照片。可以试试包含正脸、清晰的单人照。")
                .font(.system(.callout))
                .foregroundStyle(Color.softInk)
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .paperCard()
    }

    private func errorHint(_ msg: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("搜索失败")
                .font(.editorialHeadline)
                .foregroundStyle(Color.terracotta)
            Text(msg)
                .font(.system(.callout))
                .foregroundStyle(Color.softInk)
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .paperCard()
    }

    private func resultsGrid(_ results: [PersonSearchResult]) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .firstTextBaseline) {
                Text("匹配结果")
                    .font(.editorialHeadline)
                    .foregroundStyle(Color.ink)
                Spacer()
                Text("\(results.count) 张")
                    .font(.monoFootnote)
                    .foregroundStyle(Color.terracotta)
            }

            // 三列网格（与相册 tab 一致）。每行按 score 降序：list 已经排好。
            LazyVGrid(columns: [
                GridItem(.flexible(), spacing: 8),
                GridItem(.flexible(), spacing: 8),
                GridItem(.flexible(), spacing: 8),
            ], spacing: 8) {
                ForEach(results) { result in
                    ResultThumbnailCell(result: result)
                }
            }
        }
    }

    // MARK: - Search pipeline

    /// 1. JPEG 编码 → 2. 写到 data_dir/query_image_<uuid>.jpg → 3. 调
    ///    pf_ios_search_person 拿 JSON → 4. 解析结果 → 5. 切到 .ready 触发 UI。
    ///
    /// 缩略图加载不在主路径里：每个 ResultThumbnailCell 自己负责从 PHAsset
    /// 拉 300x300 缩略图，行数多时也可以错峰加载。
    private func searchPerson(image: UIImage) async {
        guard let jpegBytes = image.jpegData(compressionQuality: 0.95) else {
            await MainActor.run { searchState = .error("无法编码参考图为 JPEG") }
            return
        }

        // 把 bytes 写到 data_dir/query_image_<uuid>.jpg。和 Rust 端的
        // dirs::data_local_dir() 一致（iOS sandbox 下都解析到
        // ~/Library/Application Support/PhotoFinderNext/）。
        let writeResult = await Task.detached(priority: .userInitiated) { () -> Result<String, Error> in
            let fm = FileManager.default
            guard let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
                return .failure(NSError(domain: "FacesView", code: -1, userInfo: [NSLocalizedDescriptionKey: "找不到 Application Support 目录"]))
            }
            let dataDir = appSupport.appendingPathComponent("PhotoFinderNext")
            do {
                try fm.createDirectory(at: dataDir, withIntermediateDirectories: true)
            } catch {
                return .failure(error)
            }
            let queryFile = dataDir.appendingPathComponent("query_image_\(UUID().uuidString).jpg")
            do {
                try jpegBytes.write(to: queryFile)
                return .success(queryFile.path)
            } catch {
                return .failure(error)
            }
        }.value

        let queryPath: String
        switch writeResult {
        case .success(let path):
            queryPath = path
        case .failure(let err):
            await MainActor.run { searchState = .error("写临时文件失败：\(err.localizedDescription)") }
            return
        }

        // 调 Rust C ABI。top_k=50 与桌面 `search_by_person` 默认一致。
        let cjsonPtr: UnsafeMutablePointer<CChar>? = await Task.detached(priority: .userInitiated) { () -> UnsafeMutablePointer<CChar>? in
            queryPath.withCString { cstr -> UnsafeMutablePointer<CChar>? in
                _pf_ios_search_person(cstr, 50)
            }
        }.value

        guard let cjsonPtr = cjsonPtr else {
            await MainActor.run { searchState = .error("Rust 端搜索失败（请看 ~/Library/Application Support/PhotoFinderNext/logs/）") }
            return
        }
        defer { _pf_ios_free_string(cjsonPtr) }

        let json = String(cString: cjsonPtr)
        guard let data = json.data(using: .utf8) else {
            await MainActor.run { searchState = .error("UTF-8 解码失败") }
            return
        }
        let entries: [PersonSearchResultDTO]
        do {
            entries = try JSONDecoder().decode([PersonSearchResultDTO].self, from: data)
        } catch {
            await MainActor.run { searchState = .error("JSON 解析失败：\(error.localizedDescription)") }
            return
        }

        if entries.isEmpty {
            await MainActor.run { searchState = .empty }
            return
        }

        let results: [PersonSearchResult] = entries.map { dto in
            let localId: String? = {
                let prefix = "phasset://"
                if dto.thumbnail_path.hasPrefix(prefix) {
                    return String(dto.thumbnail_path.dropFirst(prefix.count))
                }
                return nil
            }()
            return PersonSearchResult(
                id: dto.image_id,
                faceId: dto.face_id,
                score: dto.score,
                bbox: dto.bbox,
                phAssetLocalId: localId,
                thumbnail: nil
            )
        }

        await MainActor.run { searchState = .ready(results) }
    }

    /// 烟测 hook：跳过 PHPicker 直接拿相册最新一张图跑搜索。
    /// 需要 PHPhotoLibrary 授权；首次会触发授权弹窗。
    private func autoPickFirstPhoto() {
        let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        if status == .notDetermined {
            PHPhotoLibrary.requestAuthorization(for: .readWrite) { newStatus in
                if newStatus == .authorized || newStatus == .limited {
                    DispatchQueue.main.async { self.autoPickFirstPhoto() }
                } else {
                    NSLog("[PF][AutoPick] auth denied")
                }
            }
            return
        }
        if status == .denied || status == .restricted {
            NSLog("[PF][AutoPick] auth not granted")
            return
        }
        let opts = PHFetchOptions()
        opts.fetchLimit = 1
        opts.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        let assets = PHAsset.fetchAssets(with: .image, options: opts)
        guard let asset = assets.firstObject else {
            NSLog("[PF][AutoPick] no PHAsset")
            return
        }
        PHImageManager.default().requestImage(
            for: asset,
            targetSize: PHImageManagerMaximumSize,
            contentMode: .default,
            options: nil
        ) { image, _ in
            guard let image = image else {
                NSLog("[PF][AutoPick] image load failed")
                return
            }
            DispatchQueue.main.async {
                NSLog("[PF][AutoPick] image loaded \(image.size.width)x\(image.size.height)")
                self.queryImage = image
                self.searchState = .loading
                Task { await self.searchPerson(image: image) }
            }
        }
    }
}

// MARK: - 单结果 cell：自己拉 PHAsset 缩略图

private struct ResultThumbnailCell: View {
    let result: PersonSearchResult

    @State private var thumbnail: UIImage?
    @State private var didStartLoad = false

    private static let targetSize = CGSize(width: 300, height: 300)

    var body: some View {
        ZStack(alignment: .bottomLeading) {
            // 占位（米白卡 + SF Symbol），缩略图到位后渐隐覆盖。
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color.terracottaSoft.opacity(0.18))
                .aspectRatio(1, contentMode: .fit)
                .overlay {
                    Image(systemName: "photo")
                        .font(.system(size: 22))
                        .foregroundStyle(Color.softInk.opacity(0.6))
                }

            if let thumbnail = thumbnail {
                Image(uiImage: thumbnail)
                    .resizable()
                    .scaledToFill()
                    .aspectRatio(1, contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    .transition(.opacity)
            }

            // 分数 chip：始终在最底层，覆盖在缩略图上。
            HStack(spacing: 4) {
                Image(systemName: "person.fill")
                    .font(.system(size: 9, weight: .semibold))
                Text(String(format: "%.0f%%", result.score * 100))
                    .font(.system(.caption2, design: .monospaced).weight(.semibold))
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .background(Color.terracotta)
            .clipShape(RoundedRectangle(cornerRadius: 4))
            .padding(6)
        }
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .shadow(color: .black.opacity(0.08), radius: 4, x: 0, y: 2)
        .task(id: result.phAssetLocalId) {
            await loadThumbnail()
        }
    }

    @MainActor
    private func loadThumbnail() async {
        guard !didStartLoad else { return }
        guard let localId = result.phAssetLocalId else {
            NSLog("[PF][FacesCell] no phAssetLocalId for image_id=\(result.id)")
            return
        }
        didStartLoad = true
        let assets = PHAsset.fetchAssets(withLocalIdentifiers: [localId], options: nil)
        guard let asset = assets.firstObject else {
            NSLog("[PF][FacesCell] PHAsset not found for localId=\(localId)")
            return
        }
        let options = PHImageRequestOptions()
        options.deliveryMode = .opportunistic
        options.resizeMode = .fast
        options.isNetworkAccessAllowed = true
        let targetSize = Self.targetSize
        let image: UIImage? = await withCheckedContinuation { (cont: CheckedContinuation<UIImage?, Never>) in
            var hasResumed = false
            PHImageManager.default().requestImage(
                for: asset,
                targetSize: targetSize,
                contentMode: .aspectFill,
                options: options
            ) { img, info in
                // opportunistic 会回调两次（低清预览 + 高清），只在最后一帧 resume。
                let isDegraded = (info?[PHImageResultIsDegradedKey] as? Bool) ?? false
                guard !isDegraded, !hasResumed else { return }
                hasResumed = true
                cont.resume(returning: img)
            }
        }
        if let image = image {
            withAnimation(.easeOut(duration: 0.18)) {
                thumbnail = image
            }
        }
    }
}

// MARK: - PHPickerViewController wrapper（dismiss 链路已稳定，参考 Round 2/3）

private struct FacePickerSheet: UIViewControllerRepresentable {
    @Binding var isPresented: Bool
    let onPicked: (UIImage) -> Void

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.filter = .images
        config.selectionLimit = 1
        config.preferredAssetRepresentationMode = .automatic
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    final class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let parent: FacePickerSheet
        init(_ parent: FacePickerSheet) { self.parent = parent }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            // 1) dismiss — 立即触发，独立于取图 / 权限逻辑。
            let target = picker.presentingViewController ?? Self.topPresentedViewController(from: picker)
            if let target = target {
                target.dismiss(animated: true) {
                    DispatchQueue.main.async {
                        self.parent.isPresented = false
                    }
                }
            } else {
                DispatchQueue.main.async {
                    self.parent.isPresented = false
                }
            }

            // 2) 取图 —— PHPickerResult.itemProvider 不需要授权，
            //    HEIC/RAW/iCloud 未下载都通过 loadDataRepresentation 拿到 JPEG bytes。
            guard let result = results.first else { return }
            result.itemProvider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, _ in
                guard let data = data, let image = UIImage(data: data) else {
                    NSLog("[PF][FacesPicker] data/image decode failed")
                    return
                }
                DispatchQueue.main.async {
                    self.parent.onPicked(image)
                }
            }
        }

        private static func topPresentedViewController(from vc: UIViewController) -> UIViewController? {
            var node: UIViewController? = vc
            while let next = node?.presentingViewController {
                node = next
            }
            return node
        }
    }
}