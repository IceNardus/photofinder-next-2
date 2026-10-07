//
//  SearchView.swift — 物品搜索 (按截图区域在图库中找同款)。
//
//  和桌面 V1 一致：选一张参考图 → 用户在 UI 上画一个矩形框选出感兴趣的物品
//  → Swift 把裁剪后的 JPEG 落到 ~/Library/Application Support/PhotoFinderNext/
//    query_crop_<uuid>.jpg → Rust 端 ObjectSearch::search 把整张图当作
//  单 ROI（filename 含 "query_crop" 触发此分支），跑 MobileCLIP embedding +
//  HNSW top-N + LightGlue 精排 → 返回 Top-K 命中。
//
//  数据流：
//    1. PHPicker 选图（复用 FacePickerSheet，dismiss 链路已稳）
//    2. 在 UI 上拖动 / 拉伸裁剪框
//    3. UIImage.cgImage.cropping(to:) → JPEG bytes → 写盘
//    4. _pf_ios_search_object(query_path, 20) → JSON [{image_id, image_path,
//       confidence, inlier_ratio, matched_bbox, …}]
//    5. 每个 result 把 image_path 当作 phasset://LOCALID 反查 PHAsset 缩略图
//    6. LazyVGrid 渲染 + confidence chip + matched_bbox 矩形框叠加
//
//  Rust 端实现见 `apps/ios-bridge/src/lib.rs::pf_ios_search_object`。
//  桌面 Tauri 命令 `search_objects` 同款算法。
//

import SwiftUI
import Photos
import PhotosUI
import UniformTypeIdentifiers

// MARK: - C ABI: Swift → Rust (object search)

@_silgen_name("pf_ios_search_object")
private func _pf_ios_search_object(_ path: UnsafePointer<CChar>?, _ top_k: Int32) -> UnsafeMutablePointer<CChar>?

/// 释放 Rust 端 `CString::into_raw()` 分配的字符串。
@_silgen_name("pf_ios_free_string")
private func _pf_ios_free_string(_ ptr: UnsafeMutablePointer<CChar>?)

// MARK: - DTO

/// Rust 返回的 JSON 元素。`image_path` 在 iOS 上是 `phasset://LOCALID`。
private struct ObjectSearchResultDTO: Decodable {
    let image_id: Int64
    let image_path: String
    let confidence: Float
    let inlier_ratio: Float
    let embedding_score: Float
    let matched_bbox: [Float]   // [x1, y1, x2, y2] in image pixels, top-left origin
}

struct ObjectSearchResult: Identifiable, Equatable {
    let id: Int64
    let phAssetLocalId: String?
    let confidence: Float
    let inlierRatio: Float
    let embeddingScore: Float
    let matchedBBox: [Float]    // top-left origin, image pixels
    @MainActor var thumbnail: UIImage?

    static func == (lhs: ObjectSearchResult, rhs: ObjectSearchResult) -> Bool {
        lhs.id == rhs.id && lhs.confidence == rhs.confidence
    }
}

// MARK: - SearchView

struct SearchView: View {
    @State private var isPickerPresented = false
    @State private var sourceImage: UIImage?
    @State private var imageDisplaySize: CGSize = .zero
    @State private var cropNormalized: CGRect = CGRect(x: 0.2, y: 0.2, width: 0.6, height: 0.6)
    @State private var searchState: SearchState = .idle

    enum SearchState {
        case idle                          // 未选图
        case loading                       // 搜索中（含模型加载 / 索引构建 / 搜索）
        case ready([ObjectSearchResult])   // 有结果
        case empty                         // 无命中
        case error(String)                 // 失败
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: Layout.sectionGap) {
                    masthead
                    if sourceImage == nil {
                        pickerCard
                    } else {
                        cropEditor
                        actionRow
                    }
                    searchBody
                }
                .padding(.horizontal, Layout.pagePadding)
                .padding(.bottom, 64)
            }
            .background(Color.paperBg.ignoresSafeArea())
            .toolbar(.hidden, for: .navigationBar)
            .task {
                let args = ProcessInfo.processInfo.arguments
                if args.contains("--auto-pick-first") {
                    try? await Task.sleep(nanoseconds: 600_000_000)
                    autoPickFirstPhoto()
                }
            }
            .fullScreenCover(isPresented: $isPickerPresented) {
                FacePickerSheet(isPresented: $isPickerPresented) { image in
                    self.sourceImage = image
                    self.searchState = .idle
                    self.cropNormalized = CGRect(x: 0.2, y: 0.2, width: 0.6, height: 0.6)
                }
            }
        }
    }

    private var masthead: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("AI 预览 · 桌面 MobileCLIP+LightGlue")
                .font(.monoCaption)
                .tracking(2)
                .foregroundStyle(Color.softInk)
            Text("物品搜索")
                .font(.editorialDisplay)
                .foregroundStyle(Color.ink)
            Text("圈出图中一件物品 · 找图库里所有包含它的照片")
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
                Image(systemName: "viewfinder.circle")
                    .font(.system(size: 22))
                    .foregroundStyle(Color.terracotta)
                VStack(alignment: .leading, spacing: 2) {
                    Text("选择一张参考照片")
                        .font(.system(.headline))
                        .foregroundStyle(Color.ink)
                    Text("从相册导入，在图上圈选物品")
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
    private var cropEditor: some View {
        if let img = sourceImage {
            CropCanvas(
                image: img,
                cropNormalized: $cropNormalized,
                onLayout: { imageDisplaySize = $0 }
            )
            .frame(height: 360)
        }
    }

    private var actionRow: some View {
        HStack(spacing: 10) {
            Button {
                isPickerPresented = true
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "photo.badge.plus")
                    Text("换一张")
                }
                .font(.system(.callout))
                .foregroundStyle(Color.softInk)
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .background(Color.cardBg)
                .clipShape(Capsule())
            }
            .buttonStyle(.plain)

            Button {
                resetCrop()
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "arrow.counterclockwise")
                    Text("重置选区")
                }
                .font(.system(.callout))
                .foregroundStyle(Color.softInk)
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .background(Color.cardBg)
                .clipShape(Capsule())
            }
            .buttonStyle(.plain)

            Spacer()

            Button {
                Task { await runObjectSearch() }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "magnifyingglass")
                    Text("以选中区域搜索")
                }
                .font(.system(.callout, design: .default).weight(.semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(Color.terracotta)
                .clipShape(Capsule())
                .shadow(color: Color.terracotta.opacity(0.3), radius: 6, x: 0, y: 3)
            }
            .buttonStyle(.plain)
        }
    }

    @ViewBuilder
    private var searchBody: some View {
        switch searchState {
        case .idle:
            EmptyView()
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
            Text("MobileCLIP 模型加载 + 索引构建 + 向量检索…")
                .font(.system(.callout))
                .foregroundStyle(Color.softInk)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 24)
    }

    private var emptyResultsHint: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("无匹配结果")
                .font(.editorialHeadline)
                .foregroundStyle(Color.ink)
            Text("图库里没找到包含这件物品的照片。可以试试圈出更明显、单一的特征物体（避免背景或多人合影）。")
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

    private func resultsGrid(_ results: [ObjectSearchResult]) -> some View {
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

            LazyVGrid(columns: [
                GridItem(.flexible(), spacing: 8),
                GridItem(.flexible(), spacing: 8),
                GridItem(.flexible(), spacing: 8),
            ], spacing: 8) {
                ForEach(results) { result in
                    ObjectResultCell(result: result, sourceImage: sourceImage)
                }
            }
        }
    }

    private func resetCrop() {
        cropNormalized = CGRect(x: 0.2, y: 0.2, width: 0.6, height: 0.6)
    }

    // MARK: - Search pipeline

    /// 1. 把 cropNormalized 转为 image pixel coords → 2. UIImage crop → 3.
    ///    JPEG bytes → 4. 写到 data_dir/query_crop_<uuid>.jpg（让 Rust 端把
    ///    filename 里的 "query_crop" 当作 "this is a crop, use whole image
    ///    as single ROI" 信号）→ 5. pf_ios_search_object → 6. JSON 解析 → 7.
    ///    切到 .ready 触发 UI。
    ///
    /// Rust 端做模型懒加载（MobileCLIP + SuperPoint + LightGlue，~350 MB）和
    /// 首次索引构建（HNSW，~1-3 s/100 张）。Swift 端不做进度细粒度上报，
    /// 简单显示 "加载中…" 占位。
    private func runObjectSearch() async {
        guard let source = sourceImage else { return }
        let cropDisplayRect = displayRectForCrop()
        guard cropDisplayRect.width >= 32, cropDisplayRect.height >= 32 else {
            await MainActor.run { searchState = .error("选区太小（至少 32×32 像素）") }
            return
        }
        await MainActor.run { searchState = .loading }

        // Step 1+2: 在 iOS 端把裁剪区域 crop 出来，再编码 JPEG。
        let cropImage: UIImage? = await Task.detached(priority: .userInitiated) { () -> UIImage? in
            return cropUIImage(source, normalized: cropNormalized)
        }.value

        guard let cropImage = cropImage, let jpegBytes = cropImage.jpegData(compressionQuality: 0.95) else {
            await MainActor.run { searchState = .error("无法编码裁剪图为 JPEG") }
            return
        }

        // Step 3: 写盘。文件名必须含 "query_crop"（Rust 端 ObjectSearch::search 的
        // filename token 约定），否则会被当作整图跑 selective search，
        // 再裁一遍就尴尬了。
        let writeResult = await Task.detached(priority: .userInitiated) { () -> Result<String, Error> in
            let fm = FileManager.default
            guard let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
                return .failure(NSError(domain: "SearchView", code: -1, userInfo: [NSLocalizedDescriptionKey: "找不到 Application Support 目录"]))
            }
            let dataDir = appSupport.appendingPathComponent("PhotoFinderNext")
            do {
                try fm.createDirectory(at: dataDir, withIntermediateDirectories: true)
            } catch {
                return .failure(error)
            }
            let queryFile = dataDir.appendingPathComponent("query_crop_\(UUID().uuidString).jpg")
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

        // Step 4: 调 Rust C ABI。top_k=20 与桌面 `search_objects` 默认一致。
        let cjsonPtr: UnsafeMutablePointer<CChar>? = await Task.detached(priority: .userInitiated) { () -> UnsafeMutablePointer<CChar>? in
            queryPath.withCString { cstr -> UnsafeMutablePointer<CChar>? in
                _pf_ios_search_object(cstr, 20)
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
        let entries: [ObjectSearchResultDTO]
        do {
            entries = try JSONDecoder().decode([ObjectSearchResultDTO].self, from: data)
        } catch {
            await MainActor.run { searchState = .error("JSON 解析失败：\(error.localizedDescription)") }
            return
        }

        if entries.isEmpty {
            await MainActor.run { searchState = .empty }
            return
        }

        let results: [ObjectSearchResult] = entries.map { dto in
            let localId: String? = {
                let prefix = "phasset://"
                if dto.image_path.hasPrefix(prefix) {
                    return String(dto.image_path.dropFirst(prefix.count))
                }
                return nil
            }()
            return ObjectSearchResult(
                id: dto.image_id,
                phAssetLocalId: localId,
                confidence: dto.confidence,
                inlierRatio: dto.inlier_ratio,
                embeddingScore: dto.embedding_score,
                matchedBBox: dto.matched_bbox,
                thumbnail: nil
            )
        }

        await MainActor.run { searchState = .ready(results) }
    }

    private func displayRectForCrop() -> CGRect {
        // cropNormalized is in 0..1 of the displayed image area; convert to image pixels.
        let imgSize = sourceImage?.size ?? .zero
        let pxRect = CGRect(
            x: cropNormalized.minX * imgSize.width,
            y: cropNormalized.minY * imgSize.height,
            width: cropNormalized.width * imgSize.width,
            height: cropNormalized.height * imgSize.height
        )
        return pxRect
    }

    /// 烟测 hook：跳过 PHPicker 直接拿相册最新一张图。沿用 FacesView 同款逻辑。
    private func autoPickFirstPhoto() {
        let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        if status == .notDetermined {
            PHPhotoLibrary.requestAuthorization(for: .readWrite) { newStatus in
                if newStatus == .authorized || newStatus == .limited {
                    DispatchQueue.main.async { self.autoPickFirstPhoto() }
                }
            }
            return
        }
        if status == .denied || status == .restricted { return }
        let opts = PHFetchOptions()
        opts.fetchLimit = 1
        opts.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        let assets = PHAsset.fetchAssets(with: .image, options: opts)
        guard let asset = assets.firstObject else { return }
        PHImageManager.default().requestImage(
            for: asset,
            targetSize: PHImageManagerMaximumSize,
            contentMode: .default,
            options: nil
        ) { image, _ in
            guard let image = image else { return }
            DispatchQueue.main.async {
                self.sourceImage = image
                self.cropNormalized = CGRect(x: 0.2, y: 0.2, width: 0.6, height: 0.6)
                Task { await self.runObjectSearch() }
            }
        }
    }
}

// MARK: - Crop canvas

/// 在 sourceImage 上叠加一个可拖动 + 4 角缩放的裁剪矩形。
/// `cropNormalized` 用 0..1 比例存（不依赖显示尺寸），方便后续直接换算成
/// image pixel 坐标送给 UIImage.cropping。
private struct CropCanvas: View {
    let image: UIImage
    @Binding var cropNormalized: CGRect
    let onLayout: (CGSize) -> Void

    private let handleSize: CGFloat = 28

    /// `cropNormalized` snapshot taken on the first onChanged frame of a
    /// gesture. SwiftUI's DragGesture.value.translation is **cumulative**
    /// from gesture start; applying it to the (already-updated) binding
    /// double-counts and the crop box jumps past the finger. Snapshotting
    /// once and applying the cumulative translation against the snapshot
    /// every frame keeps the box glued to the touch point. A single slot
    /// suffices — one finger = one active gesture at a time.
    @State private var gestureStartCrop: CGRect? = nil

    var body: some View {
        GeometryReader { geo in
            let canvasSize = geo.size
            let displayedSize = fittedSize(for: image.size, into: canvasSize)
            let originX = (canvasSize.width - displayedSize.width) / 2
            let originY = (canvasSize.height - displayedSize.height) / 2

            ZStack(alignment: .topLeading) {
                Color.black.opacity(0.04)

                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(width: displayedSize.width, height: displayedSize.height)
                    .position(x: canvasSize.width / 2, y: canvasSize.height / 2)
                    .onAppear { onLayout(displayedSize) }

                // 半透明蒙层 + 矩形挖空
                DimMask(cropRectInView: cropInView(normalized: cropNormalized, origin: CGPoint(x: originX, y: originY), displayedSize: displayedSize))
                    .fill(style: FillStyle(eoFill: true))
                    .foregroundColor(Color.black.opacity(0.45))

                // 矩形边框。`.fill(white.opacity(0.001))` 是 hit-test 锚点：
                // 不加填充的话，`.stroke()` 的 hit area 只有 2pt 宽的边线，
                // 用户点选区内部时 DimMask 的洞 + 矩形边框都接不到事件，
                // dragGesture 永远不触发，看起来"框挪不动"。0.001 的透明度
                // 视觉上完全看不见，但整个 frame 都变成可点击区域。
                Rectangle()
                    .fill(Color.white.opacity(0.001))
                    .overlay(
                        Rectangle()
                            .stroke(Color.terracotta, style: StrokeStyle(lineWidth: 2, dash: [4, 4]))
                    )
                    .frame(width: cropNormalized.width * displayedSize.width,
                           height: cropNormalized.height * displayedSize.height)
                    .position(x: originX + (cropNormalized.midX) * displayedSize.width,
                              y: originY + (cropNormalized.midY) * displayedSize.height)
                    .gesture(dragGesture(originX: originX, originY: originY, displayedSize: displayedSize))

                // 4 个角 handle
                cornerHandles(originX: originX, originY: originY, displayedSize: displayedSize)
            }
        }
        .background(Color.cardBg)
        .clipShape(RoundedRectangle(cornerRadius: Layout.cardCornerRadius, style: .continuous))
        .shadow(color: .black.opacity(0.08), radius: 6, x: 0, y: 2)
    }

    private func fittedSize(for imageSize: CGSize, into canvas: CGSize) -> CGSize {
        guard imageSize.width > 0, imageSize.height > 0 else { return canvas }
        let scale = min(canvas.width / imageSize.width, canvas.height / imageSize.height)
        return CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
    }

    private func cropInView(normalized: CGRect, origin: CGPoint, displayedSize: CGSize) -> CGRect {
        CGRect(
            x: origin.x + normalized.minX * displayedSize.width,
            y: origin.y + normalized.minY * displayedSize.height,
            width: normalized.width * displayedSize.width,
            height: normalized.height * displayedSize.height
        )
    }

    private func dragGesture(originX: CGFloat, originY: CGFloat, displayedSize: CGSize) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                if gestureStartCrop == nil { gestureStartCrop = cropNormalized }
                guard let start = gestureStartCrop else { return }
                let dx = value.translation.width / displayedSize.width
                let dy = value.translation.height / displayedSize.height
                var new = start
                new.origin.x = clamp(start.origin.x + dx, low: 0, high: 1 - start.width)
                new.origin.y = clamp(start.origin.y + dy, low: 0, high: 1 - start.height)
                cropNormalized = new
            }
            .onEnded { _ in
                gestureStartCrop = nil
            }
    }

    @ViewBuilder
    private func cornerHandles(originX: CGFloat, originY: CGFloat, displayedSize: CGSize) -> some View {
        let displayedCropWidth = cropNormalized.width * displayedSize.width
        let displayedCropHeight = cropNormalized.height * displayedSize.height
        let centerX = originX + cropNormalized.midX * displayedSize.width
        let centerY = originY + cropNormalized.midY * displayedSize.height

        // Top-left
        cornerHandle()
            .position(x: centerX - displayedCropWidth / 2, y: centerY - displayedCropHeight / 2)
            .gesture(resizeGestureHandle(corner: .topLeft, originX: originX, originY: originY, displayedSize: displayedSize))
        // Top-right
        cornerHandle()
            .position(x: centerX + displayedCropWidth / 2, y: centerY - displayedCropHeight / 2)
            .gesture(resizeGestureHandle(corner: .topRight, originX: originX, originY: originY, displayedSize: displayedSize))
        // Bottom-left
        cornerHandle()
            .position(x: centerX - displayedCropWidth / 2, y: centerY + displayedCropHeight / 2)
            .gesture(resizeGestureHandle(corner: .bottomLeft, originX: originX, originY: originY, displayedSize: displayedSize))
        // Bottom-right
        cornerHandle()
            .position(x: centerX + displayedCropWidth / 2, y: centerY + displayedCropHeight / 2)
            .gesture(resizeGestureHandle(corner: .bottomRight, originX: originX, originY: originY, displayedSize: displayedSize))
    }

    private func cornerHandle() -> some View {
        Circle()
            .fill(Color.terracotta)
            .frame(width: handleSize, height: handleSize)
            .overlay(Circle().stroke(Color.white, lineWidth: 2))
            .shadow(color: .black.opacity(0.2), radius: 2, x: 0, y: 1)
    }

    private enum ResizeCorner { case topLeft, topRight, bottomLeft, bottomRight }

    private func resizeGestureHandle(corner: ResizeCorner, originX: CGFloat, originY: CGFloat, displayedSize: CGSize) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                if gestureStartCrop == nil { gestureStartCrop = cropNormalized }
                guard let start = gestureStartCrop else { return }
                let dx = value.translation.width / displayedSize.width
                let dy = value.translation.height / displayedSize.height
                var new = start
                let minSize: CGFloat = 0.1
                switch corner {
                case .topLeft:
                    // Pin bottom-right (maxX, maxY) to gesture-start values.
                    new.origin.x = clamp(start.origin.x + dx, low: 0, high: start.maxX - minSize)
                    new.origin.y = clamp(start.origin.y + dy, low: 0, high: start.maxY - minSize)
                    new.size.width = start.maxX - new.origin.x
                    new.size.height = start.maxY - new.origin.y
                case .topRight:
                    // Pin bottom-left (origin.x, maxY).
                    new.size.width = clamp(start.width + dx, low: minSize, high: 1 - start.origin.x)
                    new.origin.y = clamp(start.origin.y + dy, low: 0, high: start.maxY - minSize)
                    new.size.height = start.maxY - new.origin.y
                case .bottomLeft:
                    // Pin top-right (maxX, origin.y).
                    new.origin.x = clamp(start.origin.x + dx, low: 0, high: start.maxX - minSize)
                    new.size.width = start.maxX - new.origin.x
                    new.size.height = clamp(start.height + dy, low: minSize, high: 1 - start.origin.y)
                case .bottomRight:
                    // Pin top-left (origin.x, origin.y).
                    new.size.width = clamp(start.width + dx, low: minSize, high: 1 - start.origin.x)
                    new.size.height = clamp(start.height + dy, low: minSize, high: 1 - start.origin.y)
                }
                cropNormalized = new
            }
            .onEnded { _ in
                gestureStartCrop = nil
            }
    }

    private func clamp(_ value: CGFloat, low: CGFloat, high: CGFloat) -> CGFloat {
        return Swift.max(low, Swift.min(value, high))
    }
}

/// 半透明遮罩：内部挖空一个矩形（即"crop box"外暗内亮）。
/// SwiftUI 没有原生 inverseMask，所以用 Shape + fill(eoFill: true)。
private struct DimMask: Shape {
    let cropRectInView: CGRect

    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.addRect(rect)
        path.addRect(cropRectInView)
        return path
    }
}

/// 把 UIImage 按 normalized 矩形 crop（基于 image pixel 坐标）。
/// UIImage 的像素坐标原点在左上、y 向下，和我们的 normalized 约定一致，
/// 所以可以直接换算。
private func cropUIImage(_ image: UIImage, normalized: CGRect) -> UIImage? {
    let imgSize = image.size
    let scale = image.scale
    // pixel 坐标（已应用 scale）
    let pixelRect = CGRect(
        x: normalized.minX * imgSize.width * scale,
        y: normalized.minY * imgSize.height * scale,
        width: normalized.width * imgSize.width * scale,
        height: normalized.height * imgSize.height * scale
    )
    guard let cgImage = image.cgImage,
          let cropped = cgImage.cropping(to: pixelRect) else { return nil }
    return UIImage(cgImage: cropped, scale: scale, orientation: image.imageOrientation)
}

// MARK: - 结果 cell：拉 PHAsset 缩略图 + matched_bbox 矩形框

private struct ObjectResultCell: View {
    let result: ObjectSearchResult
    let sourceImage: UIImage?

    @State private var thumbnail: UIImage?
    @State private var didStartLoad = false

    private static let targetSize = CGSize(width: 300, height: 300)

    var body: some View {
        ZStack {
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
                    .overlay {
                        // matched_bbox 矩形叠加（top-left origin，image pixels → normalized 0..1）
                        if let image = thumbnailImageSize {
                            let n = matchedBBoxNormalized(thumbSize: image, bbox: result.matchedBBox)
                            Rectangle()
                                .stroke(Color.terracotta, lineWidth: 2)
                                .frame(width: n.width, height: n.height)
                                .position(x: n.midX, y: n.midY)
                        }
                    }
            }

            VStack {
                Spacer()
                HStack {
                    Text(String(format: "%.0f%%", result.confidence * 100))
                        .font(.system(.caption2, design: .monospaced).weight(.semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(Color.terracotta)
                        .clipShape(RoundedRectangle(cornerRadius: 4))
                    Spacer()
                }
                .padding(6)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .shadow(color: .black.opacity(0.08), radius: 4, x: 0, y: 2)
        .task(id: result.phAssetLocalId) {
            await loadThumbnail()
        }
    }

    private var thumbnailImageSize: CGSize? {
        thumbnail?.size
    }

    private func matchedBBoxNormalized(thumbSize: CGSize, bbox: [Float]) -> CGRect {
        // bbox: [x1, y1, x2, y2] in original image pixels. We have thumbnail at thumbSize.
        // We don't have original image size here, so approximate via thumb aspect ratio.
        // The thumbnail is scaled to fill the cell (square), so bbox ratio to original is preserved
        // only if original was also square. For non-square originals, this is approximate but
        // good enough as a visual hint. (Production would store image_size in DTO.)
        guard bbox.count == 4, thumbSize.width > 0, thumbSize.height > 0 else { return .zero }
        let x = CGFloat(bbox[0]) / 1000 * thumbSize.width
        let y = CGFloat(bbox[1]) / 1000 * thumbSize.height
        let w = CGFloat(bbox[2] - bbox[0]) / 1000 * thumbSize.width
        let h = CGFloat(bbox[3] - bbox[1]) / 1000 * thumbSize.height
        return CGRect(x: x, y: y, width: w, height: h)
    }

    @MainActor
    private func loadThumbnail() async {
        guard !didStartLoad else { return }
        guard let localId = result.phAssetLocalId else { return }
        didStartLoad = true
        let assets = PHAsset.fetchAssets(withLocalIdentifiers: [localId], options: nil)
        guard let asset = assets.firstObject else { return }
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

// MARK: - PHPickerViewController wrapper（与 FacesView 共用 dismiss 链路）

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

            guard let result = results.first else { return }
            result.itemProvider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, _ in
                guard let data = data, let image = UIImage(data: data) else { return }
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