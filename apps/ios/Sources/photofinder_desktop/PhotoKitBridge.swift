//
//  PhotoKitBridge.swift
//  PhotoFinder Next (iOS app layer)
//
//  Implements the C ABI contract declared in platform/src/ios.rs (pf_ios_*).
//  Rust calls in through the concrete IosNativeMediaBridge and frees every
//  returned string with libc::free, so all strings are allocated with strdup.
//
//  HAND-OFF STATE (2026-08-03): written but NOT compiled against an iOS SDK —
//  this repo has no full Xcode. Validate the PhotoKit API usage on a Mac with
//  Xcode during iOS bring-up. The contract table lives in apps/ios/README.md.
//

import Foundation
import Photos
import BackgroundTasks

/// Wire format for a native photo item — mirrors platform/src/bridge.rs
/// `NativePhotoItem` (id, uri, size, modified_time).
private struct NativePhotoItem: Codable {
    let id: String
    let uri: String
    let size: UInt64
    let modified_time: Int64
}

private func assetURI(_ asset: PHAsset) -> String {
    "phasset://\(asset.localIdentifier)"
}

private func photoItem(_ asset: PHAsset) -> NativePhotoItem? {
    var size: UInt64 = 0
    if let resource = PHAssetResource.assetResources(for: asset).first {
        // fileSize is only exposed via KVC on the resource.
        size = (resource.value(forKey: "fileSize") as? NSNumber)?.uint64Value ?? 0
    }
    let mtime = Int64((asset.creationDate ?? Date()).timeIntervalSince1970)
    return NativePhotoItem(id: asset.localIdentifier, uri: assetURI(asset), size: size, modified_time: mtime)
}

/// Resolve a `phasset://` URI back to a PHAsset.
private func asset(for uriStr: String) -> PHAsset? {
    let prefix = "phasset://"
    guard uriStr.hasPrefix(prefix) else { return nil }
    let localIdentifier = String(uriStr.dropFirst(prefix.count))
    return PHAsset.fetchAssets(withLocalIdentifiers: [localIdentifier], options: nil).firstObject
}

private func diagPrint(_ msg: String) {
    let dir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
    let path = dir?.appendingPathComponent("pf_ios_diag.log").path ?? "/tmp/pf_ios_diag.log"
    let line = "[DIAG \(msg)]\n"
    if let handle = try? FileHandle(forWritingTo: URL(fileURLWithPath: path)) {
        handle.seekToEndOfFile()
        handle.write(Data(line.utf8))
        try? handle.close()
    } else {
        try? Data(line.utf8).write(to: URL(fileURLWithPath: path))
    }
}

/// Synchronous authorization for the given access level (blocks until the user
/// answers on first request).
private func authorize(_ level: PHAccessLevel) -> Bool {
    let initialStatus = PHPhotoLibrary.authorizationStatus(for: level)
    diagPrint("authorize called isMain=\(Thread.isMainThread) status=\(initialStatus.rawValue)")
    // authorizationStatus is a read-only query — safe on any thread.
    switch initialStatus {
    case .authorized, .limited:
        return true
    case .denied, .restricted:
        return false
    case .notDetermined:
        // requestAuthorization MUST be issued from the main thread; on iOS
        // Simulator a background-thread call is silently dropped (no dialog,
        // callback fires immediately with .notDetermined). On device the
        // dialog would still be presented but the semaphore on the calling
        // thread would block the main queue from delivering the callback,
        // deadlocking for the full 60s timeout. Hop to main explicitly.
        var granted = false
        let sem = DispatchSemaphore(value: 0)
        diagPrint("notDetermined branch, isMain=\(Thread.isMainThread)")
        if Thread.isMainThread {
            diagPrint("calling requestAuthorization on main directly")
            PHPhotoLibrary.requestAuthorization(for: level) { status in
                diagPrint("callback fired isMain=\(Thread.isMainThread) status=\(status.rawValue)")
                granted = (status == .authorized || status == .limited)
                sem.signal()
            }
        } else {
            diagPrint("dispatching requestAuthorization to main")
            DispatchQueue.main.async {
                diagPrint("main async block running, calling requestAuthorization")
                PHPhotoLibrary.requestAuthorization(for: level) { status in
                    diagPrint("callback fired isMain=\(Thread.isMainThread) status=\(status.rawValue)")
                    granted = (status == .authorized || status == .limited)
                    sem.signal()
                }
            }
        }
        let waitResult = sem.wait(timeout: .now() + 60)
        diagPrint("sem.wait returned, granted=\(granted) waitTimedOut=\(waitResult == .timedOut)")
        return granted
    @unknown default:
        return false
    }
}

/// JSON-encode a value into a strdup'd C string (NULL on failure).
private func strdupJSON<T: Encodable>(_ value: T) -> UnsafeMutablePointer<CChar>? {
    guard let data = try? JSONEncoder().encode(value),
          let json = String(data: data, encoding: .utf8) else { return nil }
    return strdup(json)
}

/// Record an error on the device console and return NULL (Rust treats NULL as failure).
private func fail(_ message: String) -> UnsafeMutablePointer<CChar>? {
    NSLog("PhotoKitBridge: %@", message)
    return nil
}

// MARK: - Exported C ABI (Rust contract)

@_cdecl("pf_ios_photo_permission_status")
func pf_ios_photo_permission_status() -> Int32 {
    authorize(.readWrite) ? 1 : 0
}

@_cdecl("pf_ios_request_photo_permission")
func pf_ios_request_photo_permission() -> Int32 {
    authorize(.readWrite) ? 1 : 0
}

@_cdecl("pf_ios_list_photo_items")
func pf_ios_list_photo_items() -> UnsafeMutablePointer<CChar>? {
    diagPrint("pf_ios_list_photo_items ENTERED isMain=\(Thread.isMainThread)")
    guard authorize(.readWrite) else {
        diagPrint("pf_ios_list_photo_items: authorize returned false")
        return fail("list_photo_items: photo permission not granted")
    }
    diagPrint("pf_ios_list_photo_items: authorize returned true, fetching assets")
    let assets = PHAsset.fetchAssets(with: .image, options: nil)
    diagPrint("pf_ios_list_photo_items: fetched assets count=\(assets.count)")
    var items: [NativePhotoItem] = []
    assets.enumerateObjects { asset, _, _ in
        if let item = photoItem(asset) {
            items.append(item)
        }
    }
    diagPrint("pf_ios_list_photo_items: enumerate done, items=\(items.count)")
    return strdupJSON(items)
}

@_cdecl("pf_ios_photo_item_by_uri")
func pf_ios_photo_item_by_uri(_ uriPtr: UnsafePointer<CChar>?) -> UnsafeMutablePointer<CChar>? {
    guard let uriPtr else { return fail("photo_item_by_uri: nil uri") }
    let uriStr = String(cString: uriPtr)
    guard let asset = asset(for: uriStr) else {
        return fail("photo_item_by_uri: asset not found for \(uriStr)")
    }
    guard let item = photoItem(asset) else {
        return fail("photo_item_by_uri: no item for \(uriStr)")
    }
    return strdupJSON(item)
}

@_cdecl("pf_ios_export_to_temp_file")
func pf_ios_export_to_temp_file(_ uriPtr: UnsafePointer<CChar>?) -> UnsafeMutablePointer<CChar>? {
    guard let uriPtr, let asset = asset(for: String(cString: uriPtr)) else {
        return fail("export_to_temp_file: bad uri")
    }
    return export(asset: asset, thumbnail: false, maxSize: 0)
}

@_cdecl("pf_ios_export_thumbnail_to_temp_file")
func pf_ios_export_thumbnail_to_temp_file(_ uriPtr: UnsafePointer<CChar>?, _ maxSize: UInt32) -> UnsafeMutablePointer<CChar>? {
    guard let uriPtr, let asset = asset(for: String(cString: uriPtr)) else {
        return fail("export_thumbnail_to_temp_file: bad uri")
    }
    return export(asset: asset, thumbnail: true, maxSize: maxSize)
}

@_cdecl("pf_ios_read_bytes")
func pf_ios_read_bytes(_ uriPtr: UnsafePointer<CChar>?) -> UnsafeMutablePointer<CChar>? {
    guard let uriPtr, let asset = asset(for: String(cString: uriPtr)),
          let resource = PHAssetResource.assetResources(for: asset).first else {
        return fail("read_bytes: bad uri")
    }
    var data = Data()
    var error: Error?
    let sem = DispatchSemaphore(value: 0)
    PHAssetResourceManager.default().requestData(for: resource, options: nil) { chunk in
        data.append(chunk)
    } completionHandler: { completionError in
        error = completionError
        sem.signal()
    }
    _ = sem.wait(timeout: .now() + 120)
    if let error {
        return fail("read_bytes failed: \(error)")
    }
    return strdup(data.base64EncodedString())
}

@_cdecl("pf_ios_schedule_bg_task")
func pf_ios_schedule_bg_task(_ idPtr: UnsafePointer<CChar>?, _ waitSecs: UInt64) -> Int32 {
    guard let idPtr else { return -1 }
    let identifier = String(cString: idPtr)
    let request = BGProcessingTaskRequest(identifier: identifier)
    if waitSecs > 0 {
        request.earliestBeginDate = Date(timeIntervalSinceNow: TimeInterval(waitSecs))
    }
    do {
        try BGTaskScheduler.shared.submit(request)
        return 0
    } catch {
        NSLog("PhotoKitBridge: BG task submit failed: %@", "\(error)")
        return -2
    }
}

// MARK: - Export helper

/// Write a full image or a thumbnail (clamped to `maxSize` x `maxSize`) to a
/// temp file, returning its path. `maxSize` is ignored when `thumbnail` is false.
private func export(asset: PHAsset, thumbnail: Bool, maxSize: UInt32) -> UnsafeMutablePointer<CChar>? {
    let manager = PHImageManager.default()
    let options = PHImageRequestOptions()
    options.isSynchronous = false
    options.deliveryMode = thumbnail ? .opportunistic : .highQualityFormat
    options.resizeMode = .exact

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("pf-\(UUID().uuidString).jpg")
    var outPath: String?
    let sem = DispatchSemaphore(value: 0)

    let targetSize = thumbnail
        ? CGSize(width: CGFloat(maxSize), height: CGFloat(maxSize))
        : PHImageManagerMaximumSize

    manager.requestImage(
        for: asset,
        targetSize: targetSize,
        contentMode: thumbnail ? .aspectFill : .default,
        options: options
    ) { image, _ in
        defer { sem.signal() }
        guard let image,
              let data = image.jpegData(compressionQuality: thumbnail ? 0.8 : 0.9) else {
            return
        }
        do {
            try data.write(to: url)
            outPath = url.path
        } catch {
            NSLog("PhotoKitBridge: export write failed: %@", "\(error)")
        }
    }

    _ = sem.wait(timeout: .now() + 120)
    guard let outPath else {
        return fail("export produced no image data")
    }
    return strdup(outPath)
}
