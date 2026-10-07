import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import '../ffi/rust_ffi.dart';

/// Platform-specific model path resolution
/// Handles model loading for both iOS and Android
class ModelPath {
  /// Get the models directory path for the current platform
  /// - iOS: Returns path within app bundle
  /// - Android: Extracts models from assets to documents directory on first call
  static Future<String> getModelsDir() async {
    debugPrint('[ModelPath] getModelsDir: platform=${Platform.operatingSystem}');
    if (Platform.isAndroid) {
      final dir = await _getAndroidModelsDir();
      debugPrint('[ModelPath] getModelsDir: Android dir=$dir');
      return dir;
    } else {
      final dir = await _getIOSModelsDir();
      debugPrint('[ModelPath] getModelsDir: iOS dir=$dir');
      return dir;
    }
  }

  /// Android: Extract models from assets to app documents directory
  static Future<String> _getAndroidModelsDir() async {
    debugPrint('[ModelPath] _getAndroidModelsDir: 开始提取模型');
    final appDir = await getApplicationDocumentsDirectory();
    final modelsDir = Directory('${appDir.path}/models');
    debugPrint('[ModelPath] _getAndroidModelsDir: appDir=${appDir.path}, modelsDir=${modelsDir.path}');

    // Also set data directory for database storage
    // ignore: avoid_init_with_future
    RustFFI.setDataDir(appDir.path);

    // Always recreate models dir to ensure fresh copy (in case previous copy was partial)
    if (await modelsDir.exists()) {
      debugPrint('[ModelPath] _getAndroidModelsDir: 删除旧模型目录');
      await modelsDir.delete(recursive: true);
    }
    await modelsDir.create(recursive: true);
    debugPrint('[ModelPath] _getAndroidModelsDir: 创建新模型目录成功');

    int copiedCount = 0;
    // Copy models from assets to documents directory
    // Flutter asset path: 'assets/models/xxx' is relative to flutter_assets/
    try {
      debugPrint('[ModelPath] _getAndroidModelsDir: 复制 dinov2_vits14_with_patch.onnx');
      await _copyAssetToFile(
        'assets/models/dinov2_vits14_with_patch.onnx',
        '${modelsDir.path}/dinov2_vits14_with_patch.onnx',
      );
      copiedCount++;
      debugPrint('[ModelPath] _getAndroidModelsDir: dinov2 复制成功');
    } catch (e) {
      debugPrint('[ModelPath] _getAndroidModelsDir: dinov2 复制失败: $e');
    }
    try {
      debugPrint('[ModelPath] _getAndroidModelsDir: 复制 scrfd_500m_bnkps.onnx');
      await _copyAssetToFile(
        'assets/models/scrfd_500m_bnkps.onnx',
        '${modelsDir.path}/scrfd_500m_bnkps.onnx',
      );
      copiedCount++;
      debugPrint('[ModelPath] _getAndroidModelsDir: scrfd 复制成功');
    } catch (e) {
      debugPrint('[ModelPath] _getAndroidModelsDir: scrfd 复制失败: $e');
    }
    try {
      debugPrint('[ModelPath] _getAndroidModelsDir: 复制 w600k_r50.onnx');
      await _copyAssetToFile(
        'assets/models/w600k_r50.onnx',
        '${modelsDir.path}/w600k_r50.onnx',
      );
      copiedCount++;
      debugPrint('[ModelPath] _getAndroidModelsDir: w600k 复制成功');
    } catch (e) {
      debugPrint('[ModelPath] _getAndroidModelsDir: w600k 复制失败: $e');
    }

    debugPrint('[ModelPath] _getAndroidModelsDir: copiedCount=$copiedCount');

    if (copiedCount == 0) {
      throw Exception('Failed to copy any model files from assets');
    }

    debugPrint('[ModelPath] _getAndroidModelsDir: 成功复制 $copiedCount 个模型文件到 $modelsDir');
    return modelsDir.path;
  }

  /// iOS: Find models in app bundle
  static Future<String> _getIOSModelsDir() async {
    final exePath = File(Platform.resolvedExecutable).parent.path;
    final possiblePaths = [
      exePath, // iOS simulator: Runner.app/
      '$exePath/Resources', // iOS device fallback
      '$exePath/Resources/Models', // iOS device
      '$exePath/Models', // Fallback
    ];

    for (final p in possiblePaths) {
      if (await Directory(p).exists()) {
        return p;
      }
    }

    // Fallback to first option
    return possiblePaths[0];
  }

  /// Copy asset file to destination path
  static Future<void> _copyAssetToFile(String assetPath, String destPath) async {
    try {
      final ByteData data = await rootBundle.load(assetPath);
      final List<int> bytes = data.buffer.asUint8List();
      await File(destPath).writeAsBytes(bytes);
      debugPrint('Model copied: $destPath (${bytes.length} bytes)');
    } catch (e) {
      debugPrint('Model copy failed: asset=$assetPath dest=$destPath error=$e');
      rethrow;  // Propagate error so caller knows copy failed
    }
  }
}
