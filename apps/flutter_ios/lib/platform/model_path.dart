import 'dart:io';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

/// Platform-specific model path resolution
/// Handles model loading for both iOS and Android
class ModelPath {
  /// Get the models directory path for the current platform
  /// - iOS: Returns path within app bundle
  /// - Android: Extracts models from assets to documents directory on first call
  static Future<String> getModelsDir() async {
    if (Platform.isAndroid) {
      return _getAndroidModelsDir();
    } else {
      return _getIOSModelsDir();
    }
  }

  /// Android: Extract models from assets to app documents directory
  static Future<String> _getAndroidModelsDir() async {
    final appDir = await getApplicationDocumentsDirectory();
    final modelsDir = Directory('${appDir.path}/models');

    if (!await modelsDir.exists()) {
      await modelsDir.create(recursive: true);
      // Copy models from assets to documents directory
      await _copyAssetToFile(
        'assets/models/dinov2_vits14_with_patch.onnx',
        '${modelsDir.path}/dinov2_vits14_with_patch.onnx',
      );
      await _copyAssetToFile(
        'assets/models/scrfd_500m_bnkps.onnx',
        '${modelsDir.path}/scrfd_500m_bnkps.onnx',
      );
      await _copyAssetToFile(
        'assets/models/w600k_r50.onnx',
        '${modelsDir.path}/w600k_r50.onnx',
      );
    }

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
    } catch (e) {
      // Ignore copy errors - model may already exist
    }
  }
}
