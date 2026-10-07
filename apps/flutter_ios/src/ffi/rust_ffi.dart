import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

typedef PfInitNative = Int32 Function();
typedef PfInit = int Function();

typedef PfGetVersionNative = Pointer<Utf8> Function();
typedef PfGetVersion = Pointer<Utf8> Function();

typedef PfFreeStringNative = Void Function(Pointer<Utf8>);
typedef PfFreeString = void Function(Pointer<Utf8>);

typedef PfSearchFaceNative = Pointer<Utf8> Function(
    Pointer<Uint8> bytes, IntPtr len, Int32 topK);
typedef PfSearchFace = Pointer<Utf8> Function(
    Pointer<Uint8> bytes, int len, int topK);

typedef PfGetStatisticsNative = Pointer<Utf8> Function();
typedef PfGetStatistics = Pointer<Utf8> Function();

typedef PfScanStartNative = Int32 Function();
typedef PfScanStart = int Function();

typedef PfScanProgressNative = Pointer<Utf8> Function();
typedef PfScanProgress = Pointer<Utf8> Function();

typedef PfScanStopNative = Int32 Function();
typedef PfScanStop = int Function();

typedef PfClearDatabaseNative = Int32 Function();
typedef PfClearDatabase = int Function();

typedef PfGetThumbnailNative = Pointer<Utf8> Function(Int64 imageId, Int32 maxSize);
typedef PfGetThumbnail = Pointer<Utf8> Function(int imageId, int maxSize);

class RustFFI {
  static late PfInit pf_init;
  static late PfGetVersion pf_get_version;
  static late PfFreeString pf_free_string;
  static late PfSearchFace pf_search_face;
  static late PfGetStatistics pf_get_statistics;
  static late PfScanStart pf_scan_start;
  static late PfScanProgress pf_scan_progress;
  static late PfScanStop pf_scan_stop;
  static late PfClearDatabase pf_clear_database;
  static late PfGetThumbnail pf_get_thumbnail;

  static bool _initialized = false;

  static Future<void> init() async {
    if (_initialized) return;

    DynamicLibrary dl;
    if (Platform.isIOS) {
      dl = DynamicLibrary.process();
    } else if (Platform.isMacOS) {
      dl = DynamicLibrary.open('libphotofinder_flutter_ffi.dylib');
    } else if (Platform.isLinux) {
      dl = DynamicLibrary.open('libphotofinder_flutter_ffi.so');
    } else {
      throw UnsupportedError('Unsupported platform');
    }

    pf_init = dl.lookup('pf_init').asFunction<PfInit>();
    pf_get_version = dl.lookup('pf_get_version').asFunction<PfGetVersion>();
    pf_free_string = dl.lookup('pf_free_string').asFunction<PfFreeString>();
    pf_search_face = dl.lookup('pf_search_face').asFunction<PfSearchFace>();
    pf_get_statistics = dl.lookup('pf_get_statistics').asFunction<PfGetStatistics>();
    pf_scan_start = dl.lookup('pf_scan_start').asFunction<PfScanStart>();
    pf_scan_progress = dl.lookup('pf_scan_progress').asFunction<PfScanProgress>();
    pf_scan_stop = dl.lookup('pf_scan_stop').asFunction<PfScanStop>();
    pf_clear_database = dl.lookup('pf_clear_database').asFunction<PfClearDatabase>();
    pf_get_thumbnail = dl.lookup('pf_get_thumbnail').asFunction<PfGetThumbnail>();

    _initialized = true;
  }

  static String getVersion() {
    final ptr = pf_get_version();
    if (ptr == nullptr) {
      throw Exception("pf_get_version returned null pointer");
    }
    final str = ptr.toDartString();
    pf_free_string(ptr);
    return str;
  }

  static void freeString(Pointer<Utf8> ptr) {
    pf_free_string(ptr);
  }

  static int initBackend() {
    return pf_init();
  }

  static int startScan() {
    return pf_scan_start();
  }

  static String getScanProgress() {
    final ptr = pf_scan_progress();
    if (ptr == nullptr) {
      throw Exception("pf_scan_progress returned null pointer");
    }
    final str = ptr.toDartString();
    pf_free_string(ptr);
    return str;
  }

  static int stopScan() {
    return pf_scan_stop();
  }

  static String getStatistics() {
    final ptr = pf_get_statistics();
    if (ptr == nullptr) {
      throw Exception("pf_get_statistics returned null pointer");
    }
    final str = ptr.toDartString();
    pf_free_string(ptr);
    return str;
  }

  static int clearDatabase() {
    return pf_clear_database();
  }

  static String searchFace(Uint8List bytes, int topK) {
    final ptr = malloc<Uint8>(bytes.length);
    try {
      ptr.asTypedList(bytes.length).setAll(0, bytes);

      final resultPtr = pf_search_face(ptr, bytes.length, topK);
      if (resultPtr == nullptr) {
        throw Exception("pf_search_face returned null pointer");
      }
      final result = resultPtr.toDartString();
      pf_free_string(resultPtr);
      return result;
    } finally {
      malloc.free(ptr);
    }
  }

  static String getThumbnail(int imageId, int maxSize) {
    final ptr = pf_get_thumbnail(imageId, maxSize);
    if (ptr == nullptr) {
      throw Exception("pf_get_thumbnail returned null pointer");
    }
    final result = ptr.toDartString();
    pf_free_string(ptr);
    return result;
  }
}
