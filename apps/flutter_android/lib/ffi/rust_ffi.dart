import 'dart:ffi';
import 'dart:io' show Platform;
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'package:flutter/foundation.dart';

// FFI function signatures matching Rust exports
typedef PfInitNative = Int32 Function();
typedef PfInitDart = int Function();

typedef PfGetVersionNative = Pointer<Utf8> Function();
typedef PfGetVersionDart = Pointer<Utf8> Function();

typedef PfSetModelsDirNative = Int32 Function(Pointer<Utf8> path);
typedef PfSetModelsDirDart = int Function(Pointer<Utf8> path);

typedef PfStartScanNative = Int32 Function();
typedef PfStartScanDart = int Function();

typedef PfStopScanNative = Int32 Function();
typedef PfStopScanDart = int Function();

typedef PfGetStatisticsNative = Pointer<Utf8> Function();
typedef PfGetStatisticsDart = Pointer<Utf8> Function();

typedef PfGetStatusNative = Pointer<Utf8> Function();
typedef PfGetStatusDart = Pointer<Utf8> Function();

typedef PfClearDatabaseNative = Int32 Function();
typedef PfClearDatabaseDart = int Function();

typedef PfAddPhotoNative = Int32 Function(Pointer<Uint8> bytes, IntPtr len, Int64 photoId);
typedef PfAddPhotoDart = int Function(Pointer<Uint8> bytes, int len, int photoId);

typedef PfAddPhotoFullNative = Int32 Function(Pointer<Uint8> bytes, IntPtr len, Pointer<Utf8> assetId);
typedef PfAddPhotoFullDart = int Function(Pointer<Uint8> bytes, int len, Pointer<Utf8> assetId);

typedef PfAddPhotoFromPathNative = Int32 Function(Pointer<Utf8> filePath, Pointer<Utf8> assetId);
typedef PfAddPhotoFromPathDart = int Function(Pointer<Utf8> filePath, Pointer<Utf8> assetId);

typedef PfFreeStringNative = Void Function(Pointer<Utf8> ptr);
typedef PfFreeStringDart = void Function(Pointer<Utf8> ptr);

typedef PfSearchFaceNative = Pointer<Utf8> Function(Pointer<Uint8> bytes, IntPtr len, Int32 topK, Int32 offset);
typedef PfSearchFaceDart = Pointer<Utf8> Function(Pointer<Uint8> bytes, int len, int topK, int offset);

typedef PfSearchObjectNative = Pointer<Utf8> Function(Pointer<Uint8> bytes, IntPtr len, Int32 topK, Int32 offset);
typedef PfSearchObjectDart = Pointer<Utf8> Function(Pointer<Uint8> bytes, int len, int topK, int offset);

typedef PfClusterAllNative = Pointer<Utf8> Function();
typedef PfClusterAllDart = Pointer<Utf8> Function();

typedef PfClusterAllAsyncNative = Int64 Function();
typedef PfClusterAllAsyncDart = int Function();

typedef PfClusterCheckResultNative = Pointer<Utf8> Function(Int64 taskId);
typedef PfClusterCheckResultDart = Pointer<Utf8> Function(int taskId);

typedef PfListPersonsNative = Pointer<Utf8> Function();
typedef PfListPersonsDart = Pointer<Utf8> Function();

typedef PfRenamePersonNative = Int32 Function(Int64 personId, Pointer<Utf8> name);
typedef PfRenamePersonDart = int Function(int personId, Pointer<Utf8> name);

typedef PfGetPersonFacesNative = Pointer<Utf8> Function(Int64 personId);
typedef PfGetPersonFacesDart = Pointer<Utf8> Function(int personId);

typedef PfRebuildHnswNative = Int32 Function();
typedef PfRebuildHnswDart = int Function();

typedef PfSetDataDirNative = Int32 Function(Pointer<Utf8> path);
typedef PfSetDataDirDart = int Function(Pointer<Utf8> path);

// Android scan architecture - new async scan API
typedef PfScanBeginNative = Int32 Function();
typedef PfScanBeginDart = int Function();

typedef PfEnqueuePhotoNative = Int32 Function(Int64 photoId, Pointer<Utf8> path);
typedef PfEnqueuePhotoDart = int Function(int photoId, Pointer<Utf8> path);

typedef PfScanEndNative = Int32 Function();
typedef PfScanEndDart = int Function();

typedef PfGetScanProgressNative = Pointer<Utf8> Function();
typedef PfGetScanProgressDart = Pointer<Utf8> Function();

typedef PfGetMemoryRssKbNative = IntPtr Function();
typedef PfGetMemoryRssKbDart = int Function();

typedef PfRegisterNative = Pointer<Utf8> Function(Pointer<Utf8> email, Pointer<Utf8> password);
typedef PfRegisterDart = Pointer<Utf8> Function(Pointer<Utf8> email, Pointer<Utf8> password);

typedef PfLoginNative = Pointer<Utf8> Function(Pointer<Utf8> email, Pointer<Utf8> password);
typedef PfLoginDart = Pointer<Utf8> Function(Pointer<Utf8> email, Pointer<Utf8> password);

typedef PfLogoutNative = Int32 Function();
typedef PfLogoutDart = int Function();

typedef PfGetAccountNative = Pointer<Utf8> Function();
typedef PfGetAccountDart = Pointer<Utf8> Function();

typedef PfRedeemCodeNative = Pointer<Utf8> Function(Pointer<Utf8> code);
typedef PfRedeemCodeDart = Pointer<Utf8> Function(Pointer<Utf8> code);

typedef PfGetLicenseStatusNative = Pointer<Utf8> Function();
typedef PfGetLicenseStatusDart = Pointer<Utf8> Function();

typedef PfCheckLicenseNative = Int32 Function();
typedef PfCheckLicenseDart = int Function();

typedef PfGetRemainingDaysNative = Int64 Function();
typedef PfGetRemainingDaysDart = int Function();

/// RustFFI - FFI bridge to Rust backend
/// Uses dart:ffi to call C ABI functions exported from the Rust static library
class RustFFI {
  static DynamicLibrary? _lib;
  static bool _initialized = false;
  static int _dbCount = 0;

  /// Load the Rust native library and initialize backend
  static Future<void> init() async {
    debugPrint('[FFI] init: called, _initialized=$_initialized');
    if (_initialized) {
      debugPrint('[FFI] init: already initialized, skipping');
      return;
    }

    try {
      // On Android, the Rust .so is a separate shared library that must be explicitly loaded.
      // On iOS, the library is statically linked into the Runner executable.
      if (Platform.isAndroid) {
        debugPrint('[FFI] init: Android platform, opening libphotofinder_flutter_ffi.so');
        _lib = DynamicLibrary.open('libphotofinder_flutter_ffi.so');
        debugPrint('[FFI] init: Android .so loaded successfully');
      } else {
        debugPrint('[FFI] init: Non-Android platform, using DynamicLibrary.process()');
        _lib = DynamicLibrary.process();
        debugPrint('[FFI] init: process() succeeded');
      }

      // Now call Rust pf_init() to actually initialize
      // Note: _initialized is set by initBackend() based on pf_init() result
      if (_lib != null) {
        try {
          debugPrint('[FFI] init: calling pf_init via initBackend()');
          final result = initBackend();
          if (result == 0) {
            _initialized = true;  // Only set true on success
            debugPrint('[FFI] init: Rust FFI initialized successfully');
          } else {
            debugPrint('[FFI] init: Rust FFI init returned: $result');
          }
        } catch (e) {
          debugPrint('[FFI] init: Rust FFI init error: $e');
        }
      }
    } catch (e, stack) {
      debugPrint('[FFI] init: Failed to load Rust library: $e');
      debugPrint('[FFI] init: stack: $stack');
      // Fallback: library not available, use stub mode
      _initialized = false;
    }
  }

  /// Set the models directory path for bundled models
  /// Call this BEFORE init() to specify where models are located in the app bundle
  /// Returns 0 on success, -1 on failure
  static int setModelsDir(String path) {
    debugPrint('[FFI] setModelsDir: called with path=$path');
    // Must work before init() - load the library if not already loaded
    if (_lib == null) {
      debugPrint('[FFI] setModelsDir: _lib is null, loading library');
      try {
        if (Platform.isAndroid) {
          debugPrint('[FFI] setModelsDir: opening libphotofinder_flutter_ffi.so');
          _lib = DynamicLibrary.open('libphotofinder_flutter_ffi.so');
        } else {
          debugPrint('[FFI] setModelsDir: using DynamicLibrary.process()');
          _lib = DynamicLibrary.process();
        }
        debugPrint('[FFI] setModelsDir: library loaded successfully');
      } catch (e) {
        debugPrint('[FFI] setModelsDir: failed to load library: $e');
        return -1;
      }
    }
    try {
      final pathPtr = path.toNativeUtf8().cast<Utf8>();
      debugPrint('[FFI] setModelsDir: looking up pf_set_models_dir');
      final result = _lib!
          .lookupFunction<PfSetModelsDirNative, PfSetModelsDirDart>('pf_set_models_dir')(pathPtr);
      // Free the allocated string
      calloc.free(pathPtr);
      debugPrint('[FFI] setModelsDir: pf_set_models_dir result=$result');
      return result;
    } catch (e) {
      debugPrint('[FFI] setModelsDir error: $e');
      return -1;
    }
  }

  static String getVersion() {
    if (!_initialized || _lib == null) {
      return '{"version": "1.0.0-stub"}';
    }
    try {
      final ptr = _lib!
          .lookupFunction<PfGetVersionNative, PfGetVersionDart>('pf_get_version')();
      final result = ptr.toDartString();
      _lib!
          .lookupFunction<PfFreeStringNative, PfFreeStringDart>('pf_free_string')(ptr);
      return result;
    } catch (e) {
      return '{"version": "1.0.0-error"}';
    }
  }

  static String getStatus() {
    debugPrint('[FFI] getStatus: _initialized=$_initialized, _lib=${_lib != null}');
    if (!_initialized || _lib == null) {
      return '{"error": "not initialized"}';
    }
    try {
      final ptr = _lib!
          .lookupFunction<PfGetStatusNative, PfGetStatusDart>('pf_get_status')();
      final result = ptr.toDartString();
      debugPrint('[FFI] getStatus: result=$result');
      _lib!
          .lookupFunction<PfFreeStringNative, PfFreeStringDart>('pf_free_string')(ptr);
      return result;
    } catch (e) {
      debugPrint('[FFI] getStatus: error=$e');
      return '{"error": "$e"}';
    }
  }

  static int initBackend() {
    debugPrint('[FFI] initBackend: called, _lib=${_lib != null}');
    if (_lib == null) {
      debugPrint('[FFI] initBackend: _lib is null, returning -1');
      return -1;
    }
    try {
      debugPrint('[FFI] initBackend: looking up pf_init');
      final pfInitFn = _lib!.lookupFunction<PfInitNative, PfInitDart>('pf_init');
      debugPrint('[FFI] initBackend: pf_init symbol found, calling...');
      final result = pfInitFn();
      debugPrint('[FFI] initBackend: pf_init returned $result');
      return result;
    } catch (e, stack) {
      debugPrint('[FFI] initBackend: EXCEPTION: $e');
      debugPrint('[FFI] initBackend: stack: $stack');
      return -1;
    }
  }

  static int startScan() {
    if (!_initialized || _lib == null) return 0;
    try {
      return _lib!
          .lookupFunction<PfStartScanNative, PfStartScanDart>('pf_scan_start')();
    } catch (e) {
      return -1;
    }
  }

  static String getStatistics() {
    debugPrint('[FFI] getStatistics: called, _initialized=$_initialized, _lib=${_lib != null}');
    if (!_initialized || _lib == null) {
      debugPrint('[FFI] getStatistics: using stub, _dbCount=$_dbCount');
      return '{"image_count": $_dbCount, "face_count": 0, "object_count": 0}';
    }
    try {
      debugPrint('[FFI] getStatistics: calling pf_get_statistics');
      final ptr = _lib!
          .lookupFunction<PfGetStatisticsNative, PfGetStatisticsDart>('pf_get_statistics')();
      final result = ptr.toDartString();
      debugPrint('[FFI] getStatistics: result=$result');
      _lib!
          .lookupFunction<PfFreeStringNative, PfFreeStringDart>('pf_free_string')(ptr);
      return result;
    } catch (e) {
      debugPrint('[FFI] getStatistics: error=$e');
      return '{"image_count": $_dbCount, "face_count": 0, "object_count": 0}';
    }
  }

  static int clearDatabase() {
    _dbCount = 0;
    debugPrint('[FFI] clearDatabase: _initialized=$_initialized, _lib=${_lib != null}');
    if (_initialized && _lib != null) {
      try {
        debugPrint('[FFI] clearDatabase: looking up pf_clear_database');
        final result = _lib!.lookupFunction<PfClearDatabaseNative, PfClearDatabaseDart>('pf_clear_database')();
        debugPrint('[FFI] clearDatabase: result=$result');
        return result;
      } catch (e) {
        debugPrint('[FFI] clearDatabase: error: $e');
      }
    } else {
      debugPrint('[FFI] clearDatabase: skipping - not initialized');
    }
    return 0;
  }

  /// Add a photo to the database
  /// In stub mode, just increments the count
  static int addPhoto(Uint8List bytes, int photoId) {
    _dbCount++;
    return 0;
  }

  /// Add photo via FFI (when library is available)
  /// Note: This transfers ownership of the bytes to Rust
  static int addPhotoFFI(Uint8List bytes, int photoId) {
    debugPrint('addPhotoFFI called, bytes=${bytes.length}, photoId=$photoId, _initialized=$_initialized');
    if (!_initialized || _lib == null) {
      return addPhoto(bytes, photoId); // Fallback to stub
    }
    try {
      final ptr = calloc<Uint8>(bytes.length);
      ptr.asTypedList(bytes.length).setAll(0, bytes);
      final result = _lib!
          .lookupFunction<PfAddPhotoNative, PfAddPhotoDart>('pf_add_photo')(ptr, bytes.length, photoId);
      calloc.free(ptr);
      return result;
    } catch (e) {
      return -1;
    }
  }

  /// Add photo via FFI with full asset_id string
  static int addPhotoFullFFI(Uint8List bytes, String assetId) {
    debugPrint('[FFI] addPhotoFullFFI: called, bytes=${bytes.length}, assetId=$assetId, _initialized=$_initialized');
    if (!_initialized || _lib == null) {
      debugPrint('[FFI] addPhotoFullFFI: _lib is null or not initialized, returning -1');
      return -1;
    }
    try {
      final ptr = calloc<Uint8>(bytes.length);
      ptr.asTypedList(bytes.length).setAll(0, bytes);
      final assetIdPtr = assetId.toNativeUtf8().cast<Utf8>();
      debugPrint('[FFI] addPhotoFullFFI: calling pf_add_photo_full');
      final result = _lib!
          .lookupFunction<PfAddPhotoFullNative, PfAddPhotoFullDart>('pf_add_photo_full')(ptr, bytes.length, assetIdPtr);
      debugPrint('[FFI] addPhotoFullFFI: pf_add_photo_full returned $result');
      calloc.free(ptr);
      calloc.free(assetIdPtr);
      return result;
    } catch (e, stack) {
      debugPrint('[FFI] addPhotoFullFFI: error=$e');
      debugPrint('[FFI] addPhotoFullFFI: stack=$stack');
      return -1;
    }
  }

  /// Add photo from file path (avoids temp file write)
  static int addPhotoFromPath(String filePath, String assetId) {
    debugPrint('[FFI] addPhotoFromPath: called, filePath=$filePath, assetId=$assetId, _initialized=$_initialized');
    if (!_initialized || _lib == null) {
      debugPrint('[FFI] addPhotoFromPath: _lib is null or not initialized, returning -1');
      return -1;
    }
    try {
      final filePathPtr = filePath.toNativeUtf8().cast<Utf8>();
      final assetIdPtr = assetId.toNativeUtf8().cast<Utf8>();
      debugPrint('[FFI] addPhotoFromPath: calling pf_add_photo_from_path');
      final result = _lib!
          .lookupFunction<PfAddPhotoFromPathNative, PfAddPhotoFromPathDart>('pf_add_photo_from_path')(filePathPtr, assetIdPtr);
      debugPrint('[FFI] addPhotoFromPath: pf_add_photo_from_path returned $result');
      calloc.free(filePathPtr);
      calloc.free(assetIdPtr);
      return result;
    } catch (e, stack) {
      debugPrint('[FFI] addPhotoFromPath: error=$e');
      debugPrint('[FFI] addPhotoFromPath: stack=$stack');
      return -1;
    }
  }

  static void freeString(dynamic ptr) {
    // No-op in stub
  }

  /// Search faces by image bytes
  /// Returns JSON string with search results (includes pagination info)
  static String searchFace(Uint8List bytes, int topK, {int offset = 0}) {
    if (!_initialized || _lib == null) {
      return '{"results":[],"total":0,"offset":$offset,"limit":$topK}';
    }
    try {
      final ptr = calloc<Uint8>(bytes.length);
      ptr.asTypedList(bytes.length).setAll(0, bytes);
      final resultPtr = _lib!
          .lookupFunction<PfSearchFaceNative, PfSearchFaceDart>('pf_search_face')(ptr, bytes.length, topK, offset);
      final result = resultPtr.toDartString();
      _lib!
          .lookupFunction<PfFreeStringNative, PfFreeStringDart>('pf_free_string')(resultPtr);
      calloc.free(ptr);
      return result;
    } catch (e) {
      return '{"results":[],"total":0,"offset":$offset,"limit":$topK}';
    }
  }

  /// Search objects by image bytes using color histogram features
  /// Returns JSON string with search results (includes pagination info)
  static String searchObject(Uint8List bytes, int topK, {int offset = 0}) {
    if (!_initialized || _lib == null) {
      return '{"results":[],"total":0,"offset":$offset,"limit":$topK}';
    }
    try {
      final ptr = calloc<Uint8>(bytes.length);
      ptr.asTypedList(bytes.length).setAll(0, bytes);
      final resultPtr = _lib!
          .lookupFunction<PfSearchObjectNative, PfSearchObjectDart>('pf_search_object')(ptr, bytes.length, topK, offset);
      final result = resultPtr.toDartString();
      _lib!
          .lookupFunction<PfFreeStringNative, PfFreeStringDart>('pf_free_string')(resultPtr);
      calloc.free(ptr);
      return result;
    } catch (e) {
      return '{"results":[],"total":0,"offset":$offset,"limit":$topK}';
    }
  }

  /// Execute person clustering on all unassigned faces
  /// Returns JSON: {"assigned": N, "created": M, "failed": F}
  static String clusterAll() {
    debugPrint('[FFI] clusterAll: called, _initialized=$_initialized');
    if (!_initialized || _lib == null) {
      debugPrint('[FFI] clusterAll: not initialized, returning error');
      return '{"error": "not initialized"}';
    }
    try {
      debugPrint('[FFI] clusterAll: calling pf_cluster_all');
      final ptr = _lib!
          .lookupFunction<PfClusterAllNative, PfClusterAllDart>('pf_cluster_all')();
      final result = ptr.toDartString();
      debugPrint('[FFI] clusterAll: result=$result');
      _lib!
          .lookupFunction<PfFreeStringNative, PfFreeStringDart>('pf_free_string')(ptr);
      return result;
    } catch (e) {
      debugPrint('[FFI] clusterAll: error=$e');
      return '{"error": "$e"}';
    }
  }

  /// Start async person clustering - returns task ID or -1 if already running
  static int clusterAllAsync() {
    debugPrint('[FFI] clusterAllAsync: called, _initialized=$_initialized');
    if (!_initialized || _lib == null) {
      debugPrint('[FFI] clusterAllAsync: not initialized, returning -1');
      return -1;
    }
    try {
      debugPrint('[FFI] clusterAllAsync: calling pf_cluster_all_async');
      final result = _lib!
          .lookupFunction<PfClusterAllAsyncNative, PfClusterAllAsyncDart>('pf_cluster_all_async')();
      debugPrint('[FFI] clusterAllAsync: result=$result');
      return result;
    } catch (e) {
      debugPrint('[FFI] clusterAllAsync: error=$e');
      return -1;
    }
  }

  /// Check async cluster result - returns JSON or {"status": "running"}
  static String checkClusterResult(int taskId) {
    debugPrint('[FFI] checkClusterResult: called, taskId=$taskId, _initialized=$_initialized');
    if (!_initialized || _lib == null) {
      debugPrint('[FFI] checkClusterResult: not initialized');
      return '{"error": "not initialized"}';
    }
    try {
      debugPrint('[FFI] checkClusterResult: calling pf_cluster_check_result');
      final ptr = _lib!
          .lookupFunction<PfClusterCheckResultNative, PfClusterCheckResultDart>('pf_cluster_check_result')(taskId);
      final result = ptr.toDartString();
      debugPrint('[FFI] checkClusterResult: result=$result');
      _lib!
          .lookupFunction<PfFreeStringNative, PfFreeStringDart>('pf_free_string')(ptr);
      return result;
    } catch (e) {
      debugPrint('[FFI] checkClusterResult: error=$e');
      return '{"error": "$e"}';
    }
  }

  /// List all persons
  /// Returns JSON: [{"id": 1, "name": "Person 1", "face_count": 5}, ...]
  static String listPersons() {
    debugPrint('[FFI] listPersons: called, _initialized=$_initialized');
    if (!_initialized || _lib == null) {
      debugPrint('[FFI] listPersons: not initialized, returning []');
      return '[]';
    }
    try {
      debugPrint('[FFI] listPersons: calling pf_list_persons');
      final ptr = _lib!
          .lookupFunction<PfListPersonsNative, PfListPersonsDart>('pf_list_persons')();
      final result = ptr.toDartString();
      debugPrint('[FFI] listPersons: got ${result.length} chars');
      _lib!
          .lookupFunction<PfFreeStringNative, PfFreeStringDart>('pf_free_string')(ptr);
      return result;
    } catch (e) {
      debugPrint('[FFI] listPersons: error=$e');
      return '[]';
    }
  }

  /// Rename a person
  /// Returns 0 on success, -1 on failure
  static int renamePerson(int personId, String name) {
    if (!_initialized || _lib == null) {
      return -1;
    }
    try {
      final namePtr = name.toNativeUtf8().cast<Utf8>();
      final result = _lib!
          .lookupFunction<PfRenamePersonNative, PfRenamePersonDart>('pf_rename_person')(personId, namePtr);
      calloc.free(namePtr);
      return result;
    } catch (e) {
      return -1;
    }
  }

  /// Get all faces for a person
  /// Returns JSON: [{"face_id": 1, "photo_id": 1}, ...]
  static String getPersonFaces(int personId) {
    if (!_initialized || _lib == null) {
      return '[]';
    }
    try {
      final ptr = _lib!
          .lookupFunction<PfGetPersonFacesNative, PfGetPersonFacesDart>('pf_get_person_faces')(personId);
      final result = ptr.toDartString();
      _lib!
          .lookupFunction<PfFreeStringNative, PfFreeStringDart>('pf_free_string')(ptr);
      return result;
    } catch (e) {
      return '[]';
    }
  }

  /// Rebuild HNSW indices from database
  /// Call after scan+cluster to ensure index is in sync
  static int rebuildHnswIndices() {
    debugPrint('[FFI] rebuildHnswIndices: called, _initialized=$_initialized');
    if (!_initialized || _lib == null) {
      debugPrint('[FFI] rebuildHnswIndices: not initialized, returning -1');
      return -1;
    }
    try {
      debugPrint('[FFI] rebuildHnswIndices: calling pf_rebuild_hnsw_indices');
      final result = _lib!
          .lookupFunction<PfRebuildHnswNative, PfRebuildHnswDart>('pf_rebuild_hnsw_indices')();
      debugPrint('[FFI] rebuildHnswIndices: result=$result');
      return result;
    } catch (e) {
      debugPrint('[FFI] rebuildHnswIndices: error=$e');
      return -1;
    }
  }

  /// Set the data directory path for database storage
  /// Call this BEFORE init() to specify where the database should be stored
  /// Returns 0 on success, -1 on failure
  static int setDataDir(String path) {
    debugPrint('[FFI] setDataDir: called with path=$path');
    if (_lib == null) {
      debugPrint('[FFI] setDataDir: _lib is null, returning -1');
      return -1;
    }
    try {
      final pathPtr = path.toNativeUtf8().cast<Utf8>();
      final result = _lib!
          .lookupFunction<PfSetDataDirNative, PfSetDataDirDart>('pf_set_data_dir')(pathPtr);
      calloc.free(pathPtr);
      debugPrint('[FFI] setDataDir: pf_set_data_dir result=$result');
      return result;
    } catch (e) {
      debugPrint('[FFI] setDataDir error: $e');
      return -1;
    }
  }

  // ============================================================================
  // Android Scan API - Phase 2
  // Separates scan enumeration (Flutter) from AI processing (Rust worker)
  // ============================================================================

  /// Initialize scan queue and spawn AI worker thread.
  /// Call this BEFORE sending any photos.
  /// Returns 0 on success, -1 if already scanning.
  static int scanBegin() {
    debugPrint('[FFI] scanBegin: called');
    if (!_initialized || _lib == null) {
      debugPrint('[FFI] scanBegin: not initialized, returning -1');
      return -1;
    }
    try {
      final result = _lib!
          .lookupFunction<PfScanBeginNative, PfScanBeginDart>('pf_scan_begin')();
      debugPrint('[FFI] scanBegin: result=$result');
      return result;
    } catch (e) {
      debugPrint('[FFI] scanBegin: error=$e');
      return -1;
    }
  }

  /// Enqueue a single photo for AI processing.
  /// Called by Flutter after scanBegin(), once per photo.
  /// Blocks if the queue is full (backpressure).
  /// Returns 0 on success, -1 on failure.
  static int enqueuePhoto(int photoId, String path) {
    debugPrint('[FFI] enqueuePhoto: photoId=$photoId path=$path');
    if (!_initialized || _lib == null) {
      debugPrint('[FFI] enqueuePhoto: not initialized, returning -1');
      return -1;
    }
    try {
      final pathPtr = path.toNativeUtf8().cast<Utf8>();
      final result = _lib!
          .lookupFunction<PfEnqueuePhotoNative, PfEnqueuePhotoDart>('pf_enqueue_photo')(photoId, pathPtr);
      calloc.free(pathPtr);
      debugPrint('[FFI] enqueuePhoto: result=$result');
      return result;
    } catch (e) {
      debugPrint('[FFI] enqueuePhoto: error=$e');
      return -1;
    }
  }

  /// Signal end of photo enumeration and wait for all queued photos to complete.
  /// Call this AFTER all enqueuePhoto calls.
  /// Returns 0 on success.
  static int scanEnd() {
    debugPrint('[FFI] scanEnd: called');
    if (!_initialized || _lib == null) {
      debugPrint('[FFI] scanEnd: not initialized, returning -1');
      return -1;
    }
    try {
      final result = _lib!
          .lookupFunction<PfScanEndNative, PfScanEndDart>('pf_scan_end')();
      debugPrint('[FFI] scanEnd: result=$result');
      return result;
    } catch (e) {
      debugPrint('[FFI] scanEnd: error=$e');
      return -1;
    }
  }

  /// Stop the running scan. Worker will finish current photo then stop.
  static int stopScan() {
    debugPrint('[FFI] stopScan: called');
    if (!_initialized || _lib == null) {
      debugPrint('[FFI] stopScan: not initialized, returning -1');
      return -1;
    }
    try {
      final result = _lib!
          .lookupFunction<PfStopScanNative, PfStopScanDart>('pf_stop_scan')();
      debugPrint('[FFI] stopScan: result=$result');
      return result;
    } catch (e) {
      debugPrint('[FFI] stopScan: error=$e');
      return -1;
    }
  }

  /// Get current scan progress as JSON.
  /// Format: {"enumerated":N,"queued":N,"processing":N,"completed":N,"failed":N,"total":N}
  static String getScanProgress() {
    debugPrint('[FFI] getScanProgress: called');
    if (!_initialized || _lib == null) {
      return '{"enumerated":0,"queued":0,"processing":0,"completed":0,"failed":0,"total":0}';
    }
    try {
      final ptr = _lib!
          .lookupFunction<PfGetScanProgressNative, PfGetScanProgressDart>('pf_get_scan_progress')();
      final result = ptr.toDartString();
      debugPrint('[FFI] getScanProgress: result=$result');
      _lib!
          .lookupFunction<PfFreeStringNative, PfFreeStringDart>('pf_free_string')(ptr);
      return result;
    } catch (e) {
      debugPrint('[FFI] getScanProgress: error=$e');
      return '{"enumerated":0,"queued":0,"processing":0,"completed":0,"failed":0,"total":0}';
    }
  }

  /// Get current process RSS memory in KB.
  /// Returns 0 if not available.
  static int getMemoryRssKb() {
    if (!_initialized || _lib == null) {
      return 0;
    }
    try {
      return _lib!
          .lookupFunction<PfGetMemoryRssKbNative, PfGetMemoryRssKbDart>('pf_get_memory_rss_kb')();
    } catch (e) {
      debugPrint('[FFI] getMemoryRssKb: error=$e');
      return 0;
    }
  }

  // ============================================================================
  // License / Account FFI
  // ============================================================================

  static String registerAccount(String email, String password) {
    debugPrint('[FFI] registerAccount: email=$email');
    if (!_initialized || _lib == null) {
      return '{"error": "not initialized"}';
    }
    try {
      final emailPtr = email.toNativeUtf8().cast<Utf8>();
      final passwordPtr = password.toNativeUtf8().cast<Utf8>();
      final resultPtr = _lib!
          .lookupFunction<PfRegisterNative, PfRegisterDart>('pf_register')(emailPtr, passwordPtr);
      calloc.free(emailPtr);
      calloc.free(passwordPtr);
      final result = resultPtr.toDartString();
      _lib!.lookupFunction<PfFreeStringNative, PfFreeStringDart>('pf_free_string')(resultPtr);
      return result;
    } catch (e) {
      debugPrint('[FFI] registerAccount: error=$e');
      return '{"error": "$e"}';
    }
  }

  static String loginAccount(String email, String password) {
    debugPrint('[FFI] loginAccount: email=$email');
    if (!_initialized || _lib == null) {
      return '{"error": "not initialized"}';
    }
    try {
      final emailPtr = email.toNativeUtf8().cast<Utf8>();
      final passwordPtr = password.toNativeUtf8().cast<Utf8>();
      final resultPtr = _lib!
          .lookupFunction<PfLoginNative, PfLoginDart>('pf_login')(emailPtr, passwordPtr);
      calloc.free(emailPtr);
      calloc.free(passwordPtr);
      final result = resultPtr.toDartString();
      _lib!.lookupFunction<PfFreeStringNative, PfFreeStringDart>('pf_free_string')(resultPtr);
      return result;
    } catch (e) {
      debugPrint('[FFI] loginAccount: error=$e');
      return '{"error": "$e"}';
    }
  }

  static int logoutAccount() {
    debugPrint('[FFI] logoutAccount:');
    if (!_initialized || _lib == null) {
      return -1;
    }
    try {
      return _lib!.lookupFunction<PfLogoutNative, PfLogoutDart>('pf_logout')();
    } catch (e) {
      debugPrint('[FFI] logoutAccount: error=$e');
      return -1;
    }
  }

  static String getAccount() {
    debugPrint('[FFI] getAccount:');
    if (!_initialized || _lib == null) {
      return 'null';
    }
    try {
      final ptr = _lib!.lookupFunction<PfGetAccountNative, PfGetAccountDart>('pf_get_account')();
      final result = ptr.toDartString();
      _lib!.lookupFunction<PfFreeStringNative, PfFreeStringDart>('pf_free_string')(ptr);
      return result;
    } catch (e) {
      debugPrint('[FFI] getAccount: error=$e');
      return 'null';
    }
  }

  static String redeemCode(String code) {
    debugPrint('[FFI] redeemCode: code=$code');
    if (!_initialized || _lib == null) {
      return '{"error": "not initialized"}';
    }
    try {
      final codePtr = code.toNativeUtf8().cast<Utf8>();
      final resultPtr = _lib!
          .lookupFunction<PfRedeemCodeNative, PfRedeemCodeDart>('pf_redeem_code')(codePtr);
      calloc.free(codePtr);
      final result = resultPtr.toDartString();
      _lib!.lookupFunction<PfFreeStringNative, PfFreeStringDart>('pf_free_string')(resultPtr);
      return result;
    } catch (e) {
      debugPrint('[FFI] redeemCode: error=$e');
      return '{"error": "$e"}';
    }
  }

  static String getLicenseStatus() {
    debugPrint('[FFI] getLicenseStatus:');
    if (!_initialized || _lib == null) {
      return 'null';
    }
    try {
      final ptr = _lib!
          .lookupFunction<PfGetLicenseStatusNative, PfGetLicenseStatusDart>('pf_get_license_status')();
      final result = ptr.toDartString();
      _lib!.lookupFunction<PfFreeStringNative, PfFreeStringDart>('pf_free_string')(ptr);
      return result;
    } catch (e) {
      debugPrint('[FFI] getLicenseStatus: error=$e');
      return 'null';
    }
  }

  static bool checkLicense() {
    debugPrint('[FFI] checkLicense:');
    if (!_initialized || _lib == null) {
      return false;
    }
    try {
      final result = _lib!.lookupFunction<PfCheckLicenseNative, PfCheckLicenseDart>('pf_check_license')();
      return result == 1;
    } catch (e) {
      debugPrint('[FFI] checkLicense: error=$e');
      return false;
    }
  }

  static int getRemainingDays() {
    debugPrint('[FFI] getRemainingDays:');
    if (!_initialized || _lib == null) {
      return 0;
    }
    try {
      return _lib!.lookupFunction<PfGetRemainingDaysNative, PfGetRemainingDaysDart>('pf_get_remaining_days')();
    } catch (e) {
      debugPrint('[FFI] getRemainingDays: error=$e');
      return 0;
    }
  }
}
