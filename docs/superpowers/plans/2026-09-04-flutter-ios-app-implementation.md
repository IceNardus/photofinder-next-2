# Flutter iOS App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Flutter iOS app with pure UI + Rust backend via FFI

**Architecture:** Flutter pages call Rust static library through FFI. Rust handles all backend (AI, DB, HNSW). iOS target with static library compilation.

**Tech Stack:** Flutter 3.x, Rust (staticlib), dart:ffi, image_picker

---

## Overview

```
Flutter iOS App
├── Rust FFI exports (pf_tauri_ipc commands as C ABI)
└── Flutter pages (Faces/Scan/Search/Settings)
```

---

## Task 1: Create Flutter iOS Project Structure

**Files:**
- Create: `apps/flutter_ios/src/lib.dart`
- Create: `apps/flutter_ios/src/main.dart`
- Create: `apps/flutter_ios/src/pages/root_view.dart`
- Create: `apps/flutter_ios/src/pages/faces_view.dart`
- Create: `apps/flutter_ios/src/pages/scan_view.dart`
- Create: `apps/flutter_ios/src/pages/search_view.dart`
- Create: `apps/flutter_ios/src/pages/settings_view.dart`
- Create: `apps/flutter_ios/src/ffi/rust_ffi.dart`
- Create: `apps/flutter_ios/pubspec.yaml`
- Modify: `Cargo.toml` (add flutter_ios workspace member)

- [ ] **Step 1: Create pubspec.yaml**

```yaml
name: photofinder_flutter
description: PhotoFinder iOS App
version: 1.0.0

environment:
  sdk: '>=3.0.0 <4.0.0'

dependencies:
  flutter:
    sdk: flutter
  image_picker: ^1.0.0
  ffi: ^2.0.0

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^3.0.0

flutter:
  uses-material-design: true
```

Run: `flutter create apps/flutter_ios --platforms=ios`
Expected: Creates Flutter project in apps/flutter_ios/

- [ ] **Step 2: Update workspace Cargo.toml**

Modify: `Cargo.toml` (add flutter_ios member)

```toml
members = [
    # ... existing members ...
    "apps/flutter_ios/rust",
]
```

- [ ] **Step 3: Create directory structure**

```bash
mkdir -p apps/flutter_ios/src/pages
mkdir -p apps/flutter_ios/src/ffi
mkdir -p apps/flutter_ios/rust/src
```

Run: `mkdir -p apps/flutter_ios/src/pages apps/flutter_ios/src/ffi apps/flutter_ios/rust/src && echo "Directories created"`
Expected: No output, directories exist

- [ ] **Step 4: Commit**

```bash
git add apps/flutter_ios
git commit -m "feat(flutter): scaffold Flutter iOS project structure"
```

---

## Task 2: Rust FFI Export Layer

**Files:**
- Create: `apps/flutter_ios/rust/Cargo.toml`
- Create: `apps/flutter_ios/rust/src/lib.rs`
- Create: `apps/flutter_ios/rust/src/ffi.rs`

- [ ] **Step 1: Create Rust Cargo.toml**

```toml
[package]
name = "photofinder_flutter_ffi"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["staticlib"]

[dependencies]
pf_tauri_ipc = { path = "../../crates/pf_tauri_ipc" }
pf_application = { path = "../../crates/pf_application" }
pf_config = { path = "../../crates/pf_config" }
pf_ai = { path = "../../crates/pf_ai" }
pf_database = { path = "../../crates/pf_database" }
pf_vector = { path = "../../crates/pf_vector" }
pf_platform = { path = "../../crates/pf_platform" }
pf_core = { path = "../../crates/pf_core" }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tracing = "0.1"
anyhow = "1"
```

Run: `cat > apps/flutter_ios/rust/Cargo.toml << 'EOF'
[package]
name = "photofinder_flutter_ffi"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["staticlib"]

[dependencies]
pf_tauri_ipc = { path = "../../crates/pf_tauri_ipc" }
pf_application = { path = "../../crates/pf_application" }
pf_config = { path = "../../crates/pf_config" }
pf_ai = { path = "../../crates/pf_ai" }
pf_database = { path = "../../crates/pf_database" }
pf_vector = { path = "../../crates/pf_vector" }
pf_platform = { path = "../../crates/pf_platform" }
pf_core = { path = "../../crates/pf_core" }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tracing = "0.1"
anyhow = "1"
EOF`
Expected: File created

- [ ] **Step 2: Create FFI module**

```rust
//! FFI export layer for Flutter iOS
//! All functions use C ABI and are exported with #[no_mangle]

use std::os::raw::c_char;
use std::ptr::null_mut;

fn null_str(s: &str) -> *mut c_char {
    std::CString::new(s).unwrap().into_raw()
}

fn json_ok<T: serde::Serialize>(data: &T) -> *mut c_char {
    null_str(&serde_json::to_string(data).unwrap())
}

fn json_err(err: &str) -> *mut c_char {
    null_str(err)
}

/// Initialize the Rust backend
/// Returns: 0 on success, -1 on failure
#[no_mangle]
pub extern "C" fn pf_init() -> i32 {
    // TODO: Call bootstrap
    0
}

/// Get version string
#[no_mangle]
pub extern "C" fn pf_get_version() -> *mut c_char {
    null_str("PhotoFinder 2.0")
}

/// Start scanning photos
#[no_mangle]
pub extern "C" fn pf_scan_start() -> i32 {
    // TODO: Call scan command
    0
}

/// Get scan progress as JSON
#[no_mangle]
pub extern "C" fn pf_scan_progress() -> *mut c_char {
    json_ok(&serde_json::json!({
        "processed": 0,
        "total": 0,
        "current_file": ""
    }))
}

/// Stop scan
#[no_mangle]
pub extern "C" fn pf_scan_stop() -> i32 {
    0
}

/// Get database statistics
#[no_mangle]
pub extern "C" fn pf_get_statistics() -> *mut c_char {
    json_ok(&serde_json::json!({
        "image_count": 0,
        "face_count": 0
    }))
}

/// Clear database
#[no_mangle]
pub extern "C" fn pf_clear_database() -> i32 {
    0
}

/// Search faces by image bytes
#[no_mangle]
pub extern "C" fn pf_search_face(
    _image_bytes: *const u8,
    _len: usize,
    _top_k: i32,
) -> *mut c_char {
    // TODO: Implement actual search
    json_ok(&serde_json::json!([]))
}

/// Get thumbnail as base64
#[no_mangle]
pub extern "C" fn pf_get_thumbnail(_image_id: i64, _max_size: i32) -> *mut c_char {
    json_err("not implemented")
}

/// Free string allocated by Rust
#[no_mangle]
pub extern "C" fn pf_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe { std::CString::from_raw(ptr) };
    }
}
```

Run: `cat > apps/flutter_ios/rust/src/lib.rs << 'EOF'
//! PhotoFinder Flutter FFI Library
#![deny(unsafe_code)]

pub mod ffi;
EOF`
Expected: File created

- [ ] **Step 3: Create FFI exports**

```rust
//! FFI exports for Flutter iOS

use std::os::raw::c_char;
use std::ptr::null_mut;

fn null_str(s: &str) -> *mut c_char {
    std::CString::new(s).unwrap().into_raw()
}

fn json_ok<T: serde::Serialize>(data: &T) -> *mut c_char {
    null_str(&serde_json::to_string(data).unwrap())
}

#[no_mangle]
pub extern "C" fn pf_init() -> i32 {
    0
}

#[no_mangle]
pub extern "C" fn pf_get_version() -> *mut c_char {
    null_str("PhotoFinder 2.0")
}

#[no_mangle]
pub extern "C" fn pf_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe { std::CString::from_raw(ptr) };
    }
}
```

Run: `cat > apps/flutter_ios/rust/src/ffi.rs << 'EOF'
//! FFI exports
use std::os::raw::c_char;

fn null_str(s: &str) -> *mut c_char {
    std::CString::new(s).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn pf_init() -> i32 {
    0
}

#[no_mangle]
pub extern "C" fn pf_get_version() -> *mut c_char {
    null_str("PhotoFinder 2.0")
}

#[no_mangle]
pub extern "C" fn pf_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe { std::CString::from_raw(ptr) };
    }
}
EOF`
Expected: File created

- [ ] **Step 4: Commit**

```bash
git add apps/flutter_ios/rust
git commit -m "feat(flutter_ffi): add Rust FFI export layer"
```

---

## Task 3: Flutter FFI Bridge

**Files:**
- Create: `apps/flutter_ios/src/ffi/rust_ffi.dart`

- [ ] **Step 1: Create RustFFI class**

```dart
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
    } else {
      dl = DynamicLibrary.open('libphotofinder_flutter_ffi.so');
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
    final str = ptr.toDartString();
    pf_free_string(ptr);
    return str;
  }

  static int stopScan() {
    return pf_scan_stop();
  }

  static String getStatistics() {
    final ptr = pf_get_statistics();
    final str = ptr.toDartString();
    pf_free_string(ptr);
    return str;
  }

  static int clearDatabase() {
    return pf_clear_database();
  }

  static String searchFace(Uint8List bytes, int topK) {
    final ptr = malloc<Uint8>(bytes.length);
    ptr.asTypedList(bytes.length).setAll(0, bytes);

    final resultPtr = pf_search_face(ptr, bytes.length, topK);
    malloc.free(ptr);

    final result = resultPtr.toDartString();
    pf_free_string(resultPtr);
    return result;
  }

  static String getThumbnail(int imageId, int maxSize) {
    final ptr = pf_get_thumbnail(imageId, maxSize);
    final result = ptr.toDartString();
    pf_free_string(ptr);
    return result;
  }
}
```

Run: `cat > apps/flutter_ios/src/ffi/rust_ffi.dart << 'EOFFI'
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

typedef PfInitNative = Int32 Function();
typedef PfInit = int Function();

typedef PfGetVersionNative = Pointer<Utf8> Function();
typedef PfGetVersion = Pointer<Utf8> Function();

typedef PfFreeStringNative = Void Function(Pointer<Utf8>);
typedef PfFreeString = void Function(Pointer<Utf8>);

class RustFFI {
  static late PfInit pf_init;
  static late PfGetVersion pf_get_version;
  static late PfFreeString pf_free_string;

  static bool _initialized = false;

  static Future<void> init() async {
    if (_initialized) return;

    DynamicLibrary dl;
    if (Platform.isIOS) {
      dl = DynamicLibrary.process();
    } else {
      dl = DynamicLibrary.open('libphotofinder_flutter_ffi.so');
    }

    pf_init = dl.lookup('pf_init').asFunction<PfInit>();
    pf_get_version = dl.lookup('pf_get_version').asFunction<PfGetVersion>();
    pf_free_string = dl.lookup('pf_free_string').asFunction<PfFreeString>();

    _initialized = true;
  }

  static String getVersion() {
    final ptr = pf_get_version();
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
}
EOFFI`
Expected: File created

- [ ] **Step 2: Commit**

```bash
git add apps/flutter_ios/src/ffi/rust_ffi.dart
git commit -m "feat(flutter): add Rust FFI bridge layer"
```

---

## Task 4: Flutter Main Entry and RootView

**Files:**
- Create: `apps/flutter_ios/src/main.dart`
- Create: `apps/flutter_ios/src/pages/root_view.dart`

- [ ] **Step 1: Create main.dart**

```dart
import 'package:flutter/material.dart';
import 'pages/root_view.dart';

void main() {
  runApp(const PhotoFinderApp());
}

class PhotoFinderApp extends StatelessWidget {
  const PhotoFinderApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PhotoFinder',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFFE8D5C4)),
        useMaterial3: true,
      ),
      home: const RootView(),
    );
  }
}
```

Run: `cat > apps/flutter_ios/src/main.dart << 'EOFMAIN'
import 'package:flutter/material.dart';
import 'pages/root_view.dart';

void main() {
  runApp(const PhotoFinderApp());
}

class PhotoFinderApp extends StatelessWidget {
  const PhotoFinderApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PhotoFinder',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFFE8D5C4)),
        useMaterial3: true,
      ),
      home: const RootView(),
    );
  }
}
EOFMAIN`
Expected: File created

- [ ] **Step 2: Create RootView with TabBar**

```dart
import 'package:flutter/material.dart';
import 'faces_view.dart';
import 'scan_view.dart';
import 'search_view.dart';
import 'settings_view.dart';

class RootView extends StatefulWidget {
  const RootView({super.key});

  @override
  State<RootView> createState() => _RootViewState();
}

class _RootViewState extends State<RootView> {
  int _selectedIndex = 0;

  static const List<Widget> _pages = [
    FacesView(),
    ScanView(),
    SearchView(),
    SettingsView(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _pages[_selectedIndex],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (index) {
          setState(() {
            _selectedIndex = index;
          });
        },
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.person_outline),
            selectedIcon: Icon(Icons.person),
            label: '面孔',
          ),
          NavigationDestination(
            icon: Icon(Icons.scanner_outlined),
            selectedIcon: Icon(Icons.scanner),
            label: '扫描',
          ),
          NavigationDestination(
            icon: Icon(Icons.search),
            label: '搜索',
          ),
          NavigationDestination(
            icon: Icon(Icons.settings_outlined),
            selectedIcon: Icon(Icons.settings),
            label: '设置',
          ),
        ],
      ),
    );
  }
}
```

Run: `cat > apps/flutter_ios/src/pages/root_view.dart << 'EOFROOT'
import 'package:flutter/material.dart';
import 'faces_view.dart';
import 'scan_view.dart';
import 'search_view.dart';
import 'settings_view.dart';

class RootView extends StatefulWidget {
  const RootView({super.key});

  @override
  State<RootView> createState() => _RootViewState();
}

class _RootViewState extends State<RootView> {
  int _selectedIndex = 0;

  static const List<Widget> _pages = [
    FacesView(),
    ScanView(),
    SearchView(),
    SettingsView(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _pages[_selectedIndex],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (index) {
          setState(() {
            _selectedIndex = index;
          });
        },
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.person_outline),
            selectedIcon: Icon(Icons.person),
            label: '面孔',
          ),
          NavigationDestination(
            icon: Icon(Icons.scanner_outlined),
            selectedIcon: Icon(Icons.scanner),
            label: '扫描',
          ),
          NavigationDestination(
            icon: Icon(Icons.search),
            label: '搜索',
          ),
          NavigationDestination(
            icon: Icon(Icons.settings_outlined),
            selectedIcon: Icon(Icons.settings),
            label: '设置',
          ),
        ],
      ),
    );
  }
}
EOFROOT`
Expected: File created

- [ ] **Step 3: Commit**

```bash
git add apps/flutter_ios/src/main.dart apps/flutter_ios/src/pages/root_view.dart
git commit -m "feat(flutter): add main entry and RootView with TabBar"
```

---

## Task 5: Flutter FacesView Page

**Files:**
- Create: `apps/flutter_ios/src/pages/faces_view.dart`

- [ ] **Step 1: Create FacesView**

```dart
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class FacesView extends StatelessWidget {
  const FacesView({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'AI 预览 · 桌面 InsightFace',
                style: TextStyle(fontSize: 12, color: Colors.grey),
              ),
              const SizedBox(height: 4),
              const Text(
                '人物搜索',
                style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 4),
              const Text(
                '选一张参考面孔 · 找图库里同一人所有照片',
                style: TextStyle(fontSize: 14, color: Colors.grey),
              ),
              const SizedBox(height: 24),
              _buildPickCard(context),
              const SizedBox(height: 24),
              _buildEmptyHint(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildPickCard(BuildContext context) {
    return InkWell(
      onTap: () async {
        final picker = ImagePicker();
        final image = await picker.pickImage(source: ImageSource.gallery);
        if (image != null) {
          // TODO: Call Rust FFI to search
        }
      },
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: const Color(0xFFF5F0E8),
          borderRadius: BorderRadius.circular(16),
        ),
        child: const Row(
          children: [
            Icon(Icons.person_add_alt_1, size: 24, color: Color(0xFFC67B5C)),
            SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '选择一张参考照片',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                  ),
                  SizedBox(height: 2),
                  Text(
                    '从相册导入，系统自动找最大人脸',
                    style: TextStyle(fontSize: 12, color: Colors.grey),
                  ),
                ],
              ),
            ),
            Icon(Icons.chevron_right, color: Colors.grey),
          ],
        ),
      ),
    );
  }

  Widget _buildEmptyHint() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFFF5F0E8),
        borderRadius: BorderRadius.circular(16),
      ),
      child: const Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '暂未选择参考照片',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
          ),
          SizedBox(height: 8),
          Text(
            '点击上方卡片从相册导入含人脸的照片。SCRFD 模型首次调用会懒加载，所有计算在本地完成。',
            style: TextStyle(fontSize: 14, color: Colors.grey),
          ),
        ],
      ),
    );
  }
}
```

Run: `cat > apps/flutter_ios/src/pages/faces_view.dart << 'EOFFACES'
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class FacesView extends StatelessWidget {
  const FacesView({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'AI 预览 · 桌面 InsightFace',
                style: TextStyle(fontSize: 12, color: Colors.grey),
              ),
              const SizedBox(height: 4),
              const Text(
                '人物搜索',
                style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 4),
              const Text(
                '选一张参考面孔 · 找图库里同一人所有照片',
                style: TextStyle(fontSize: 14, color: Colors.grey),
              ),
              const SizedBox(height: 24),
              _buildPickCard(context),
              const SizedBox(height: 24),
              _buildEmptyHint(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildPickCard(BuildContext context) {
    return InkWell(
      onTap: () async {
        final picker = ImagePicker();
        final image = await picker.pickImage(source: ImageSource.gallery);
        if (image != null) {
          // TODO: Call Rust FFI to search
        }
      },
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: const Color(0xFFF5F0E8),
          borderRadius: BorderRadius.circular(16),
        ),
        child: const Row(
          children: [
            Icon(Icons.person_add_alt_1, size: 24, color: Color(0xFFC67B5C)),
            SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '选择一张参考照片',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                  ),
                  SizedBox(height: 2),
                  Text(
                    '从相册导入，系统自动找最大人脸',
                    style: TextStyle(fontSize: 12, color: Colors.grey),
                  ),
                ],
              ),
            ),
            Icon(Icons.chevron_right, color: Colors.grey),
          ],
        ),
      ),
    );
  }

  Widget _buildEmptyHint() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFFF5F0E8),
        borderRadius: BorderRadius.circular(16),
      ),
      child: const Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '暂未选择参考照片',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
          ),
          SizedBox(height: 8),
          Text(
            '点击上方卡片从相册导入含人脸的照片。SCRFD 模型首次调用会懒加载，所有计算在本地完成。',
            style: TextStyle(fontSize: 14, color: Colors.grey),
          ),
        ],
      ),
    );
  }
}
EOFFACES`
Expected: File created

- [ ] **Step 2: Commit**

```bash
git add apps/flutter_ios/src/pages/faces_view.dart
git commit -m "feat(flutter): add FacesView page"
```

---

## Task 6: Flutter ScanView Page

**Files:**
- Create: `apps/flutter_ios/src/pages/scan_view.dart`

- [ ] **Step 1: Create ScanView**

```dart
import 'package:flutter/material.dart';

class ScanView extends StatefulWidget {
  const ScanView({super.key});

  @override
  State<ScanView> createState() => _ScanViewState();
}

class _ScanViewState extends State<ScanView> {
  int _dbCount = 0;
  bool _isScanning = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                '把相册装进数据库 · SCAN',
                style: TextStyle(fontSize: 12, color: Colors.grey),
              ),
              const SizedBox(height: 4),
              const Text(
                '扫描',
                style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 4),
              const Text(
                'iOS 端录入：枚举 PHAsset → SHA256 → SQLite',
                style: TextStyle(fontSize: 14, color: Colors.grey),
              ),
              const SizedBox(height: 24),
              _buildDbStatusCard(),
              const SizedBox(height: 24),
              _buildActionCard(),
              const Spacer(),
              _buildClearButton(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildDbStatusCard() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFFF5F0E8),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  '数据库状态',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                ),
                const SizedBox(height: 4),
                Text(
                  _dbCount >= 0 ? '$_dbCount 张' : '尚未初始化',
                  style: TextStyle(
                    fontSize: 14,
                    color: _dbCount >= 0 ? const Color(0xFFC67B5C) : Colors.grey,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildActionCard() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFFF5F0E8),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        children: [
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: _isScanning ? null : _startScan,
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFFC67B5C),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              child: Text(_isScanning ? '正在扫描…' : '开始扫描'),
            ),
          ),
          const SizedBox(height: 12),
          Text(
            _isScanning ? '正在枚举照片库并写入数据库' : '第一次进 app 时 Rust 后台已初始化',
            style: const TextStyle(fontSize: 12, color: Colors.grey),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildClearButton() {
    return SizedBox(
      width: double.infinity,
      child: OutlinedButton(
        onPressed: _isScanning ? null : _showClearConfirm,
        style: OutlinedButton.styleFrom(
          foregroundColor: const Color(0xFFC67B5C),
          side: const BorderSide(color: Color(0xFFC67B5C)),
          padding: const EdgeInsets.symmetric(vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
        child: const Text('清除扫描记录'),
      ),
    );
  }

  void _startScan() {
    setState(() {
      _isScanning = true;
    });
    // TODO: Call Rust FFI
  }

  void _showClearConfirm() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('清除全部扫描记录？'),
        content: Text('将删除 $_dbCount 张已入库照片的元数据。'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('取消'),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              _clearDb();
            },
            child: const Text('清除'),
          ),
        ],
      ),
    );
  }

  void _clearDb() {
    // TODO: Call Rust FFI
  }
}
```

Run: `cat > apps/flutter_ios/src/pages/scan_view.dart << 'EOFSCAN'
import 'package:flutter/material.dart';

class ScanView extends StatefulWidget {
  const ScanView({super.key});

  @override
  State<ScanView> createState() => _ScanViewState();
}

class _ScanViewState extends State<ScanView> {
  int _dbCount = 0;
  bool _isScanning = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                '把相册装进数据库 · SCAN',
                style: TextStyle(fontSize: 12, color: Colors.grey),
              ),
              const SizedBox(height: 4),
              const Text(
                '扫描',
                style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 4),
              const Text(
                'iOS 端录入：枚举 PHAsset → SHA256 → SQLite',
                style: TextStyle(fontSize: 14, color: Colors.grey),
              ),
              const SizedBox(height: 24),
              _buildDbStatusCard(),
              const SizedBox(height: 24),
              _buildActionCard(),
              const Spacer(),
              _buildClearButton(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildDbStatusCard() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFFF5F0E8),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  '数据库状态',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                ),
                const SizedBox(height: 4),
                Text(
                  _dbCount >= 0 ? '\$_dbCount 张' : '尚未初始化',
                  style: TextStyle(
                    fontSize: 14,
                    color: _dbCount >= 0 ? const Color(0xFFC67B5C) : Colors.grey,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildActionCard() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFFF5F0E8),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        children: [
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: _isScanning ? null : _startScan,
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFFC67B5C),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              child: Text(_isScanning ? '正在扫描…' : '开始扫描'),
            ),
          ),
          const SizedBox(height: 12),
          Text(
            _isScanning ? '正在枚举照片库并写入数据库' : '第一次进 app 时 Rust 后台已初始化',
            style: const TextStyle(fontSize: 12, color: Colors.grey),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildClearButton() {
    return SizedBox(
      width: double.infinity,
      child: OutlinedButton(
        onPressed: _isScanning ? null : _showClearConfirm,
        style: OutlinedButton.styleFrom(
          foregroundColor: const Color(0xFFC67B5C),
          side: const BorderSide(color: Color(0xFFC67B5C)),
          padding: const EdgeInsets.symmetric(vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
        child: const Text('清除扫描记录'),
      ),
    );
  }

  void _startScan() {
    setState(() {
      _isScanning = true;
    });
  }

  void _showClearConfirm() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('清除全部扫描记录？'),
        content: Text('将删除 \$_dbCount 张已入库照片的元数据。'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('取消'),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              _clearDb();
            },
            child: const Text('清除'),
          ),
        ],
      ),
    );
  }

  void _clearDb() {}
}
EOFSCAN`
Expected: File created

- [ ] **Step 2: Commit**

```bash
git add apps/flutter_ios/src/pages/scan_view.dart
git commit -m "feat(flutter): add ScanView page"
```

---

## Task 7: Flutter SearchView and SettingsView

**Files:**
- Create: `apps/flutter_ios/src/pages/search_view.dart`
- Create: `apps/flutter_ios/src/pages/settings_view.dart`

- [ ] **Step 1: Create SearchView**

```dart
import 'package:flutter/material.dart';

class SearchView extends StatelessWidget {
  const SearchView({super.key});

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text('物品搜索'),
          ],
        ),
      ),
    );
  }
}
```

Run: `cat > apps/flutter_ios/src/pages/search_view.dart << 'EOFSEARCH'
import 'package:flutter/material.dart';

class SearchView extends StatelessWidget {
  const SearchView({super.key});

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text('物品搜索'),
          ],
        ),
      ),
    );
  }
}
EOFSEARCH`
Expected: File created

- [ ] **Step 2: Create SettingsView**

```dart
import 'package:flutter/material.dart';

class SettingsView extends StatelessWidget {
  const SettingsView({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                '卷末 · Settings',
                style: TextStyle(fontSize: 12, color: Colors.grey),
              ),
              const SizedBox(height: 4),
              const Text(
                'PhotoFinder Next',
                style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 4),
              const Text(
                'iOS 预览版',
                style: TextStyle(fontSize: 14, color: Colors.grey),
              ),
              const SizedBox(height: 24),
              _buildInfoRow('照片访问', '完整访问'),
              const SizedBox(height: 12),
              _buildInfoRow('名称', 'PhotoFinder Next'),
              const SizedBox(height: 12),
              _buildInfoRow('版本', '1.0'),
              const Spacer(),
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: const Color(0xFFF5F0E8),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: const Text(
                  'iOS 端目前为只读预览壳。完整功能请使用 macOS / Windows 桌面版本。',
                  style: TextStyle(fontSize: 14, color: Colors.grey),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildInfoRow(String title, String value) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFF5F0E8),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        children: [
          const Icon(Icons.info_outline, color: Color(0xFFC67B5C)),
          const SizedBox(width: 14),
          Expanded(child: Text(title)),
          Text(value, style: const TextStyle(color: Colors.grey)),
        ],
      ),
    );
  }
}
```

Run: `cat > apps/flutter_ios/src/pages/settings_view.dart << 'EOFSETTINGS'
import 'package:flutter/material.dart';

class SettingsView extends StatelessWidget {
  const SettingsView({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                '卷末 · Settings',
                style: TextStyle(fontSize: 12, color: Colors.grey),
              ),
              const SizedBox(height: 4),
              const Text(
                'PhotoFinder Next',
                style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 4),
              const Text(
                'iOS 预览版',
                style: TextStyle(fontSize: 14, color: Colors.grey),
              ),
              const SizedBox(height: 24),
              _buildInfoRow('照片访问', '完整访问'),
              const SizedBox(height: 12),
              _buildInfoRow('名称', 'PhotoFinder Next'),
              const SizedBox(height: 12),
              _buildInfoRow('版本', '1.0'),
              const Spacer(),
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: const Color(0xFFF5F0E8),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: const Text(
                  'iOS 端目前为只读预览壳。完整功能请使用 macOS / Windows 桌面版本。',
                  style: TextStyle(fontSize: 14, color: Colors.grey),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildInfoRow(String title, String value) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFF5F0E8),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        children: [
          const Icon(Icons.info_outline, color: Color(0xFFC67B5C)),
          const SizedBox(width: 14),
          Expanded(child: Text(title)),
          Text(value, style: const TextStyle(color: Colors.grey)),
        ],
      ),
    );
  }
}
EOFSETTINGS`
Expected: File created

- [ ] **Step 3: Commit**

```bash
git add apps/flutter_ios/src/pages/search_view.dart apps/flutter_ios/src/pages/settings_view.dart
git commit -m "feat(flutter): add SearchView and SettingsView pages"
```

---

## Verification

### Compilation
```bash
cd apps/flutter_ios
flutter pub get
flutter build ios --simulator --no-codesign
```
Expected: BUILD SUCCEEDED

### Test
```bash
xcrun simctl list devices available | grep iPhone
# Expected: List of available iPhone simulators
flutter run -d <simulator-id>
```
Expected: App runs on iOS simulator

---

## Implementation Order

1. Task 1: Create Flutter iOS Project Structure
2. Task 2: Rust FFI Export Layer
3. Task 3: Flutter FFI Bridge
4. Task 4: Flutter Main Entry and RootView
5. Task 5: Flutter FacesView Page
6. Task 6: Flutter ScanView Page
7. Task 7: Flutter SearchView and SettingsView
