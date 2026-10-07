# Flutter iOS App 架构设计

> **Date:** 2026-09-04
> **Status:** Approved
> **Author:** PhotoFinder Next Team

## 概述

Flutter iOS App 采用纯前端 + Rust 后端的架构：
- **Flutter**: 仅负责 UI 页面展示，不含业务逻辑
- **Rust**: 处理所有后台工作（照片处理、AI 推理、数据库）

通过 **FFI (Foreign Function Interface)** 直接调用 Rust 静态库。

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      Flutter iOS App                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐         ┌──────────────────────────┐  │
│  │   Flutter 页面   │ ◄─────► │   Rust 静态库 (.a)      │  │
│  │                 │   FFI   │                          │  │
│  │  - 面孔搜索     │         │  - 照片处理              │  │
│  │  - 物品搜索     │         │  - AI 推理 (ArcFace)    │  │
│  │  - 扫描入库     │         │  - SQLite 数据库         │  │
│  │  - 设置         │         │  - HNSW 向量索引        │  │
│  │  - 相册访问     │         │                          │  │
│  └─────────────────┘         └──────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 前端框架 | Flutter 3.x | iOS 页面开发 |
| 后端 | Rust | 所有业务逻辑处理 |
| 通信方式 | dart:ffi | Flutter 调用 Rust 函数 |
| AI 模型 | ArcFace + SCRFD | 人脸检测/识别 |
| 数据库 | SQLite | 本地数据存储 |
| 向量索引 | HNSW | 高维向量检索 |
| 照片访问 | image_picker | Flutter 选图 |
| 编译目标 | iOS Simulator/Device | arm64 |

## Rust 静态库导出

### 编译配置

```toml
# Rust Cargo.toml
[lib]
crate-type = ["staticlib"]  # 编译为 iOS 静态库

[target.aarch64-apple-ios]
rustflags = ["-C", "link-arg=-dead_strip"]
```

### 导出函数列表

```rust
use std::os::raw::c_char;
use std::ptr::{self, null_mut};

// ============ 初始化 ============

/// 初始化 Rust 后端
/// 返回: 0=成功, -1=失败
#[no_mangle]
pub extern "C" fn pf_init() -> i32;

/// 获取库版本
#[no_mangle]
pub extern "C" fn pf_get_version() -> *mut c_char {
    let version = CString::new("PhotoFinder 2.0").unwrap();
    version.into_raw()
}

// ============ 扫描 ============

/// 扫描照片入库
/// 返回: task_id (用于查询进度)
#[no_mangle]
pub extern "C" fn pf_scan_start() -> i32;

/// 查询扫描进度
/// 返回: JSON 字符串 {"processed": 100, "total": 500}
#[no_mangle]
pub extern "C" fn pf_scan_progress() -> *mut c_char;

/// 停止扫描
#[no_mangle]
pub extern "C" fn pf_scan_stop() -> i32;

/// 获取扫描结果
/// 返回: JSON 字符串 {"total_found": 100, "new_images": 50}
#[no_mangle]
pub extern "C" fn pf_scan_result() -> *mut c_char;

// ============ 人脸搜索 ============

/// 用图片搜索人脸
/// image_bytes: JPEG 图片字节
/// len: 字节长度
/// top_k: 返回数量
/// 返回: JSON 字符串数组 [SearchHit]
#[no_mangle]
pub extern "C" fn pf_search_face(
    image_bytes: *const u8,
    len: usize,
    top_k: i32,
) -> *mut c_char;

// ============ 数据库操作 ============

/// 获取数据库统计
/// 返回: JSON {"image_count": 1000, "face_count": 500}
#[no_mangle]
pub extern "C" fn pf_get_statistics() -> *mut c_char;

/// 获取缩略图
/// image_id: 图片 ID
/// max_size: 最大边长
/// 返回: Base64 编码的 JPEG
#[no_mangle]
pub extern "C" fn pf_get_thumbnail(image_id: i64, max_size: i32) -> *mut c_char;

/// 清除数据库
#[no_mangle]
pub extern "C" fn pf_clear_database() -> i32;

// ============ 内存管理 ============

/// 释放字符串内存（由 Rust 分配的需要用此释放）
#[no_mangle]
pub extern "C" fn pf_free_string(ptr: *mut c_char);
```

### Rust 端错误处理

```rust
fn null_str(s: &str) -> *mut c_char {
    CString::new(s).unwrap().into_raw()
}

fn json_result<T: serde::Serialize>(result: &Result<T, String>) -> *mut c_char {
    match result {
        Ok(data) => null_str(&serde_json::to_string(data).unwrap()),
        Err(e) => null_str(e),
    }
}
```

## Flutter FFI 桥接层

### pubspec.yaml 依赖

```yaml
dependencies:
  flutter:
    sdk: flutter
  image_picker: ^1.0.0
  ffi: ^2.0.0

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^3.0.0
```

### Rust FFI 绑定 (rust_ffi.dart)

```dart
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

typedef PfInitNative = Int32 Function();
typedef PfInit = int Function();

typedef PfSearchFaceNative = Pointer<Utf8> Function(
    Pointer<Uint8> bytes, IntPtr len, Int32 topK);
typedef PfSearchFace = Pointer<Utf8> Function(
    Pointer<Uint8> bytes, int len, int topK);

typedef PfFreeStringNative = Void Function(Pointer<Utf8>);
typedef PfFreeString = void Function(Pointer<Utf8>);

class RustFFI {
  static late PfInit pf_init;
  static late PfSearchFace pf_search_face;
  static late PfFreeString pf_free_string;

  static void init() {
    final dl = Platform.isIOS
        ? DynamicLibrary.process()
        : DynamicLibrary.open('libphotofinder.so');

    pf_init = dl.lookup('pf_init').asFunction<PfInit>();
    pf_search_face = dl.lookup('pf_search_face').asFunction<PfSearchFace>();
    pf_free_string = dl.lookup('pf_free_string').asFunction<PfFreeString>();
  }
}
```

## Flutter 页面设计

### 页面列表

| 页面 | 路由 | 功能 |
|------|------|------|
| 面孔搜索 | `/faces` | 选择图片 → 搜索相似人脸 |
| 扫描 | `/scan` | 扫描相册入库 |
| 搜索 | `/search` | 物品搜索 |
| 设置 | `/settings` | App 设置 |

### 主界面 (RootView)

```dart
class RootView extends StatefulWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: TabBarView(
        children: [
          FacesView(),
          ScanView(),
          SearchView(),
          SettingsView(),
        ],
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: (index) => setState(() => _selectedIndex = index),
        items: [
          BottomNavigationBarItem(
            icon: Icon(Icons.person),
            label: '面孔',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.scanner),
            label: '扫描',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.search),
            label: '搜索',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.settings),
            label: '设置',
          ),
        ],
      ),
    );
  }
}
```

### 面孔搜索页面 (FacesView)

```dart
class FacesView extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // 顶部导航
        Text('人物搜索'),

        // 选择图片按钮
        ElevatedButton.icon(
          onPressed: () async {
            final picker = ImagePicker();
            final image = await picker.pickImage(source: ImageSource.gallery);
            if (image != null) {
              // 调用 FFI 搜索
              final bytes = await image.readAsBytes();
              final result = RustFFI.searchFace(bytes, 50);
              // 显示结果
            }
          },
          icon: Icon(Icons.add_photo_alternate),
          label: Text('选择参考照片'),
        ),

        // 结果网格
        Expanded(
          child: GridView.builder(
            itemCount: _results.length,
            gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 3,
            ),
            itemBuilder: (context, index) {
              return ResultCard(result: _results[index]);
            },
          ),
        ),
      ],
    );
  }
}
```

### 扫描页面 (ScanView)

```dart
class ScanView extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text('扫描入库'),

        // 数据库状态
        Card(
          child: ListTile(
            leading: Icon(Icons.storage),
            title: Text('数据库'),
            subtitle: Text('${_dbCount} 张照片'),
          ),
        ),

        // 开始扫描按钮
        ElevatedButton(
          onPressed: _isScanning ? null : () {
            RustFFI.startScan();
          },
          child: Text(_isScanning ? '扫描中...' : '开始扫描'),
        ),

        // 进度条
        if (_isScanning)
          LinearProgressIndicator(value: _progress),

        Spacer(),

        // 清除按钮
        OutlinedButton(
          onPressed: () => _showClearConfirm(context),
          child: Text('清除记录'),
        ),
      ],
    );
  }
}
```

## 开发顺序

### Phase 1: Rust 静态库编译
1. 创建 `apps/flutter_ios/` 目录
2. 编写 Rust 导出函数 (`src/lib.rs`, `src/ffi.rs`)
3. 配置 `Cargo.toml` 编译为 `.a`
4. 测试编译成功

### Phase 2: Flutter FFI 桥接
1. 创建 Flutter 项目
2. 实现 `rust_ffi.dart` 绑定
3. 实现基本调用流程

### Phase 3: Flutter UI 页面
1. 主页框架 (TabView)
2. 面孔搜索页面
3. 扫描页面
4. 搜索页面
5. 设置页面

### Phase 4: 集成测试
1. iOS 模拟器运行
2. 照片选择测试
3. 搜索功能测试

## 文件结构

```
apps/
└── flutter_ios/
    ├── src/
    │   ├── main.dart
    │   ├── rust_ffi.dart          # FFI 桥接
    │   ├── pages/
    │   │   ├── root_view.dart
    │   │   ├── faces_view.dart
    │   │   ├── scan_view.dart
    │   │   ├── search_view.dart
    │   │   └── settings_view.dart
    │   └── widgets/
    │       └── result_card.dart
    └── rust/                       # Rust 静态库源码
        ├── Cargo.toml
        └── src/
            ├── lib.rs
            ├── ffi.rs
            └── ...
```

## 技术风险

| 风险 | 缓解措施 |
|------|----------|
| Rust 静态库编译复杂 | 先在主机测试，再交叉编译 iOS |
| FFI 内存管理 | 严格遵循谁分配谁释放原则 |
| 异步处理 | Rust 端使用 blocking 模式，Flutter 端 Future |
| iOS 权限 | 使用 image_picker 处理照片访问权限 |

## 验证标准

1. `cargo build --target aarch64-apple-ios` 成功
2. Flutter iOS 模拟器运行成功
3. 照片选择功能正常
4. 搜索结果显示正常
