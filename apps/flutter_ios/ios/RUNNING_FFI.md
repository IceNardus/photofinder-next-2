# PhotoFinder Flutter iOS FFI 手动链接指南

## 当前状态

- Rust FFI库已编译：`target/aarch64-apple-ios-sim/debug/libphotofinder_flutter_ffi.a`
- Flutter iOS app可以构建（stub模式运行）
- CocoaPods系统有问题，需要手动链接

## 手动链接步骤

### 方法1: 使用Xcode手动链接（推荐）

1. **打开Xcode项目**
   ```bash
   open ios/Runner.xcworkspace
   ```

2. **添加静态库到项目**
   - 在Project Navigator中右键点击Runner项目
   - 选择 "Add Files to Runner..."
   - 导航到：`target/aarch64-apple-ios-sim/debug/`
   - 选择 `libphotofinder_flutter_ffi.a`
   - 勾选 "Copy items if needed" 和 "Create folder references"
   - Add to targets: Runner

3. **设置链接器标志**
   - 选择Runner项目 → Build Settings
   - 搜索 "Other Linker Flags"
   - 添加：`-force_load $(SRCROOT)/../target/aarch64-apple-ios-sim/debug/libphotofinder_flutter_ffi.a`

4. **构建项目**
   - Product → Build (Cmd+B)

### 方法2: 修复CocoaPods后运行脚本

1. **修复CocoaPods安装**
   ```bash
   # 重新安装CocoaPods
   sudo gem install cocoapods
   # 或使用Homebrew
   brew reinstall cocoapods
   ```

2. **运行链接脚本**
   ```bash
   cd apps/flutter_ios/scripts
   ./link_rust_ffi.sh
   ```

3. **重新构建**
   ```bash
   flutter build ios --simulator --no-codesign
   ```

## 验证FFI是否工作

在模拟器上运行app后：
1. 打开Scan页面
2. 点击"开始扫描"
3. 选择几张照片
4. 如果看到数据库状态显示照片数量增加，说明FFI正常工作

## 如果FFI未链接（stub模式）

App仍然可以运行，但：
- 数据库状态始终显示0或上次数值
- 照片不会被实际处理
- 所有FFI调用返回stub值

## 调试FFI

可以在Rust端添加日志：
```rust
eprintln!("pf_add_photo called with {} bytes, photo_id: {}", len, photo_id);
```

重新编译后运行：
```bash
cargo build -p photofinder_flutter_ffi --target aarch64-apple-ios-sim
# 重新复制库文件到项目
```
