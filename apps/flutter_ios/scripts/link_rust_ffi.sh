#!/bin/bash
# Script to link Rust FFI library into iOS project
# Run this after building the Rust library for iOS

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FLUTTER_FFI_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"  # Workspace root
IOS_DIR="$FLUTTER_FFI_DIR/apps/flutter_ios/ios"
RUST_FFI_DIR="$FLUTTER_FFI_DIR/apps/flutter_ios/rust"

echo "PhotoFinder FFI Linker"
echo "====================="

# Step 1: Build Rust library for iOS simulator
echo "Building Rust library for iOS simulator..."
cd "$FLUTTER_FFI_DIR"
cargo build -p photofinder_flutter_ffi --target aarch64-apple-ios-sim 2>&1

RUST_LIB="$FLUTTER_FFI_DIR/target/aarch64-apple-ios-sim/debug/libphotofinder_flutter_ffi.a"
if [ ! -f "$RUST_LIB" ]; then
    echo "ERROR: Rust library not found at $RUST_LIB"
    exit 1
fi
echo "Rust library built: $RUST_LIB"

# Step 2: Copy library to LocalPods
POD_LIB_DIR="$IOS_DIR/LocalPods/PhotoFinderFFI/lib"
mkdir -p "$POD_LIB_DIR"
cp "$RUST_LIB" "$POD_LIB_DIR/"
echo "Library copied to: $POD_LIB_DIR"

# Step 3: Create Podfile if it doesn't exist
PODFILE="$IOS_DIR/Podfile"
if [ ! -f "$PODFILE" ]; then
    echo "Creating Podfile..."
    cat > "$PODFILE" << 'EOF'
# Uncomment this line to define a global platform for this project
platform :ios, '13.0'

# Flutter iOS pod dependencies
install_all_flutter_pods(Flutter)

# Local PhotoFinder FFI pod
pod 'PhotoFinderFFI', :path => './LocalPods/PhotoFinderFFI'

post_install do |installer|
  installer.pods_project.targets.each do |target|
    flutter_additional_ios_build_settings(target)
  end
end
EOF
    echo "Podfile created"
else
    echo "Podfile already exists, skipping creation"
fi

# Step 4: Run pod install
echo "Running pod install..."
cd "$IOS_DIR"
pod install --repo-update

echo ""
echo "Linking complete!"
echo ""
echo "Note: After pod install, you may need to:"
echo "  1. Open ios/Runner.xcworkspace in Xcode"
echo "  2. Select the Runner target"
echo "  3. Build the project (Cmd+B)"
echo ""
echo "Or simply run: flutter run -d <simulator>"
