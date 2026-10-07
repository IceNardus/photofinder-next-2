#!/bin/bash
# Copies models from project root to assets/models for APK packaging
# Run this before: flutter build apk

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.. && pwd)"
FLUTTER_DIR="$PROJECT_ROOT/apps/flutter_android"

MODELS_SRC="$PROJECT_ROOT/models"
MODELS_DST="$FLUTTER_DIR/assets/models"

echo "[copy-models] Source: $MODELS_SRC"
echo "[copy-models] Dest: $MODELS_DST"

if [ ! -d "$MODELS_SRC" ]; then
    echo "[copy-models] ERROR: models/ not found at $MODELS_SRC"
    exit 1
fi

mkdir -p "$MODELS_DST"

for model in "$MODELS_SRC"/*.onnx; do
    if [ -f "$model" ]; then
        filename=$(basename "$model")
        echo "[copy-models] Copying $filename..."
        cp "$model" "$MODELS_DST/"
    fi
done

echo "[copy-models] Done. $(ls -1 $MODELS_DST | wc -l) model(s) copied"
