#!/usr/bin/env python3
"""
Python helper for person_reid_bench.rs: extracts body appearance features.

This script is called by the Rust harness as a subprocess. It:
1. Creates body crops from face bbox + expansion heuristics
2. Extracts MobileNetV4_Conv_Large features (via ConvNeXt proxy in timm)
3. Outputs embeddings as JSON

Input (stdin): JSON with fields:
  - images: list of {id, path, face_bbox: [x, y, w, h]}
  - crop_type: "face" | "upper_body" | "full_body" | "loose"
  - model_name: "mobilenetv4_conv_large" | "convnext_tiny"
  - model_path: path to ONNX model (if pre-exported)

Output (stdout): JSON with fields:
  - embeddings: dict of id -> [float] feature vector
  - crop_type: echoed back
  - model_name: echoed back
  - error: error message if any
"""

import sys
import json
import os
import numpy as np
import cv2

# Try to import onnxruntime for ONNX model
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    print("WARNING: onnxruntime not available", file=sys.stderr)

# Try to import timm for model export
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("WARNING: timm not available", file=sys.stderr)


def expand_face_to_body(face_bbox, img_h, img_w, crop_type):
    """Expand face bbox to body region based on crop_type."""
    fx, fy, fw, fh = face_bbox

    if crop_type == "face":
        # Just the face with small margin
        margin = 0.1
        mx, my = fw * margin, fh * margin
        cx, cy = fx + fw / 2, fy + fh / 2
        size = max(fw, fh) * (1 + 2 * margin)
        x = int(max(0, cx - size / 2))
        y = int(max(0, cy - size / 2))
        w = int(min(img_w - x, size))
        h = int(min(img_h - y, size))
        return x, y, w, h

    elif crop_type == "upper_body":
        # Face + upper body: expand 2x down, 30% wider
        cx = fx + fw / 2
        # Center of expansion: at mid-face
        mid_face_y = fy + fh / 2
        # Height: ~2.5x face height (face is ~40% of upper body)
        height = fh * 2.5
        # Width: 1.3x face width
        width = fw * 1.3
        x = int(max(0, cx - width / 2))
        y = int(max(0, mid_face_y - height * 0.3))  # Start slightly above face center
        w = int(min(img_w - x, width))
        h = int(min(img_h - y, height))
        return x, y, w, h

    elif crop_type == "full_body":
        # Full body: expand 4x down, same width
        cx = fx + fw / 2
        mid_face_y = fy + fh / 2
        height = fh * 4.0
        width = fw * 1.2
        x = int(max(0, cx - width / 2))
        y = int(max(0, mid_face_y - height * 0.15))
        w = int(min(img_w - x, width))
        h = int(min(img_h - y, height))
        return x, y, w, h

    elif crop_type == "loose":
        # Loose: expand 3x down, 50% wider
        cx = fx + fw / 2
        mid_face_y = fy + fh / 2
        height = fh * 3.0
        width = fw * 1.5
        x = int(max(0, cx - width / 2))
        y = int(max(0, mid_face_y - height * 0.2))
        w = int(min(img_w - x, width))
        h = int(min(img_h - y, height))
        return x, y, w, h

    return fx, fy, fw, fh


def load_image(path):
    """Load image as RGB numpy array."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_and_normalize(img, target_h, target_w, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Resize image and normalize with ImageNet mean/std."""
    img = cv2.resize(img, (target_w, target_h))
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(mean)) / np.array(std)
    # HWC -> CHW
    img = img.transpose(2, 0, 1)
    return img


def extract_mobilenetv4_features(body_crops, model_path=None):
    """Extract features using MobileNetV4_Conv_Large via timm ONNX export."""
    if HAS_TIMM and not os.path.exists(model_path):
        print(f"Exporting MobileNetV4_Conv_Large to {model_path}...", file=sys.stderr)
        import torch
        m = timm.create_model('mobilenetv4_conv_large', pretrained=False, num_classes=0, global_pool='avg')
        m.eval()
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = m(x)
        print(f"MobileNetV4_Conv_Large: {out.shape[1]} features", file=sys.stderr)
        torch.onnx.export(m, x, model_path, input_names=['input'], output_names=['output'], opset_version=18)
        print(f"Exported to ONNX", file=sys.stderr)

    if os.path.exists(model_path):
        sess = ort.InferenceSession(model_path)
        input_name = sess.get_inputs()[0].name
        embeddings = {}
        for img_id, crop in body_crops.items():
            img = resize_and_normalize(crop, 256, 256)
            img = img.astype(np.float32).reshape(1, 3, 256, 256)
            out = sess.run(None, {input_name: img})[0]
            # L2 normalize
            feat = out[0]
            norm = np.linalg.norm(feat)
            if norm > 1e-8:
                feat = feat / norm
            embeddings[str(img_id)] = feat.tolist()
        return embeddings, "mobilenetv4_conv_large", 1280

    return None, None, None


def extract_convnext_features(body_crops, model_path=None):
    """Extract features using ConvNeXt_tiny via timm ONNX export."""
    if HAS_TIMM and not os.path.exists(model_path):
        print(f"Exporting ConvNeXt_tiny to {model_path}...", file=sys.stderr)
        import torch
        m = timm.create_model('convnext_tiny', pretrained=False, num_classes=0, global_pool='avg')
        m.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = m(x)
        print(f"ConvNeXt_tiny: {out.shape[1]} features", file=sys.stderr)
        torch.onnx.export(m, x, model_path, input_names=['input'], output_names=['output'], opset_version=18)
        print(f"Exported to ONNX", file=sys.stderr)

    if os.path.exists(model_path):
        sess = ort.InferenceSession(model_path)
        input_name = sess.get_inputs()[0].name
        embeddings = {}
        for img_id, crop in body_crops.items():
            img = resize_and_normalize(crop, 224, 224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            img = img.astype(np.float32).reshape(1, 3, 224, 224)
            out = sess.run(None, {input_name: img})[0]
            # L2 normalize
            feat = out[0]
            norm = np.linalg.norm(feat)
            if norm > 1e-8:
                feat = feat / norm
            embeddings[str(img_id)] = feat.tolist()
        return embeddings, "convnext_tiny", 768

    return None, None, None


def main():
    try:
        data = json.loads(sys.stdin.read())
    except Exception as e:
        print(json.dumps({"error": f"Failed to parse input: {e}"}))
        sys.exit(1)

    images = data.get("images", [])
    crop_type = data.get("crop_type", "upper_body")
    model_name = data.get("model_name", "convnext_tiny")
    model_path = data.get("model_path", "/tmp/body_feature.onnx")

    # Create body crops
    body_crops = {}
    for item in images:
        img_id = item["id"]
        path = item["path"]
        face_bbox = item["face_bbox"]  # [x, y, w, h]

        try:
            img = load_image(path)
            x, y, w, h = expand_face_to_body(face_bbox, img.shape[0], img.shape[1], crop_type)
            crop = img[y:y+h, x:x+w]
            body_crops[img_id] = crop
        except Exception as e:
            print(f"WARNING: Failed to create crop for img {img_id}: {e}", file=sys.stderr)
            continue

    # Extract features
    embeddings = None
    actual_model = None
    dim = None

    if model_name == "mobilenetv4_conv_large":
        embeddings, actual_model, dim = extract_mobilenetv4_features(body_crops, model_path)
    elif model_name == "convnext_tiny":
        embeddings, actual_model, dim = extract_convnext_features(body_crops, model_path)
    else:
        print(f"WARNING: Unknown model {model_name}, trying convnext_tiny", file=sys.stderr)
        embeddings, actual_model, dim = extract_convnext_features(body_crops, model_path)

    result = {
        "embeddings": embeddings or {},
        "crop_type": crop_type,
        "model_name": actual_model or model_name,
        "feature_dim": dim,
        "num_crops": len(body_crops),
        "error": None
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
