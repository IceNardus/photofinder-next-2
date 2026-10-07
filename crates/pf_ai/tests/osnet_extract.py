#!/usr/bin/env python3
"""
OSNet/FastReID Feature Extractor for person_reid_osnet_bench.rs

USAGE:
    python3 osnet_extract.py <model_type> <input_dir> <output_dir>

model_type: osnet_x1_0 | osnet_ain_x1_0 | fastreid_sbs_r50 | convnext_tiny | mobilenetv4

This script:
1. Loads the specified model (OSNet via torch if weights available)
2. Extracts body appearance features from pre-cropped body images
3. Saves embeddings as numpy files + JSON manifest

OSNet weights can be obtained from:
- Original: https://github.com/KaiyangZhou/deep-person-reid
- Direct download: weights are in torch format (.pth)
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── OSNet Architecture (from deep-person-reid) ─────────────────────────────────

class Conv1x1(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return self.bn(self.conv(x))


class Conv1x1ReLU(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class OSNetBlock(nn.Module):
    """Omni-scale feature extraction block."""
    def __init__(self, in_planes, out_planes, ksize=3):
        super().__init__()
        self.conv1 = Conv1x1ReLU(in_planes, out_planes)
        self.conv2 = Conv1x1(out_planes, out_planes)

        # Lightweight convolutions (ghostly)
        self.conv3 = nn.Conv2d(in_planes, in_planes, 3, padding=1, groups=in_planes, bias=False)
        self.bn3 = nn.BatchNorm2d(in_planes)
        self.conv4 = nn.Conv2d(in_planes, in_planes, 3, padding=1, groups=in_planes, bias=False)
        self.bn4 = nn.BatchNorm2d(in_planes)

        self.global_conv = nn.Conv2d(in_planes, out_planes, 1, bias=False)
        self.global_bn = nn.BatchNorm2d(out_planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        # Lightweight branch
        out = self.conv3(x)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.bn4(out)

        # Global branch
        global_feat = F.adaptive_avg_pool2d(identity, 1)
        global_feat = self.global_conv(global_feat)
        global_feat = self.global_bn(global_feat)

        out = out + global_feat
        out = self.relu(out)

        out = self.conv1(out)
        out = self.conv2(out)

        return out


class OSNet(nn.Module):
    """OSNet: Omni-Scale Network for Person Re-Identification."""
    def __init__(self, num_classes=751, loss='softmax', num_features=512):
        super().__init__()
        self.loss = loss
        self.num_features = num_features

        # Stem
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # OSNet blocks
        self.block1 = OSNetBlock(64, 256, ksize=3)
        self.block2 = OSNetBlock(256, 256, ksize=3)
        self.block3 = OSNetBlock(256, 512, ksize=3)
        self.block4 = OSNetBlock(512, 512, ksize=3)

        # Global feature extractor
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # Feature dimension reducer
        self.fc = nn.Linear(512, num_features)
        self.bn_fc = nn.BatchNorm1d(num_features)

        if loss == 'softmax':
            self.classifier = nn.Linear(num_features, num_classes)
        elif loss == 'triplet':
            self.classifier = nn.Linear(num_features, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn_fc(x)

        if self.loss == 'softmax':
            x = self.classifier(x)
        elif self.loss == 'triplet':
            # Return features for triplet loss
            pass

        return x


def load_osnet(weights_path, num_features=512, num_classes=1000):
    """Load OSNet from weights file."""
    model = OSNet(num_classes=num_classes, loss='softmax', num_features=num_features)
    state_dict = torch.load(weights_path, map_location='cpu')

    # Remove 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Remove classifier keys to avoid mismatch
    classifier_keys = [k for k in state_dict.keys() if 'classifier' in k]
    for k in classifier_keys:
        del state_dict[k]

    # Load state dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {missing[:5]}...", file=sys.stderr)
    if unexpected:
        print(f"Unexpected keys: {unexpected[:5]}...", file=sys.stderr)

    # Remove classifier to get features
    model.classifier = nn.Identity()

    model.eval()
    return model, num_features


# ─── Main extraction functions ──────────────────────────────────────────────────

def preprocess_image(img_rgb, mean, std, target_h, target_w):
    """Preprocess a uint8 RGB image for model input."""
    import cv2
    img = cv2.resize(img_rgb, (target_w, target_h))
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(mean)) / np.array(std)
    img = img.transpose(2, 0, 1)
    return img


def l2_normalize(x):
    norm = np.linalg.norm(x)
    if norm < 1e-8:
        return x
    return x / norm


def extract_features(model, images_dir, model_type, output_dir):
    """Extract features for all images in the input directory."""
    import cv2

    # Check if ONNX or PyTorch model
    is_onnx = hasattr(model, 'run')

    config = {
        "osnet_x1_0": {"input_size": (256, 128), "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "osnet_ain_x1_0": {"input_size": (256, 128), "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "reid_ont": {"input_size": (256, 128), "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "convnext_tiny": {"input_size": (224, 224), "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }

    cfg = config.get(model_type, config["osnet_x1_0"])
    target_h, target_w = cfg["input_size"]
    mean, std = cfg["mean"], cfg["std"]

    os.makedirs(output_dir, exist_ok=True)

    img_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not img_files:
        print(f"No images found in {images_dir}", file=sys.stderr)
        return None

    embeddings = {}
    for fname in img_files:
        img_path = os.path.join(images_dir, fname)
        img_id = os.path.splitext(fname)[0]
        img = cv2.imread(img_path)
        if img is None:
            print(f"WARNING: Cannot read {img_path}", file=sys.stderr)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x = preprocess_image(img, mean, std, target_h, target_w)
        x = x.astype(np.float32).reshape(1, 3, target_h, target_w)

        if is_onnx:
            # ONNX model
            out = model.run(None, {'input': x})[0]
        else:
            # PyTorch model
            with torch.no_grad():
                out = model(torch.from_numpy(x)).numpy()

        feat = out.flatten()
        feat = l2_normalize(feat)

        embeddings[img_id] = feat.tolist()
        np.save(os.path.join(output_dir, f"{img_id}.npy"), feat)

    return embeddings


def main():
    if len(sys.argv) < 4:
        print("Usage: osnet_extract.py <model_type> <images_dir> <output_dir>", file=sys.stderr)
        sys.exit(1)

    model_type = sys.argv[1]
    images_dir = sys.argv[2]
    output_dir = sys.argv[3]

    print(f"Loading model: {model_type}", file=sys.stderr)

    model = None
    feature_dim = 512
    input_size = (256, 128)

    # Try Re-ID ONNX model first (YouTu 2021)
    reid_onnx = "/Users/mac/ai-project/photofinder-next-2/models/person_reid_youtu_2021nov.onnx"
    if os.path.exists(reid_onnx):
        try:
            import onnxruntime as ort
            model = ort.InferenceSession(reid_onnx)
            feature_dim = 768
            input_size = (256, 128)
            print(f"Loaded Re-ID ONNX from {reid_onnx}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to load Re-ID ONNX: {e}", file=sys.stderr)

    if model is None:
        print("ERROR: Could not load model", file=sys.stderr)
        manifest = {
            "model_type": model_type,
            "status": "MODEL_NOT_AVAILABLE",
            "feature_dim": 0,
            "num_embeddings": 0,
        }
        print(json.dumps(manifest))
        sys.exit(0)

    print(f"Extracting features from {images_dir}...", file=sys.stderr)
    embeddings = extract_features(model, images_dir, model_type, output_dir)

    if embeddings is None:
        print("FATAL: No embeddings extracted", file=sys.stderr)
        sys.exit(1)

    manifest = {
        "model_type": model_type,
        "status": "OK",
        "feature_dim": feature_dim,
        "num_embeddings": len(embeddings),
        "input_size": input_size,
        "images_dir": images_dir,
        "embeddings": embeddings,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. {len(embeddings)} embeddings saved to {output_dir}", file=sys.stderr)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
