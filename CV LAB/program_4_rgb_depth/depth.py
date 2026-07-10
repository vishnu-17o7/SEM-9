"""
Monocular depth estimation via MiDaS-small (ONNX, ~25 MB).

Weights auto-downloaded on first run into
  assets/models/midas_v21_small_256.onnx
"""
from __future__ import annotations

import cv2
import numpy as np

from _common import MODEL_DIR, ensure_download

WEIGHT_URLS = [
    "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx",
]
WEIGHT_PATH = MODEL_DIR / "midas_v21_small_256.onnx"


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Map a single-channel depth map to a colourised BGR image (inferno)."""
    d = depth.copy()
    d = (d - d.min()) / max(1e-6, d.max() - d.min())
    d = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)


class MiDaSSmall:
    def __init__(self):
        ensure_download(WEIGHT_PATH, WEIGHT_URLS,
                        hint=f"Place the MiDaS ONNX at {WEIGHT_PATH}")
        self.net = cv2.dnn.readNetFromONNX(str(WEIGHT_PATH))

    def infer(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        resized = cv2.resize(img, (256, 256))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        blob = rgb.transpose(2, 0, 1)[None].astype(np.float32)
        self.net.setInput(blob)
        out = self.net.forward()
        depth = out[0, 0]
        depth = cv2.resize(depth, (w, h))
        depth = 1.0 / np.clip(depth, 1e-3, None)
        return depth.astype(np.float32)
