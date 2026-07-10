"""
Lightweight OpenCV-DNN object detector.

Uses YOLOv3-tiny (~34 MB total). Auto-downloads weights on first run
into  assets/models/yolov3-tiny.{weights,cfg}, and assets/models/coco.names.

Why YOLOv3-tiny over MobileNet-SSD here:
  - URLs are stable and the model + config are guaranteed to match.
  - 80 COCO classes out of the box.
  - Runs on CPU at usable speeds.
"""
from __future__ import annotations

import cv2
import numpy as np

from _common import MODEL_DIR, ensure_download

WEIGHTS = MODEL_DIR / "yolov3-tiny.weights"
CFG = MODEL_DIR / "yolov3-tiny.cfg"
NAMES = MODEL_DIR / "coco.names"

WEIGHTS_URLS = [
    "https://data.pjreddie.com/files/yolov3-tiny.weights",
    "https://github.com/hank-ai/darknet/releases/download/v2.0/yolov3-tiny.weights",
]
CFG_URLS = [
    "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
    "https://github.com/pjreddie/darknet/raw/master/cfg/yolov3-tiny.cfg",
]
NAMES_URLS = [
    "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
    "https://github.com/pjreddie/darknet/raw/master/data/coco.names",
]


class YOLOv3Tiny:
    """Tiny YOLO detector — same 80 COCO classes as full YOLOv3."""

    def __init__(self, conf_threshold: float = 0.4):
        ensure_download(WEIGHTS, WEIGHTS_URLS,
                        hint=f"Download yolov3-tiny.weights into {MODEL_DIR}")
        ensure_download(CFG, CFG_URLS,
                        hint=f"Download yolov3-tiny.cfg into {MODEL_DIR}")
        ensure_download(NAMES, NAMES_URLS,
                        hint=f"Download coco.names into {MODEL_DIR}")
        self.net = cv2.dnn.readNetFromDarknet(str(CFG), str(WEIGHTS))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.ln = self.net.getUnconnectedOutLayersNames()
        with open(NAMES, encoding="utf-8") as f:
            self.classes = [c.strip() for c in f if c.strip()]
        self.conf_threshold = conf_threshold

    def detect(self, img: np.ndarray) -> list[dict]:
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_out = self.net.forward(self.ln)

        boxes, confs, ids = [], [], []
        for out in layer_out:
            if len(out.shape) == 3:
                out = out[0]
            for det in out:
                scores = det[5:]
                cid = int(np.argmax(scores))
                # Scale class score by objectness score (det[4])
                s = float(det[4] * scores[cid])
                if s < self.conf_threshold:
                    continue
                cx, cy, bw, bh = det[0:4] * np.array([w, h, w, h])
                x, y = int(cx - bw / 2), int(cy - bh / 2)
                boxes.append([x, y, int(bw), int(bh)])
                confs.append(s)
                ids.append(cid)

        idxs = cv2.dnn.NMSBoxes(boxes, confs, self.conf_threshold, 0.4)
        if len(idxs) == 0:
            return []
        idxs = idxs.flatten() if hasattr(idxs, "flatten") else idxs
        out = []
        for i in idxs:
            x, y, bw, bh = boxes[i]
            out.append({
                "label": self.classes[ids[i]] if ids[i] < len(self.classes) else str(ids[i]),
                "conf": round(confs[i], 3),
                "box": [x, y, bw, bh],
            })
        return out
