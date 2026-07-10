"""YOLOv3 via OpenCV DNN (darknet-53 backbone, 80 COCO classes).

Weights auto-download on first run (~248 MB) into
  assets/models/yolov3.weights
  assets/models/yolov3.cfg
  assets/models/coco.names
"""
from __future__ import annotations

import cv2
import numpy as np

from _common import MODEL_DIR, ensure_download

WEIGHTS = MODEL_DIR / "yolov3.weights"
CFG = MODEL_DIR / "yolov3.cfg"
NAMES = MODEL_DIR / "coco.names"

WEIGHTS_URLS = [
    "https://data.pjreddie.com/files/yolov3.weights",
    "https://github.com/hank-ai/darknet/releases/download/v2.0/yolov3.weights",
]
CFG_URLS = [
    "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    "https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg",
]
NAMES_URLS = [
    "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
    "https://github.com/pjreddie/darknet/raw/master/data/coco.names",
]


class YOLOv3:
    def __init__(self):
        ensure_download(WEIGHTS, WEIGHTS_URLS,
                        hint=f"Download yolov3.weights (~248 MB) into {MODEL_DIR}")
        ensure_download(CFG, CFG_URLS,
                        hint=f"Download yolov3.cfg into {MODEL_DIR}")
        ensure_download(NAMES, NAMES_URLS,
                        hint=f"Download coco.names into {MODEL_DIR}")
        self.net = cv2.dnn.readNetFromDarknet(str(CFG), str(WEIGHTS))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.ln = self.net.getUnconnectedOutLayersNames()
        with open(NAMES, encoding="utf-8") as f:
            self.classes = [c.strip() for c in f if c.strip()]

    def detect(self, img, conf=0.4, nms=0.4):
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_out = self.net.forward(self.ln)

        boxes, confs, ids = [], [], []
        for out in layer_out:
            for det in out:
                scores = det[5:]
                cid = int(np.argmax(scores))
                s = float(scores[cid])
                if s < conf:
                    continue
                cx, cy, bw, bh = det[0:4] * np.array([w, h, w, h])
                x, y = int(cx - bw / 2), int(cy - bh / 2)
                boxes.append([x, y, int(bw), int(bh)])
                confs.append(s)
                ids.append(cid)
        idxs = cv2.dnn.NMSBoxes(boxes, confs, conf, nms)
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

    def draw(self, img, dets):
        out = img.copy()
        for d in dets:
            x, y, w, h = d["box"]
            cv2.rectangle(out, (x, y), (x + w, y + h), (80, 220, 255), 2)
            cv2.putText(out, f"{d['label']} {d['conf']:.2f}",
                        (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (80, 220, 255), 1, cv2.LINE_AA)
        return out
