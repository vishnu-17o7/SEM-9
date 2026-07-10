"""Mask R-CNN — uses torchvision (ResNet-50 FPN, COCO weights).

Avoids the ~170 MB ONNX download by relying on the torchvision model
zoo, which fetches weights on first use to the standard torch hub cache.
If torchvision is not installed the class prints a clear error.
"""
from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image

COCO_NAMES = [
    "__bg__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "N/A", "N/A",
    "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


class MaskRCNN:
    def __init__(self, device_name: str = "cpu"):
        try:
            import torch
            import torchvision
        except Exception as e:
            raise RuntimeError(
                f"Mask R-CNN needs torch + torchvision. Install with:\n"
                f"  pip install torch torchvision\n(underlay error: {e})"
            ) from e
        self.torch = torch
        self.device = device_name
        weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
        self.model.eval().to(self.device)
        self.transform = weights.transforms()

    def segment(self, img, conf=0.5):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = self.transform(Image.fromarray(rgb)).to(self.device)
        with self.torch.no_grad():
            out = self.model([t])[0]
        results = []
        for i, score in enumerate(out["scores"].tolist()):
            if score < conf:
                continue
            box = out["boxes"][i].cpu().numpy().astype(int)
            mask = out["masks"][i, 0].cpu().numpy()
            label_id = int(out["labels"][i].item())
            results.append({
                "label": COCO_NAMES[label_id] if label_id < len(COCO_NAMES) else str(label_id),
                "conf": round(score, 3),
                "box": [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])],
                "mask": (mask > 0.5).astype(np.uint8),
            })
        return results

    def draw(self, img, segs):
        out = img.copy()
        rng = random.Random(0)
        for s in segs:
            color = tuple(int(c) for c in rng.choice([
                (255, 80, 80), (80, 255, 80), (80, 80, 255),
                (255, 255, 80), (255, 80, 255), (80, 255, 255),
            ]))
            m = s["mask"].astype(bool)
            overlay = out.copy()
            overlay[m] = color
            out = cv2.addWeighted(out, 0.7, overlay, 0.3, 0)
            x, y, w, h = s["box"]
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cv2.putText(out, f"{s['label']} {s['conf']:.2f}",
                        (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
        return out
