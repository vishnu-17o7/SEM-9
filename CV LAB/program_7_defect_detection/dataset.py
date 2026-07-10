"""Synthetic defect dataset + tiny MVTec-style loader.

If `assets/datasets/mvtec_subset/<class>/<split>/*.png` exists, we use
that. Otherwise we generate a synthetic dataset of clean metal
backgrounds with painted defects (scratches, dents, holes).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


CLASSES = ["scratch", "dent", "hole"]


class DefectDataset:
    def __init__(self, samples: list[dict], classes: list[str]):
        self.samples = samples
        self.classes = classes
        self.cls_to_idx = {c: i + 1 for i, c in enumerate(classes)}  # 0 = background

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = cv2.imread(str(s["image"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, labels = [], []
        for b, c in zip(s["boxes"], s["labels"]):
            x, y, w, h = b
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cls_to_idx[c])
        return img, {"boxes": boxes, "labels": labels}


def _metal_bg(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """Plausible-looking brushed-metal background."""
    base = rng.integers(140, 200, size=(size, size), dtype=np.uint8)
    # horizontal brush strokes
    for _ in range(60):
        y = rng.integers(0, size)
        x0 = rng.integers(0, size // 2)
        x1 = rng.integers(size // 2, size)
        delta = rng.integers(-25, 25)
        base[y, x0:x1] = np.clip(base[y, x0:x1].astype(int) + delta, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    return img


def _paint_defect(img: np.ndarray, kind: str, rng: np.random.Generator) -> tuple[list, str]:
    h, w = img.shape[:2]
    if kind == "scratch":
        x0, y0 = rng.integers(20, w - 60), rng.integers(20, h - 20)
        length = rng.integers(40, 100)
        angle = rng.integers(0, 180)
        x1 = int(x0 + length * np.cos(np.radians(angle)))
        y1 = int(y0 + length * np.sin(np.radians(angle)))
        cv2.line(img, (x0, y0), (x1, y1), (40, 40, 40), 2, cv2.LINE_AA)
        x, y = min(x0, x1), min(y0, y1)
        return [[x, y, abs(x1 - x0) + 4, abs(y1 - y0) + 4]], "scratch"
    if kind == "dent":
        cx, cy = rng.integers(40, w - 40), rng.integers(40, h - 40)
        r = rng.integers(10, 25)
        cv2.circle(img, (cx, cy), r, (90, 90, 100), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r, (60, 60, 70), 2, cv2.LINE_AA)
        return [[cx - r - 2, cy - r - 2, 2 * r + 4, 2 * r + 4]], "dent"
    if kind == "hole":
        cx, cy = rng.integers(40, w - 40), rng.integers(40, h - 40)
        r = rng.integers(6, 14)
        cv2.circle(img, (cx, cy), r, (0, 0, 0), -1, cv2.LINE_AA)
        return [[cx - r - 2, cy - r - 2, 2 * r + 4, 2 * r + 4]], "hole"
    return [], "scratch"


def _generate_split(root: Path, split: str, n: int) -> list[dict]:
    (root / split).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng({"train": 0, "val": 1}[split])
    samples = []
    for i in range(n):
        bg = _metal_bg(rng)
        n_defects = rng.integers(1, 3)
        boxes, labels = [], []
        for _ in range(n_defects):
            cls = CLASSES[rng.integers(0, len(CLASSES))]
            b, l = _paint_defect(bg, cls, rng)
            boxes.extend(b); labels.extend([l])
        if not boxes:
            cls = CLASSES[rng.integers(0, len(CLASSES))]
            boxes, labels = _paint_defect(bg, cls, rng)
        path = root / split / f"{i:04d}.jpg"
        cv2.imwrite(str(path), bg)
        samples.append({"image": path, "boxes": boxes, "labels": labels})
    return samples


def build_or_load_dataset(root: Path, n_synth: int = 120) -> tuple:
    """Returns (train_ds, val_ds, classes). Uses MVTec-style folder if present."""
    # MVTec-style: root/<class>/<split>/*.png  — single-class only
    mvtec = root.parent / "mvtec_subset"
    if mvtec.exists() and any(mvtec.iterdir()):
        classes = sorted([p.name for p in mvtec.iterdir() if p.is_dir()])
        train_samples, val_samples = [], []
        for cls in classes:
            for split, out in (("train", train_samples), ("val", val_samples)):
                d = mvtec / cls / split
                if not d.exists():
                    continue
                for p in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
                    out.append({"image": p, "boxes": [[20, 20, 60, 60]],
                                "labels": [cls]})
        return DefectDataset(train_samples, classes), DefectDataset(val_samples, classes), classes

    # synthetic fallback
    train_samples = _generate_split(root, "train", n_synth)
    val_samples = _generate_split(root, "val", max(20, n_synth // 5))
    return DefectDataset(train_samples, CLASSES), DefectDataset(val_samples, CLASSES), CLASSES
