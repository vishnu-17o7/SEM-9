"""Face dataset loader.

Prefers `assets/datasets/lfw_subset/<name>/*.jpg`. Falls back to
sklearn's fetch_lfw_people, downsampled to a small subset.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_face_dataset(local_root: Path, min_faces: int = 12) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Returns (X_images, y, classes). X_images are BGR uint8 arrays."""
    if local_root.exists() and any(local_root.iterdir()):
        return _load_local(local_root, min_faces)
    print(f"[setup] {local_root} empty; fetching LFW_people via sklearn")
    return _load_lfw(min_faces)


def _load_local(root: Path, min_faces: int) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    classes, imgs, labels = [], [], []
    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue
        files = sorted(person_dir.glob("*.jpg")) + sorted(person_dir.glob("*.png"))
        if len(files) < min_faces // 2:
            continue
        cls = len(classes)
        for fp in files:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            imgs.append(img)
            labels.append(cls)
        classes.append(person_dir.name)
    return imgs, np.array(labels), classes


def _load_lfw(min_faces: int) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    from sklearn.datasets import fetch_lfw_people
    lfw = fetch_lfw_people(min_faces_per_person=min_faces, resize=0.5, color=True)
    classes = list(lfw.target_names)
    imgs = []
    # lfw.images is (N, H, W, 3) RGB float in [0, 1] — convert to BGR uint8
    for arr in lfw.images:
        bgr = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        imgs.append(bgr)
    return imgs, lfw.target, classes
