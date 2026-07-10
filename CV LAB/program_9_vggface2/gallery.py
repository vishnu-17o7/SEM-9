"""Gallery and probe loaders with auto-seed from LFW.

On first run with an empty gallery, downloads a small LFW_people subset.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from _common import (
    GALLERY_DIR, PROBES_DIR, ensure_download, print_banner,
)


def _ensure_lfw_seed() -> None:
    """Populate gallery/ and probes/ with a few LFW identities if empty."""
    has_gallery = any(GALLERY_DIR.iterdir()) if GALLERY_DIR.exists() else False
    has_probes = any(PROBES_DIR.iterdir()) if PROBES_DIR.exists() else False
    if has_gallery and has_probes:
        return

    print("[setup] Seeding face gallery/probes from LFW dataset...")
    # LFW people — pick a few well-known identities
    # Use fetch_lfw_people which downloads to sklearn cache (~200 MB)
    try:
        from sklearn.datasets import fetch_lfw_people
        lfw = fetch_lfw_people(min_faces_per_person=40, resize=0.5, color=True)
    except Exception as e:
        print(f"[warn] Could not fetch LFW: {e}")
        return

    # Pick 4 identities for gallery, 2 more as probes
    target_names = list(lfw.target_names)
    rng = np.random.default_rng(42)

    # Find indices with enough samples
    gallery_names = []
    probe_names = []
    used = set()
    for name in target_names:
        ids = [i for i, n in enumerate(lfw.target) if lfw.target_names[n] == name]
        if len(ids) >= 6 and name not in used:
            if len(gallery_names) < 4:
                gallery_names.append((name, ids))
            elif len(probe_names) < 2:
                probe_names.append((name, ids))
            used.add(name)

    if not gallery_names:
        print("[warn] Not enough LFW identities found.")
        return

    for name, ids in gallery_names:
        person_dir = GALLERY_DIR / name.replace(" ", "_")
        person_dir.mkdir(parents=True, exist_ok=True)
        chosen = rng.choice(ids, size=min(3, len(ids)), replace=False)
        for j, idx in enumerate(chosen):
            # lfw.images is (N, H, W, 3) RGB float [0, 1]
            img_rgb = (lfw.images[idx] * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            path = person_dir / f"{j:02d}.jpg"
            cv2.imwrite(str(path), img_bgr)
        print(f"  gallery: {person_dir.name} ({len(chosen)} images)")

    for name, ids in probe_names:
        chosen = rng.choice(ids, size=min(2, len(ids)), replace=False)
        for j, idx in enumerate(chosen):
            img_rgb = (lfw.images[idx] * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            stem = name.replace(" ", "_")
            path = PROBES_DIR / f"{stem}_{j:02d}.jpg"
            cv2.imwrite(str(path), img_bgr)
        print(f"  probes:  {name} ({len(chosen)} images)")


def build_gallery(gallery_dir: Path, detector, embedder) -> dict:
    """Read assets/gallery/<person>/*.jpg and produce averaged embeddings."""
    _ensure_lfw_seed()

    names: list[str] = []
    embeddings: list[np.ndarray] = []

    for person_dir in sorted(gallery_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        imgs = sorted(person_dir.glob("*.jpg")) + sorted(person_dir.glob("*.png"))
        if not imgs:
            continue
        per_person = []
        for img_path in imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            boxes = detector.detect(img)
            for (x, y, w, h) in boxes:
                crop = img[max(0, y):y + h, max(0, x):x + w]
                if crop.size == 0:
                    continue
                per_person.append(embedder.embed(crop))
        if per_person:
            avg = np.mean(per_person, axis=0)
            avg = avg / (np.linalg.norm(avg) + 1e-10)
            embeddings.append(avg)
            names.append(person_dir.name)
    emb_arr = np.stack(embeddings, axis=0) if embeddings else np.zeros((0, embedder.dim))
    return {"names": names, "embeddings": emb_arr}


def load_probes() -> list[Path]:
    _ensure_lfw_seed()
    return sorted(PROBES_DIR.glob("*.jpg")) + sorted(PROBES_DIR.glob("*.png"))
