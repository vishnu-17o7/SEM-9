"""
Program 9 — Face Recognition with VGGFace2 (LIVE UI)
====================================================

Live UI
-------
  Slider  threshold  cosine threshold (0..100, /100)
  Keys    m          toggle mode (detect / identify / verify)
          n / p      next / previous probe
          s          save current frame
          q / esc    quit
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from _common import (
    GALLERY_DIR, OUTPUT_DIR, PROBES_DIR, print_banner, save_image, save_json,
    live_show, live_keys, make_slider, read_slider, device, fit_to,
)

from .detector import FaceDetector
from .embedder import FaceEmbedder
from .gallery import build_gallery, load_probes

EXPERIMENT = "experiment_9_vggface2"
WIN = "Experiment 9 — Face Recognition"


def main(mode: str = "identify", threshold: float = 0.6) -> None:
    print_banner(f"Experiment 9 — Face Recognition ({mode})")

    detector = FaceDetector(device_name=device())
    embedder = FaceEmbedder()

    gallery = build_gallery(GALLERY_DIR, detector, embedder)
    if not gallery["names"]:
        print(f"[warn] gallery is empty. Add folders under {GALLERY_DIR}/<person>/*.jpg")
    print(f"[gallery] {len(gallery['names'])} identities")

    # Pre-compute probe faces (detection is the slow part)
    probes = load_probes()
    probe_faces: list[list[dict]] = []
    for p in probes:
        img = cv2.imread(str(p))
        if img is None:
            probe_faces.append([]); continue
        boxes = detector.detect(img)
        faces = []
        for (x, y, w, h) in boxes:
            crop = img[max(0, y):y + h, max(0, x):x + w]
            if crop.size == 0: continue
            emb = embedder.embed(crop)
            sims = (embedder.cosine(emb, gallery["embeddings"])
                    if gallery["embeddings"].size else None)
            if sims is not None and len(sims) > 0:
                best = int(np.argmax(sims))
                name = gallery["names"][best]
                score = float(sims[best])
            else:
                name, score = "unknown", 0.0
            faces.append({"box": [x, y, w, h], "name": name, "score": score})
        probe_faces.append(faces)

    out_dir = OUTPUT_DIR / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1024, 768)
    make_slider(WIN, "threshold x100", 0, 100, int(threshold * 100))

    modes = ["detect", "identify", "verify"]
    if mode not in modes: mode = "identify"
    mode_idx = modes.index(mode)

    idx = 0
    while probes:
        thresh = read_slider(WIN, "threshold x100", int(threshold * 100)) / 100.0
        path = probes[idx]
        img = cv2.imread(str(path))
        if img is None:
            idx = (idx + 1) % len(probes); continue
        cur_mode = modes[mode_idx]
        vis = img.copy()
        for f in probe_faces[idx]:
            x, y, w, h = f["box"]
            color = (80, 220, 255)
            label = ""
            if cur_mode == "identify":
                label = f"{f['name']} {f['score']:.2f}"
            elif cur_mode == "verify":
                accept = f["score"] >= thresh
                color = (80, 255, 100) if accept else (80, 80, 255)
                label = f"{'ACCEPT' if accept else 'REJECT'} ({f['name']} {f['score']:.2f})"
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis, label, (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        view = fit_to(vis, 1024, 768)
        info = [
            f"Mode: {cur_mode}    Probe: {path.name}  ({idx+1}/{len(probes)})",
            f"threshold: {thresh:.2f}   faces: {len(probe_faces[idx])}",
            "m mode   n/p next/prev probe   s save   q quit",
        ]
        live_show(WIN, view, info, font_scale=0.55)
        k = live_keys(50)
        if k.get("q"): break
        elif k.get("m"): mode_idx = (mode_idx + 1) % len(modes)
        elif k.get("n"): idx = (idx + 1) % len(probes)
        elif k.get("p"): idx = (idx - 1) % len(probes)
        elif k.get("s"):
            save_image(f"ui_capture_{cur_mode}_{path.stem}.png", view, EXPERIMENT)
    cv2.destroyAllWindows()

    # Static outputs
    rows = []
    for path, faces in zip(probes, probe_faces):
        for i, f in enumerate(faces):
            rows.append({"probe": path.name, "face": i, **f})
    save_json(f"{modes[mode_idx]}_results.json", rows, EXPERIMENT)
    print(f"[done] outputs in outputs/{EXPERIMENT}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["detect", "identify", "verify"], default="identify")
    p.add_argument("--threshold", type=float, default=0.6)
    a = p.parse_args()
    main(mode=a.mode, threshold=a.threshold)
