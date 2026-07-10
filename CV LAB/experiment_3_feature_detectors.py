"""
Experiment 3 — SIFT / AKAZE / ORB Landmark Keypoint Detection (LIVE UI)
========================================================================

4-panel display per landmark: original image + keypoint locations for SIFT, AKAZE, and ORB.
No connecting lines — only coloured circles at detected keypoint positions.
Max keypoint accuracy with nfeatures=5000 (SIFT), default (AKAZE), 120000 (ORB).

Landmarks: Statue of Liberty, Eiffel Tower, Colosseum, Taj Mahal

Live UI
-------
  Keys    n / p      next / previous landmark
          s          save current frame
          q / esc    quit

Outputs (under outputs/experiment_3_feature_detectors/)
  - panel_<landmark>.png         4-panel composite per landmark
  - keypoints.json               all counts + timings
"""
from __future__ import annotations

import time
import urllib.request
import json
from pathlib import Path

import cv2
import numpy as np

from _common import (
    LANDMARKS_DIR, OUTPUT_DIR, print_banner, save_image,
    live_show, live_keys, fit_to,
)

EXPERIMENT = "experiment_3_feature_detectors"
WIN = "Experiment 3 — Landmark Keypoint Detectors"

LANDMARKS = [
    ("Statue of Liberty", "https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fedc.h-cdn.co%2Fassets%2F16%2F25%2Famerican-landmarks-statue-of-liberty_1.jpg&f=1&nofb=1&ipt=d5e3f03460c7f2c48cf2fc09dfd33e2cbc1e169c3fd3a8728dec9212e83c43b2"),
    ("Eiffel Tower",       "https://picsum.photos/seed/eiffel/640/480"),
    ("Colosseum",          "https://picsum.photos/seed/colosseum/640/480"),
    ("Taj Mahal",          "https://picsum.photos/seed/tajmahal/640/480"),
]

BG = (18, 18, 28)
TEXT_CLR = (220, 220, 235)
ACCENT = (100, 220, 255)

SIFT_COLOR  = (80, 220, 255)
AKAZE_COLOR = (100, 255, 150)
ORB_COLOR   = (255, 180, 80)

PANEL_W = 600


def fetch_images() -> list[tuple[str, np.ndarray]]:
    out = []
    for name, url in LANDMARKS:
        cache_path = LANDMARKS_DIR / f"{name.lower().replace(' ', '_')}.png"
        if not cache_path.exists():
            print(f"[setup] downloading {name} ...")
            try:
                urllib.request.urlretrieve(url, cache_path)
            except Exception as e:
                print(f"[warn] failed to download {name}: {e}")
        img = cv2.imread(str(cache_path))
        if img is None:
            print(f"[skip] could not read {name}")
            continue
        scale = PANEL_W / img.shape[1]
        img = cv2.resize(img, (PANEL_W, int(img.shape[0] * scale)))
        out.append((name, img))
        print(f"[info] {name}: {img.shape[1]}x{img.shape[0]}")
    return out


def make_sift():
    return cv2.SIFT_create(nfeatures=5000)


def make_akaze():
    return cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB)


def make_orb():
    return cv2.ORB_create(nfeatures=120000)


def detect_keypoints(detector, img: np.ndarray) -> tuple[list[cv2.KeyPoint], float]:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    kp = detector.detect(g, None)
    dt = (time.perf_counter() - t0) * 1000.0
    return kp, dt


def draw_keypoints_panel(img: np.ndarray, kp: list[cv2.KeyPoint], color: tuple,
                         label: str, dt_ms: float) -> np.ndarray:
    out = img.copy()
    for k in kp:
        pt = (int(k.pt[0]), int(k.pt[1]))
        r = max(2, int(k.size * 0.15))
        cv2.circle(out, pt, r, color, -1, cv2.LINE_AA)
    h, w = out.shape[:2]
    bar = np.full((48, w, 3), BG, dtype=np.uint8)
    cv2.putText(bar, label, (8, 18),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, ACCENT, 1, cv2.LINE_AA)
    cv2.putText(bar, f"kp={len(kp)}  {dt_ms:.0f}ms", (8, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_CLR, 1, cv2.LINE_AA)
    return np.vstack([bar, out])


def build_original_panel(img: np.ndarray, name: str) -> np.ndarray:
    h, w = img.shape[:2]
    bar = np.full((48, w, 3), BG, dtype=np.uint8)
    cv2.putText(bar, f"ORIGINAL — {name}", (8, 18),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, ACCENT, 1, cv2.LINE_AA)
    cv2.putText(bar, f"{w}x{h}", (8, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_CLR, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def build_composite(landmark_name: str, img: np.ndarray,
                    detectors: dict, results: dict) -> np.ndarray:
    orig_panel = build_original_panel(img, landmark_name)

    panels = {}
    for name, (detector, color) in detectors.items():
        kp, dt = detect_keypoints(detector, img)
        results[name] = {"keypoints": len(kp), "latency_ms": round(dt, 2)}
        panels[name] = draw_keypoints_panel(img, kp, color, name, dt)
        print(f"  {name:>5}: kp={len(kp):>5}  {dt:.0f} ms")

    min_h = min(p.shape[0] for p in [orig_panel, *panels.values()])
    orig_panel = orig_panel[:min_h]
    for n in panels:
        panels[n] = panels[n][:min_h]

    sep_v = np.full((min_h, 4, 3), BG, dtype=np.uint8)
    top_row = np.hstack([orig_panel, sep_v, panels["SIFT"]])
    bot_row = np.hstack([panels["AKAZE"], sep_v, panels["ORB"]])

    row_w = min(top_row.shape[1], bot_row.shape[1])
    top_row = top_row[:, :row_w]
    bot_row = bot_row[:, :row_w]
    sep_h = np.full((4, row_w, 3), BG, dtype=np.uint8)
    return np.vstack([top_row, sep_h, bot_row])


def main() -> None:
    print_banner("Experiment 3 — Landmark Keypoint Detection")

    images = fetch_images()
    if not images:
        raise FileNotFoundError("No landmark images could be loaded.")

    detectors = {
        "SIFT":  (make_sift(), SIFT_COLOR),
        "AKAZE": (make_akaze(), AKAZE_COLOR),
        "ORB":   (make_orb(), ORB_COLOR),
    }

    all_results: dict[str, dict] = {}
    composites: list[tuple[str, np.ndarray]] = []

    for name, img in images:
        print(f"\n[landmark] {name}")
        results: dict[str, dict] = {}
        composite = build_composite(name, img, detectors, results)
        composites.append((name, composite))
        all_results[name] = results

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1600, 900)
    idx = 0

    while True:
        name, composite = composites[idx]
        r = all_results[name]
        view = fit_to(composite, 1600, 900)
        info = [
            f"{name}  ({idx+1}/{len(composites)})   "
            f"SIFT: kp={r['SIFT']['keypoints']} {r['SIFT']['latency_ms']}ms   "
            f"AKAZE: kp={r['AKAZE']['keypoints']} {r['AKAZE']['latency_ms']}ms   "
            f"ORB: kp={r['ORB']['keypoints']} {r['ORB']['latency_ms']}ms",
            "n next   p prev   s save   q quit",
        ]
        live_show(WIN, view, info, font_scale=0.55)
        k = live_keys(80)
        if k.get("q"):
            break
        elif k.get("n"):
            idx = (idx + 1) % len(composites)
        elif k.get("p"):
            idx = (idx - 1) % len(composites)
        elif k.get("s"):
            save_image(f"panel_{name.lower().replace(' ', '_')}.png", view, EXPERIMENT)
    cv2.destroyAllWindows()

    out_dir = OUTPUT_DIR / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, composite in composites:
        save_image(f"panel_{name.lower().replace(' ', '_')}.png", composite, EXPERIMENT)
    json_path = out_dir / "keypoints.json"
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\n[done] outputs in outputs/{EXPERIMENT}/")


if __name__ == "__main__":
    main()
