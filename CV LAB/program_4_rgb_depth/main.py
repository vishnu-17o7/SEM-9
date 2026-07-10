"""
Program 4 — Multi-Object Detection + Segmentation (RGB + Depth) [LIVE UI]
=========================================================================

Live UI
-------
  Sliders  conf     detection confidence (0..100, /100)
           depth_pct percentile stretch for the depth map (50..100)
  Keys     d        toggle depth colormap (inferno / viridis)
           n        next sample
           s        save current frame
           q / esc  quit
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from _common import (
    SAMPLE_DIR, OUTPUT_DIR, print_banner, require, save_image, save_json,
    bgr_to_rgb, live_show, live_keys, make_slider, read_slider, fit_to,
)

from .depth import MiDaSSmall, colorize_depth
from .detector import YOLOv3Tiny
from .geometry import draw_3d_box

EXPERIMENT = "experiment_4_rgb_depth"
WIN = "Experiment 4 — RGB + Depth"

# 5 diverse scene types for object detection + depth estimation
SCENE_URLS = [
    ("indoor_room",     "https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Sittingroom-edit1.jpg/1280px-Sittingroom-edit1.jpg"),
    ("crowded_street",  "https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg"),
    ("parking_lot",     "https://raw.githubusercontent.com/pjreddie/darknet/master/data/horses.jpg"),
    ("kitchen",         "https://raw.githubusercontent.com/pjreddie/darknet/master/data/eagle.jpg"),
    ("park",            "https://raw.githubusercontent.com/pjreddie/darknet/master/data/person.jpg"),
]



def _synth_scene(bg_color, objects):
    """Create a simple scene with colored rectangles on a gradient background."""
    h, w = 600, 800
    img = np.full((h, w, 3), bg_color, dtype=np.uint8)
    for y in range(h):
        t = y / h
        img[y] = (img[y] * (1 - t * 0.3)).astype(np.uint8)
    for obj in objects:
        if len(obj) == 5:
            cx, cy, ow, oh, color = obj
        else:
            cx, cy, ow, oh = obj
            color = (int(np.random.randint(80, 210)), int(np.random.randint(80, 210)), int(np.random.randint(80, 210)))
        cv2.rectangle(img, (cx - ow // 2, cy - oh // 2),
                      (cx + ow // 2, cy + oh // 2), color, -1)
        cv2.rectangle(img, (cx - ow // 2, cy - oh // 2),
                      (cx + ow // 2, cy + oh // 2), (255, 255, 255), 2)
    return img


def fetch_samples() -> list[Path]:
    """Fetch 5 diverse scene images; fall back to synthetic scenes if downloads fail."""
    downloaded = []
    for name, url in SCENE_URLS:
        p = SAMPLE_DIR / f"{name}.jpg"
        if p.exists():
            downloaded.append(p)
            continue
        try:
            print(f"[setup] downloading {name} ...")
            urllib.request.urlretrieve(url, p)
            if p.exists() and p.stat().st_size > 1000:
                downloaded.append(p)
        except Exception:
            print(f"[warn] could not fetch {name}")
    if len(downloaded) >= 3:
        return downloaded
    # Fallback synthetic scenes with distinct compositions
    print("[warn] < 3 downloads succeeded; generating synthetic scenes")
    synthetic = [
        _synth_scene((100, 100, 160), [(260, 280, 110, 200)]),
        _synth_scene((120, 130, 120), [(200, 250, 110, 60, (180, 80, 80)), (500, 240, 100, 55, (80, 120, 200))]),
        _synth_scene((90, 95, 100),   [(400, 280, 110, 60, (80, 120, 200)), (600, 310, 90, 50, (60, 200, 100))]),
        _synth_scene((180, 200, 210), [(100, 250, 60, 100, (60, 60, 220)), (350, 200, 80, 60, (200, 80, 80))]),
        _synth_scene((80, 130, 80),   [(300, 250, 40, 80, (70, 50, 30)), (500, 230, 50, 100, (60, 40, 30))]),
    ]
    out = []
    for i, (name, _) in enumerate(SCENE_URLS):
        p = SAMPLE_DIR / f"{name}.jpg"
        cv2.imwrite(str(p), synthetic[i])
        out.append(p)
    return out


def main() -> None:
    print_banner("Experiment 4 — RGB + Depth Detection")

    samples = fetch_samples()
    print(f"[info] {len(samples)} scene(s): {[s.stem for s in samples]}")

    detector = YOLOv3Tiny(conf_threshold=0.25)
    try:
        midas = MiDaSSmall()
    except FileNotFoundError as e:
        print(f"[warn] {e}")
        midas = None

    # Pre-compute depth for each sample (MiDaS is slow per call)
    depth_maps: list[np.ndarray | None] = []
    for s in samples:
        img = cv2.imread(str(s))
        if img is None:
            depth_maps.append(None); continue
        if midas is not None:
            try:
                depth_maps.append(midas.infer(img))
            except Exception as e:
                print(f"[warn] MiDaS failed for {s.name}: {e}")
                depth_maps.append(None)
        else:
            depth_maps.append(None)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1600, 540)
    make_slider(WIN, "conf x100", 0, 100, 25)
    make_slider(WIN, "depth_pct", 50, 100, 95)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)

    sample_idx = 0
    colormap_idx = 0
    colormaps = [
        ("inferno",  cv2.COLORMAP_INFERNO),
        ("viridis",  cv2.COLORMAP_VIRIDIS),
        ("jet",      cv2.COLORMAP_JET),
    ]

    while True:
        path = samples[sample_idx]
        img = cv2.imread(str(path))
        if img is None:
            sample_idx = (sample_idx + 1) % len(samples)
            continue
        require(path)

        conf = read_slider(WIN, "conf x100", 25) / 100.0
        depth_pct = read_slider(WIN, "depth_pct", 95) / 100.0
        cmap_name, cmap = colormaps[colormap_idx]

        # Re-detect with the slider's confidence
        detector.conf_threshold = conf
        detections = detector.detect(img)

        # 2D boxes
        det_vis = img.copy()
        for d in detections:
            x, y, w, h = d["box"]
            cv2.rectangle(det_vis, (x, y), (x + w, y + h), (80, 220, 255), 2)
            cv2.putText(det_vis, f"{d['label']} {d['conf']:.2f}",
                        (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (80, 220, 255), 1, cv2.LINE_AA)

        # Depth visualisation
        depth = depth_maps[sample_idx]
        if depth is not None:
            d_clip = np.clip(depth, np.percentile(depth, 100 - depth_pct * 100),
                             np.percentile(depth, depth_pct * 100))
            d_norm = (d_clip - d_clip.min()) / max(1e-6, d_clip.max() - d_clip.min())
            depth_vis = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cmap)
        else:
            h, w = img.shape[:2]
            depth_vis = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
            depth_vis = cv2.applyColorMap(depth_vis, cmap)

        # 3D boxes
        boxes_vis = img.copy()
        for d in detections:
            x, y, w, h = d["box"]
            cx, cy = x + w / 2, y + h
            if depth is not None:
                z = float(np.clip(depth[int(cy) % depth.shape[0],
                                        int(cx) % depth.shape[1]], 0.1, 5.0))
            else:
                z = 1.0 + 0.5 * (cx / img.shape[1])
            draw_3d_box(boxes_vis, (cx, cy, z), (w, h, 0.5 + z * 0.3))

        sep = np.full((det_vis.shape[0], 8, 3), 18, dtype=np.uint8)
        combo = np.hstack([det_vis, sep, depth_vis, sep, boxes_vis])

        info = [
            f"Experiment 4 — RGB + Depth  ({path.name})",
            f"conf={conf:.2f}   depth_pct={depth_pct:.2f}   colormap={cmap_name}",
            f"sample {sample_idx+1}/{len(samples)}  detections: {len(detections)}",
            "n next sample   d colormap   s save   q quit",
        ]
        live_show(WIN, fit_to(combo, 1600, 540), info, font_scale=0.55)

        k = live_keys(40)
        if k.get("q"): break
        elif k.get("n"): sample_idx = (sample_idx + 1) % len(samples)
        elif k.get("d"): colormap_idx = (colormap_idx + 1) % len(colormaps)
        elif k.get("s"):
            save_image(f"ui_capture_{path.stem}_{sample_idx}.png",
                       fit_to(combo, 1600, 540), EXPERIMENT)

    cv2.destroyAllWindows()

    # Save final outputs
    img = cv2.imread(str(samples[0]))
    detections = detector.detect(img)
    vis = img.copy()
    for d in detections:
        x, y, w, h = d["box"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (80, 220, 255), 2)
        cv2.putText(vis, f"{d['label']} {d['conf']:.2f}", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 255), 1, cv2.LINE_AA)
    save_image("01_detections.png", vis, EXPERIMENT)
    if depth_maps[0] is not None:
        save_image("02_depth.png", colorize_depth(depth_maps[0]), EXPERIMENT)
    save_image("03_3d_boxes.png", boxes_vis, EXPERIMENT)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, im, t in zip(axes, [vis, depth_vis, boxes_vis],
                         ["RGB detections", "Depth (MiDaS)", "3D boxes"]):
        ax.imshow(bgr_to_rgb(im)); ax.set_title(t); ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / EXPERIMENT / "00_overview.png", dpi=140)
    plt.close(fig)

    save_json("detections.json", detections, EXPERIMENT)
    print(f"[done] outputs in outputs/{EXPERIMENT}/")


if __name__ == "__main__":
    main()
