"""
Program 6 — YOLOv3 + Mask R-CNN on traffic video (LIVE UI)
==========================================================

Live UI
-------
  Sliders  yolo_conf  YOLO confidence (0..100, /100)
           mrcnn_conf  Mask R-CNN confidence (0..100, /100)
  Keys     space      pause / resume
           s          save current frame
           r          rewind to start
           q / esc    quit
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from _common import (
    VIDEOS_DIR, OUTPUT_DIR, print_banner, save_image,
    live_show, live_keys, make_slider, read_slider, device, fit_to,
)

from .yolo import YOLOv3
from .maskrcnn import MaskRCNN
from .video_source import open_video, synth_traffic_clip

EXPERIMENT = "experiment_6_yolo_maskrcnn"
WIN = "Experiment 6 — YOLOv3 + Mask R-CNN"


def main() -> None:
    print_banner("Experiment 6 — YOLOv3 + Mask R-CNN")

    video_path = VIDEOS_DIR / "traffic_sample.mp4"
    if not video_path.exists():
        print(f"[setup] {video_path} not found. Generating a synthetic clip...")
        video_path = synth_traffic_clip(video_path, n_frames=60, fps=10)
    print(f"[info] video: {video_path}")

    yolo = YOLOv3()
    rcnn = MaskRCNN(device_name=device())

    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_dir = OUTPUT_DIR / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = out_dir / "out.mp4"
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W * 2, H))

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1600, 480)
    make_slider(WIN, "yolo_conf x100", 0, 100, 40)
    make_slider(WIN, "mrcnn_conf x100", 0, 100, 50)

    total_counts: Counter = Counter()
    per_frame: list[dict] = []
    frame_idx = 0
    paused = False
    saved_at_least_once = False
    # Read first frame immediately so we always have a valid image
    ret, current_frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, current_frame = cap.read()
    frame_idx = 1 if ret else 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # End of video — seek back to start and pause
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if ret:
                    frame_idx = 1
                    current_frame = frame.copy()
                    paused = True
                continue
            frame_idx += 1
            current_frame = frame.copy()

        # live confidence from sliders
        yolo.conf_threshold = read_slider(WIN, "yolo_conf x100", 40) / 100.0
        mrcnn_conf = read_slider(WIN, "mrcnn_conf x100", 50) / 100.0

        dets = yolo.detect(current_frame, conf=yolo.conf_threshold, nms=0.4)
        segs = rcnn.segment(current_frame, conf=mrcnn_conf)

        det_vis = yolo.draw(current_frame, dets)
        seg_vis = rcnn.draw(current_frame, segs)
        combo = np.hstack([det_vis, seg_vis])

        # progress bar
        if total_frames > 0:
            bar_w = combo.shape[1] - 20
            filled = int(bar_w * frame_idx / total_frames)
            cv2.rectangle(combo, (10, combo.shape[0] - 18),
                          (10 + bar_w, combo.shape[0] - 6), (60, 60, 80), 1)
            cv2.rectangle(combo, (10, combo.shape[0] - 18),
                          (10 + filled, combo.shape[0] - 6), (100, 220, 255), -1)

        # save to MP4
        writer.write(combo)

        # live counts
        names = [d["label"] for d in dets] + [s["label"] for s in segs]
        c = Counter(names)
        total_counts.update(names)
        per_frame.append({"frame": frame_idx, **dict(c)})

        info = [
            f"Frame {frame_idx}/{total_frames}   {'PAUSED' if paused else 'playing'}  "
            f"  fps={fps:.1f}",
            f"yolo conf={yolo.conf_threshold:.2f}  mrcnn conf={mrcnn_conf:.2f}  "
            f"dets={len(dets)}  segs={len(segs)}",
            (f"live counts: {dict(c.most_common(3))}" if c else "live counts: —"),
            "space pause/resume   s save frame   r rewind   q quit",
        ]
        live_show(WIN, fit_to(combo, 1600, 480), info, font_scale=0.5)
        k = live_keys(int(1000 / max(1, fps)) if not paused else 80)
        if k.get("q"): break
        elif k.get("space"): paused = not paused
        elif k.get("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, current_frame = cap.read()
            frame_idx = 1 if ret else 0
            paused = True
        elif k.get("s"):
            save_image(f"ui_frame_{frame_idx:05d}.png",
                       fit_to(combo, 1600, 480), EXPERIMENT)
            saved_at_least_once = True

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    with open(out_dir / "total_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["class", "count"])
        for k, v in total_counts.most_common():
            w.writerow([k, v])
    with open(out_dir / "per_frame_counts.csv", "w", newline="", encoding="utf-8") as f:
        cols = sorted({k for row in per_frame for k in row if k != "frame"})
        w = csv.DictWriter(f, fieldnames=["frame"] + cols)
        w.writeheader()
        for row in per_frame:
            w.writerow(row)
    print(f"[done] video: {out_path}  saved_frame={saved_at_least_once}")


if __name__ == "__main__":
    main()
