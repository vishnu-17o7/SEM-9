"""Video I/O + a small synthetic traffic-clip generator."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    return cap


def synth_traffic_clip(out_path: Path, n_frames: int = 80, fps: int = 10) -> Path:
    """Generate a richer 'traffic' clip with gradient sky, buildings, trees, and detailed vehicles."""
    W, H = 640, 360
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    rng = np.random.default_rng(0)

    # Building silhouettes (static)
    buildings = []
    for x in range(0, W, rng.integers(40, 80)):
        bw = rng.integers(30, 70)
        bh = rng.integers(40, 120)
        buildings.append((x, 180 - bh, bw, bh))

    # Trees along the road
    trees = [(x, 175) for x in range(10, W, rng.integers(50, 100))]

    # Vehicles with more detail
    vehicles = [
        {"x": 50,  "y": 200, "w": 110, "h": 55, "speed": 4,  "color": (40, 120, 220)},
        {"x": 300, "y": 280, "w": 90,  "h": 48, "speed": -3, "color": (60, 200, 80)},
        {"x": 500, "y": 130, "w": 80,  "h": 42, "speed": 5,  "color": (200, 80, 80)},
    ]

    for t in range(n_frames):
        # ── Sky with gradient ──
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        for y in range(180):
            t_sky = y / 180.0
            b = int(60 + 60 * (1 - t_sky))
            g = int(100 + 80 * (1 - t_sky))
            r = int(140 + 60 * (1 - t_sky))
            frame[y, :] = (b, g, r)

        # ── Clouds (moving) ──
        for cx in range(0, W, 120):
            cx = (cx + t * 3) % W
            cv2.ellipse(frame, (cx, 40), (60, 18), 0, 0, 360, (210, 220, 230), -1)
            cv2.ellipse(frame, (cx + 30, 30), (40, 14), 0, 0, 360, (220, 230, 240), -1)

        # ── Building silhouettes ──
        for bx, by, bw, bh in buildings:
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 45, 55), -1)
            # Windows
            for wy in range(by + 6, by + bh - 4, 12):
                for wx in range(bx + 4, bx + bw - 4, 14):
                    win_bright = 180 + int(30 * np.sin(wx * 0.5 + t * 0.3))
                    cv2.rectangle(frame, (wx, wy), (wx + 6, wy + 6),
                                  (win_bright, win_bright, win_bright - 20), -1)

        # ── Ground / grass strip ──
        frame[175:185, :] = (40, 100, 50)

        # ── Road surface ──
        cv2.rectangle(frame, (0, 185), (W, 320), (60, 65, 70), -1)
        # Road edge lines
        cv2.line(frame, (0, 185), (W, 185), (200, 200, 190), 2)
        cv2.line(frame, (0, 320), (W, 320), (200, 200, 190), 2)
        # Dashed lane markings (scrolling)
        for x in range(0, W, 35):
            cx = (x + t * 5) % W
            cv2.rectangle(frame, (cx, 248), (cx + 16, 258), (220, 220, 210), -1)

        # ── Trees (trunk + crown) ──
        for tx, ty in trees:
            cv2.rectangle(frame, (tx - 2, ty), (tx + 2, ty + 12), (50, 40, 30), -1)
            cv2.circle(frame, (tx, ty - 6), 14, (30, 110, 40), -1)
            cv2.circle(frame, (tx + 6, ty - 10), 10, (40, 130, 50), -1)

        # ── Vehicles ──
        for v in vehicles:
            v["x"] += v["speed"]
            if v["x"] < -v["w"]:
                v["x"] = W
            if v["x"] > W:
                v["x"] = -v["w"]
            vx, vy = int(v["x"]), int(v["y"])
            # Body
            cv2.rectangle(frame, (vx, vy), (vx + v["w"], vy + v["h"]), v["color"], -1)
            # Roof (slightly lighter)
            roof_color = tuple(min(255, c + 30) for c in v["color"])
            cv2.rectangle(frame, (vx + 10, vy - 8), (vx + v["w"] - 10, vy), roof_color, -1)
            # Windows
            for wx in (vx + 15, vx + v["w"] - 30):
                cv2.rectangle(frame, (wx, vy + 4), (wx + 14, vy + 20), (180, 200, 220), -1)
            # Wheels
            for wx in (vx + 12, vx + v["w"] - 20):
                cv2.circle(frame, (wx, vy + v["h"]), 6, (30, 30, 30), -1)
            # Outline
            cv2.rectangle(frame, (vx, vy), (vx + v["w"], vy + v["h"]), (255, 255, 255), 1)

        cv2.putText(frame, f"frame {t}/{n_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    return out_path
