"""
Experiment 8 — Feature Matching + Image Registration → Flipbook (LIVE UI)
===========================================================================

Live UI
-------
  Sliders  none
  Keys     space     pause / resume playback
           ← / →     step back / forward one frame
           + / -     increase / decrease playback FPS
           r         reset to first frame
           f         toggle flip horizontally
           s         save current frame
           q / esc   quit
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from _common import (
    OUTPUT_DIR, SEQUENCE_DIR, print_banner, require, save_image, save_json,
    live_show, live_keys, fit_to,
)

EXPERIMENT = "experiment_8_flipbook_registration"
WIN = "Experiment 8 — Flipbook Player"


def synth_sequence(out_dir: Path, n: int = 12, size: int = 480) -> list[Path]:
    """Generate a flipbook sequence: landscape with a moving car and animated sun."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        frame = np.full((size, size, 3), (220, 220, 235), dtype=np.uint8)

        # Sky gradient
        for y in range(size // 2):
            t = y / (size // 2)
            b = int(180 + 70 * (1 - t))
            g = int(200 + 50 * (1 - t))
            r = int(220 + 30 * (1 - t))
            frame[y, :] = (min(b, 255), min(g, 255), min(r, 255))

        # Ground
        ground_y = size // 2
        frame[ground_y:, :] = (60, 130, 50)
        for y in range(ground_y, size):
            t = (y - ground_y) / (size - ground_y)
            frame[y, :] = (60, int(130 - t * 30), int(50 - t * 20))

        # Road
        road_y = ground_y + size // 6
        road_h = size // 6
        frame[road_y:road_y + road_h, :] = (55, 60, 65)
        # Lane markings (scrolling)
        dash_offset = (i * 25) % 60
        for dx in range(0, size, 60):
            frame[road_y + road_h // 2 - 2:road_y + road_h // 2 + 2,
                  max(0, dx + dash_offset):min(size, dx + dash_offset + 30)] = (200, 200, 200)

        # Sun (moves across the sky)
        sun_angle = i * (360 / n)
        sun_x = int(size * 0.85 * (1 + np.cos(np.radians(sun_angle - 180))) / 2)
        sun_y = int(size * 0.35 * (1 + np.sin(np.radians(sun_angle - 180))) / 2)
        sun_y = min(sun_y, size // 2 - 30)
        cv2.circle(frame, (sun_x, max(10, sun_y)), 28, (200, 220, 240), -1)
        cv2.circle(frame, (sun_x, max(10, sun_y)), 22, (240, 240, 200), -1)

        # House
        hx, hy = size // 5, road_y - 70
        cv2.rectangle(frame, (hx, hy), (hx + 50, hy + 50), (120, 90, 70), -1)
        cv2.rectangle(frame, (hx + 18, hy + 25), (hx + 32, hy + 50), (40, 50, 60), -1)
        cv2.rectangle(frame, (hx + 6, hy + 8), (hx + 18, hy + 20), (180, 200, 220), -1)
        # Roof triangle
        roof_pts = np.array([[hx - 6, hy], [hx + 25, hy - 25], [hx + 56, hy]], dtype=np.int32)
        cv2.fillPoly(frame, [roof_pts], (100, 60, 50))
        # Tree
        tx, ty = size // 3 + 20, road_y - 50
        cv2.rectangle(frame, (tx - 3, ty), (tx + 3, ty + 30), (50, 40, 30), -1)
        cv2.circle(frame, (tx, ty - 15), 22, (30, 100, 40), -1)
        cv2.circle(frame, (tx + 10, ty - 8), 16, (40, 130, 50), -1)
        cv2.circle(frame, (tx - 8, ty - 6), 14, (50, 120, 45), -1)

        # Car (moving along the road)
        car_x = int((i / n) * (size + 100)) - 50
        car_y = road_y + road_h // 2 - 10
        car_w, car_h = 60, 20
        cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), (60, 120, 220), -1)
        cv2.rectangle(frame, (car_x + 6, car_y - 5), (car_x + car_w - 6, car_y), (100, 160, 240), -1)
        for wx in (car_x + 8, car_x + car_w - 22):
            cv2.rectangle(frame, (wx, car_y + 3), (wx + 12, car_y + car_h - 2), (180, 200, 220), -1)
        for cx in (car_x + 8, car_x + car_w - 12):
            cv2.circle(frame, (min(cx, size - 1), car_y + car_h), 5, (30, 30, 30), -1)
        cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), (255, 255, 255), 1)

        path = out_dir / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(path), frame)
        paths.append(path)
    return paths


def load_sequence() -> list[Path]:
    user_frames = sorted(SEQUENCE_DIR.glob("*.jpg")) + sorted(SEQUENCE_DIR.glob("*.png"))
    user_frames = [p for p in user_frames if p.is_file()]
    if len(user_frames) >= 3:
        return user_frames
    print(f"[setup] {SEQUENCE_DIR} has <3 frames. Generating synthetic sequence...")
    return synth_sequence(SEQUENCE_DIR, n=12)


def align_pair(img_a, img_b, detector):
    ga = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    kp_a, des_a = detector.detectAndCompute(ga, None)
    kp_b, des_b = detector.detectAndCompute(gb, None)
    if des_a is None or des_b is None or len(des_a) < 4 or len(des_b) < 4:
        return img_b, 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des_a, des_b)
    if len(matches) < 4:
        return img_b, 0
    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 3.0)
    inliers = int(mask.ravel().sum()) if mask is not None else 0
    if H is None:
        return img_b, 0
    h, w = img_a.shape[:2]
    return cv2.warpPerspective(img_b, H, (w, h)), inliers


def main() -> None:
    print_banner("Experiment 8 — Flipbook Image Registration")

    frames = load_sequence()
    print(f"[info] sequence length: {len(frames)}")

    detector = cv2.ORB_create(nfeatures=1500)
    aligned: list[np.ndarray] = []
    inlier_counts: list[int] = []
    prev = cv2.imread(str(frames[0]))
    aligned.append(prev)
    inlier_counts.append(0)

    for i in range(1, len(frames)):
        cur = cv2.imread(str(frames[i]))
        if cur is None:
            aligned.append(prev); inlier_counts.append(0); continue
        if cur.shape != prev.shape:
            cur = cv2.resize(cur, (prev.shape[1], prev.shape[0]))
        warped, inliers = align_pair(prev, cur, detector)
        aligned.append(warped)
        inlier_counts.append(inliers)
        prev = warped
        print(f"  [align] {frames[i].name}  inliers={inliers}")

    out_dir = OUTPUT_DIR / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)

    # Contact sheet (for static output)
    rows, cols = 3, 4
    h, w = aligned[0].shape[:2]
    sheet = np.full((h * rows, w * cols, 3), 18, dtype=np.uint8)
    for i, im in enumerate(aligned):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    save_image("contact_sheet.png", sheet, EXPERIMENT)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1024, 768)

    # Live playback
    fps = 8.0
    paused = False
    flip = False
    i = 0
    last = 0.0
    import time
    while True:
        now = time.perf_counter()
        if not paused and (now - last) >= 1.0 / fps:
            i = (i + 1) % len(aligned)
            last = now

        frame = aligned[i]
        if flip:
            frame = cv2.flip(frame, 1)
        view = fit_to(frame, 1024, 768)

        info = [
            f"Frame {i+1}/{len(aligned)}  ({frames[i].name})",
            f"inliers to prev: {inlier_counts[i]}    fps: {fps:.1f}    "
            f"{'PAUSED' if paused else 'playing'}    flip: {flip}",
            "space pause   ←/→ step   +/- speed   f flip   r reset   s save   q quit",
        ]
        live_show(WIN, view, info, font_scale=0.55)

        k = live_keys(20)
        if k.get("q"): break
        elif k.get("space"): paused = not paused
        elif k.get("right"): i = (i + 1) % len(aligned); paused = True
        elif k.get("left"):  i = (i - 1) % len(aligned); paused = True
        elif k.get("plus"):  fps = min(30.0, fps + 1)
        elif k.get("minus"): fps = max(1.0, fps - 1)
        elif k.get("f"):     flip = not flip
        elif k.get("r"):     i = 0; paused = True
        elif k.get("s"):
            save_image(f"ui_capture_{i:03d}.png", view, EXPERIMENT)
    cv2.destroyAllWindows()

    # Export static GIF / MP4
    try:
        import imageio.v2 as imageio
        rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in aligned]
        imageio.mimsave(str(out_dir / "flipbook.gif"), rgb, duration=1.0 / fps)
        print(f"[done] {out_dir / 'flipbook.gif'}")
    except Exception as e:
        print(f"[warn] could not write GIF: {e}")
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(out_dir / "flipbook.mp4"), fourcc, fps,
                              (aligned[0].shape[1], aligned[0].shape[0]))
        for im in aligned:
            vw.write(im)
        vw.release()
        print(f"[done] {out_dir / 'flipbook.mp4'}")
    except Exception as e:
        print(f"[warn] could not write MP4: {e}")

    save_json("summary.json", {
        "n_frames": len(aligned),
        "inlier_counts": inlier_counts,
        "outputs": ["contact_sheet.png", "flipbook.gif", "flipbook.mp4"],
    }, EXPERIMENT)
    print(f"[done] outputs in outputs/{EXPERIMENT}/")


if __name__ == "__main__":
    main()
