"""
Experiment 2 — Image Restoration Pipeline (LIVE UI)
==================================================

Three-stage restoration:
  (A) Image blending   — linear (addWeighted) and Poisson (seamlessClone)
  (B) Morphing         — cross-dissolve between damaged and final
  (C) Edge enhancement — unsharp mask + Sobel/Scharr

Live UI
-------
  Slider  alpha      blend weight (0..100)  — only used for linear blend
  Keys    q/esc      quit
          n          cycle views (damaged / linear / poisson / morph sheet / final)
          s          save current view as PNG
          +  /  -    bump slider step

Inputs / Outputs
----------------
  assets/cache/{reference,damaged}.jpg    (auto-fetched or synthesised)
  outputs/experiment_2_image_restoration/  (6 PNGs + summary.json)
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

from _common import (
    CACHE_DIR, print_banner, require, save_image, save_json, save_figure, bgr_to_rgb,
    live_show, live_keys, make_slider, read_slider, draw_panel, fit_to,
)

EXPERIMENT = "experiment_2_image_restoration"

REFERENCE_URLS = [
    "https://pixnio.com/free-images/2017/10/27/2017-10-27-06-30-07.jpg",
]
WIN = "Experiment 2 — Image Restoration"


def fetch_reference() -> Path:
    ref = CACHE_DIR / "reference.jpg"
    if ref.exists():
        return ref
    for url in REFERENCE_URLS:
        try:
            print(f"[setup] downloading reference to {ref}  ({url})")
            req = urllib.request.Request(url, headers={"User-Agent": "SEM9-LabHub/1.0"})
            with urllib.request.urlopen(req) as resp, open(ref, "wb") as f:
                f.write(resp.read())
            if ref.exists() and ref.stat().st_size > 1000:
                return ref
        except Exception:
            continue
    # Fallback: generate an interesting synthetic urban scene
    h, w = 600, 800
    sky = np.full((h, w, 3), (180, 140, 100), dtype=np.uint8)
    ground = np.full((h, w, 3), (60, 80, 70), dtype=np.uint8)
    img = np.vstack([sky[:h//2], ground[h//2:]])
    for x in range(100, w, 140):
        rect_w = 60
        rect_h = np.random.randint(100, 200)
        y0 = h // 2 - rect_h
        cv2.rectangle(img, (x, y0), (x + rect_w, h // 2),
                      (np.random.randint(60, 140),) * 3, -1)
        # windows
        for wy in range(y0 + 10, h // 2 - 10, 25):
            for wx in range(x + 6, x + rect_w - 6, 20):
                cv2.rectangle(img, (wx, wy), (wx + 10, wy + 12),
                              (220, 230, 240), -1)
    cv2.line(img, (0, h // 2), (w, h // 2), (50, 70, 60), 3)
    cv2.imwrite(str(ref), img)
    return ref


def synth_damage(reference: Path) -> Path:
    """Apply vintage-style damage: yellowing, scratches, dust, and fading."""
    img = cv2.imread(str(reference))
    h, w = img.shape[:2]
    rng = np.random.default_rng(42)
    out = img.copy().astype(np.float32)

    # Yellowing / sepia tone in vignette pattern
    xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    vignette = 1 - 0.4 * np.sqrt(xx**2 + yy**2)
    vignette = np.clip(vignette, 0.3, 1.0)[:, :, None]
    yellow = np.array([0.6, 1.0, 1.3], dtype=np.float32)  # BGR: boost yellow/red
    out = out * vignette * yellow

    # Fading / washed out regions (random patches)
    n_patches = rng.integers(3, 8)
    for _ in range(n_patches):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        rx, ry = rng.integers(50, 180), rng.integers(50, 150)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
        blur = cv2.GaussianBlur(mask, (51, 51), 25)
        out += blur[:, :, None] * rng.uniform(30, 80) * 0.5

    # Scratches (thin lines)
    n_scratches = rng.integers(8, 20)
    for _ in range(n_scratches):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        length = rng.integers(30, 200)
        angle = rng.uniform(0, np.pi)
        x1 = int(x0 + length * np.cos(angle))
        y1 = int(y0 + length * np.sin(angle))
        color = (rng.uniform(0, 30), rng.uniform(0, 30), rng.uniform(0, 30))
        cv2.line(out, (x0, y0), (min(x1, w-1), min(y1, h-1)),
                 color, rng.integers(1, 2), cv2.LINE_AA)

    # Dust specks
    n_dust = rng.integers(50, 150)
    for _ in range(n_dust):
        px, py = rng.integers(0, w), rng.integers(0, h)
        radius = rng.integers(1, 3)
        cv2.circle(out, (px, py), radius, (200, 200, 200), -1, cv2.LINE_AA)

    out = np.clip(out, 0, 255).astype(np.uint8)
    out_path = CACHE_DIR / "damaged.jpg"
    cv2.imwrite(str(out_path), out)
    return out_path


# ─────────────────────────── A. Blending ───────────────────────────
def linear_blend(damaged: np.ndarray, reference: np.ndarray, alpha: float) -> np.ndarray:
    if damaged.shape != reference.shape:
        reference = cv2.resize(reference, (damaged.shape[1], damaged.shape[0]))
    return cv2.addWeighted(damaged, 1 - alpha, reference, alpha, 0)


def poisson_blend(damaged: np.ndarray, reference: np.ndarray) -> np.ndarray:
    h, w = damaged.shape[:2]
    if reference.shape[:2] != (h, w):
        reference = cv2.resize(reference, (w, h))
    cx, cy = w // 2, h // 2
    half_w, half_h = int(w * 0.30), int(h * 0.30)
    src = reference[cy - half_h:cy + half_h, cx - half_w:cx + half_w].copy()
    mask = 255 * np.ones(src.shape[:2], dtype=np.uint8)
    try:
        return cv2.seamlessClone(src, damaged, mask, (cx, cy), cv2.NORMAL_CLONE)
    except cv2.error:
        return linear_blend(damaged, reference, 0.6)


# ─────────────────────────── B. Morphing ───────────────────────────
def cross_dissolve(a: np.ndarray, b: np.ndarray, steps: int = 8) -> list[np.ndarray]:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return [cv2.addWeighted(a, 1 - i / steps, b, i / steps, 0)
            for i in range(steps + 1)]


def morph_contact_sheet(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    h, w = frames[0].shape[:2]
    cols = 4
    rows = (len(frames) + cols - 1) // cols
    sheet = np.full((h * rows, w * cols, 3), 18, dtype=np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
    return sheet


# ─────────────────────────── C. Edge enhancement ───────────────────────────
def unsharp_mask(img: np.ndarray, sigma: float = 1.4, strength: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharp = cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def sobel_edges(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    mag = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)


def edge_enhance_pipeline(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sharp = unsharp_mask(img)
    edges = sobel_edges(sharp)
    overlay = cv2.addWeighted(sharp, 1.0, edges, 0.35, 0)
    return sharp, edges, overlay


# ─────────────────────────── Live UI ───────────────────────────
def main() -> None:
    print_banner("Experiment 2 — Image Restoration Pipeline")

    ref_path = fetch_reference()
    require(ref_path, "Reference image missing — check fetch_reference()")
    dmg_path = CACHE_DIR / "damaged.jpg"
    if not dmg_path.exists():
        dmg_path = synth_damage(ref_path)
    require(dmg_path, "Damaged image missing — check synth_damage()")
    reference = cv2.imread(str(ref_path))
    damaged = cv2.imread(str(dmg_path))
    if reference is None or damaged is None:
        raise FileNotFoundError(f"Could not decode reference ({ref_path}) or damaged ({dmg_path})")
    print(f"[info] reference: {ref_path}  shape={reference.shape}")
    print(f"[info] damaged  : {dmg_path}  shape={damaged.shape}")

    # Precompute all variants (cheap; depth-1 pipeline)
    poisson = poisson_blend(damaged, reference)
    morph_frames = cross_dissolve(damaged, poisson, steps=8)
    morph_sheet = morph_contact_sheet(morph_frames)
    _, _, final = edge_enhance_pipeline(poisson)
    edges_only = sobel_edges(unsharp_mask(poisson))

    # Save static outputs
    save_image("01_blend_linear.png", linear_blend(damaged, reference, 0.5), EXPERIMENT)
    save_image("02_blend_poisson.png", poisson, EXPERIMENT)
    save_image("03_morph_sequence.png", morph_sheet, EXPERIMENT)
    save_image("04_edges.png", edges_only, EXPERIMENT)
    save_image("05_final_restored.png", final, EXPERIMENT)

    # Live window setup
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1440, 760)
    make_slider(WIN, "alpha x100", 0, 100, 50)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)

    views = [
        ("Damaged (input)",        damaged),
        ("Linear blend (slider)",  None),       # recomputed each frame
        ("Poisson blend",          poisson),
        ("Morph contact sheet",    morph_sheet),
        ("Edges (Scharr)",         edges_only),
        ("Final restored",         final),
    ]
    idx = 0

    help_lines = [
        "Experiment 2 — Image Restoration",
        "n next view   p prev   s save   q quit",
        "slider 'alpha x100' controls the linear-blend view",
    ]

    while True:
        title, view = views[idx]
        if view is None:
            alpha = read_slider(WIN, "alpha x100", default=50) / 100.0
            view = linear_blend(damaged, reference, alpha)

        # Resize to match the window's widescreen aspect ratio
        view = fit_to(view, 1440, 760)
        info = [
            title,
            f"view {idx+1}/{len(views)}  ({view.shape[1]}x{view.shape[0]})",
        ] + help_lines[1:]
        if "slider" in title.lower():
            info.insert(1, f"alpha = {read_slider(WIN, 'alpha x100', 50) / 100.0:.2f}")
        live_show(WIN, view, info)

        k = live_keys(50)
        if k.get("q"): break
        elif k.get("n"): idx = (idx + 1) % len(views)
        elif k.get("p"): idx = (idx - 1) % len(views)
        elif k.get("s"):
            p = save_image(f"ui_capture_{idx:02d}_{int(read_slider(WIN,'alpha x100',50))}.png",
                           view, EXPERIMENT)
            print(f"[ui] saved {p}")

    cv2.destroyAllWindows()

    # Static overview
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    titles = ["Damaged", "Linear blend", "Poisson blend",
              "Morph contact sheet", "Edges (Scharr)", "Final restored"]
    panels = [damaged, linear_blend(damaged, reference, 0.5), poisson,
              morph_sheet, edges_only, final]
    for ax, img, t in zip(axes.ravel(), panels, titles):
        ax.imshow(bgr_to_rgb(img)); ax.set_title(t); ax.axis("off")
    fig.suptitle("Image Restoration Pipeline", fontsize=14, fontweight="bold")
    save_figure("00_overview.png", fig, EXPERIMENT)
    plt.close(fig)

    save_json("summary.json", {
        "reference": str(ref_path), "damaged": str(dmg_path),
        "shapes": {"damaged": list(damaged.shape),
                   "reference": list(reference.shape)},
    }, EXPERIMENT)
    print(f"[done] outputs in outputs/{EXPERIMENT}/")


if __name__ == "__main__":
    main()
