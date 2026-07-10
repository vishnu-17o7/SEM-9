"""
Experiment 5 — OCR + Image Captioning (LIVE UI)
================================================

Live UI
-------
  Keys    o        toggle OCR boxes
          c        toggle caption overlay
          n / p    next / previous image
          s        save current composite
          q / esc  quit
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

from _common import (
    SAMPLE_DIR, print_banner, require, save_image, save_json,
    live_show, live_keys, fit_to,
)

EXPERIMENT = "experiment_5_ocr_captioning"
WIN = "Experiment 5 — OCR + Captioning"


def fetch_samples() -> list[Path]:
    """Generate 3 distinctive synthetic scenes with embedded text for OCR."""
    out = []
    scenes_data = [
        # 1. Restaurant menu chalkboard
        ("menu.jpg", (45, 40, 35), [
            (30, 40, "MENU", 1.4, (240, 220, 100)),
            (30, 90, "1. Margherita ...... $12", 0.55, (210, 210, 200)),
            (30, 120, "2. Pepperoni  ...... $14", 0.55, (210, 210, 200)),
            (30, 150, "3. Carbonara ...... $16", 0.55, (210, 210, 200)),
            (30, 180, "4. Caprese  ........ $11", 0.55, (210, 210, 200)),
            (30, 220, "Desserts", 1.0, (240, 200, 120)),
            (30, 260, "Tiramisu ........... $7", 0.55, (210, 210, 200)),
            (30, 290, "Panna Cotta ....... $6", 0.55, (210, 210, 200)),
            (30, 340, "OPEN 11 AM - 10 PM", 0.6, (200, 220, 100)),
        ]),
        # 2. Directional street sign
        ("sign.jpg", (200, 210, 220), [
            (50, 40, "INTERSECTION AHEAD", 0.7, (20, 20, 20)),
            (40, 100, "<- Museum     0.3 km", 0.65, (50, 50, 50)),
            (40, 140, "<- Library     0.8 km", 0.65, (50, 50, 50)),
            (40, 180, "-> Airport    12 km", 0.65, (50, 50, 50)),
            (40, 220, "-> Station     2.5 km", 0.65, (50, 50, 50)),
            (60, 300, "SPEED LIMIT 40", 1.0, (200, 50, 50)),
            (60, 350, "RADAR ENFORCED", 0.55, (180, 50, 50)),
        ]),
        # 3. Newspaper/book page spread
        ("book.jpg", (230, 225, 210), [
            (40, 30, "THE DAILY GAZETTE", 1.2, (10, 10, 10)),
            (40, 65, "-----------------------------", 0.5, (60, 60, 60)),
            (40, 90, "CV Lab Breakthrough", 1.0, (10, 10, 10)),
            (40, 120, "Researchers at the Computer", 0.5, (30, 30, 30)),
            (40, 140, "Vision Lab have developed a", 0.5, (30, 30, 30)),
            (40, 160, "new approach to feature", 0.5, (30, 30, 30)),
            (40, 180, "detection using deep neural", 0.5, (30, 30, 30)),
            (40, 200, "networks. The system can", 0.5, (30, 30, 30)),
            (40, 230, "identify objects in real-time", 0.5, (30, 30, 30)),
            (40, 260, "with 94.7% accuracy.", 0.5, (30, 30, 30)),
            (40, 310, "Volume XII - No. 42", 0.5, (100, 100, 100)),
        ]),
    ]
    for name, bg, text_lines in scenes_data:
        p = SAMPLE_DIR / name
        img = _make_scene(bg, text_lines)
        cv2.imwrite(str(p), img)
        out.append(p)
    return out


def _make_scene(bg, text_lines):
    """Create a synthetic image with embedded text for OCR testing."""
    h, w = 400, 600
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    for y in range(h):
        t = y / h
        img[y] = (np.array(bg) * (1 - t * 0.15)).astype(np.uint8)
    noise = np.random.randint(-8, 8, img.shape, dtype=np.int8)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for x, y, text, scale, color in text_lines:
        cv2.putText(img, text, (x, y),
                    cv2.FONT_HERSHEY_DUPLEX if scale > 1.0 else cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, max(1, int(scale * 1.5)), cv2.LINE_AA)
    return img


# ─────────────────────────── OCR ───────────────────────────
def _pytesseract_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def ocr_pytesseract(img: np.ndarray) -> dict:
    import pytesseract
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)
    blocks = []
    for i, txt in enumerate(data["text"]):
        txt = (txt or "").strip()
        if not txt:
            continue
        conf = float(data["conf"][i])
        if conf < 30:
            continue
        x, y, w, h = (int(v) for v in
                      (data["left"][i], data["top"][i],
                       data["width"][i], data["height"][i]))
        blocks.append({"text": txt, "conf": round(conf, 1), "box": [x, y, w, h]})
    return {"engine": "pytesseract", "blocks": blocks}


def ocr_mser_fallback(img: np.ndarray) -> dict:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, "MSER_create"):
        regions, _ = cv2.MSER_create().detectRegions(g)
    else:
        regions = []
    blocks = []
    for r in regions[:30]:
        x, y, w, h = cv2.boundingRect(r.reshape(-1, 1, 2))
        if w * h < 100:
            continue
        blocks.append({"text": "", "conf": 0.0, "box": [x, y, w, h]})
    return {"engine": "mser-fallback", "blocks": blocks}


def run_ocr(img: np.ndarray) -> dict:
    if _pytesseract_available():
        return ocr_pytesseract(img)
    return ocr_mser_fallback(img)


def draw_ocr(img: np.ndarray, ocr: dict) -> np.ndarray:
    out = img.copy()
    for b in ocr["blocks"]:
        x, y, w, h = b["box"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (60, 200, 255), 2)
        if b["text"]:
            cv2.putText(out, b["text"], (x, max(0, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 200, 255), 1, cv2.LINE_AA)
    return out


# ─────────────────────────── Captioning ───────────────────────────
def _blip_available() -> bool:
    try:
        import transformers  # noqa
        return True
    except Exception:
        return False


def caption_blip(img: np.ndarray) -> dict:
    from transformers import pipeline
    cap = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return {"engine": "blip-base", "caption": cap(rgb)[0]["generated_text"]}


def caption_template(img: np.ndarray) -> dict:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = float(hsv[..., 2].mean()) / 255.0
    sat = float(hsv[..., 1].mean()) / 255.0
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = float(cv2.Canny(g, 80, 160).mean()) / 255.0
    h, w = img.shape[:2]
    aspect = ("portrait" if h > w else "landscape" if w > h else "square")
    b = "dark" if brightness < 0.4 else "bright" if brightness > 0.6 else "moderately lit"
    c = "vivid" if sat > 0.5 else "muted" if sat < 0.2 else "natural"
    t = "highly detailed" if edges > 0.08 else "smooth" if edges < 0.02 else "textured"
    return {"engine": "template-fallback",
            "caption": f"A {aspect} {b} scene with {c} colours and {t} surfaces.",
            "brightness": round(brightness, 3),
            "saturation": round(sat, 3),
            "edge_density": round(edges, 4)}


def run_caption(img: np.ndarray) -> dict:
    if _blip_available():
        try:
            return caption_blip(img)
        except Exception as e:
            print(f"[warn] BLIP failed ({e}); using template fallback")
    return caption_template(img)


def draw_caption(img: np.ndarray, caption: str, engine: str) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    # Caption strip at the bottom
    band_h = 60
    band = np.zeros((band_h, w, 3), dtype=np.uint8)
    cv2.putText(band, caption, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(band, f"[engine: {engine}]", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    return np.vstack([out, band])


def main() -> None:
    print_banner("Experiment 5 — OCR + Image Captioning")

    samples = fetch_samples()
    print(f"[info] {len(samples)} sample image(s)")

    # Pre-compute OCR + captions (BLIP is slow on first call)
    cache: list[dict] = []
    for p in samples:
        img = cv2.imread(str(p))
        if img is None: continue
        print(f"[run] {p.name}")
        ocr = run_ocr(img)
        cap = run_caption(img)
        cache.append({"path": p, "img": img, "ocr": ocr, "caption": cap})
        print(f"  OCR engine: {ocr['engine']:14s}  blocks: {len(ocr['blocks'])}")
        print(f"  Caption   : {cap['engine']:14s}  text:   {cap['caption']}")

    if not cache:
        raise FileNotFoundError("No images available.")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    idx = 0
    show_ocr = True
    show_caption = True

    while True:
        entry = cache[idx]
        img = entry["img"].copy()
        if show_ocr:
            img = draw_ocr(img, entry["ocr"])
        if show_caption:
            img = draw_caption(img, entry["caption"]["caption"],
                               entry["caption"]["engine"])
        img = fit_to(img, 1280, 720)

        info = [
            f"{entry['path'].name}  ({idx+1}/{len(cache)})",
            f"OCR: {entry['ocr']['engine']}  blocks: {len(entry['ocr']['blocks'])}",
            f"Cap: {entry['caption']['engine']}",
            f"text: {entry['caption']['caption']}",
            "n next   p prev   o OCR   c caption   s save   q quit",
        ]
        live_show(WIN, img, info, font_scale=0.55)

        k = live_keys(40)
        if k.get("q"): break
        elif k.get("n"): idx = (idx + 1) % len(cache)
        elif k.get("p"): idx = (idx - 1) % len(cache)
        elif k.get("o"): show_ocr = not show_ocr
        elif k.get("c"): show_caption = not show_caption
        elif k.get("s"):
            save_image(f"ui_capture_{entry['path'].stem}.png", img, EXPERIMENT)
    cv2.destroyAllWindows()

    # Static outputs
    for entry in cache:
        stem = entry["path"].stem
        save_image(f"{stem}_ocr.png", draw_ocr(entry["img"], entry["ocr"]), EXPERIMENT)
        save_image(f"{stem}_caption.png",
                   draw_caption(entry["img"], entry["caption"]["caption"],
                                entry["caption"]["engine"]), EXPERIMENT)
    save_json("report.json",
              [{"image": e["path"].name, "ocr": e["ocr"], "caption": e["caption"]}
               for e in cache], EXPERIMENT)
    print(f"[done] outputs in outputs/{EXPERIMENT}/")


if __name__ == "__main__":
    main()
