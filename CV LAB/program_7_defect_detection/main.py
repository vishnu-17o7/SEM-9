"""
Program 7 — Custom Object Detection for Industrial Defect Detection [LIVE UI]
==============================================================================

Live UI
-------
  During training: a matplotlib window that updates loss curves in
                   real time, alongside the live cv2 info panel.
  After training : 2x3 grid of val images with GT (green) vs predictions
                   (cyan). `n` for next batch, `q` to quit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from _common import (
    DATA_DIR, OUTPUT_DIR, print_banner, save_image, save_json, device,
    live_show, live_keys, LivePlot, draw_panel, fit_to,
)

from .dataset import build_or_load_dataset
from .model import build_faster_rcnn
from .train import train
from .infer import evaluate_and_visualise

EXPERIMENT = "experiment_7_defect_detection"
WIN = "Experiment 7 — Defect Detection (inference)"


def main(epochs: int = 4, n_synth: int = 120) -> None:
    print_banner("Experiment 7 — Industrial Defect Detection")

    dev = device()
    print("[1/4] preparing dataset...")
    train_ds, val_ds, classes = build_or_load_dataset(DATA_DIR / "defect_synth",
                                                     n_synth=n_synth)
    print(f"      classes: {classes}  train={len(train_ds)} val={len(val_ds)}")

    print("[2/4] building Faster R-CNN...")
    model = build_faster_rcnn(num_classes=len(classes) + 1, device_name=dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      params: {n_params / 1e6:.1f} M")

    print("[3/4] training...")
    out_dir = OUTPUT_DIR / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)
    history = train(model, train_ds, val_ds, epochs=epochs, device_name=dev,
                    out_dir=out_dir, live=False)
    print(f"      final train loss: {history['train_loss'][-1]:.3f}")

    print("[4/4] evaluating with live grid...")
    metrics = evaluate_and_visualise(model, val_ds, classes, out_dir=out_dir,
                                     device_name=dev)

    # Live inference loop
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    saved = False
    while True:
        live_show(WIN, _read_grid(out_dir), [
            f"Experiment 7 — inference grid (epoch {epochs})",
            f"train loss: {history['train_loss'][-1]:.3f}   "
            f"val loss:   {history['val_loss'][-1]:.3f}",
            "s save current grid   q quit",
        ], font_scale=0.55)
        k = live_keys(50)
        if k.get("q"): break
        elif k.get("s"):
            save_image("ui_capture.png", _read_grid(out_dir), EXPERIMENT)
            saved = True
    cv2.destroyAllWindows()

    save_json("metrics.json", {"history": history, "eval": metrics}, EXPERIMENT)
    print(f"[done] outputs in outputs/{EXPERIMENT}/  saved_grid_capture={saved}")


def _read_grid(out_dir: Path) -> np.ndarray:
    p = out_dir / "test_predictions.png"
    if p.exists():
        img = cv2.imread(str(p))
        if img is not None:
            return fit_to(img, 1280, 720)
    return np.full((720, 1280, 3), 18, dtype=np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--n_synth", type=int, default=120)
    args = parser.parse_args()
    main(epochs=args.epochs, n_synth=args.n_synth)
