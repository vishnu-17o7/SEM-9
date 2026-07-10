"""
Program 12 — Small-object + Satellite multi-label classification (LIVE UI)
===========================================================================

Live UI
-------
  During training: one matplotlib window per sub-task updating loss curves
                   in real time, alongside a cv2 info panel.
  After training : Grad-CAM grid in a cv2 window.
                   `s` save, `q` quit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from _common import (
    DATA_DIR, OUTPUT_DIR, print_banner, save_image, save_json, device,
    live_show, live_keys, LivePlot, fit_to,
)

from . import cifar as cifar_mod
from . import satellite as sat_mod

EXPERIMENT = "experiment_12_smallobj_satellite"
WIN = "Experiment 12 — Grad-CAM"


def main(epochs: int = 3, quick: bool = False) -> None:
    print_banner("Experiment 12 — Small-object + Satellite multi-label")
    dev = device()
    print(f"[info] device: {dev}")

    out_dir = OUTPUT_DIR / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Task A: CIFAR-10 (small objects) ---")
    plot_a = LivePlot("Program 12A — CIFAR-10 (training)", subplots=2)
    plot_a.add_series("train_loss", 0)
    plot_a.add_series("val_acc", 1)
    cifar_metrics = cifar_mod.run_cifar(epochs=epochs, quick=quick,
                                        device_name=dev, out_dir=out_dir,
                                        live_plot=plot_a)
    plot_a.save(str(out_dir / "cifar_curves_live.png"))
    plot_a.close()
    print(f"  final CIFAR acc: {cifar_metrics['final_accuracy']:.3f}")

    print("\n--- Task B: EuroSAT (satellite, multi-label) ---")
    plot_b = LivePlot("Program 12B — EuroSAT (training)", subplots=2)
    plot_b.add_series("train_loss", 0)
    plot_b.add_series("val_f1_micro", 1)
    sat_metrics = sat_mod.run_satellite(epochs=epochs, quick=quick,
                                        device_name=dev, out_dir=out_dir,
                                        live_plot=plot_b)
    plot_b.save(str(out_dir / "satellite_curves_live.png"))
    plot_b.close()
    print(f"  final EuroSAT micro-F1: {sat_metrics['micro_f1']:.3f}")

    gradcam_path = out_dir / "satellite_gradcam.png"
    saved = False
    if gradcam_path.exists():
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 1280, 600)
        img = cv2.imread(str(gradcam_path))
        info = [
            f"Program 12 — small-object + satellite multi-label",
            f"CIFAR acc: {cifar_metrics['final_accuracy']:.3f}   "
            f"Sat micro-F1: {sat_metrics['micro_f1']:.3f}",
            "s save   q quit",
        ]
        while True:
            live_show(WIN, fit_to(img, 1280, 600), info, font_scale=0.55)
            k = live_keys(80)
            if k.get("q"): break
            elif k.get("s"):
                save_image("ui_capture_gradcam.png", fit_to(img, 1280, 600), EXPERIMENT)
                saved = True
        cv2.destroyAllWindows()

    save_json("summary.json",
              {"cifar": cifar_metrics, "satellite": sat_metrics}, EXPERIMENT)
    print(f"[done] outputs in outputs/{EXPERIMENT}/  saved={saved}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    main(epochs=a.epochs, quick=a.quick)
