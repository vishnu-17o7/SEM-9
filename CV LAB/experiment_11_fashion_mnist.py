"""
Experiment 11 — CNN Classifier on Fashion-MNIST (LIVE UI)
==========================================================

Live UI
-------
  During training: matplotlib window updating loss/accuracy curves
                   in real time, plus a live cv2 info panel.
  After training : sample predictions grid (3x6) in a cv2 window.
                   `s` save, `q` quit.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from _common import (
    DATA_DIR, OUTPUT_DIR, print_banner, save_figure, save_image, save_json, device,
    live_show, live_keys, LivePlot, fit_to,
)

EXPERIMENT = "experiment_11_fashion_mnist"
WIN = "Experiment 11 — Fashion-MNIST"

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def fetch_fashion_mnist(cache_dir: Path, quick: bool):
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [cache_dir / f"{n}.npy" for n in ("train_x", "train_y", "test_x", "test_y")]
    if all(p.exists() for p in paths):
        x_tr, y_tr, x_te, y_te = (np.load(p) for p in paths)
    else:
        print("[setup] downloading Fashion-MNIST from torchvision ...")
        from torchvision import datasets
        root = cache_dir / "_raw"
        root.mkdir(exist_ok=True)
        tr = datasets.FashionMNIST(root=str(root), train=True, download=True)
        te = datasets.FashionMNIST(root=str(root), train=False, download=True)
        x_tr = tr.data.numpy().astype(np.uint8)
        y_tr = tr.targets.numpy().astype(np.int64)
        x_te = te.data.numpy().astype(np.uint8)
        y_te = te.targets.numpy().astype(np.int64)
        np.save(paths[0], x_tr); np.save(paths[1], y_tr)
        np.save(paths[2], x_te); np.save(paths[3], y_te)
    if quick:
        rng = np.random.default_rng(0)
        idx_tr = rng.choice(len(x_tr), 5000, replace=False)
        idx_te = rng.choice(len(x_te), 1000, replace=False)
        x_tr, y_tr = x_tr[idx_tr], y_tr[idx_tr]
        x_te, y_te = x_te[idx_te], y_te[idx_te]
    return x_tr, y_tr, x_te, y_te


def train_one(model, loader, opt, crit, dev):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward(); opt.step()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        correct += (out.argmax(1) == y).sum().item()
    return loss_sum / total, correct / total


@torch.no_grad()
def eval_one(model, loader, crit, dev):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    preds, targets = [], []
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        out = model(x)
        loss_sum += crit(out, y).item() * x.size(0)
        total += x.size(0)
        p = out.argmax(1)
        correct += (p == y).sum().item()
        preds.append(p.cpu().numpy()); targets.append(y.cpu().numpy())
    return loss_sum / total, correct / total, np.concatenate(preds), np.concatenate(targets)


def main(epochs: int = 5, batch_size: int = 128, quick: bool = False) -> None:
    print_banner("Experiment 11 — Fashion-MNIST CNN")
    dev = device()
    print(f"[info] device: {dev}")

    cache = DATA_DIR / "fashion_mnist"
    x_tr, y_tr, x_te, y_te = fetch_fashion_mnist(cache, quick=quick)
    print(f"[data] train: {x_tr.shape}  test: {x_te.shape}")

    x_tr = (x_tr.astype(np.float32) / 255.0)[:, None]
    x_te = (x_te.astype(np.float32) / 255.0)[:, None]
    tr_loader = DataLoader(TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
                           batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te)),
                           batch_size=batch_size)

    model = SmallCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    out_dir = OUTPUT_DIR / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    plot = LivePlot("Program 11 — Fashion-MNIST (training)", subplots=2)
    plot.add_series("train_loss", 0)
    plot.add_series("val_loss", 0)
    plot.add_series("train_acc", 1)
    plot.add_series("val_acc", 1)

    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        tl, ta = train_one(model, tr_loader, opt, crit, dev)
        vl, va, _, _ = eval_one(model, te_loader, crit, dev)
        history["train_loss"].append(tl); history["train_acc"].append(ta)
        history["val_loss"].append(vl);   history["val_acc"].append(va)
        plot.update(train_loss=tl, val_loss=vl, train_acc=ta, val_acc=va)
        print(f"  ep {ep}/{epochs}  train loss={tl:.3f} acc={ta:.3f}  "
              f"val loss={vl:.3f} acc={va:.3f}")
    dt = time.perf_counter() - t0
    print(f"[time] training: {dt:.1f} s")

    plot.save(str(out_dir / "training_curves_live.png"))
    plot.close()

    torch.save(model.state_dict(), out_dir / "fashion_mnist_cnn.pth")

    # Static figures
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train"); axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train"); axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy"); axes[1].legend()
    fig.tight_layout()
    save_figure("training_curves.png", fig, EXPERIMENT)
    plt.close(fig)

    _, acc, y_pred, y_true = eval_one(model, te_loader, crit, dev)
    print(f"\nFinal test accuracy: {acc:.3f}")
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES, fontsize=8)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion matrix (acc={acc:.2f})")
    fig.tight_layout()
    save_figure("confusion_matrix.png", fig, EXPERIMENT)
    plt.close(fig)

    fig, axes = plt.subplots(3, 6, figsize=(10, 6))
    idxs = np.random.default_rng(0).choice(len(x_te), size=18, replace=False)
    for ax, i in zip(axes.ravel(), idxs):
        ax.imshow(x_te[i, 0], cmap="gray")
        ax.set_title(f"{CLASSES[y_pred[i]]}\n(true: {CLASSES[y_true[i]]})", fontsize=7)
        ax.axis("off")
    fig.suptitle("Sample predictions", fontsize=12)
    fig.tight_layout()
    save_figure("sample_predictions.png", fig, EXPERIMENT)
    plt.close(fig)

    # Live window
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 800)
    grid_img = cv2.imread(str(out_dir / "sample_predictions.png"))
    cm_img = cv2.imread(str(out_dir / "confusion_matrix.png"))
    view_mode = "grid"
    saved = False
    while True:
        cur = grid_img if view_mode == "grid" else cm_img
        info = [
            f"Program 11 — Fashion-MNIST (acc={acc:.3f}, {dt:.0f}s)",
            f"view: {view_mode}    press 'g' grid, 'c' confusion",
            "g grid   c confusion   s save   q quit",
        ]
        live_show(WIN, fit_to(cur, 1280, 800), info, font_scale=0.55)
        k = live_keys(80)
        if k.get("q"): break
        elif k.get("s"):
            save_image(f"ui_capture_{view_mode}.png", fit_to(cur, 1280, 800), EXPERIMENT)
            saved = True
        elif k.get("code") == ord('g'):
            view_mode = "grid"
        elif k.get("code") == ord('c'):
            view_mode = "confusion"
    cv2.destroyAllWindows()

    save_json("metrics.json", {
        "final_test_accuracy": acc, "epochs": epochs, "history": history,
        "training_seconds": round(dt, 1),
    }, EXPERIMENT)
    print(f"[done] outputs in outputs/{EXPERIMENT}/  saved={saved}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    main(epochs=a.epochs, batch_size=a.batch_size, quick=a.quick)
