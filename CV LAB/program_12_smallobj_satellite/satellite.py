"""EuroSAT satellite multi-label classification + Grad-CAM visualisation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from _common import DATA_DIR, save_figure
from .gradcam import gradcam

SAT_LABELS = ["water", "forest", "urban", "agricultural", "barren"]
_EXP = "experiment_12_smallobj_satellite"


def _synth_multilabel_dataset(n: int, n_labels: int = len(SAT_LABELS), size: int = 32):
    """Generate synthetic satellite-like patches with random multi-labels.

    Each class is a colour-coded noise pattern; multiple classes can be
    mixed to mimic overlapping land-cover types.
    """
    rng = np.random.default_rng(42)
    imgs = np.zeros((n, size, size, 3), dtype=np.float32)
    labels = np.zeros((n, n_labels), dtype=np.float32)
    # Per-class RGB centers (in [0, 1] range)
    class_colors = np.array([
        [0.15, 0.25, 0.78],  # water — blue
        [0.10, 0.47, 0.15],  # forest — green
        [0.55, 0.55, 0.55],  # urban — grey
        [0.78, 0.70, 0.30],  # agricultural — tan/yellow
        [0.65, 0.50, 0.35],  # barren — brown
    ], dtype=np.float32)
    for i in range(n):
        # pick 1..3 classes
        n_active = rng.integers(1, 4)
        active = rng.choice(n_labels, size=n_active, replace=False)
        labels[i, active] = 1.0
        # Divide the patch into rectangular blocks, each assigned to a class
        patch = np.zeros((size, size, 3), dtype=np.float32)
        if n_active == 1:
            # Single class fills the whole patch with texture
            color = class_colors[active[0]]
            noise = rng.normal(0, 0.05, (size, size, 3)).astype(np.float32)
            patch = np.clip(color[None, None, :] + noise, 0, 1)
        else:
            # Split into horizontal bands — one per active class
            band_edges = sorted(rng.choice(range(1, size), size=n_active - 1, replace=False))
            band_starts = [0] + band_edges
            band_ends = band_edges + [size]
            for a, y0, y1 in zip(active, band_starts, band_ends):
                color = class_colors[a]
                noise = rng.normal(0, 0.05, (y1 - y0, size, 3)).astype(np.float32)
                patch[y0:y1] = np.clip(color[None, None, :] + noise, 0, 1)
        imgs[i] = patch
    imgs = imgs.transpose(0, 3, 1, 2)
    return imgs.astype(np.float32), labels.astype(np.float32)


def _build_model(n_classes: int):
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def run_satellite(epochs: int, quick: bool, device_name: str, out_dir: Path,
                  live_plot=None) -> dict:
    n = 800 if quick else 4000
    x, y = _synth_multilabel_dataset(n=n)
    x_te, y_te = _synth_multilabel_dataset(n=200)
    print(f"  data: train={x.shape} labels={y.shape}  test={x_te.shape}")

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x_tr_t  = (torch.from_numpy(x)  - mean) / std
    x_te_t  = (torch.from_numpy(x_te) - mean) / std
    y_tr_t  = torch.from_numpy(y)
    y_te_t  = torch.from_numpy(y_te)

    tr_loader = DataLoader(TensorDataset(x_tr_t, y_tr_t), batch_size=64, shuffle=True)
    te_loader = DataLoader(TensorDataset(x_te_t, y_te_t), batch_size=128)

    model = _build_model(len(SAT_LABELS)).to(device_name)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_f1_micro": []}
    for ep in range(1, epochs + 1):
        model.train()
        running, n_seen = 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device_name), yb.to(device_name)
            opt.zero_grad()
            out = model(xb)
            loss = bce(out, yb)
            loss.backward(); opt.step()
            running += loss.item() * xb.size(0); n_seen += xb.size(0)
        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for xb, yb in te_loader:
                xb, yb = xb.to(device_name), yb.to(device_name)
                p = (torch.sigmoid(model(xb)) > 0.5).cpu().numpy()
                all_p.append(p); all_y.append(yb.cpu().numpy())
        P = np.concatenate(all_p); Y = np.concatenate(all_y)
        f1 = _f1(Y, P)
        history["train_loss"].append(running / n_seen)
        history["val_f1_micro"].append(f1)
        if live_plot is not None:
            live_plot.update(train_loss=running / n_seen, val_f1_micro=f1)
        print(f"  ep {ep}/{epochs}  loss={running/n_seen:.3f}  val_f1_micro={f1:.3f}")

    # Per-label F1
    per_label = []
    for j, name in enumerate(SAT_LABELS):
        per_label.append({"label": name,
                          "f1": _f1(Y[:, j], P[:, j]),
                          "support": int(Y[:, j].sum())})
    print("  per-label F1:")
    for r in per_label:
        print(f"    {r['label']:>14}  f1={r['f1']:.3f}  support={r['support']}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train_loss"]); ax[0].set_title("EuroSAT train loss")
    ax[1].plot(history["val_f1_micro"]); ax[1].set_title("EuroSAT val F1 (micro)")
    fig.tight_layout()
    save_figure("satellite_curves.png", fig, _EXP)
    plt.close(fig)

    torch.save(model.state_dict(), out_dir / "eurosat_resnet18.pth")

    # Grad-CAM grid on a few test patches
    sample = x_te_t[:8].to(device_name)
    heatmaps = gradcam(model, sample, target_layer=model.layer4)
    _mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    for i in range(8):
        inp = x_te_t[i] * _std.squeeze()[:, None, None] + _mean.squeeze()[:, None, None]
        inp = inp.permute(1, 2, 0).clip(0, 1).numpy()
        axes[0, i].imshow(inp)
        axes[0, i].axis("off")
        axes[0, i].set_title(SAT_LABELS[int(y_te_t[i].argmax())], fontsize=7)
        axes[1, i].imshow(heatmaps[i], cmap="jet")
        axes[1, i].axis("off")
    fig.suptitle("Input (top) vs Grad-CAM (bottom)", fontsize=11)
    fig.tight_layout()
    save_figure("satellite_gradcam.png", fig, _EXP)
    plt.close(fig)

    return {"per_label_f1": per_label, "micro_f1": history["val_f1_micro"][-1], "history": history}


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true.astype(bool)
    yp = y_pred.astype(bool)
    tp = float((yt & yp).sum())
    fp = float((~yt & yp).sum())
    fn = float((yt & ~yp).sum())
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    p = tp / (tp + fp); r = tp / (tp + fn)
    return 2 * p * r / (p + r)
