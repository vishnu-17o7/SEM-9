"""CIFAR-10 small-object classifier."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from _common import DATA_DIR, save_figure

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]

_EXP = "experiment_12_smallobj_satellite"


def _fetch_cifar(cache: Path, quick: bool):
    cache.mkdir(parents=True, exist_ok=True)
    paths = [cache / f"{n}.npy" for n in ("x_tr", "y_tr", "x_te", "y_te")]
    if all(p.exists() for p in paths):
        return tuple(np.load(p) for p in paths)
    print("[cifar] downloading via torchvision ...")
    from torchvision import datasets
    root = cache / "_raw"
    root.mkdir(exist_ok=True)
    tr = datasets.CIFAR10(str(root), train=True, download=True)
    te = datasets.CIFAR10(str(root), train=False, download=True)
    x_tr = tr.data.astype(np.float32) / 255.0
    y_tr = np.array(tr.targets, dtype=np.int64)
    x_te = te.data.astype(np.float32) / 255.0
    y_te = np.array(te.targets, dtype=np.int64)
    np.save(paths[0], x_tr); np.save(paths[1], y_tr)
    np.save(paths[2], x_te); np.save(paths[3], y_te)
    if quick:
        x_tr, y_tr = x_tr[:5000], y_tr[:5000]
        x_te, y_te = x_te[:1000], y_te[:1000]
    return x_tr, y_tr, x_te, y_te


def _build_model():
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m


def run_cifar(epochs: int, quick: bool, device_name: str, out_dir: Path,
              live_plot=None) -> dict:
    x_tr, y_tr, x_te, y_te = _fetch_cifar(DATA_DIR / "cifar10", quick=quick)
    print(f"  data: train={x_tr.shape}  test={x_te.shape}")

    # NHWC -> NCHW
    x_tr_t = torch.from_numpy(x_tr.transpose(0, 3, 1, 2))
    x_te_t = torch.from_numpy(x_te.transpose(0, 3, 1, 2))
    y_tr_t = torch.from_numpy(y_tr)
    y_te_t = torch.from_numpy(y_te)

    # ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x_tr_t = (x_tr_t - mean) / std
    x_te_t = (x_te_t - mean) / std

    tr_loader = DataLoader(TensorDataset(x_tr_t, y_tr_t), batch_size=128, shuffle=True)
    te_loader = DataLoader(TensorDataset(x_te_t, y_te_t), batch_size=256)

    model = _build_model().to(device_name)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_acc": []}
    for ep in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        for x, y in tr_loader:
            x, y = x.to(device_name), y.to(device_name)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward(); opt.step()
            running += loss.item() * x.size(0); n += x.size(0)
        model.eval()
        correct, total = 0, 0
        preds = []
        with torch.no_grad():
            for x, y in te_loader:
                x, y = x.to(device_name), y.to(device_name)
                p = model(x).argmax(1)
                correct += (p == y).sum().item(); total += x.size(0)
                preds.append(p.cpu().numpy())
        va = correct / total
        history["train_loss"].append(running / n)
        history["val_acc"].append(va)
        if live_plot is not None:
            live_plot.update(train_loss=running / n, val_acc=va)
        print(f"  ep {ep}/{epochs}  loss={running/n:.3f}  val_acc={va:.3f}")

    preds = np.concatenate(preds)
    acc = (preds == y_te).mean()

    # Curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train_loss"]); ax[0].set_title("CIFAR-10 train loss")
    ax[1].plot(history["val_acc"]); ax[1].set_title("CIFAR-10 val accuracy")
    fig.tight_layout()
    save_figure("cifar_curves.png", fig, _EXP)
    plt.close(fig)

    # Confusion
    cm = confusion_matrix(y_te, preds, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_xticklabels(CIFAR_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(10)); ax.set_yticklabels(CIFAR_CLASSES, fontsize=8)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"CIFAR-10 confusion (acc={acc:.2f})")
    fig.tight_layout()
    save_figure("cifar_confusion.png", fig, EXP)
    plt.close(fig)

    torch.save(model.state_dict(), out_dir / "cifar_resnet18.pth")
    print(f"  final accuracy: {acc:.3f}")
    print(classification_report(y_te, preds, target_names=CIFAR_CLASSES, zero_division=0))
    return {"final_accuracy": acc, "history": history}
