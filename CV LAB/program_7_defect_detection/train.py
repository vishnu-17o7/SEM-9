"""Lightweight training loop for Faster R-CNN with DataLoader batching."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


def collate(batch):
    imgs, tgts = list(zip(*batch))
    return list(imgs), list(tgts)


def train(model, train_ds, val_ds, epochs: int, device_name: str,
          out_dir: Path, live: bool = False) -> dict:
    model.train()
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=1e-4, weight_decay=1e-4)
    history = {"train_loss": [], "val_loss": []}

    plot = None
    if live:
        from _common import LivePlot
        plot = LivePlot("Program 7 — Defect Detection (training)", subplots=2)
        plot.add_series("train_loss", 0)
        plot.add_series("val_loss", 0)

    from torch.utils.data import DataLoader
    for epoch in range(1, epochs + 1):
        # ── Training with batched DataLoader ──
        loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate)
        model.train()
        running, n = 0.0, 0
        for imgs, tgts in loader:
            # Convert each image to tensor
            batch_x = []
            batch_t = []
            for img, tgt in zip(imgs, tgts):
                x = torch.from_numpy(img / 255.0).permute(2, 0, 1).float().to(device_name)
                t = {k: (torch.tensor(v).to(device_name) if isinstance(v, (list, np.ndarray))
                         else torch.tensor([v]).to(device_name))
                     for k, v in tgt.items()}
                batch_x.append(x)
                batch_t.append(t)
            loss_dict = model(batch_x, batch_t)
            total = sum(l for l in loss_dict.values())
            optim.zero_grad()
            total.backward()
            optim.step()
            running += float(total.item())
            n += len(imgs)
        history["train_loss"].append(running / max(1, n))

        # ── Validation on full val set ──
        model.train()
        running_val, nv = 0.0, 0
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate)
        with torch.no_grad():
            for imgs, tgts in val_loader:
                batch_x = []
                batch_t = []
                for img, tgt in zip(imgs, tgts):
                    x = torch.from_numpy(img / 255.0).permute(2, 0, 1).float().to(device_name)
                    t = {k: (torch.tensor(v).to(device_name) if isinstance(v, (list, np.ndarray))
                             else torch.tensor([v]).to(device_name))
                         for k, v in tgt.items()}
                    batch_x.append(x)
                    batch_t.append(t)
                loss_dict = model(batch_x, batch_t)
                running_val += sum(l.item() for l in loss_dict.values())
                nv += len(imgs)
        history["val_loss"].append(running_val / max(1, nv))

        if plot is not None:
            plot.update(train_loss=history["train_loss"][-1],
                        val_loss=history["val_loss"][-1])
        print(f"  epoch {epoch}/{epochs}  "
              f"train={history['train_loss'][-1]:.3f}  "
              f"val={history['val_loss'][-1]:.3f}")

    torch.save(model.state_dict(), out_dir / "fasterrcnn_defect.pth")
    model.eval()

    if plot is not None:
        plot.save(str(out_dir / "training_curves_live.png"))
        plot.close()
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history["train_loss"], label="train")
        ax.plot(history["val_loss"], label="val")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
        ax.set_title("Faster R-CNN training")
        fig.tight_layout()
        fig.savefig(out_dir / "training_curves.png", dpi=140)
        plt.close(fig)

    return history
