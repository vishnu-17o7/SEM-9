"""Run inference on the val set and produce a prediction grid."""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch


def evaluate_and_visualise(model, val_ds, classes, out_dir: Path, device_name: str) -> dict:
    model.eval()
    rng = random.Random(0)
    indices = rng.sample(range(len(val_ds)), k=min(6, len(val_ds)))

    panels = []
    metrics = {"per_image": [], "total_gt": 0, "total_pred": 0, "tp": 0, "matches": []}
    for idx in indices:
        img, tgt = val_ds[idx]
        x = torch.from_numpy(img / 255.0).permute(2, 0, 1).float().to(device_name)
        with torch.no_grad():
            out = model([x])[0]
        vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gt_boxes = [(b, l) for b, l in zip(tgt["boxes"], tgt["labels"])]
        pred_boxes = []
        for box, lbl, sc in zip(out["boxes"].cpu().numpy().astype(int),
                                out["labels"].cpu().numpy(),
                                out["scores"].cpu().numpy()):
            if sc < 0.5:
                continue
            pred_boxes.append((box, lbl, sc))
        # GT in green
        for b, l in gt_boxes:
            x1, y1, x2, y2 = b
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
            name = classes[l - 1] if 1 <= l <= len(classes) else str(l)
            cv2.putText(vis, f"GT:{name}", (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1, cv2.LINE_AA)
            metrics["total_gt"] += 1
        # Pred in cyan with IoU-based TP counting
        gt_assigned = [False] * len(gt_boxes)
        for box, lbl, sc in pred_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 200, 80), 2)
            name = classes[lbl - 1] if 1 <= lbl <= len(classes) else str(lbl)
            cv2.putText(vis, f"PR:{name} {sc:.2f}", (x1, min(vis.shape[0] - 4, y2 + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 80), 1, cv2.LINE_AA)
            metrics["total_pred"] += 1
            # Match to best GT box
            best_iou = 0.0
            best_gi = -1
            for gi, (gb, gl) in enumerate(gt_boxes):
                if gl != lbl:
                    continue
                iou = _box_iou(box, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= 0.5 and best_gi >= 0 and not gt_assigned[best_gi]:
                gt_assigned[best_gi] = True
                metrics["tp"] += 1
                metrics.setdefault("matches", []).append({"gt_index": best_gi, "iou": round(best_iou, 3)})
        panels.append(vis)

    # 2x3 grid
    rows, cols = 2, 3
    h, w = panels[0].shape[:2]
    sheet = np.full((h * rows, w * cols, 3), 18, dtype=np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = p
    cv2.imwrite(str(out_dir / "test_predictions.png"), sheet)

    metrics["precision_proxy"] = round(metrics["tp"] / max(1, metrics["total_pred"]), 3)
    metrics["recall_proxy"] = round(metrics["tp"] / max(1, metrics["total_gt"]), 3)
    return metrics


def _box_iou(a, b):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xi1 = max(ax1, bx1); yi1 = max(ay1, by1)
    xi2 = min(ax2, bx2); yi2 = min(ay2, by2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / max(union, 1)
