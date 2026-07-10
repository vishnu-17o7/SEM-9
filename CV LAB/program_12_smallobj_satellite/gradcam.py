"""Minimal Grad-CAM implementation (no external dependency)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def gradcam(model, x: torch.Tensor, target_layer) -> np.ndarray:
    """Compute Grad-CAM heatmaps for a batch `x`.

    Returns an (N, H, W) array normalised to [0, 1].
    """
    activations, gradients = [], []

    def fwd_hook(_, inp, out):
        activations.append(out.detach())

    def bwd_hook(_, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        model.eval()
        out = model(x)
        # Use the strongest logit per sample
        idx = out.argmax(dim=1)
        score = out.gather(1, idx.unsqueeze(1)).sum()
        model.zero_grad()
        score.backward()
        a = activations[0]      # (N, C, h, w)
        g = gradients[0]        # (N, C, h, w)
        weights = g.mean(dim=(2, 3), keepdim=True)  # (N, C, 1, 1)
        cam = F.relu((weights * a).sum(dim=1))       # (N, h, w)
        cam = cam.cpu().numpy()
        cam = np.stack([_normalise(m) for m in cam], axis=0)
        return cam
    finally:
        h1.remove(); h2.remove()


def _normalise(m: np.ndarray) -> np.ndarray:
    m = m - m.min()
    m = m / max(1e-8, m.max())
    return m
