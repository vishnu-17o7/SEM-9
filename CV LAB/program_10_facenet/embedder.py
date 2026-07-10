"""FaceNet embedder (InceptionResnet-V1) via facenet-pytorch."""
from __future__ import annotations

import cv2
import numpy as np


class FaceNetEmbedder:
    def __init__(self, device_name: str = "cpu"):
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1
        except Exception as e:
            raise RuntimeError(
                "facenet-pytorch is required. Install with:\n"
                "  pip install facenet-pytorch torch torchvision\n"
                f"(underlay error: {e})"
            ) from e
        self.torch = torch
        self.device = device_name
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device_name)
        self.dim = 512

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (160, 160)).astype("float32") / 255.0
        t = self.torch.from_numpy((rgb - 0.5) / 0.5).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with self.torch.no_grad():
            emb = self.model(t).squeeze(0).cpu().numpy()
        return (emb / (np.linalg.norm(emb) + 1e-10)).astype("float32")
