"""Face embedder.

Tries to use keras-vggface (the literal VGGFace2 ResNet-50 SE) first,
falls back to facenet-pytorch's InceptionResnet-V1 (FaceNet) which
ships with VGGFace2-pretrained weights as a free alternative.

Either way the public API is identical: 512-D L2-normalised embedding.
"""
from __future__ import annotations

import cv2
import numpy as np


class FaceEmbedder:
    def __init__(self):
        self.backend = "facenet-pytorch"
        self._keras_model = None
        self._torch_model = None
        self._torch_transform = None
        self.dim = 512

        # Try keras-vggface first (literal VGGFace2)
        try:
            from keras_vggface.vggface import VGGFace
            from keras_vggface.utils import preprocess_input
            self._keras_model = VGGFace(model="resnet50", include_top=False,
                                        input_shape=(224, 224, 3), pooling="avg")
            self._keras_pre = preprocess_input
            self.backend = "keras-vggface-resnet50"
            self.dim = 2048  # ResNet-50 SE pooled
            print(f"[embedder] using {self.backend}")
            return
        except Exception as e:
            print(f"[info] keras-vggface unavailable ({e}); trying facenet-pytorch")

        # Fallback: facenet-pytorch (VGGFace2-pretrained InceptionResnet)
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1
            self._torch_model = InceptionResnetV1(pretrained="vggface2").eval()
            self._torch_transform = torch  # alias for compose()
            print(f"[embedder] using facenet-pytorch (vggface2 pretrained)")
        except Exception as e:
            raise RuntimeError(
                "No face embedder available. Install one of:\n"
                "  pip install keras-vggface tensorflow\n"
                "  pip install facenet-pytorch torch\n"
                f"(underlay error: {e})"
            ) from e

    def _preprocess_keras(self, face_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224)).astype("float32")
        return self._keras_pre(rgb[None], version=2)

    def _preprocess_torch(self, face_bgr: np.ndarray):
        import torch
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (160, 160)).astype("float32") / 255.0
        t = torch.from_numpy((rgb - 0.5) / 0.5).permute(2, 0, 1).unsqueeze(0).float()
        return t

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        if self._keras_model is not None:
            x = self._preprocess_keras(face_bgr)
            emb = self._keras_model.predict(x, verbose=0)[0]
        else:
            import torch
            t = self._preprocess_torch(face_bgr)
            with torch.no_grad():
                emb = self._torch_model(t).squeeze(0).numpy()
        # L2 normalise
        n = np.linalg.norm(emb) + 1e-10
        return (emb / n).astype("float32")

    @staticmethod
    def cosine(a: np.ndarray, B: np.ndarray) -> np.ndarray:
        # both already L2 normalised
        return B @ a
