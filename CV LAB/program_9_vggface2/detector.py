"""Face detector — uses MTCNN (facenet-pytorch) if available, else OpenCV DNN."""
from __future__ import annotations

import cv2
import numpy as np


class FaceDetector:
    def __init__(self, device_name: str = "cpu"):
        self.device = device_name
        self.backend = "opencv-haar"
        self._mtcnn = None
        try:
            from facenet_pytorch import MTCNN
            self._mtcnn = MTCNN(keep_all=True, device=device_name)
            self.backend = "mtcnn"
        except Exception as e:
            print(f"[warn] facenet-pytorch not available ({e}); using OpenCV Haar cascade")

    def detect(self, img: np.ndarray) -> list[list[int]]:
        """Return list of [x, y, w, h] boxes in image coordinates."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self._mtcnn is not None:
            boxes, _ = self._mtcnn.detect(rgb)
            if boxes is None:
                return []
            out = []
            for b in boxes:
                if b is None:
                    continue
                x1, y1, x2, y2 = b
                out.append([int(max(0, x1)), int(max(0, y1)),
                            int(x2 - x1), int(y2 - y1)])
            return out

        # OpenCV Haar fallback
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(g, 1.2, 5)
        return [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in faces]
