"""
3D bounding-box projection from a depth pixel.

This is a deliberately simple pinhole projection — the goal is to
visualise a 3D wireframe on top of the 2D detection, not to recover
metric pose. Tweak FOCAL_LENGTH for your dataset.
"""
from __future__ import annotations

import cv2
import numpy as np

FOCAL_LENGTH = 700.0  # pixels (works for ~640px wide frames)
DEFAULT_HEIGHT_M = 1.7


def draw_3d_box(img, center_xyz, size_xyz, color=(80, 220, 255), thickness=2):
    """Draw a 3D wireframe box on `img`.

    center_xyz = (cx, cy, z)        # image-pixel x, image-pixel y, depth (m)
    size_xyz   = (w, h, depth_size) # 2D box width/height (pixels), depth size (m)
    """
    cx, cy, z = center_xyz
    bw, bh, bd = size_xyz
    # 8 corners in camera frame (z forward, x right, y down)
    if z <= 0:
        z = 0.5
    fx = fy = FOCAL_LENGTH
    s = bd / 2.0
    corners = np.array([
        [-bw / 2 / fx * z, -bh / 2 / fy * z,  s],
        [ bw / 2 / fx * z, -bh / 2 / fy * z,  s],
        [ bw / 2 / fx * z,  bh / 2 / fy * z,  s],
        [-bw / 2 / fx * z,  bh / 2 / fy * z,  s],
        [-bw / 2 / fx * z, -bh / 2 / fy * z, -s],
        [ bw / 2 / fx * z, -bh / 2 / fy * z, -s],
        [ bw / 2 / fx * z,  bh / 2 / fy * z, -s],
        [-bw / 2 / fx * z,  bh / 2 / fy * z, -s],
    ], dtype=np.float32)
    # Project: u = cx + X*fx/z,  v = cy + Y*fy/z
    proj = []
    for X, Y, Z in corners:
        zz = z + Z
        if zz <= 0.1:
            zz = 0.1
        u = int(cx + X * fx / zz)
        v = int(cy + Y * fy / zz)
        proj.append((u, v))
    proj = np.array(proj, dtype=np.int32)
    # 12 edges of the box
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        cv2.line(img, tuple(proj[a]), tuple(proj[b]), color, thickness, cv2.LINE_AA)
    return img
