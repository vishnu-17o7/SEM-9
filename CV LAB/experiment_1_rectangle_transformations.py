"""
Experiment 1: 2D Geometric Transformations on Rectangles
(a) Keyboard-Controlled Rectangle Movement (Translation)
(b) Mouse-Controlled Rectangle Transformation (Translation, Scaling, Rotation, Affine)
"""

import cv2
import numpy as np
import ctypes
import time
from ctypes import wintypes

from _common import draw_panel, print_banner

user32 = ctypes.windll.user32

def key_down(vk):
    """Return True if the virtual key is physically pressed right now."""
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)

# ── Palette ────────────────────────────────────────────
BG       = (18, 18, 28)    # dark background
GRID_LINE = (40, 40, 55)
RECT_FILL = (100, 180, 255)  # warm orange-gold
RECT_EDGE = (200, 230, 255)
ACCENT    = (100, 220, 255)
MARKER     = (80, 80, 100)

# ── Constants ──────────────────────────────────────────
WINDOW_W, WINDOW_H = 800, 600
RECT_W, RECT_H = 100, 60

# Part (a) — keyboard
k_x, k_y = WINDOW_W // 2 - RECT_W // 2, WINDOW_H // 2 - RECT_H // 2

# Part (b) — mouse
m_x, m_y = WINDOW_W // 2 - RECT_W // 2, WINDOW_H // 2 - RECT_H // 2
m_scale = 1.0
m_angle = 0.0
m_affine = False

# ── Helpers ────────────────────────────────────────────
def make_canvas():
    img = np.full((WINDOW_H, WINDOW_W, 3), BG, dtype=np.uint8)
    # Subtle grid every 40 px
    for x in range(0, WINDOW_W, 40):
        cv2.line(img, (x, 0), (x, WINDOW_H), GRID_LINE, 1)
    for y in range(0, WINDOW_H, 40):
        cv2.line(img, (0, y), (WINDOW_W, y), GRID_LINE, 1)
    return img

def draw_rounded_rect(img, pts, fill, edge, thickness=2):
    """Draw a filled polygon with anti-aliased outline (simulated rounded look)."""
    cv2.fillPoly(img, [pts], fill, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, edge, thickness, cv2.LINE_AA)

def draw_origin_cross(img, cx, cy):
    """Small crosshair at centre."""
    cv2.line(img, (cx - 12, cy), (cx + 12, cy), MARKER, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 12), (cx, cy + 12), MARKER, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 3, ACCENT, -1, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────
# Part (a) — Keyboard-controlled translation
# ─────────────────────────────────────────────────────────────
def part_a_keyboard_control():
    global k_x, k_y
    print("[Part A] Use W/A/S/D to move (supports simultaneous keys for diagonals). ESC to quit.")

    MOVE_INTERVAL = 0.016
    last_move = 0.0

    while True:
        now = time.perf_counter()
        dx, dy = 0, 0

        if key_down(0x57): dy -= 1
        if key_down(0x53): dy += 1
        if key_down(0x41): dx -= 1
        if key_down(0x44): dx += 1
        if key_down(0x1B): break

        if dx != 0 and dy != 0:
            dx = int(round(dx * 0.707))
            dy = int(round(dy * 0.707))

        if (dx != 0 or dy != 0) and (now - last_move >= MOVE_INTERVAL):
            k_x = np.clip(k_x + dx, 0, WINDOW_W - RECT_W)
            k_y = np.clip(k_y + dy, 0, WINDOW_H - RECT_H)
            last_move = now

        img = make_canvas()
        # Rectangle with depth shadow
        shadow = np.array([[k_x + 4, k_y + 4],
                           [k_x + RECT_W + 4, k_y + 4],
                           [k_x + RECT_W + 4, k_y + RECT_H + 4],
                           [k_x + 4, k_y + RECT_H + 4]], dtype=np.int32)
        cv2.fillPoly(img, [shadow], (30, 30, 45), cv2.LINE_AA)
        cv2.rectangle(img, (k_x, k_y), (k_x + RECT_W, k_y + RECT_H), RECT_FILL, -1, cv2.LINE_AA)
        cv2.rectangle(img, (k_x, k_y), (k_x + RECT_W, k_y + RECT_H), RECT_EDGE, 2, cv2.LINE_AA)
        draw_origin_cross(img, k_x + RECT_W // 2, k_y + RECT_H // 2)

        draw_panel(img, [
            "KEYBOARD CONTROL",
            f"Position: ({k_x}, {k_y})",
            "W A S D  ·  ESC to exit",
        ])

        cv2.imshow("Part A — Keyboard Control (ESC to exit)", img)
        cv2.waitKey(1)

    cv2.destroyWindow("Part A — Keyboard Control (ESC to exit)")

# ─────────────────────────────────────────────────────────────
# Part (b) — Mouse-controlled transformations
# ─────────────────────────────────────────────────────────────
def get_transformed_rect():
    """Return the four corners of the current rectangle, possibly rotated/scaled/affine."""
    sw, sh = int(RECT_W * m_scale), int(RECT_H * m_scale)
    cx, cy = m_x + sw / 2, m_y + sh / 2
    w, h = sw, sh
    pts = np.array([[-w/2, -h/2], [ w/2, -h/2],
                    [ w/2,  h/2], [-w/2,  h/2]], dtype=np.float32)

    theta = np.radians(m_angle)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    pts = pts @ R.T

    if m_affine:
        shear = np.array([[1.0, 0.4],
                          [0.2, 1.0]])
        pts = pts @ shear.T

    pts += np.array([cx, cy])
    return pts.astype(np.int32)


def mouse_callback(event, x, y, flags, param):
    global m_x, m_y, m_scale, m_angle, m_affine

    if event == cv2.EVENT_MOUSEMOVE:
        m_x = x - int(RECT_W * m_scale) // 2
        m_y = y - int(RECT_H * m_scale) // 2
        w_s, h_s = int(RECT_W * m_scale), int(RECT_H * m_scale)
        m_x = np.clip(m_x, 0, WINDOW_W - w_s)
        m_y = np.clip(m_y, 0, WINDOW_H - h_s)

    elif event == cv2.EVENT_LBUTTONDOWN:
        m_scale = min(round(m_scale + 0.5, 1), 3.0)

    elif event == cv2.EVENT_RBUTTONDOWN:
        m_angle = (m_angle + 30) % 360

    elif event == cv2.EVENT_MBUTTONDOWN:
        m_affine = not m_affine


def part_b_mouse_control():
    global m_x, m_y, m_scale, m_angle, m_affine

    cv2.namedWindow("Part B — Mouse Control (ESC to exit)")
    cv2.setMouseCallback("Part B — Mouse Control (ESC to exit)", mouse_callback)

    print("[Part B] Move mouse to translate. Left=Scale  Right=Rotate  Middle=Affine toggle.")
    print("         Keyboard:  R=Rotate  S=Scale  A=Affine  ESC=Exit\n")

    while True:
        # Process keyboard shortcuts as well
        key = cv2.waitKeyEx(1)
        if key == 27:           # ESC
            break
        elif key in (ord('r'), ord('R')):
            m_angle = (m_angle + 30) % 360
        elif key in (ord('s'), ord('S')):
            m_scale = min(round(m_scale + 0.5, 1), 3.0)
        elif key in (ord('z'), ord('Z')):
            m_scale = max(round(m_scale - 0.5, 1), 0.5)
        elif key in (ord('a'), ord('A')):
            m_affine = not m_affine

        img = make_canvas()

        pts = get_transformed_rect()
        shadow = pts + np.array([[4, 4]], dtype=np.int32)
        cv2.fillPoly(img, [shadow], (30, 30, 45), cv2.LINE_AA)
        draw_rounded_rect(img, pts, RECT_FILL, RECT_EDGE)

        cx, cy = int(m_x + RECT_W / 2), int(m_y + RECT_H / 2)
        draw_origin_cross(img, cx, cy)

        draw_panel(img, [
            "MOUSE CONTROL  (keyboard shortcuts also available)",
            f"Translation: ({m_x}, {m_y})",
            f"Scale: {m_scale:.1f}x  (L-click / S / Z)",
            f"Rotation: {m_angle:.0f}°  (R-click / R)",
            f"Affine: {'ON' if m_affine else 'OFF'}  (M-click / A)",
            "ESC to exit",
        ])

        # Full-window crosshair through centre
        h, w = img.shape[:2]
        cv2.line(img, (cx, 0), (cx, h), (60, 60, 80), 1, cv2.LINE_AA)
        cv2.line(img, (0, cy), (w, cy), (60, 60, 80), 1, cv2.LINE_AA)

        cv2.imshow("Part B — Mouse Control (ESC to exit)", img)

    cv2.destroyWindow("Part B — Mouse Control (ESC to exit)")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("Experiment 1 — 2D Geometric Transformations on Rectangles")
    print("=" * 55)

    part_a_keyboard_control()
    part_b_mouse_control()

    cv2.destroyAllWindows()
    print("\nExperiment 1 completed.")
