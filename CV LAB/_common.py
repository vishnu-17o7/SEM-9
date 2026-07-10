"""
Shared helpers for CV Lab experiments.

Provides:
  - ASSET_DIR, OUTPUT_DIR  : standard project folders
  - require(path, hint)    : raises FileNotFoundError with a useful message
  - device                 : torch device string ("cuda" if available, else "cpu")
  - save_image / save_figure / save_json
  - print_banner
"""
from __future__ import annotations

import atexit
import json
import signal
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── Global cleanup (ensure all OpenCV windows close on exit / Ctrl+C) ──
_cleanup_registered = False

def _ensure_cleanup() -> None:
    global _cleanup_registered
    if _cleanup_registered:
        return
    _cleanup_registered = True

    def _cleanup():
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    atexit.register(_cleanup)
    # Also catch SIGINT (Ctrl+C) inside interactive loops
    try:
        signal.signal(signal.SIGINT, lambda s, f: sys.exit(130))
    except (ValueError, AttributeError):
        pass

CV_LAB_DIR = Path(__file__).resolve().parent
ASSET_DIR = CV_LAB_DIR / "assets"
OUTPUT_DIR = CV_LAB_DIR / "outputs"

# Asset subfolders (used by the individual programs)
MODEL_DIR = ASSET_DIR / "models"
DATA_DIR = ASSET_DIR / "datasets"
SAMPLE_DIR = ASSET_DIR / "samples"
SEQUENCE_DIR = ASSET_DIR / "sequence"
LANDMARKS_DIR = ASSET_DIR / "landmarks"
GALLERY_DIR = ASSET_DIR / "gallery"
PROBES_DIR = ASSET_DIR / "probes"
VIDEOS_DIR = ASSET_DIR / "videos"
CACHE_DIR = ASSET_DIR / "cache"

for _d in (ASSET_DIR, OUTPUT_DIR, MODEL_DIR, DATA_DIR, SAMPLE_DIR,
           SEQUENCE_DIR, LANDMARKS_DIR, GALLERY_DIR, PROBES_DIR,
           VIDEOS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def require(path: Path, hint: str = "") -> Path:
    """Raise FileNotFoundError with a clear hint if a required asset is missing."""
    if not path.exists():
        msg = f"Missing asset: {path}\n"
        if hint:
            msg += f"Hint: {hint}\n"
        msg += f"Create the file/folder and rerun. Project assets live under: {ASSET_DIR}"
        raise FileNotFoundError(msg)
    return path


def ensure_download(path: Path, urls: list[str] | str, hint: str = "") -> Path:
    """Download `path` from the first reachable URL in `urls` if it is missing.

    Tries each URL in order; on success returns `path`. If every URL
    fails, raises FileNotFoundError with the supplied hint.
    """
    if path.exists():
        return path
    if isinstance(urls, str):
        urls = [urls]
    path.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    last_err: Exception | None = None
    for url in urls:
        try:
            print(f"[setup] downloading {path.name}  ({url})")
            urllib.request.urlretrieve(url, path)
            if path.exists() and path.stat().st_size > 0:
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"[setup]   saved {path}  ({size_mb:.1f} MB)")
                return path
        except Exception as e:
            last_err = e
            print(f"[setup]   failed: {type(e).__name__}: {e}")
    msg = f"Could not download {path} from any of:\n  " + "\n  ".join(urls)
    if hint:
        msg += f"\nHint: {hint}"
    if last_err is not None:
        msg += f"\nLast error: {last_err}"
    raise FileNotFoundError(msg)


def out_dir(experiment: str) -> Path:
    """Return (and create) the per-experiment output folder."""
    d = OUTPUT_DIR / experiment
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_image(name: str, img: np.ndarray, experiment: str) -> Path:
    """Save a BGR or grayscale image to outputs/<experiment>/<name>."""
    p = out_dir(experiment) / name
    cv2.imwrite(str(p), img)
    return p


def save_figure(name: str, fig, experiment: str) -> Path:
    """Save a matplotlib figure to outputs/<experiment>/<name>."""
    p = out_dir(experiment) / name
    fig.savefig(p, dpi=140, bbox_inches="tight")
    return p


def save_json(name: str, data, experiment: str) -> Path:
    """Save a JSON-serialisable object to outputs/<experiment>/<name>."""
    p = out_dir(experiment) / name
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return p


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def print_banner(title: str) -> None:
    _ensure_cleanup()
    bar = "=" * 64
    print(bar)
    print(f"  {title}")
    print(bar)


def device(prefer_cuda: bool = True) -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    try:
        import torch
        if prefer_cuda and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def side_by_side(*imgs: np.ndarray, pad: int = 6, color=(20, 20, 28)) -> np.ndarray:
    """Concatenate images horizontally with a thin vertical separator."""
    h = max(i.shape[0] for i in imgs)
    norm = []
    for i in imgs:
        if i.ndim == 2:
            i = cv2.cvtColor(i, cv2.COLOR_GRAY2BGR)
        if i.shape[0] != h:
            scale = h / i.shape[0]
            i = cv2.resize(i, (int(i.shape[1] * scale), h))
        norm.append(i)
    sep = np.full((h, pad, 3), color, dtype=np.uint8)
    parts = []
    for idx, i in enumerate(norm):
        parts.append(i)
        if idx != len(norm) - 1:
            parts.append(sep)
    return np.hstack(parts)


# ─────────────────────────── Live UI helpers ───────────────────────────
# Match Program 1's palette so the look is consistent across the lab.
_PANEL_BG  = (10, 10, 18)
_PANEL_BORDER = (60, 60, 80)
_TEXT     = (220, 220, 235)
_TEXT_DIM = (140, 140, 155)
_ACCENT   = (100, 220, 255)


def draw_panel(img: np.ndarray, lines: list[str], x: int = 12, y: int = 12,
               font_scale: float = 0.5, accent_first: bool = True) -> np.ndarray:
    """Draw a semi-transparent info panel with text lines. Returns the image."""
    line_h = 22
    pad = 10
    font = cv2.FONT_HERSHEY_DUPLEX
    panel_w = max(
        cv2.getTextSize(l, font, font_scale, 1)[0][0] for l in lines
    ) + pad * 2
    panel_h = len(lines) * line_h + pad * 2
    h, w = img.shape[:2]
    if x + panel_w > w:
        x = max(0, w - panel_w - 6)
    if y + panel_h > h:
        y = max(0, h - panel_h - 6)
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), _PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.rectangle(img, (x, y), (x + panel_w, y + panel_h), _PANEL_BORDER, 1)
    for i, line in enumerate(lines):
        ty = y + pad + i * line_h + 14
        col = _ACCENT if (accent_first and i == 0) else _TEXT
        cv2.putText(img, line, (x + pad, ty), font, font_scale, col, 1, cv2.LINE_AA)
    return img


def live_show(win: str, img: np.ndarray, lines: list[str] | None = None,
              x: int = 12, y: int = 12, font_scale: float = 0.5) -> None:
    """imshow + draw an info panel on top."""
    out = img.copy()
    if lines:
        draw_panel(out, lines, x=x, y=y, font_scale=font_scale)
    cv2.imshow(win, out)


def live_keys(delay_ms: int = 1) -> dict:
    """Poll keys; return a small dict of standard bindings plus the raw code.

    Returned keys (lowercased): 'q' (quit), 's' (save), 'n' (next),
    'p' (prev), 'r' (reset), 'space' (pause), 'esc' (quit).
    """
    k = cv2.waitKeyEx(delay_ms)
    if k == -1:
        return {"code": -1}
    out = {"code": int(k), "raw": k}
    if k in (ord('q'), 27):            out['q'] = True
    if k == ord('s'):                  out['s'] = True
    if k == ord('n'):                  out['n'] = True
    if k == ord('p'):                  out['p'] = True
    if k == ord('r'):                  out['r'] = True
    if k == ord(' '):                  out['space'] = True
    if k == ord('m'):                  out['m'] = True
    if k in (83, 65363):               out['right'] = True
    if k in (81, 65361):               out['left']  = True
    if k == ord('h'):                  out['h'] = True
    if k in (43, 61, ord('+')):        out['plus'] = True   # +, Shift+=, Numpad+
    if k in (45, 95, ord('-'), 189):   out['minus'] = True  # -, _, Numpad-, OEM minus
    return out


def make_slider(win: str, name: str, minimum: int, maximum: int,
                default: int) -> None:
    """Wrap cv2.createTrackbar with a sensible default."""
    cv2.createTrackbar(name, win, max(default, minimum), maximum,
                       lambda v: None)


def read_slider(win: str, name: str, default: int, minimum: int = 0) -> int:
    v = cv2.getTrackbarPos(name, win)
    return max(minimum, v if v > 0 else default)


def fit_to(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize image to fit within (w, h) keeping aspect ratio, centered on canvas."""
    ih, iw = img.shape[:2]
    s = min(w / max(iw, 1), h / max(ih, 1))
    nw, nh = int(iw * s), int(ih * s)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((h, w, 3), 18, dtype=np.uint8)
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


# ─────────────────────────── Live matplotlib for training ───────────────────────────
class LivePlot:
    """Matplotlib window that auto-refreshes. Used by training programs."""

    def __init__(self, title: str = "Training", subplots: int = 2):
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()
        self.fig, self.axes = plt.subplots(1, subplots, figsize=(11, 4))
        if subplots == 1:
            self.axes = [self.axes]
        self.fig.suptitle(title)
        self.lines: dict[str, object] = {}
        self.data: dict[str, list] = {}
        self.fig.show()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def add_series(self, name: str, ax_idx: int, color: str = None) -> None:
        ax = self.axes[ax_idx]
        if color:
            line, = ax.plot([], [], label=name, color=color)
        else:
            line, = ax.plot([], [], label=name)
        line.set_label(name)
        ax.legend()
        self.lines[name] = line
        self.data[name] = []

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if k in self.data:
                self.data[k].append(v)
                self.lines[k].set_data(range(1, len(self.data[k]) + 1),
                                        self.data[k])
                ax = self.lines[k].axes
                ax.relim(); ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save(self, path: str) -> None:
        self.fig.savefig(path, dpi=140, bbox_inches="tight")

    def close(self) -> None:
        self.plt.ioff()
        self.plt.close(self.fig)
