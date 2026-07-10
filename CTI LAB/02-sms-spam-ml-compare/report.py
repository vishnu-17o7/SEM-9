"""Render the comparison results into a self-contained HTML report.

Reads results/metrics.json + results/predictions/*.npz and writes
results/report.html. No CDN dependencies. Matplotlib generates the
confusion-matrix thumbnails; the comparison table is hand-rolled HTML
so it sorts by any column via a tiny vanilla-JS handler.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

# Default font on Windows lacks a 600-weight variant, which would emit
# "Failed to find font weight 600" warnings. Snap all weights to 700 (bold)
# so every label that says fontweight=600 actually renders bold.
fm._load_fontmanager(try_read_cache=False)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["figure.titleweight"] = "bold"

from sms_preprocessing import load_split

RESULTS_DIR = Path(__file__).parent / "results"
PRED_DIR = RESULTS_DIR / "predictions"
REPORT_PATH = RESULTS_DIR / "report.html"

# Hallmark-locked design tokens (mirrored from project 1's tokens.css)
TOKENS = {
    "bg":        "#0f1115",
    "panel":     "#171a21",
    "panel-2":   "#1f242d",
    "border":    "#262b36",
    "text":      "#e6e8eb",
    "muted":     "#8a93a3",
    "faint":     "#5b6473",
    "accent":    "#5b8cff",
    "spam":      "#ef4444",
    "ham":       "#22c55e",
    "warn":      "#f59e0b",
}

# Per-model colour palette — base models in a single cool hue, ensembles in
# the warm accent so the eye finds them instantly. Colour-blind safe pairing.
MODEL_COLORS = {
    "MultinomialNB":      "#7aa2f7",  # cool blue
    "ComplementNB":       "#9ece6a",  # sage green
    "LogisticRegression": "#7dcfff",  # cyan
    "LinearSVC":          "#bb9af7",  # lavender
    "RandomForest":       "#ff9e64",  # soft orange
    "GradientBoosting":   "#f7768e",  # soft pink
    "SoftVoting":         "#5b8cff",  # accent blue (ensemble)
    "Stacking":           "#f59e0b",  # warm amber (ensemble)
}

# Stable order for chart legends and bar sorting
METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
}


def _cm_png(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> str:
    """Return a base64-encoded PNG of the confusion matrix for `name`."""
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(2.6, 2.4), dpi=110)
    fig.patch.set_facecolor(TOKENS["panel"])
    ax.set_facecolor(TOKENS["panel"])
    im = ax.imshow(cm, cmap="RdYlGn", vmin=0, vmax=max(cm[0, 0], cm[1, 1], 1))
    for i in range(2):
        for j in range(2):
            color = "#0f1115" if cm[i, j] > cm.max() * 0.55 else TOKENS["text"]
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=12, fontweight="600")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["ham", "spam"], color=TOKENS["muted"], fontsize=9)
    ax.set_yticklabels(["ham", "spam"], color=TOKENS["muted"], fontsize=9)
    ax.set_xlabel("predicted", color=TOKENS["muted"], fontsize=9)
    ax.set_ylabel("actual", color=TOKENS["muted"], fontsize=9)
    ax.set_title(name, color=TOKENS["text"], fontsize=10, pad=6)
    for s in ax.spines.values():
        s.set_color(TOKENS["border"])
    ax.tick_params(colors=TOKENS["muted"], length=0)
    fig.tight_layout(pad=0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=TOKENS["panel"])
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=TOKENS["panel"], bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _style_axes(ax, *, grid: bool = True) -> None:
    ax.set_facecolor(TOKENS["panel-2"])
    for s in ax.spines.values():
        s.set_color(TOKENS["border"])
    ax.tick_params(colors=TOKENS["muted"], length=0)
    if grid:
        ax.grid(True, color=TOKENS["border"], linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)


def _chart_bar_metrics(metrics: list[dict]) -> str:
    """Grouped bar chart: one bar per (model, metric). Models on x, metric hue."""
    fig, ax = plt.subplots(figsize=(11.5, 4.6), dpi=120)
    fig.patch.set_facecolor(TOKENS["panel"])
    _style_axes(ax)

    names = [m["name"] for m in metrics]
    n = len(names)
    k = len(METRIC_KEYS)
    x = np.arange(n)
    width = 0.16

    bar_colors = ["#7aa2f7", "#22c55e", "#f59e0b", "#5b8cff", "#bb9af7"]
    for i, key in enumerate(METRIC_KEYS):
        vals = [m[key] for m in metrics]
        offset = (i - (k - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=METRIC_LABELS[key],
               color=bar_colors[i], edgecolor="none", alpha=0.92)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", color=TOKENS["muted"], fontsize=9)
    ax.set_ylim(0.7, 1.01)
    ax.set_yticks(np.arange(0.7, 1.01, 0.05))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0.7, 1.01, 0.05)], color=TOKENS["muted"], fontsize=9)
    ax.set_ylabel("score", color=TOKENS["muted"], fontsize=10)
    ax.set_title("Per-metric scores across all 8 models",
                 color=TOKENS["text"], fontsize=12, pad=10, loc="left")
    leg = ax.legend(loc="lower right", ncol=5, frameon=False,
                    labelcolor=TOKENS["muted"], fontsize=9, bbox_to_anchor=(1.0, -0.32))
    return _png_b64(fig)


def _chart_pr_scatter(metrics: list[dict]) -> str:
    """Precision-Recall scatter, point size = ROC-AUC, color by family."""
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=120)
    fig.patch.set_facecolor(TOKENS["panel"])
    _style_axes(ax)

    for m in metrics:
        size = 80 + (m["roc_auc"] - 0.95) * 1800
        size = max(60, min(size, 360))
        edge = "#5b8cff" if m["family"] == "ensemble" else "none"
        ax.scatter(m["recall"], m["precision"], s=size,
                   color=MODEL_COLORS.get(m["name"], TOKENS["muted"]),
                   edgecolors=edge, linewidths=1.5, alpha=0.9, zorder=3)
        ax.annotate(m["name"], (m["recall"], m["precision"]),
                    xytext=(6, 4), textcoords="offset points",
                    color=TOKENS["muted"], fontsize=8.5)

    ax.set_xlim(0.6, 1.01)
    ax.set_ylim(0.65, 1.01)
    ax.set_xlabel("Recall", color=TOKENS["muted"], fontsize=10)
    ax.set_ylabel("Precision", color=TOKENS["muted"], fontsize=10)
    ax.set_xticks(np.arange(0.6, 1.01, 0.1))
    ax.set_yticks(np.arange(0.65, 1.01, 0.05))
    ax.set_xticklabels([f"{v:.1f}" for v in np.arange(0.6, 1.01, 0.1)], color=TOKENS["muted"])
    ax.set_yticklabels([f"{v:.2f}" for v in np.arange(0.65, 1.01, 0.05)], color=TOKENS["muted"])
    ax.set_title("Precision vs Recall — point size proportional to ROC-AUC",
                 color=TOKENS["text"], fontsize=12, pad=10, loc="left")
    # manual legend for "ensemble" rings
    ring = ax.scatter([], [], s=120, color=TOKENS["muted"],
                      edgecolors="#5b8cff", linewidths=1.5, label="ensemble")
    base = ax.scatter([], [], s=120, color=TOKENS["muted"],
                      edgecolors="none", label="base")
    ax.legend(handles=[ring, base], loc="lower left", frameon=False,
              labelcolor=TOKENS["muted"], fontsize=9)
    return _png_b64(fig)


def _chart_speed_vs_f1(metrics: list[dict]) -> str:
    """Train time (log x) vs F1, point size = ROC-AUC, color by family."""
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=120)
    fig.patch.set_facecolor(TOKENS["panel"])
    _style_axes(ax)

    for m in metrics:
        size = 80 + (m["roc_auc"] - 0.95) * 1800
        size = max(60, min(size, 360))
        ax.scatter(m["train_time_s"], m["f1"], s=size,
                   color=MODEL_COLORS.get(m["name"], TOKENS["muted"]),
                   edgecolors="#5b8cff" if m["family"] == "ensemble" else "none",
                   linewidths=1.5, alpha=0.9, zorder=3)
        # label offset to dodge the point
        ax.annotate(m["name"], (m["train_time_s"], m["f1"]),
                    xytext=(7, 4), textcoords="offset points",
                    color=TOKENS["muted"], fontsize=8.5)

    ax.set_xscale("log")
    ax.set_xlim(0.15, 30)
    ax.set_ylim(0.78, 0.96)
    ax.set_xlabel("Train time (s, log scale)", color=TOKENS["muted"], fontsize=10)
    ax.set_ylabel("F1 score", color=TOKENS["muted"], fontsize=10)
    ax.set_yticks(np.arange(0.78, 0.96, 0.02))
    ax.set_yticklabels([f"{v:.2f}" for v in np.arange(0.78, 0.96, 0.02)], color=TOKENS["muted"])
    ax.set_title("Speed vs F1 — fastest top performer wins",
                 color=TOKENS["text"], fontsize=12, pad=10, loc="left")
    return _png_b64(fig)


def _chart_roc_curves(metrics: list[dict]) -> str:
    """One ROC curve per model on a single chart."""
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=120)
    fig.patch.set_facecolor(TOKENS["panel"])
    _style_axes(ax)

    for m in metrics:
        data = np.load(PRED_DIR / f"{m['name']}.npz")
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        lw = 2.2 if m["family"] == "ensemble" else 1.4
        ax.plot(fpr, tpr, color=MODEL_COLORS.get(m["name"], TOKENS["muted"]),
                linewidth=lw, label=f"{m['name']} (AUC {m['roc_auc']:.3f})",
                alpha=0.92)

    ax.plot([0, 1], [0, 1], color=TOKENS["border"], linestyle="--", linewidth=1)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    ax.set_xlabel("False positive rate", color=TOKENS["muted"], fontsize=10)
    ax.set_ylabel("True positive rate", color=TOKENS["muted"], fontsize=10)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title("ROC curves — all 8 models",
                 color=TOKENS["text"], fontsize=12, pad=10, loc="left")
    ax.legend(loc="lower right", frameon=False, labelcolor=TOKENS["muted"],
              fontsize=8, ncol=2)
    return _png_b64(fig)


def _chart_pr_curves(metrics: list[dict]) -> str:
    """One Precision-Recall curve per model. Spam is 13.6% — PR-AUC matters here."""
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=120)
    fig.patch.set_facecolor(TOKENS["panel"])
    _style_axes(ax)

    # baseline = positive rate
    data0 = np.load(PRED_DIR / f"{metrics[0]['name']}.npz")
    baseline = float(data0["y_true"].mean())
    ax.axhline(baseline, color=TOKENS["border"], linestyle="--", linewidth=1,
               label=f"baseline = {baseline:.2%}")

    for m in metrics:
        data = np.load(PRED_DIR / f"{m['name']}.npz")
        prec, rec, _ = precision_recall_curve(data["y_true"], data["y_prob"])
        lw = 2.2 if m["family"] == "ensemble" else 1.4
        ax.plot(rec, prec, color=MODEL_COLORS.get(m["name"], TOKENS["muted"]),
                linewidth=lw, label=m["name"], alpha=0.92)

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.4, 1.01)
    ax.set_xlabel("Recall", color=TOKENS["muted"], fontsize=10)
    ax.set_ylabel("Precision", color=TOKENS["muted"], fontsize=10)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0.4, 1.01, 0.1))
    ax.set_title("Precision-Recall curves — spam is 13.6% of the corpus",
                 color=TOKENS["text"], fontsize=12, pad=10, loc="left")
    ax.legend(loc="lower left", frameon=False, labelcolor=TOKENS["muted"],
              fontsize=8, ncol=2)
    return _png_b64(fig)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _badge(family: str) -> str:
    color = TOKENS["accent"] if family == "ensemble" else TOKENS["muted"]
    return (
        f'<span class="badge" style="color:{color};border-color:{color}">'
        f'{family}</span>'
    )


def _row(r: dict, cm_b64: str) -> str:
    winner = ""  # filled in by JS after sort
    est = r.get("estimator", "")
    est_html = f' <span class="est">{est}</span>' if est else ""
    return (
        f'<tr data-family="{r["family"]}" '
        f'data-accuracy="{r["accuracy"]}" data-precision="{r["precision"]}" '
        f'data-recall="{r["recall"]}" data-f1="{r["f1"]}" data-roc="{r["roc_auc"]}" '
        f'data-train="{r["train_time_s"]}" data-predict="{r["predict_time_s"]}">'
        f'<td class="name"><strong>{r["name"]}</strong> {_badge(r["family"])}{est_html}</td>'
        f'<td class="num">{_fmt_pct(r["accuracy"])}</td>'
        f'<td class="num">{_fmt_pct(r["precision"])}</td>'
        f'<td class="num">{_fmt_pct(r["recall"])}</td>'
        f'<td class="num"><strong>{_fmt_pct(r["f1"])}</strong></td>'
        f'<td class="num">{_fmt_pct(r["roc_auc"])}</td>'
        f'<td class="num">{r["train_time_s"]:.2f}s</td>'
        f'<td class="num">{r["predict_time_s"]:.3f}s</td>'
        f'<td class="cm"><img alt="cm-{r["name"]}" src="data:image/png;base64,{cm_b64}"></td>'
        f"</tr>"
    )


def main() -> None:
    from sms_preprocessing import (
        DATA_FILE,
        DATA_DIR,
        DATASET_URL,
        load_dataframe,
    )
    import hashlib, platform

    metrics = json.loads((RESULTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    _, _, _, y_test = load_split()
    n_test = int(np.asarray(y_test).shape[0])

    # dataset stats — measured from the file the report was generated from
    df = load_dataframe()
    total = int(len(df))
    n_spam = int(df["label"].sum())
    n_ham = total - n_spam
    spam_ratio = df["label"].mean()

    msg_lens = df["message"].str.len()
    word_lens = df["clean_message"].str.split().str.len()
    raw_stats = {
        "min": int(msg_lens.min()),
        "median": int(msg_lens.median()),
        "mean": round(float(msg_lens.mean()), 1),
        "max": int(msg_lens.max()),
    }
    word_stats = {
        "min": int(word_lens.min()),
        "median": int(word_lens.median()),
        "mean": round(float(word_lens.mean()), 1),
        "max": int(word_lens.max()),
    }
    n_dups = int(df.duplicated(subset=["message"]).sum())
    file_bytes = DATA_FILE.stat().st_size if DATA_FILE.exists() else 0
    file_sha = hashlib.sha256(DATA_FILE.read_bytes()).hexdigest() if DATA_FILE.exists() else ""
    py_ver = platform.python_version()
    sk_ver = __import__("sklearn").__version__

    rows = []
    for r in metrics:
        data = np.load(PRED_DIR / f"{r['name']}.npz")
        cm_b64 = _cm_png(data["y_true"], data["y_pred"], r["name"])
        rows.append(_row(r, cm_b64))

    best = max(metrics, key=lambda r: r["f1"])
    best_idx = metrics.index(best)

    by_name = {m["name"]: m for m in metrics}
    boosting = by_name.get("GradientBoosting")
    voting = by_name.get("SoftVoting")
    stacking = by_name.get("Stacking")

    chart_bar   = _chart_bar_metrics(metrics)
    chart_prs   = _chart_pr_scatter(metrics)
    chart_speed = _chart_speed_vs_f1(metrics)
    chart_roc   = _chart_roc_curves(metrics)
    chart_pr    = _chart_pr_curves(metrics)

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SMS Spam · Model Comparison</title>
<style>
  :root {{
    --bg: {TOKENS["bg"]}; --panel: {TOKENS["panel"]}; --panel-2: {TOKENS["panel-2"]};
    --border: {TOKENS["border"]}; --text: {TOKENS["text"]}; --muted: {TOKENS["muted"]};
    --faint: {TOKENS["faint"]}; --accent: {TOKENS["accent"]};
    --spam: {TOKENS["spam"]}; --ham: {TOKENS["ham"]};
    --font-mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    --font-body: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 0; background: var(--bg); color: var(--text);
    font-family: var(--font-body); font-size: 14px; line-height: 1.5; }}
  main {{ max-width: 1280px; margin: 0 auto; padding: 32px 28px 64px; }}

  h1 {{ margin: 0 0 4px; font-size: 30px; font-weight: 600; letter-spacing: -0.02em; }}
  h1 .accent {{ color: var(--accent); }}
  .eyebrow {{ font-family: var(--font-mono); font-size: 11px;
    letter-spacing: 0.16em; text-transform: uppercase; color: var(--faint); }}
  .sub {{ color: var(--muted); margin-top: 6px; }}
  hr {{ border: 0; border-top: 1px solid var(--border); margin: 24px 0; }}

  .metrics {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0 28px;
  }}
  .card {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 18px;
  }}
  .card .label {{ font-family: var(--font-mono); font-size: 10px;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--faint); }}
  .card .value {{ font-size: 26px; font-weight: 600; margin-top: 4px;
    font-variant-numeric: tabular-nums; }}
  .card .value.best {{ color: var(--ham); }}
  .card .value.family {{ color: var(--accent); }}
  .card .delta {{ color: var(--muted); font-family: var(--font-mono); font-size: 11px;
    margin-top: 4px; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border);
    vertical-align: middle; }}
  th {{ font-family: var(--font-mono); font-size: 10px; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.12em; color: var(--faint);
    background: var(--panel-2); position: sticky; top: 0; cursor: pointer;
    user-select: none; }}
  th:hover {{ color: var(--text); }}
  th .arrow {{ opacity: 0.4; margin-left: 4px; }}
  th.active .arrow {{ opacity: 1; color: var(--accent); }}
  td.num {{ font-family: var(--font-mono); font-variant-numeric: tabular-nums;
    text-align: right; }}
  td.name {{ min-width: 220px; }}
  td.cm img {{ display: block; width: 120px; height: auto; border-radius: 4px; }}
  tbody tr:hover {{ background: rgba(255,255,255,0.02); }}
  tbody tr.best {{ background: rgba(34, 197, 94, 0.06); }}

  .badge {{ display: inline-block; font-family: var(--font-mono);
    font-size: 9px; letter-spacing: 0.1em; text-transform: uppercase;
    border: 1px solid; border-radius: 999px;
    padding: 1px 7px; margin-left: 6px; vertical-align: middle; }}
  .est {{ display: inline-block; font-family: var(--font-mono);
    font-size: 10px; color: var(--faint); margin-left: 6px; }}
  .lineup {{ margin-top: -12px; margin-bottom: 24px; }}
  .lineup-card .value {{ font-size: 18px; }}
  .lineup-card code {{ background: transparent; color: var(--accent);
    font-family: var(--font-mono); font-size: 10px;
    padding: 0; border: 0; }}

  .footer {{ margin-top: 24px; color: var(--faint); font-family: var(--font-mono);
    font-size: 11px; line-height: 1.8; }}
  .footer code {{ background: var(--panel-2); padding: 1px 6px; border-radius: 4px;
    border: 1px solid var(--border); color: var(--text); }}
  @media (max-width: 900px) {{
    .metrics {{ grid-template-columns: repeat(2, 1fr); }}
    .dataset-grid {{ grid-template-columns: repeat(2, 1fr); }}
    td.cm {{ display: none; }}
    th.cm-head {{ display: none; }}
  }}

  /* chart panels */
  .charts {{ display: flex; flex-direction: column; gap: 12px; margin-bottom: 24px; }}
  .chart-card {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 18px 12px;
  }}
  .chart-head {{
    display: flex; align-items: baseline; justify-content: space-between;
    gap: 16px; margin-bottom: 6px;
  }}
  .chart-head h2 {{ margin: 0; font-size: 14px; font-weight: 600; letter-spacing: -0.005em; }}
  .chart-head .sub {{ color: var(--faint); font-family: var(--font-mono);
    font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; }}
  .chart-body {{ background: var(--panel); border-radius: 6px; overflow: hidden; }}
  .chart-body img {{ display: block; width: 100%; height: auto; }}
  .chart-card.full .chart-body img {{ width: 100%; }}
  .chart-grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
  }}
  @media (max-width: 900px) {{
    .chart-grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<main>
  <div class="eyebrow">// sms spam · text classification</div>
  <h1>Model <span class="accent">comparison</span></h1>
  <p class="sub">All 8 models trained on the same train/test split of the UCI SMS Spam Collection. Sort any column by clicking its header.</p>

  <hr/>

  <section class="dataset">
    <div class="dataset-head">
      <div class="eyebrow">// dataset</div>
      <h2>UCI SMS Spam Collection</h2>
      <p class="dataset-source">
        Source: <a href="{DATASET_URL}" target="_blank" rel="noopener">{DATASET_URL}</a>
      </p>
    </div>
    <div class="dataset-grid">
      <div class="card"><div class="label">Total messages</div><div class="value">{total:,}</div><div class="delta">file: {DATA_FILE.name}</div></div>
      <div class="card"><div class="label">Ham</div><div class="value ham">{n_ham:,}</div><div class="delta">{(1 - spam_ratio) * 100:.1f}% of corpus</div></div>
      <div class="card"><div class="label">Spam</div><div class="value spam">{n_spam:,}</div><div class="delta">{spam_ratio * 100:.1f}% of corpus</div></div>
      <div class="card"><div class="label">Class imbalance</div><div class="value">{spam_ratio * 100:.1f}% : {(1 - spam_ratio) * 100:.1f}%</div><div class="delta">spam : ham ratio</div></div>
      <div class="card"><div class="label">Message length (chars)</div><div class="value num">{raw_stats["median"]}<span class="unit"> median</span></div><div class="delta">min {raw_stats["min"]} · mean {raw_stats["mean"]} · max {raw_stats["max"]}</div></div>
      <div class="card"><div class="label">Message length (words)</div><div class="value num">{word_stats["median"]}<span class="unit"> median</span></div><div class="delta">min {word_stats["min"]} · mean {word_stats["mean"]} · max {word_stats["max"]}</div></div>
      <div class="card"><div class="label">Duplicate messages</div><div class="value num">{n_dups:,}</div><div class="delta">exact text duplicates (kept in set)</div></div>
      <div class="card"><div class="label">Train / test split</div><div class="value num">{total - n_test:,} / {n_test:,}</div><div class="delta">80 / 20 stratified on label</div></div>
    </div>
    <details class="dataset-meta">
      <summary>File metadata &amp; runtime</summary>
      <table class="meta">
        <tr><th>File</th><td><code>{DATA_FILE}</code></td></tr>
        <tr><th>Size</th><td>{file_bytes:,} bytes</td></tr>
        <tr><th>SHA-256</th><td><code>{file_sha}</code></td></tr>
        <tr><th>Python</th><td>{py_ver}</td></tr>
        <tr><th>scikit-learn</th><td>{sk_ver}</td></tr>
        <tr><th>Stratified holdout</th><td>80% train ({total - n_test:,} msgs) · 20% test ({n_test:,} msgs) · seed 42</td></tr>
      </table>
    </details>
  </section>

  <hr/>

  <section class="metrics">
    <div class="card"><div class="label">Test set</div><div class="value">{n_test:,}</div><div class="delta">20% stratified holdout</div></div>
    <div class="card"><div class="label">Models compared</div><div class="value">{len(metrics)}</div><div class="delta">6 base · 2 ensembles</div></div>
    <div class="card"><div class="label">Best F1</div><div class="value best">{_fmt_pct(best["f1"])}</div><div class="delta">{best["name"]} (family: {best["family"]})</div></div>
    <div class="card"><div class="label">Best ROC-AUC</div><div class="value best">{_fmt_pct(max(r["roc_auc"] for r in metrics))}</div><div class="delta">{max(metrics, key=lambda r: r["roc_auc"])["name"]}</div></div>
  </section>

  <section class="metrics lineup">
    <div class="card lineup-card">
      <div class="label">Boosting</div>
      <div class="value">GradientBoosting</div>
      <div class="delta"><code>sklearn.ensemble.GradientBoostingClassifier</code></div>
      <div class="delta">sequential boosting on TF-IDF · F1 {_fmt_pct(boosting["f1"])} · {boosting["train_time_s"]:.2f}s</div>
    </div>
    <div class="card lineup-card">
      <div class="label">Voting</div>
      <div class="value">SoftVoting</div>
      <div class="delta"><code>sklearn.ensemble.VotingClassifier(voting="soft")</code></div>
      <div class="delta">averages predict_proba of NB + CNB + LR + calibrated LinearSVC</div>
    </div>
    <div class="card lineup-card">
      <div class="label">Stacking</div>
      <div class="value">Stacking</div>
      <div class="delta"><code>sklearn.ensemble.StackingClassifier</code></div>
      <div class="delta">NB + CNB + LR + GBDT → LogisticRegression meta-learner (5-fold CV)</div>
    </div>
    <div class="card lineup-card">
      <div class="label">Best F1 winner</div>
      <div class="value best">{_fmt_pct(best["f1"])}</div>
      <div class="delta">{best["name"]} · F1 {_fmt_pct(best["f1"])} · trained in {best["train_time_s"]:.2f}s</div>
    </div>
  </section>

  <section class="charts">
    <div class="chart-card full">
      <div class="chart-head">
        <h2>Per-metric scores</h2>
        <span class="sub">one bar per (model, metric) · ensembles ringed in accent</span>
      </div>
      <div class="chart-body"><img alt="bar chart" src="data:image/png;base64,{chart_bar}"></div>
    </div>

    <div class="chart-grid">
      <div class="chart-card">
        <div class="chart-head">
          <h2>Precision vs Recall</h2>
          <span class="sub">point size = ROC-AUC</span>
        </div>
        <div class="chart-body"><img alt="PR scatter" src="data:image/png;base64,{chart_prs}"></div>
      </div>
      <div class="chart-card">
        <div class="chart-head">
          <h2>Speed vs F1</h2>
          <span class="sub">train time (log) vs F1</span>
        </div>
        <div class="chart-body"><img alt="speed vs f1" src="data:image/png;base64,{chart_speed}"></div>
      </div>
      <div class="chart-card">
        <div class="chart-head">
          <h2>ROC curves</h2>
          <span class="sub">all 8 models overlaid</span>
        </div>
        <div class="chart-body"><img alt="roc curves" src="data:image/png;base64,{chart_roc}"></div>
      </div>
      <div class="chart-card">
        <div class="chart-head">
          <h2>Precision-Recall curves</h2>
          <span class="sub">spam is 13.6% of corpus</span>
        </div>
        <div class="chart-body"><img alt="pr curves" src="data:image/png;base64,{chart_pr}"></div>
      </div>
    </div>
  </section>

  <div style="overflow-x:auto; border:1px solid var(--border); border-radius:10px; background:var(--panel)">
  <table id="results">
    <thead>
      <tr>
        <th data-key="name">Model <span class="arrow">↕</span></th>
        <th data-key="accuracy" class="num">Accuracy <span class="arrow">↕</span></th>
        <th data-key="precision" class="num">Precision <span class="arrow">↕</span></th>
        <th data-key="recall" class="num">Recall <span class="arrow">↕</span></th>
        <th data-key="f1" class="num">F1 <span class="arrow">↕</span></th>
        <th data-key="roc" class="num">ROC-AUC <span class="arrow">↕</span></th>
        <th data-key="train" class="num">Train <span class="arrow">↕</span></th>
        <th data-key="predict" class="num">Predict <span class="arrow">↕</span></th>
        <th class="cm-head">Confusion matrix</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
  </div>

  <p class="footer">
    Reproduce with <code>python model_comparison.py</code> then <code>python report.py</code>.<br/>
    Dataset: <a href="{DATASET_URL}" target="_blank" rel="noopener">UCI SMS Spam Collection</a> · {total:,} messages · {spam_ratio * 100:.1f}% spam · split 80/20 stratified (seed 42).
  </p>
</main>
<script>
  (function () {{
    const table = document.getElementById('results');
    const tbody = table.querySelector('tbody');
    const headers = table.querySelectorAll('th[data-key]');
    let sortKey = 'f1';
    let sortDir = -1; // descending by default
    const KEY_MAP = {{
      accuracy: 'accuracy', precision: 'precision', recall: 'recall',
      f1: 'f1', roc: 'roc', train: 'train', predict: 'predict', name: 'name'
    }};
    function sortBy(key, dir) {{
      const rows = Array.from(tbody.querySelectorAll('tr'));
      rows.sort((a, b) => {{
        const av = a.dataset[KEY_MAP[key]];
        const bv = b.dataset[KEY_MAP[key]];
        const an = parseFloat(av);
        const bn = parseFloat(bv);
        const both = !isNaN(an) && !isNaN(bn);
        if (both) return dir * (an - bn);
        return dir * String(av).localeCompare(String(bv));
      }});
      rows.forEach((r) => tbody.appendChild(r));
    }}
    function updateHeader() {{
      headers.forEach((h) => {{
        const arrow = h.querySelector('.arrow');
        if (h.dataset.key === sortKey) {{
          h.classList.add('active');
          arrow.textContent = sortDir === 1 ? '↑' : '↓';
        }} else {{
          h.classList.remove('active');
          arrow.textContent = '↕';
        }}
      }});
    }}
    function markBest() {{
      const firstRow = tbody.querySelector('tr');
      if (firstRow) firstRow.classList.add('best');
    }}
    headers.forEach((h) => {{
      h.addEventListener('click', () => {{
        if (sortKey === h.dataset.key) sortDir = -sortDir;
        else {{ sortKey = h.dataset.key; sortDir = -1; }}
        sortBy(sortKey, sortDir);
        updateHeader();
        markBest();
      }});
    }});
    sortBy(sortKey, sortDir);
    updateHeader();
    markBest();
  }})();
</script>
</body>
</html>
"""
    REPORT_PATH.write_text(page, encoding="utf-8")
    print(f"Wrote {REPORT_PATH} ({REPORT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
