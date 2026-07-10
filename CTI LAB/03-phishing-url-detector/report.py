"""
Generate a self-contained HTML report comparing phishing URL detection models.

Depends on results/metrics.json and model predictions saved by model_comparison.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

RESULTS_DIR = Path(__file__).parent / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
REPORT_FILE = RESULTS_DIR / "report.html"

rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#4a4a6a",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#a0a0b0",
    "ytick.color": "#a0a0b0",
    "legend.facecolor": "#16213e",
    "legend.edgecolor": "#4a4a6a",
    "grid.color": "#2a2a4a",
    "grid.alpha": 0.3,
})

CSS = """
:root {
    --bg: #0f0f1a;
    --card: #1a1a2e;
    --card-hover: #222240;
    --border: #2a2a4a;
    --text: #e0e0e0;
    --text-dim: #8888aa;
    --accent: #00d4aa;
    --accent2: #7c3aed;
    --danger: #ef4444;
    --success: #22c55e;
    --warn: #f59e0b;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.6; padding: 2rem;
}
h1 { font-size: 2rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.75rem; }
h1 small { font-size: 1rem; color: var(--text-dim); font-weight: 400; }
h2 { font-size: 1.25rem; margin: 1.5rem 0 0.75rem; color: var(--accent); }
.subtitle { color: var(--text-dim); margin-bottom: 1.5rem; }
.card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.25rem; margin-bottom: 1.25rem;
}
.card table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.card th, .card td { padding: 0.6rem 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }
.card th { color: var(--text-dim); font-weight: 500; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }
.card tr:hover td { background: var(--card-hover); }
.badge {
    display: inline-block; padding: 0.15rem 0.5rem; border-radius: 6px;
    font-size: 0.75rem; font-weight: 600;
}
.badge-best { background: #065f4622; color: var(--accent); border: 1px solid var(--accent); }
.badge-warn { background: #f59e0b22; color: var(--warn); border: 1px solid var(--warn); }
.flex-row { display: flex; gap: 1rem; flex-wrap: wrap; }
.stat { flex: 1; min-width: 120px; text-align: center; padding: 0.75rem; }
.stat-value { font-size: 1.75rem; font-weight: 700; }
.stat-label { font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
@media (max-width: 768px) { .grid-2 { grid-template-columns: 1fr; } body { padding: 1rem; } }
.chart-container { text-align: center; }
.chart-container img { max-width: 100%; border-radius: 8px; }
th.sortable { cursor: pointer; user-select: none; }
th.sortable::after { content: ' ↕'; opacity: 0.3; }
th.sortable.sorted-asc::after { content: ' ↑'; opacity: 1; }
th.sortable.sorted-desc::after { content: ' ↓'; opacity: 1; }
"""


def _bar_chart(df: pd.DataFrame, x: str, y: str, title: str, filename: str,
               color: str = "#00d4aa", sort: bool = True) -> str:
    """Create a bar chart and return the base64-encoded PNG path."""
    if sort:
        df = df.sort_values(y, ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(df[x], df[y], color=color, height=0.7, edgecolor="#4a4a6a", linewidth=0.5)
    ax.set_xlabel(y.replace("_", " ").title())
    ax.set_title(title, color="#e0e0e0", fontsize=13, pad=12)
    for bar, val in zip(bars, df[y]):
        ax.text(val + 0.005 * df[y].max(), bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color="#a0a0b0")
    ax.margins(x=0.15)
    fig.tight_layout()
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    return path.name


def _confusion_matrix(cm: list[list[int]], model_name: str) -> str:
    """Plot confusion matrix and return filename."""
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                xticklabels=["Legitimate", "Phishing"],
                yticklabels=["Legitimate", "Phishing"],
                ax=ax, cbar=False,
                annot_kws={"color": "white", "fontsize": 13})
    ax.set_xlabel("Predicted", color="#e0e0e0")
    ax.set_ylabel("Actual", color="#e0e0e0")
    ax.set_title(f"{model_name}", color="#e0e0e0", fontsize=11, pad=8)
    fig.tight_layout()
    filename = f"cm_{model_name}.png"
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    return path.name


def _feature_importance_chart() -> str | None:
    """Plot top feature importances from Random Forest."""
    imp_path = RESULTS_DIR / "feature_importance.csv"
    if not imp_path.exists():
        return None
    df = pd.read_csv(imp_path).head(12)
    return _bar_chart(df, "Feature", "Importance",
                      "Top-12 Feature Importances (Random Forest)", "feature_importance.png",
                      color="#7c3aed")


def _roc_curves() -> str:
    """Plot ROC curves for all models that have predictions saved."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    from sklearn.metrics import roc_curve
    colors = ["#00d4aa", "#7c3aed", "#ef4444", "#f59e0b", "#22c55e", "#3b82f6", "#ec4899", "#14b8a6"]

    for i, npz_path in enumerate(sorted(PREDICTIONS_DIR.glob("*.npz"))):
        data = np.load(npz_path)
        y_true = data["y_true"]
        y_prob = data["y_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = np.trapz(tpr, fpr)
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, color=color, lw=1.5,
                label=f"{npz_path.stem} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "--", lw=1, alpha=0.4, color="#4a4a6a")
    ax.set_xlabel("False Positive Rate", color="#e0e0e0")
    ax.set_ylabel("True Positive Rate", color="#e0e0e0")
    ax.set_title("ROC Curves — All Models", color="#e0e0e0", fontsize=13, pad=12)
    ax.legend(fontsize=8, loc="lower right", labelcolor="#a0a0b0")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    filename = "roc_curves.png"
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    return path.name


def generate_report():
    """Generate the HTML report."""
    RESULTS_DIR.mkdir(exist_ok=True)

    print("  Generating phishing URL detection report...")

    # Load metrics
    with open(RESULTS_DIR / "metrics.json") as f:
        metrics = json.load(f)

    df = pd.DataFrame(metrics).T
    df = df.reset_index().rename(columns={"index": "Model"})

    # Charts
    print("  Plotting charts...")
    _bar_chart(df, "Model", "f1", "F1 Score by Model", "f1_chart.png")
    _bar_chart(df, "Model", "roc_auc", "ROC-AUC by Model", "roc_auc_chart.png")
    _bar_chart(df, "Model", "accuracy", "Accuracy by Model", "accuracy_chart.png")
    _bar_chart(df, "Model", "precision", "Precision by Model", "precision_chart.png", color="#7c3aed")
    _bar_chart(df, "Model", "recall", "Recall by Model", "recall_chart.png", color="#22c55e")

    # ROC curves
    roc_chart = _roc_curves()

    # Confusion matrices
    cm_files = {}
    for name, data in metrics.items():
        if "confusion_matrix" in data:
            cm = data["confusion_matrix"]
            cm_files[name] = _confusion_matrix(cm, name)

    # Feature importance
    fi_path = _feature_importance_chart()

    # Build the HTML
    best_model = df.loc[df["f1"].idxmax()]

    def _img_tag(name):
        return f'<img src="{name}" alt="{name}" />'

    def _row_badge(val, field="f1", higher_is_better=True):
        best_val = df[field].max() if higher_is_better else df[field].min()
        if val == best_val:
            return '<span class="badge badge-best">BEST</span>'
        return ""

    rows = ""
    for _, r in df.sort_values("f1", ascending=False).iterrows():
        cm_img = cm_files.get(r["Model"], "")
        cm_cell = f'<img src="{cm_img}" style="width:160px;border-radius:6px;" />' if cm_img else ""
        rows += f"""
        <tr>
            <td><strong>{r['Model']}</strong><br><span style="color:var(--text-dim);font-size:0.8rem;">{r.get('family','')}</span></td>
            <td>{r['accuracy']:.4f} {_row_badge(r['accuracy'],'accuracy')}</td>
            <td>{r['precision']:.4f} {_row_badge(r['precision'],'precision')}</td>
            <td>{r['recall']:.4f} {_row_badge(r['recall'],'recall')}</td>
            <td>{r['f1']:.4f} {_row_badge(r['f1'],'f1')}</td>
            <td>{r['roc_auc']:.4f} {_row_badge(r['roc_auc'],'roc_auc')}</td>
            <td>{r.get('train_time_s', 0):.3f}s</td>
            <td>{cm_cell}</td>
        </tr>"""

    cm_section = ""
    for name in sorted(cm_files.keys()):
        cm_section += f'<div class="chart-container">{_img_tag(cm_files[name])}</div>'

    fi_section = f'<div class="chart-container">{_img_tag(fi_path)}</div>' if fi_path else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Phishing URL Detection — Model Comparison Report</title>
<style>{CSS}</style>
</head>
<body>

<h1>🕵️ Phishing URL Detection <small>Model Comparison Report</small></h1>
<p class="subtitle">
    URL-based feature extraction · {len(df)} classifiers evaluated ·
    {len(features := []) or df.iloc[0].get('notes', '')}
    Dataset: synthetic phishing + legitimate URLs
</p>

<!-- Summary stats -->
<div class="card">
    <div class="flex-row">
        <div class="stat">
            <div class="stat-value" style="color:var(--accent);">{best_model['f1']:.4f}</div>
            <div class="stat-label">Best F1 Score</div>
            <div style="font-size:0.8rem;color:var(--text-dim);">{best_model['Model']}</div>
        </div>
        <div class="stat">
            <div class="stat-value" style="color:var(--accent2);">{best_model['roc_auc']:.4f}</div>
            <div class="stat-label">Best ROC-AUC</div>
            <div style="font-size:0.8rem;color:var(--text-dim);">{best_model['Model']}</div>
        </div>
        <div class="stat">
            <div class="stat-value" style="color:var(--warn);">{df['train_time_s'].sum():.1f}s</div>
            <div class="stat-label">Total Train Time</div>
            <div style="font-size:0.8rem;color:var(--text-dim);">all models</div>
        </div>
        <div class="stat">
            <div class="stat-value" style="color:var(--success);">{len(df)}</div>
            <div class="stat-label">Models Compared</div>
            <div style="font-size:0.8rem;color:var(--text-dim);">{', '.join(df['Model'])}</div>
        </div>
    </div>
</div>

<!-- Model comparison table -->
<div class="card">
    <h2>📊 Model Comparison</h2>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th class="sortable">Accuracy</th>
                <th class="sortable">Precision</th>
                <th class="sortable">Recall</th>
                <th class="sortable">F1</th>
                <th class="sortable">ROC-AUC</th>
                <th>Train Time</th>
                <th>Confusion Matrix</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
</div>

<!-- Charts -->
<div class="grid-2">
    <div class="card chart-container">
        <h2>F1 Score</h2>
        {_img_tag('f1_chart.png')}
    </div>
    <div class="card chart-container">
        <h2>ROC-AUC</h2>
        {_img_tag('roc_auc_chart.png')}
    </div>
    <div class="card chart-container">
        <h2>Accuracy</h2>
        {_img_tag('accuracy_chart.png')}
    </div>
    <div class="card chart-container">
        <h2>Precision</h2>
        {_img_tag('precision_chart.png')}
    </div>
    <div class="card chart-container">
        <h2>Recall</h2>
        {_img_tag('recall_chart.png')}
    </div>
    <div class="card chart-container">
        <h2>ROC Curves</h2>
        {_img_tag(roc_chart)}
    </div>
</div>

<!-- Confusion matrices -->
<div class="card">
    <h2>🔲 Confusion Matrices</h2>
    <div class="grid-2">
        {cm_section}
    </div>
</div>

<!-- Feature importance -->
{fi_section and f'<div class="card chart-container"><h2>⭐ Feature Importances</h2>{fi_section}</div>' or ''}

<!-- How it works -->
<div class="card">
    <h2>🔬 How It Works</h2>
    <p style="color:var(--text-dim);margin-bottom:0.5rem;">
        28 features are extracted from each URL, capturing structural patterns
        that distinguish phishing from legitimate websites:
    </p>
    <div class="grid-2">
        <div>
            <strong style="color:var(--accent);">Length-based</strong>
            <ul style="color:var(--text-dim);font-size:0.85rem;padding-left:1.25rem;">
                <li>URL length, path length, domain length</li>
                <li>Digit ratio, special character ratio</li>
                <li>URL entropy (randomness measure)</li>
            </ul>
            <strong style="color:var(--accent2);margin-top:0.5rem;display:block;">Structural</strong>
            <ul style="color:var(--text-dim);font-size:0.85rem;padding-left:1.25rem;">
                <li>Number of dots, hyphens, slashes, @ symbols</li>
                <li>Number of subdomains, path depth</li>
                <li>Presence of IP address, port, HTTPS</li>
            </ul>
        </div>
        <div>
            <strong style="color:var(--accent);">Content-based</strong>
            <ul style="color:var(--text-dim);font-size:0.85rem;padding-left:1.25rem;">
                <li>Suspicious keywords (login, verify, secure...)</li>
                <li>URL shortener detection</li>
                <li>Double-slash redirect patterns</li>
            </ul>
            <strong style="color:var(--accent2);margin-top:0.5rem;display:block;">Domain-based</strong>
            <ul style="color:var(--text-dim);font-size:0.85rem;padding-left:1.25rem;">
                <li>TLD length, has-www flag</li>
                <li>Domain-based features</li>
            </ul>
        </div>
    </div>
</div>

<script>
document.querySelectorAll('.sortable').forEach(th => {{
    th.addEventListener('click', () => {{
        const table = th.closest('table');
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const idx = Array.from(th.parentNode.children).indexOf(th);
        const desc = th.classList.contains('sorted-asc');
        th.classList.toggle('sorted-asc', !desc);
        th.classList.toggle('sorted-desc', desc);
        rows.sort((a, b) => {{
            let va = a.children[idx].textContent.trim();
            let vb = b.children[idx].textContent.trim();
            let na = parseFloat(va), nb = parseFloat(vb);
            if (!isNaN(na) && !isNaN(nb)) {{ va = na; vb = nb; }}
            if (va < vb) return desc ? 1 : -1;
            if (va > vb) return desc ? -1 : 1;
            return 0;
        }});
        rows.forEach(r => tbody.appendChild(r));
    }});
}});
</script>

</body>
</html>"""

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  ✓ Report: {REPORT_FILE}")


if __name__ == "__main__":
    generate_report()
