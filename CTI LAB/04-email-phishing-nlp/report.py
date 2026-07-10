"""
Generate self-contained HTML report for Email Phishing Detection model comparison.
"""

import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
METRICS_JSON = RESULTS_DIR / "metrics.json"
REPORT_HTML = RESULTS_DIR / "report.html"


def plot_confusion_matrix(cm, labels, title):
    """Plot confusion matrix and return as base64 PNG."""
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=12, pad=10)
    tick_marks = range(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    thresh = cm.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot_bar_chart(results_df, metric, title, color="#2563eb"):
    """Plot horizontal bar chart for a metric and return base64 PNG."""
    df_sorted = results_df.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(df_sorted["Model"], df_sorted[metric], color=color, height=0.6)
    ax.set_xlabel(metric)
    ax.set_title(title, fontsize=12)
    for i, v in enumerate(df_sorted[metric]):
        ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.05)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot_roc_comparison(predictions_dir, results_df):
    """Plot ROC curves for all models and return base64 PNG."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_df)))

    from sklearn.metrics import roc_curve
    for i, (_, row) in enumerate(results_df.iterrows()):
        model_name = row["Model"]
        npz_path = predictions_dir / f"{model_name}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        y_true = data["y_true"]
        y_prob = data["y_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = row["ROC-AUC"]
        ax.plot(fpr, tpr, color=colors[i], lw=1.5,
                label=f"{model_name} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves Comparison")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def generate_report():
    if not METRICS_JSON.exists():
        print("No metrics.json found. Run train_phishing_model.py first.")
        return

    with open(METRICS_JSON) as f:
        metrics = json.load(f)

    rows = []
    for name, data in metrics.items():
        rows.append({
            "Model": name,
            "Family": data.get("family", ""),
            "Accuracy": data["accuracy"],
            "Precision": data["precision"],
            "Recall": data["recall"],
            "F1": data["f1"],
            "ROC-AUC": data["roc_auc"],
            "Train_Time_s": data["train_time_s"],
            "Predict_Time_s": data["predict_time_s"],
        })
    df = pd.DataFrame(rows)

    # Generate plots
    cm_images = {}
    for name, data in metrics.items():
        cm = np.array(data.get("confusion_matrix", [[0, 0], [0, 0]]))
        cm_images[name] = plot_confusion_matrix(cm, ["Legit", "Phishing"], f"{name}")

    f1_img = plot_bar_chart(df, "F1", "F1 Score by Model", "#2563eb")
    roc_img = plot_bar_chart(df, "ROC-AUC", "ROC-AUC by Model", "#059669")
    acc_img = plot_bar_chart(df, "Accuracy", "Accuracy by Model", "#d97706")

    roc_curves = plot_roc_comparison(
        Path(__file__).parent / "results" / "predictions", df)

    # Build table rows
    table_rows = []
    for _, row in df.sort_values("F1", ascending=False).iterrows():
        table_rows.append(f"""
        <tr>
            <td><strong>{row["Model"]}</strong></td>
            <td>{row["Family"]}</td>
            <td>{row["Accuracy"]:.4f}</td>
            <td>{row["Precision"]:.4f}</td>
            <td>{row["Recall"]:.4f}</td>
            <td>{row["F1"]:.4f}</td>
            <td>{row["ROC-AUC"]:.4f}</td>
            <td>{row["Train_Time_s"]:.3f}s</td>
        </tr>""")

    # Build confusion matrices section
    cm_section = ""
    for name in sorted(metrics.keys()):
        cm_section += f"""
        <div class="cm-card">
            <h3>{name}</h3>
            <img src="data:image/png;base64,{cm_images[name]}" alt="{name} CM">
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Email Phishing Detection — Model Comparison</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f8fafc; color: #1e293b; padding: 2rem; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 0.25rem; color: #0f172a; }}
    .subtitle {{ color: #64748b; margin-bottom: 2rem; font-size: 0.95rem; }}
    .summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
    .summary-card {{ background: white; border-radius: 12px; padding: 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    .summary-card .label {{ font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
    .summary-card .value {{ font-size: 1.5rem; font-weight: 700; color: #0f172a; margin-top: 0.25rem; }}
    .summary-card .value.gold {{ color: #d97706; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 2rem; }}
    th {{ background: #f1f5f9; text-align: left; padding: 0.75rem 1rem; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: #475569; cursor: pointer; }}
    th:hover {{ background: #e2e8f0; }}
    td {{ padding: 0.75rem 1rem; border-bottom: 1px solid #f1f5f9; font-size: 0.9rem; }}
    tr:hover td {{ background: #f8fafc; }}
    .best {{ color: #059669; font-weight: 600; }}
    .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }}
    .chart-card {{ background: white; border-radius: 12px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    .chart-card h2 {{ font-size: 1rem; margin-bottom: 0.75rem; color: #334155; }}
    .chart-card img {{ width: 100%; height: auto; }}
    .cm-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
    .cm-card {{ background: white; border-radius: 12px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center; }}
    .cm-card h3 {{ font-size: 0.9rem; color: #475569; margin-bottom: 0.5rem; }}
    .cm-card img {{ max-width: 100%; height: auto; }}
    .section-title {{ font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem; color: #0f172a; }}
    @media (max-width: 768px) {{ body {{ padding: 1rem; }} .charts {{ grid-template-columns: 1fr; }} .cm-grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="container">
    <h1>Email Phishing Detection</h1>
    <p class="subtitle">Machine Learning Model Comparison &mdash; Spambase Dataset &bull; {len(metrics)} models</p>

    <div class="summary-cards">
        <div class="summary-card">
            <div class="label">Best Model</div>
            <div class="value">{df.sort_values("F1", ascending=False).iloc[0]["Model"]}</div>
        </div>
        <div class="summary-card">
            <div class="label">Best F1 Score</div>
            <div class="value gold">{df["F1"].max():.4f}</div>
        </div>
        <div class="summary-card">
            <div class="label">Best Accuracy</div>
            <div class="value gold">{df["Accuracy"].max():.4f}</div>
        </div>
        <div class="summary-card">
            <div class="label">Best ROC-AUC</div>
            <div class="value gold">{df["ROC-AUC"].max():.4f}</div>
        </div>
    </div>

    <h2 class="section-title">Model Performance</h2>
    <table id="metrics-table">
        <thead>
            <tr>
                <th onclick="sortTable(0)">Model</th>
                <th onclick="sortTable(1)">Family</th>
                <th onclick="sortTable(2)">Accuracy</th>
                <th onclick="sortTable(3)">Precision</th>
                <th onclick="sortTable(4)">Recall</th>
                <th onclick="sortTable(5)">F1 Score</th>
                <th onclick="sortTable(6)">ROC-AUC</th>
                <th onclick="sortTable(7)">Train Time</th>
            </tr>
        </thead>
        <tbody>
            {''.join(table_rows)}
        </tbody>
    </table>

    <h2 class="section-title">ROC Curves</h2>
    <div class="chart-card">
        <img src="data:image/png;base64,{roc_curves}" alt="ROC Curves">
    </div>

    <h2 class="section-title">Model Comparison Charts</h2>
    <div class="charts">
        <div class="chart-card">
            <h2>F1 Score</h2>
            <img src="data:image/png;base64,{f1_img}" alt="F1 Scores">
        </div>
        <div class="chart-card">
            <h2>ROC-AUC</h2>
            <img src="data:image/png;base64,{roc_img}" alt="ROC-AUC">
        </div>
        <div class="chart-card">
            <h2>Accuracy</h2>
            <img src="data:image/png;base64,{acc_img}" alt="Accuracy">
        </div>
    </div>

    <h2 class="section-title">Confusion Matrices</h2>
    <div class="cm-grid">
        {cm_section}
    </div>
</div>

<script>
function sortTable(col) {{
    const table = document.getElementById("metrics-table");
    const tbody = table.querySelector("tbody");
    const rows = Array.from(tbody.querySelectorAll("tr"));
    const ascending = table.getAttribute("data-sort-col") !== String(col) || table.getAttribute("data-sort-dir") === "desc";
    rows.sort((a, b) => {{
        let va = a.children[col].textContent.trim();
        let vb = b.children[col].textContent.trim();
        let na = parseFloat(va), nb = parseFloat(vb);
        if (!isNaN(na) && !isNaN(nb)) {{ va = na; vb = nb; }}
        else {{ va = va.toLowerCase(); vb = vb.toLowerCase(); }}
        if (va < vb) return ascending ? -1 : 1;
        if (va > vb) return ascending ? 1 : -1;
        return 0;
    }});
    rows.forEach(r => tbody.appendChild(r));
    table.setAttribute("data-sort-col", col);
    table.setAttribute("data-sort-dir", ascending ? "asc" : "desc");
}}
</script>
</body>
</html>"""

    with open(REPORT_HTML, "w") as f:
        f.write(html)
    print(f"  Report saved to {REPORT_HTML}")


if __name__ == "__main__":
    generate_report()
