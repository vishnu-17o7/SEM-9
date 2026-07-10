"""
Generate HTML report for credential stuffing detection results.
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
DATA_DIR = Path(__file__).parent / "data"


def plot_confusion_matrix(cm, labels, title):
    fig, ax = plt.subplots(figsize=(3.5, 3))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Oranges)
    ax.set_title(title, fontsize=11, pad=8)
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


def plot_bar_chart(results_df, metric, title, color="#dc2626"):
    df_sorted = results_df.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.barh(df_sorted["Detector"], df_sorted[metric], color=color, height=0.5)
    ax.set_xlabel(metric)
    ax.set_title(title, fontsize=11)
    for i, v in enumerate(df_sorted[metric]):
        ax.text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def generate_report():
    if not METRICS_JSON.exists():
        print("No metrics found. Run detect_stuffing.py first.")
        return

    with open(METRICS_JSON) as f:
        metrics = json.load(f)

    rows = []
    for name, data in metrics.items():
        rows.append({
            "Detector": name,
            "Accuracy": data["accuracy"],
            "Precision": data["precision"],
            "Recall": data["recall"],
            "F1": data["f1"],
            "FPR": data["false_positive_rate"],
            "Detections": data["detection_count"],
        })
    df = pd.DataFrame(rows)

    cm_images = {}
    for name, data in metrics.items():
        cm = np.array(data.get("confusion_matrix", [[0, 0], [0, 0]]))
        cm_images[name] = plot_confusion_matrix(cm, ["Normal", "Attack"], f"{name}")

    f1_img = plot_bar_chart(df, "F1", "F1 Score by Detector", "#2563eb")
    prec_img = plot_bar_chart(df, "Precision", "Precision by Detector", "#059669")
    recall_img = plot_bar_chart(df, "Recall", "Recall by Detector", "#dc2626")

    table_rows = []
    for _, row in df.sort_values("F1", ascending=False).iterrows():
        table_rows.append(f"""
        <tr>
            <td><strong>{row["Detector"]}</strong></td>
            <td>{row["Accuracy"]:.4f}</td>
            <td>{row["Precision"]:.4f}</td>
            <td>{row["Recall"]:.4f}</td>
            <td>{row["F1"]:.4f}</td>
            <td>{row["FPR"]:.4f}</td>
            <td>{int(row["Detections"])}</td>
        </tr>""")

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
<title>Credential Stuffing Detection — Model Comparison</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f8fafc; color: #1e293b; padding: 2rem; }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    h1 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: 0.25rem; color: #0f172a; }}
    .subtitle {{ color: #64748b; margin-bottom: 2rem; font-size: 0.85rem; }}
    .summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
    .summary-card {{ background: white; border-radius: 12px; padding: 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    .summary-card .label {{ font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
    .summary-card .value {{ font-size: 1.3rem; font-weight: 700; color: #0f172a; margin-top: 0.25rem; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 2rem; }}
    th {{ background: #f1f5f9; text-align: left; padding: 0.7rem 1rem; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #475569; cursor: pointer; }}
    td {{ padding: 0.7rem 1rem; border-bottom: 1px solid #f1f5f9; font-size: 0.85rem; }}
    tr:hover td {{ background: #f8fafc; }}
    .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }}
    .chart-card {{ background: white; border-radius: 12px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    .chart-card h2 {{ font-size: 0.9rem; margin-bottom: 0.5rem; color: #334155; }}
    .chart-card img {{ width: 100%; height: auto; }}
    .cm-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }}
    .cm-card {{ background: white; border-radius: 12px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center; }}
    .cm-card h3 {{ font-size: 0.85rem; color: #475569; margin-bottom: 0.5rem; }}
    .cm-card img {{ max-width: 100%; height: auto; }}
    .section-title {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; margin-top: 1.5rem; color: #0f172a; }}
</style>
</head>
<body>
<div class="container">
    <h1>Credential Stuffing Detection</h1>
    <p class="subtitle">Anomaly Detection on Synthetic Login Logs &bull; {len(metrics)} detectors</p>

    <div class="summary-cards">
        <div class="summary-card">
            <div class="label">Best Detector</div>
            <div class="value">{df.sort_values("F1", ascending=False).iloc[0]["Detector"]}</div>
        </div>
        <div class="summary-card">
            <div class="label">Best F1</div>
            <div class="value">{df["F1"].max():.4f}</div>
        </div>
        <div class="summary-card">
            <div class="label">Best Precision</div>
            <div class="value">{df["Precision"].max():.4f}</div>
        </div>
        <div class="summary-card">
            <div class="label">Best Recall</div>
            <div class="value">{df["Recall"].max():.4f}</div>
        </div>
    </div>

    <h2 class="section-title">Detection Performance</h2>
    <table>
        <thead><tr>
            <th>Detector</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>FPR</th><th>Detections</th>
        </tr></thead>
        <tbody>{''.join(table_rows)}</tbody>
    </table>

    <h2 class="section-title">Comparison Charts</h2>
    <div class="charts">
        <div class="chart-card"><h2>F1 Score</h2><img src="data:image/png;base64,{f1_img}"></div>
        <div class="chart-card"><h2>Precision</h2><img src="data:image/png;base64,{prec_img}"></div>
        <div class="chart-card"><h2>Recall</h2><img src="data:image/png;base64,{recall_img}"></div>
    </div>

    <h2 class="section-title">Confusion Matrices</h2>
    <div class="cm-grid">{cm_section}</div>
</div>
</body>
</html>"""

    with open(REPORT_HTML, "w") as f:
        f.write(html)
    print(f"  Report saved to {REPORT_HTML}")


if __name__ == "__main__":
    generate_report()
