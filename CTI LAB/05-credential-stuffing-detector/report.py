"""Generate a standalone evidence-led HTML report for CTI Lab 05."""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).parent / "results"
METRICS_PATH = RESULTS_DIR / "metrics.json"
REPORT_PATH = RESULTS_DIR / "report.html"


def plot_metric(results: pd.DataFrame, metric: str, color: str) -> str:
    """Render one detector comparison as an embedded PNG."""
    ordered = results.sort_values(metric, ascending=True)
    fig, axis = plt.subplots(figsize=(5.4, 3.1))
    axis.barh(ordered["Detector"], ordered[metric], color=color, height=0.56)
    axis.set_xlim(0, 1.05)
    axis.set_xlabel(metric)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", alpha=0.18)
    axis.set_axisbelow(True)
    for index, value in enumerate(ordered[metric]):
        axis.text(value + 0.015, index, f"{value:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, transparent=True)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def plot_confusion_matrix(matrix: list[list[int]], name: str) -> str:
    """Render a labelled normal-versus-attack confusion matrix."""
    values = np.array(matrix)
    fig, axis = plt.subplots(figsize=(3.1, 2.8))
    image = axis.imshow(values, interpolation="nearest", cmap="YlOrBr")
    axis.set_title(name, fontsize=10, pad=8)
    axis.set_xticks([0, 1], ["Normal", "Attack"])
    axis.set_yticks([0, 1], ["Normal", "Attack"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    threshold = values.max() / 2 if values.size else 0
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "#1f2937",
            )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, transparent=True)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def dataset_cards(datasets: list[dict[str, object]]) -> str:
    """Build source cards from the training manifest."""
    cards: list[str] = []
    for dataset in datasets:
        counts = dataset.get("counts", {})
        count_text = ", ".join(f"{key.replace('_', ' ')}: {value:,}" for key, value in counts.items()) or "No local events imported"
        status = str(dataset.get("status", "used in this run")).replace("_", " ")
        cards.append(
            f"""
            <article class="source-card">
              <div class="source-topline"><span class="status">{html.escape(status)}</span><span>{html.escape(str(dataset['role']))}</span></div>
              <h3>{html.escape(str(dataset['name']))}</h3>
              <p class="count">{html.escape(count_text)}</p>
              <p>{html.escape(str(dataset['limitations']))}</p>
              <a href="{html.escape(str(dataset['official_url']), quote=True)}">Official dataset page</a>
            </article>"""
        )
    return "".join(cards)


def generate_report() -> None:
    """Generate the report from the current real-source metrics payload."""
    if not METRICS_PATH.exists():
        print("No metrics found. Run 'python run.py train' first.")
        return

    payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    detectors = payload.get("detectors", {})
    if not detectors:
        print("Metrics do not contain detector results. Run 'python run.py train' again.")
        return

    rows = [
        {
            "Detector": name,
            "Accuracy": values["accuracy"],
            "Precision": values["precision"],
            "Recall": values["recall"],
            "F1": values["f1"],
            "FPR": values["false_positive_rate"],
            "Detections": values["detection_count"],
        }
        for name, values in detectors.items()
    ]
    results = pd.DataFrame(rows)
    best = results.sort_values("F1", ascending=False).iloc[0]
    f1_chart = plot_metric(results, "F1", "#0f766e")
    precision_chart = plot_metric(results, "Precision", "#2563eb")
    recall_chart = plot_metric(results, "Recall", "#b45309")

    table_rows = "".join(
        f"""<tr><th scope="row">{html.escape(row.Detector)}</th><td>{row.Accuracy:.4f}</td>
        <td>{row.Precision:.4f}</td><td>{row.Recall:.4f}</td><td>{row.F1:.4f}</td>
        <td>{row.FPR:.4f}</td><td>{int(row.Detections):,}</td></tr>"""
        for row in results.sort_values("F1", ascending=False).itertuples()
    )
    matrices = "".join(
        f"""<figure class="matrix"><img src="data:image/png;base64,{plot_confusion_matrix(values['confusion_matrix'], name)}" alt="{html.escape(name)} confusion matrix"></figure>"""
        for name, values in sorted(detectors.items())
    )
    evaluation = payload["evaluation"]
    datasets = payload["dataset"]["datasets"]
    html_document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CTI 05 | Credential Stuffing Detection Report</title>
<style>
  :root {{ --ink:#15202b; --muted:#596775; --paper:#fbfaf7; --line:#d8dedc; --teal:#0f766e; --amber:#b45309; --blue:#2563eb; --soft-teal:#e4f1ee; --soft-amber:#fff1dd; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--paper); color:var(--ink); font-family:Georgia, "Times New Roman", serif; line-height:1.5; }}
  .shell {{ width:min(1120px, calc(100% - 36px)); margin:0 auto; padding:42px 0 60px; }}
  .masthead {{ border-top:8px solid var(--teal); border-bottom:1px solid var(--line); padding:20px 0 24px; display:grid; grid-template-columns:1fr auto; gap:24px; align-items:end; }}
  .eyebrow {{ margin:0 0 8px; font:700 12px/1.2 ui-monospace, Consolas, monospace; color:var(--teal); letter-spacing:.08em; text-transform:uppercase; }}
  h1 {{ font-size:clamp(30px, 5vw, 52px); line-height:1.02; margin:0; letter-spacing:0; }}
  .subtitle {{ color:var(--muted); font-size:17px; max-width:680px; margin:12px 0 0; }}
  .run-note {{ font:12px/1.4 ui-monospace, Consolas, monospace; color:var(--muted); text-align:right; max-width:250px; }}
  h2 {{ font-size:22px; margin:42px 0 12px; letter-spacing:0; }}
  h3 {{ margin:10px 0 4px; font-size:18px; }}
  .cards {{ display:grid; grid-template-columns:repeat(4, 1fr); border:1px solid var(--line); }}
  .metric {{ min-height:118px; padding:17px; border-right:1px solid var(--line); background:#fff; }}
  .metric:last-child {{ border-right:0; }}
  .metric-label {{ display:block; color:var(--muted); font:700 11px/1.2 ui-monospace, Consolas, monospace; letter-spacing:.06em; text-transform:uppercase; }}
  .metric-value {{ display:block; margin-top:10px; font-size:26px; font-weight:700; }}
  .lede {{ padding:16px 18px; border-left:4px solid var(--amber); background:var(--soft-amber); color:#633a0b; }}
  .sources {{ display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:14px; }}
  .source-card {{ padding:18px; border:1px solid var(--line); background:#fff; }}
  .source-card p {{ margin:8px 0; color:var(--muted); font-size:14px; }}
  .source-card a {{ color:var(--teal); font-weight:bold; }}
  .source-topline {{ display:flex; justify-content:space-between; gap:12px; color:var(--muted); font:11px/1.3 ui-monospace, Consolas, monospace; text-transform:uppercase; }}
  .status {{ color:var(--teal); font-weight:700; }} .count {{ color:var(--ink)!important; font-weight:bold; }}
  table {{ border-collapse:collapse; width:100%; background:#fff; border:1px solid var(--line); font-size:14px; }}
  th, td {{ padding:12px 10px; border-bottom:1px solid var(--line); text-align:right; }} th:first-child, td:first-child {{ text-align:left; }} thead th {{ background:var(--soft-teal); font:700 11px/1.2 ui-monospace, Consolas, monospace; text-transform:uppercase; color:#315550; }} tbody tr:last-child th, tbody tr:last-child td {{ border-bottom:0; }}
  .charts {{ display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:14px; }} .chart {{ margin:0; padding:12px; border:1px solid var(--line); background:#fff; }} .chart figcaption {{ font:700 14px/1.2 ui-monospace, Consolas, monospace; color:var(--muted); }} .chart img {{ width:100%; display:block; }}
  .matrices {{ display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:10px; }} .matrix {{ margin:0; padding:8px; border:1px solid var(--line); background:#fff; }} .matrix img {{ display:block; width:100%; }}
  .method {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }} .method div {{ padding:18px; border-top:2px solid var(--teal); background:#fff; }} .method p {{ margin:8px 0 0; color:var(--muted); }}
  @media (max-width:760px) {{ .masthead {{ grid-template-columns:1fr; }} .run-note {{ text-align:left; }} .cards, .charts, .matrices {{ grid-template-columns:1fr 1fr; }} .sources, .method {{ grid-template-columns:1fr; }} .metric:nth-child(2) {{ border-right:0; }} .metric {{ border-bottom:1px solid var(--line); }} }}
  @media (max-width:480px) {{ .shell {{ width:min(100% - 24px, 1120px); padding-top:24px; }} .cards, .charts, .matrices {{ grid-template-columns:1fr; }} .metric {{ border-right:0; }} table {{ font-size:12px; }} th, td {{ padding:9px 5px; }} }}
</style>
</head>
<body><main class="shell">
  <header class="masthead"><div><p class="eyebrow">CTI Lab Exercise 05 / Evidence Report</p><h1>Credential Stuffing Detection</h1><p class="subtitle">Model comparison on a labelled public-security benchmark, with source provenance and limits shown alongside the results.</p></div><p class="run-note">{html.escape(str(evaluation['split']))}<br>{int(evaluation['sample_count']):,} prepared events</p></header>
  <section><h2>At a glance</h2><div class="cards"><div class="metric"><span class="metric-label">Best detector</span><strong class="metric-value">{html.escape(str(best.Detector))}</strong></div><div class="metric"><span class="metric-label">Best F1</span><strong class="metric-value">{best.F1:.4f}</strong></div><div class="metric"><span class="metric-label">Labelled brute-force</span><strong class="metric-value">{int(evaluation['attack_count']):,}</strong></div><div class="metric"><span class="metric-label">Models compared</span><strong class="metric-value">{len(results)}</strong></div></div></section>
  <section><h2>Interpretation boundary</h2><p class="lede">{html.escape(str(payload['dataset']['training_note']))}</p></section>
  <section><h2>Dataset provenance</h2><div class="sources">{dataset_cards(datasets)}</div></section>
  <section><h2>Detection performance</h2><table><thead><tr><th>Detector</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>FPR</th><th>Detections</th></tr></thead><tbody>{table_rows}</tbody></table></section>
  <section><h2>Comparison</h2><div class="charts"><figure class="chart"><figcaption>F1 score</figcaption><img src="data:image/png;base64,{f1_chart}" alt="F1 score by detector"></figure><figure class="chart"><figcaption>Precision</figcaption><img src="data:image/png;base64,{precision_chart}" alt="Precision by detector"></figure><figure class="chart"><figcaption>Recall</figcaption><img src="data:image/png;base64,{recall_chart}" alt="Recall by detector"></figure></div></section>
  <section><h2>Confusion matrices</h2><div class="matrices">{matrices}</div></section>
  <section><h2>Method notes</h2><div class="method"><div><h3>Feature mapping</h3><p>{html.escape(str(evaluation['feature_note']))}</p></div><div><h3>Deployment note</h3><p>The local Flask form remains a lab decision aid. Feed it actual authentication telemetry in deployment; do not use these benchmark scores as a production threshold.</p></div></div></section>
</main></body></html>"""
    REPORT_PATH.write_text(html_document, encoding="utf-8")
    print(f"  Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    generate_report()
