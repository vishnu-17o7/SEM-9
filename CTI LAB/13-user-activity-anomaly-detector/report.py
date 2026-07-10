"""Generate HTML report for user activity anomaly detection."""
import base64, io, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
RESULTS_DIR = Path(__file__).parent / "results"
METRICS_JSON = RESULTS_DIR / "metrics.json"
REPORT_HTML = RESULTS_DIR / "report.html"

def plot_cm(cm, labels, title):
    fig, ax = plt.subplots(figsize=(3.5,3))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.YlOrRd)
    ax.set_title(title, fontsize=11, pad=8); ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    thresh = cm.max()/2
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j,i,format(cm[i,j],"d"),ha="center",va="center",color="white" if cm[i,j]>thresh else "black")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted"); plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def plot_bar(df, metric, title, color="#2563eb"):
    df_s = df.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(5,3.5))
    ax.barh(df_s["Detector"], df_s[metric], color=color, height=0.5)
    ax.set_xlabel(metric); ax.set_title(title, fontsize=11)
    for i, v in enumerate(df_s[metric]): ax.text(v+0.01,i,f"{v:.4f}",va="center",fontsize=9)
    ax.set_xlim(0,1.05); plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def generate_report():
    if not METRICS_JSON.exists(): print("Run detect_anomalies.py first."); return
    with open(METRICS_JSON) as f: metrics = json.load(f)
    rows = [{"Detector":n,"Accuracy":d["accuracy"],"Precision":d["precision"],"Recall":d["recall"],"F1":d["f1"],"ROC-AUC":d["roc_auc"],"Anomalies":d["anomaly_count"]} for n,d in metrics.items()]
    df = pd.DataFrame(rows)
    cm_imgs = {n: plot_cm(np.array(d.get("confusion_matrix",[[0,0],[0,0]])),["Normal","Anomaly"],n) for n,d in metrics.items()}
    f1_img = plot_bar(df,"F1","F1 Score"); prec_img = plot_bar(df,"Precision","Precision","#059669")
    trows = "".join(f'<tr><td><strong>{r["Detector"]}</strong></td><td>{r["Accuracy"]:.4f}</td><td>{r["Precision"]:.4f}</td><td>{r["Recall"]:.4f}</td><td>{r["F1"]:.4f}</td><td>{r["ROC-AUC"]:.4f}</td><td>{int(r["Anomalies"])}</td></tr>' for _,r in df.sort_values("F1",ascending=False).iterrows())
    cm_sec = "".join(f'<div class="cm-card"><h3>{n}</h3><img src="data:image/png;base64,{v}"></div>' for n,v in sorted(cm_imgs.items()))
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>User Activity Anomaly Detection</title><style>*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#f8fafc;color:#1e293b;padding:2rem}}
.container{{max-width:1100px;margin:0 auto}}h1{{font-size:1.6rem;font-weight:700;margin-bottom:.25rem;color:#0f172a}}
.subtitle{{color:#64748b;margin-bottom:2rem;font-size:.85rem}}
.summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin-bottom:2rem}}
.summary-card{{background:white;border-radius:12px;padding:1.25rem;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.summary-card .label{{font-size:.75rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em}}
.summary-card .value{{font-size:1.3rem;font-weight:700;color:#0f172a;margin-top:.25rem}}
table{{width:100%;border-collapse:collapse;background:white;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.08);margin-bottom:2rem}}
th{{background:#f1f5f9;text-align:left;padding:.7rem 1rem;font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;color:#475569}}
td{{padding:.7rem 1rem;border-bottom:1px solid #f1f5f9;font-size:.85rem}}tr:hover td{{background:#f8fafc}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1.5rem;margin-bottom:2rem}}
.chart-card{{background:white;border-radius:12px;padding:1rem;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.chart-card h2{{font-size:.9rem;margin-bottom:.5rem;color:#334155}}.chart-card img{{width:100%;height:auto}}
.cm-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem;margin-bottom:2rem}}
.cm-card{{background:white;border-radius:12px;padding:1rem;box-shadow:0 1px 3px rgba(0,0,0,.08);text-align:center}}
.cm-card h3{{font-size:.85rem;color:#475569;margin-bottom:.5rem}}.cm-card img{{max-width:100%}}
.section-title{{font-size:1.1rem;font-weight:600;margin-bottom:1rem;margin-top:1.5rem;color:#0f172a}}</style></head><body>
<div class="container">
<h1>User Activity Anomaly Detection</h1>
<p class="subtitle">10 Command-Line Session Features — 3 Detectors</p>
<div class="summary-cards"><div class="summary-card"><div class="label">Best F1</div><div class="value">{df["F1"].max():.4f}</div></div><div class="summary-card"><div class="label">Best ROC-AUC</div><div class="value">{df["ROC-AUC"].max():.4f}</div></div></div>
<table><thead><tr><th>Detector</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>Anomalies</th></tr></thead><tbody>{trows}</tbody></table>
<div class="charts"><div class="chart-card"><h2>F1 Score</h2><img src="data:image/png;base64,{f1_img}"></div><div class="chart-card"><h2>Precision</h2><img src="data:image/png;base64,{prec_img}"></div></div>
<div class="cm-grid">{cm_sec}</div>
</div></body></html>"""
    with open(REPORT_HTML,"w") as f: f.write(html)
    print(f"  Report saved to {REPORT_HTML}")

if __name__ == "__main__":
    generate_report()
