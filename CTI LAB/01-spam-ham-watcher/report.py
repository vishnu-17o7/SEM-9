"""Render a compact evidence report for Project 01."""
from __future__ import annotations

import json
from pathlib import Path


RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    """Write the HTML report from persisted metrics and evaluation metadata."""
    metrics = json.loads((RESULTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((RESULTS_DIR / "evaluation_manifest.json").read_text(encoding="utf-8"))
    mode = manifest["dataset"]["mode"].replace("_", " ").upper()
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Spam/Ham Evidence</title><style>body{{font-family:system-ui;max-width:900px;margin:40px auto;line-height:1.5}}.badge{{padding:5px 10px;background:#0f766e;color:#fff;border-radius:12px}}pre{{background:#f3f4f6;padding:16px;overflow:auto}}</style></head><body><h1>Spam/Ham Watcher</h1><p><span class='badge'>{mode}</span></p><p>{manifest['dataset']['limitations']}</p><h2>Locked evaluation</h2><pre>{json.dumps(manifest['split'], indent=2)}</pre><h2>Metrics</h2><pre>{json.dumps(metrics, indent=2)}</pre></body></html>"""
    (RESULTS_DIR / "report.html").write_text(html, encoding="utf-8")
    print(RESULTS_DIR / "report.html")


if __name__ == "__main__":
    main()
