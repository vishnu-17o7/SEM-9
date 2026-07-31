"""Add a provenance badge to a generated CTI HTML report."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LABELS = {
    "real": "REAL DATA",
    "synthetic_fallback": "SYNTHETIC FALLBACK",
    "legacy": "LEGACY INVALID",
}


def add_badge(project: Path) -> None:
    """Inject or refresh a visible dataset-provenance badge."""
    manifest = json.loads((project / "results" / "evaluation_manifest.json").read_text(encoding="utf-8"))
    report_path = project / "results" / "report.html"
    try:
        html = report_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        html = report_path.read_text(encoding="cp1252")
    label = LABELS.get(manifest["dataset"]["mode"], manifest["dataset"]["mode"].upper())
    badge = (
        f'<div data-evaluation-badge="true" style="position:relative;margin:0 auto 1rem;'
        f'padding:.65rem 1rem;border-radius:999px;background:#0f172a;color:#f8fafc;'
        f'border:2px solid #14b8a6;font:700 0.82rem/1.2 system-ui;letter-spacing:.08em;'
        f'text-align:center;max-width:28rem">{label}</div>'
    )
    html = re.sub(r'<div data-evaluation-badge="true".*?</div>', badge, html, count=1, flags=re.DOTALL)
    if 'data-evaluation-badge="true"' not in html:
        html = re.sub(r"(<body[^>]*>)", r"\1" + badge, html, count=1, flags=re.IGNORECASE)
    report_path.write_text(html, encoding="utf-8")
    print(f"  Badge: {label} -> {report_path}")


def main() -> int:
    """Inject a provenance badge into one project report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()
    add_badge(args.project)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
