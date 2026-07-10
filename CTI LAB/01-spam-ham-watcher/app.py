"""FastAPI dashboard for the spam/ham IMAP watcher.

Run:
    uvicorn app:app --reload --port 8000

Then open http://localhost:8000/
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import db

app = FastAPI(title="Spam/Ham Watcher Dashboard")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _window_to_since(window: str) -> str | None:
    now = datetime.now(timezone.utc)
    return {
        "1h": (now - timedelta(hours=1)).isoformat(timespec="seconds"),
        "24h": (now - timedelta(hours=24)).isoformat(timespec="seconds"),
        "7d": (now - timedelta(days=7)).isoformat(timespec="seconds"),
        "30d": (now - timedelta(days=30)).isoformat(timespec="seconds"),
        "all": None,
    }.get(window, None)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return html


@app.get("/api/data")
def api_data(
    window: str = Query("7d", pattern="^(1h|24h|7d|30d|all)$"),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    since = _window_to_since(window)

    summary = db.fetch_summary(since)
    buckets = db.fetch_probability_buckets(since)
    recent = [dict(r) for r in db.fetch_recent(limit=limit, since=since)]

    return {
        "window": window,
        "total": summary["total"],
        "spam": summary["spam"],
        "ham": summary["ham"],
        "spam_ratio": summary["spam_ratio"],
        "daily": [
            {"day": d, "spam": v["spam"], "ham": v["ham"]}
            for d, v in sorted(summary["daily"].items())
        ],
        "buckets": [
            {"bucket": b, "spam": s, "ham": h} for (b, s, h) in buckets
        ],
        "recent": recent,
        "db_path": str(db.DB_PATH),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
