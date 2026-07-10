"""SQLite layer for the spam/ham classification dashboard.

Stores one row per email processed by gmail_watcher.py and provides
query helpers used by dashboard.py.
"""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.getenv("DB_PATH", "data/classifications.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS classifications (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    received_at  TEXT    NOT NULL,
    sender       TEXT,
    subject      TEXT,
    label        TEXT    NOT NULL CHECK (label IN ('spam', 'ham')),
    probability  REAL    NOT NULL,
    body_preview TEXT
);
CREATE INDEX IF NOT EXISTS idx_received_at ON classifications(received_at);
CREATE INDEX IF NOT EXISTS idx_label       ON classifications(label);
"""


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def _conn():
    c = _connect(DB_PATH)
    try:
        yield c
    finally:
        c.close()


def init_db(path: Path | str | None = None) -> None:
    target = Path(path) if path else DB_PATH
    with _connect(target) as c:
        c.executescript(_SCHEMA)


def record_classification(
    sender: str,
    subject: str,
    label: str,
    probability: float,
    body_preview: str = "",
    received_at: str | None = None,
) -> int:
    if label not in ("spam", "ham"):
        raise ValueError(f"label must be 'spam' or 'ham', got {label!r}")
    ts = received_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _conn() as c:
        cur = c.execute(
            """
            INSERT INTO classifications
                (received_at, sender, subject, label, probability, body_preview)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ts, sender, subject, label, float(probability), body_preview),
        )
        return cur.lastrowid


def fetch_recent(limit: int = 100, since: str | None = None) -> list[sqlite3.Row]:
    sql = "SELECT * FROM classifications"
    params: tuple = ()
    if since:
        sql += " WHERE received_at >= ?"
        params = (since,)
    sql += " ORDER BY id DESC LIMIT ?"
    params = params + (int(limit),)
    with _conn() as c:
        return list(c.execute(sql, params))


def fetch_summary(since: str | None = None) -> dict:
    where = ""
    params: tuple = ()
    if since:
        where = " WHERE received_at >= ?"
        params = (since,)

    with _conn() as c:
        totals = c.execute(
            f"SELECT label, COUNT(*) AS n FROM classifications{where} GROUP BY label",
            params,
        ).fetchall()
        spam = 0
        ham = 0
        for row in totals:
            if row["label"] == "spam":
                spam = row["n"]
            elif row["label"] == "ham":
                ham = row["n"]
        total = spam + ham
        ratio = (spam / total) if total else 0.0

        daily = c.execute(
            f"""
            SELECT substr(received_at, 1, 10) AS day, label, COUNT(*) AS n
            FROM classifications
            {where}
            GROUP BY day, label
            ORDER BY day
            """,
            params,
        ).fetchall()

    daily_dict: dict[str, dict[str, int]] = {}
    for row in daily:
        daily_dict.setdefault(row["day"], {"spam": 0, "ham": 0})
        daily_dict[row["day"]][row["label"]] = row["n"]

    return {
        "total": total,
        "spam": spam,
        "ham": ham,
        "spam_ratio": ratio,
        "daily": daily_dict,
    }


def fetch_probability_buckets(since: str | None = None) -> list[tuple[str, int, int]]:
    """Return list of (bucket_label, spam_count, ham_count) for 10 bins."""
    bins = [(i / 10, (i + 1) / 10) for i in range(10)]
    out = []
    with _conn() as c:
        for lo, hi in bins:
            row = c.execute(
                """
                SELECT label, COUNT(*) AS n
                FROM classifications
                WHERE probability >= ? AND probability < ?
                GROUP BY label
                """,
                (lo, hi),
            ).fetchall()
            spam = 0
            ham = 0
            for r in row:
                if r["label"] == "spam":
                    spam = r["n"]
                elif r["label"] == "ham":
                    ham = r["n"]
            label = f"{lo:.1f}-{hi:.1f}"
            out.append((label, spam, ham))
    return out
