"""
CLI to query the Threat Intelligence Repository.

Usage:
    python cli_query.py --type ip
    python cli_query.py --severity critical
    python cli_query.py --search "185.130"
    python cli_query.py --stats
    python cli_query.py --add "ip" "1.2.3.4" --source manual --severity high
"""

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import hashlib
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "threat_intel.db"


def get_conn():
    if not DB_PATH.exists():
        print("Database not found. Run build_repository.py first.")
        sys.exit(1)
    return sqlite3.connect(str(DB_PATH))


def main():
    parser = argparse.ArgumentParser(description="Threat Intelligence Repository CLI")
    parser.add_argument("--type", help="IoC type (ip, domain, url, hash, cve)")
    parser.add_argument("--severity", default="medium", help="Severity (low, medium, high, critical)")
    parser.add_argument("--search", help="Search keyword in value or description")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--add", nargs=2, metavar=("TYPE", "VALUE"), help="Add a new IoC")
    parser.add_argument("--source", default="manual", help="Source for --add")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence for --add")
    parser.add_argument("--tags", default="", help="Tags for --add")
    parser.add_argument("--limit", type=int, default=30, help="Result limit")
    args = parser.parse_args()

    conn = get_conn()

    if args.stats:
        cursor = conn.execute("""
            SELECT ioc_type, COUNT(*), ROUND(AVG(confidence),2), ROUND(AVG(CASE severity WHEN 'critical' THEN 4 WHEN 'high' THEN 3 WHEN 'medium' THEN 2 WHEN 'low' THEN 1 END),2)
            FROM indicators GROUP BY ioc_type ORDER BY COUNT(*) DESC
        """)
        print(f"\n  {'Type':15s} {'Count':8s} {'Avg Conf':10s} {'Avg Severity':12s}")
        print(f"  {'─' * 45}")
        for row in cursor:
            sev_map = {4: "critical", 3: "high", 2: "medium", 1: "low"}
            sev_label = sev_map.get(int(row[3]), "unknown")
            print(f"  {row[0]:15s} {row[1]:8d} {row[2]:.2f}      {sev_label}")
        total = conn.execute("SELECT COUNT(*) FROM indicators").fetchone()[0]
        print(f"  {'─' * 45}")
        print(f"  {'TOTAL':15s} {total:8d}")
        conn.close()
        return

    if args.add:
        ioc_type, value = args.add
        ioc_id = hashlib.md5(f"{ioc_type}:{value}".encode()).hexdigest()
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            conn.execute("""
                INSERT OR IGNORE INTO indicators (id, ioc_type, value, source, confidence, severity, tags, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ioc_id, ioc_type, value, args.source, args.confidence, args.severity, args.tags, now, now))
            conn.commit()
            print(f"  Added {ioc_type}: {value}")
        except Exception as e:
            print(f"  Error: {e}")
        conn.close()
        return

    # Query
    query = "SELECT ioc_type, value, source, confidence, severity, tags, description, first_seen, reference_url FROM indicators WHERE 1=1"
    params = []
    if args.type:
        query += " AND ioc_type = ?"; params.append(args.type)
    if args.severity:
        query += " AND severity = ?"; params.append(args.severity)
    if args.search:
        query += " AND (value LIKE ? OR description LIKE ?)"; params.append(f"%{args.search}%"); params.append(f"%{args.search}%")
    query += f" ORDER BY first_seen DESC LIMIT {args.limit}"

    df = pd.read_sql(query, conn, params=params)
    if len(df) == 0:
        print("  No matching indicators found.")
    else:
        print(f"\n  Found {len(df)} indicators:")
        for _, row in df.iterrows():
            print(f"  [{row['ioc_type']:6s}] {row['value']:40s}  (src: {row['source']:12s} sev: {row['severity']:8s} conf: {row['confidence']:.2f})")
    conn.close()


if __name__ == "__main__":
    main()
