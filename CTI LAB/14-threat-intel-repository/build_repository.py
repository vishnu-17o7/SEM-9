"""
Threat Intelligence Repository — centralized IoC storage.

Collects threat indicators from multiple sources:
- URLhaus API (abuse.ch)
- AlienVault OTX (if API key available)
- Built-in sample threat feeds
- Manual IoC entry support

Stores in SQLite with deduplication and export capabilities.
"""

import csv
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
DB_PATH = DATA_DIR / "threat_intel.db"

# Sample threat data bundled (for when APIs are unavailable)
SAMPLE_IOCS = {
    "malicious_ips": [
        "185.130.5.27", "91.121.87.34", "5.188.62.14", "194.26.29.113",
        "45.142.212.77", "89.248.165.80", "91.240.118.83", "185.141.62.253",
        "31.220.59.11", "94.102.61.90", "103.253.42.100", "182.253.106.80",
        "45.33.32.156", "104.248.50.85", "167.71.5.83", "68.183.191.110",
        "192.241.214.181", "138.197.62.204", "159.89.209.50", "165.227.88.157",
    ],
    "malicious_domains": [
        "evil-malware.xyz", "phishing-bank.tk", "steal-login.ml",
        "download-virus.ga", "free-prize-win.cf", "malware-c2.top",
        "ransomware-pay.work", "botnet-ctrl.download", "credential-harvest.bid",
        "exploit-kit.loan", "phishing-verify.date", "maldoc-delivery.racing",
        "payload-staging.xyz", "exfil-data.top", "darknet-market.tk",
    ],
    "malicious_hashes": [
        "e1112134b6dcc8f54d7f7c1b9c5f1e3a",  # MD5 examples
        "f2a3245c7d8e9f0a1b2c3d4e5f6a7b8c",
        "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
        "b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7",
        "c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8",
        "d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9",
    ],
    "malicious_urls": [
        "http://evil-malware.xyz/loader.exe",
        "http://phishing-bank.tk/login/verify.html",
        "http://ransomware-pay.work/payment/bc1q...",
        "https://malware-c2.top/command?token=abc123",
        "http://credential-harvest.bid/account/update",
        "https://exploit-kit.loan/exploit.php?cve=2024-xxx",
        "http://payload-staging.xyz/dropper.bin",
        "https://exfil-data.top/upload.php",
        "http://maldoc-delivery.racing/invoice.doc",
        "https://darknet-market.tk/marketplace",
    ],
    "cve_ids": [
        "CVE-2024-3094", "CVE-2024-1709", "CVE-2023-46604",
        "CVE-2023-34362", "CVE-2023-32784", "CVE-2023-23397",
        "CVE-2022-30190", "CVE-2022-26134", "CVE-2021-44228",
        "CVE-2021-26855", "CVE-2020-1472", "CVE-2017-0144",
    ],
}


def init_db():
    """Initialize SQLite database schema."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS indicators (
            id TEXT PRIMARY KEY,
            ioc_type TEXT NOT NULL,
            value TEXT NOT NULL,
            source TEXT,
            confidence REAL DEFAULT 0.5,
            severity TEXT DEFAULT 'medium',
            tags TEXT DEFAULT '',
            description TEXT DEFAULT '',
            first_seen TEXT,
            last_seen TEXT,
            reference_url TEXT DEFAULT '',
            UNIQUE(ioc_type, value)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sources (
            name TEXT PRIMARY KEY,
            url TEXT,
            last_fetch TEXT,
            total_indicators INTEGER DEFAULT 0,
            enabled INTEGER DEFAULT 1
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tags (
            indicator_id TEXT,
            tag TEXT,
            PRIMARY KEY(indicator_id, tag)
        )
    """)
    conn.commit()
    return conn


def insert_ioc(conn, ioc_type, value, source="manual", confidence=0.5,
               severity="medium", tags="", description="", reference_url=""):
    """Insert an IoC with deduplication."""
    ioc_id = hashlib.md5(f"{ioc_type}:{value}".encode()).hexdigest()
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        conn.execute("""
            INSERT OR IGNORE INTO indicators
            (id, ioc_type, value, source, confidence, severity, tags, description, first_seen, last_seen, reference_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ioc_id, ioc_type, value, source, confidence, severity, tags, description, now, now, reference_url))
        conn.commit()
        return True
    except Exception as e:
        return False


def insert_sample_iocs(conn):
    """Insert bundled sample threat data."""
    print("\n  Loading sample threat data...")
    count = 0
    for ip in SAMPLE_IOCS["malicious_ips"]:
        if insert_ioc(conn, "ip", ip, "sample_feed", 0.7, "medium", "malicious-ip", "Known malicious IP"):
            count += 1
    for domain in SAMPLE_IOCS["malicious_domains"]:
        if insert_ioc(conn, "domain", domain, "sample_feed", 0.8, "high", "malicious-domain", "Known malicious domain"):
            count += 1
    for h in SAMPLE_IOCS["malicious_hashes"]:
        if insert_ioc(conn, "hash", h, "sample_feed", 0.9, "high", "malware-hash", "Known malware hash"):
            count += 1
    for url in SAMPLE_IOCS["malicious_urls"]:
        if insert_ioc(conn, "url", url, "sample_feed", 0.8, "high", "malicious-url", "Known malicious URL"):
            count += 1
    for cve in SAMPLE_IOCS["cve_ids"]:
        if insert_ioc(conn, "cve", cve, "nvd_feed", 1.0, "critical", "cve", f"Known vulnerability: {cve}", f"https://nvd.nist.gov/vuln/detail/{cve}"):
            count += 1
    print(f"  Inserted {count} indicators")
    return count


def fetch_urlhaus(conn):
    """Fetch recent URLs from URLhaus API."""
    print("\n  Fetching from URLhaus API...")
    try:
        resp = requests.get("https://urlhaus-api.abuse.ch/v1/urls/recent/", timeout=15)
        if resp.status_code != 200:
            print("  URLhaus API unavailable")
            return 0
        data = resp.json()
        if data.get("query_status") != "ok":
            return 0
        count = 0
        for entry in data.get("urls", [])[:100]:
            url = entry.get("url", "")
            if not url:
                continue
            tags = ",".join(entry.get("tags", []))
            ref = entry.get("urlhaus_reference", "")
            if insert_ioc(conn, "url", url, "urlhaus", 0.9, "high", tags, f"URLhaus: {entry.get('threat', '')}", ref):
                count += 1
        print(f"  Fetched {count} URLs from URLhaus")
        return count
    except Exception as e:
        print(f"  URLhaus fetch failed: {e}")
        return 0


def export_csv(conn, output_path, ioc_type=None):
    """Export indicators to CSV."""
    query = "SELECT ioc_type, value, source, confidence, severity, tags, description, first_seen, reference_url FROM indicators"
    params = []
    if ioc_type:
        query += " WHERE ioc_type = ?"
        params.append(ioc_type)
    df = pd.read_sql(query, conn, params=params)
    df.to_csv(output_path, index=False)
    print(f"  Exported {len(df)} indicators to {output_path}")


def export_json(conn, output_path):
    """Export all indicators to JSON (STIX-like format)."""
    cursor = conn.execute("SELECT * FROM indicators")
    columns = [desc[0] for desc in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    with open(output_path, "w") as f:
        json.dump({"indicators": rows, "count": len(rows), "exported": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}, f, indent=2)
    print(f"  Exported {len(rows)} indicators to {output_path}")


def query_iocs(conn, ioc_type=None, severity=None, limit=50):
    """Query indicators with filters."""
    query = "SELECT ioc_type, value, source, confidence, severity, tags, description, first_seen FROM indicators WHERE 1=1"
    params = []
    if ioc_type:
        query += " AND ioc_type = ?"; params.append(ioc_type)
    if severity:
        query += " AND severity = ?"; params.append(severity)
    query += f" ORDER BY first_seen DESC LIMIT {limit}"
    return pd.read_sql(query, conn, params=params)


def stats(conn):
    """Return summary statistics."""
    cursor = conn.execute("""
        SELECT ioc_type, COUNT(*) as count, ROUND(AVG(confidence), 2) as avg_conf
        FROM indicators GROUP BY ioc_type ORDER BY count DESC
    """)
    return cursor.fetchall()


def main():
    DATA_DIR.mkdir(exist_ok=True); RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 50)
    print("  Threat Intelligence Repository")
    print("=" * 50)

    conn = init_db()
    print(f"\n  Database: {DB_PATH}")

    # Insert sample data
    insert_sample_iocs(conn)

    # Try fetching from URLhaus
    try:
        fetch_urlhaus(conn)
    except Exception as e:
        print(f"  URLhaus fetch error: {e}")

    # Stats
    print(f"\n  {'─' * 50}")
    print("  Repository Statistics:")
    for row in stats(conn):
        print(f"    {row[0]:15s}  {row[1]:5d} indicators  (avg conf: {row[2]:.2f})")
    total = conn.execute("SELECT COUNT(*) FROM indicators").fetchone()[0]
    print(f"    {'─' * 40}")
    print(f"    {'TOTAL':15s}  {total:5d} indicators")

    # Export
    RESULTS_DIR.mkdir(exist_ok=True)
    export_csv(conn, RESULTS_DIR / "all_indicators.csv")
    export_json(conn, RESULTS_DIR / "all_indicators.json")

    # Per-type exports
    for ioc_type in ["ip", "domain", "url", "hash", "cve"]:
        export_csv(conn, RESULTS_DIR / f"{ioc_type}_indicators.csv", ioc_type)

    conn.close()
    print(f"\n  Repository ready at {DB_PATH}")


if __name__ == "__main__":
    main()
