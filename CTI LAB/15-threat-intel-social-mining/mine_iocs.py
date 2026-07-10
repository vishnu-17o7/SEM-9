"""
Mine indicators of compromise from social media, research feeds, and public sources.

Extracts IoCs using regex patterns from text sources.
Includes bundled sample social media feeds for demonstration.
"""

import json
import re
import hashlib
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

# Regular expressions for IoC extraction
IOC_PATTERNS = {
    "ip": re.compile(
        r"(?<![0-9])(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?![0-9])"
    ),
    "domain": re.compile(
        r"(?:(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
        r"(?:com|org|net|io|gov|edu|tk|ml|ga|cf|gq|xyz|top|work|download|bid|loan|"
        r"racing|accountant|science|click|review|trade|webcam|men|win|site|online|"
        r"tech|space|store|cloud|app|live|help|info|pro|name|mobi|me|tv|cc|ws|"
        r"in|uk|de|fr|jp|au|br|ca|cn|ru|ch|nl|se|no|dk|fi|pl|at|be|es|it|pt|"
        r"eu|asia|co|us|biz|su|xxx))"
    ),
    "url": re.compile(
        r"https?://[^\s<>\"'{}|\\^`[\]]+(?:\.[^\s<>\"'{}|\\^`[\]]+)+"
    ),
    "md5": re.compile(r"\b[a-fA-F0-9]{32}\b"),
    "sha1": re.compile(r"\b[a-fA-F0-9]{40}\b"),
    "sha256": re.compile(r"\b[a-fA-F0-9]{64}\b"),
    "cve": re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
}

# Sample social media posts (simulating Twitter/X, Reddit, security feeds)
SAMPLE_FEEDS = [
    {"source": "twitter", "content": "New #phishing campaign targeting banking customers. C2 server at 185.130.5.27 using domain evil-malware.xyz. #infosec"},
    {"source": "twitter", "content": "Alert: CVE-2024-3094 exploit detected in the wild. Payload hosted at http://exploit-kit.loan/exploit.php Check your systems!"},
    {"source": "reddit", "content": "r/netsec - Found a new malware sample. MD5: e1112134b6dcc8f54d7f7c1b9c5f1e3a. Communicates with 91.121.87.34:443 via TLS."},
    {"source": "twitter", "content": "Ransomware group using double extortion. Check for connections to 45.142.212.77 or 89.248.165.80. #ransomware #cybersecurity"},
    {"source": "blog", "content": "Analysis of recent supply chain attack: initial access via CVE-2023-46604, lateral movement through 192.241.214.181, exfiltration to 138.197.62.204."},
    {"source": "twitter", "content": "New IoC feed: phishing-bank.tk targeting major banks. Steals credentials via fake login pages. Also seen: steal-login.ml"},
    {"source": "reddit", "content": "r/blueteamsec - Anyone seeing traffic to 5.188.62.14:8443? Looks like Mirai variant C2. SHA256: a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2"},
    {"source": "twitter", "content": "CVE-2021-44228 (Log4Shell) still being exploited. New campaign detected targeting unpatched servers. Payload: http://payload-staging.xyz/dropper.bin"},
    {"source": "blog", "content": "APT group using watering hole attacks. Compromised sites redirect to 194.26.29.113 which serves malicious JavaScript. Related domains: maldoc-delivery.racing"},
    {"source": "twitter", "content": "Credential stuffing attack targeting financial sector. Login attempts from distributed IPs including 103.253.42.100, 182.253.106.80, 45.33.32.156."},
    {"source": "reddit", "content": "r/malware - New stealer variant exfiltrates to exfil-data.top via HTTPS. MD5: f2a3245c7d8e9f0a1b2c3d4e5f6a7b8c. Uses Telegram bot for C2."},
    {"source": "twitter", "content": "CVE-2023-34362 (MOVEit Transfer) exploitation continues. Check for outbound connections to 167.71.5.83 or 68.183.191.110 on port 443."},
    {"source": "blog", "content": "Darknet marketplace advertising stolen credentials. Market domain: darknet-market.tk. Bitcoin wallet for ransomware payments also identified."},
    {"source": "twitter", "content": "New phishing kit discovered. Uses open redirect on legit sites to bypass filters. Delivers payload from http://free-prize-win.cf/redirect.php"},
    {"source": "reddit", "content": "r/cybersecurity - Botnet controller at 159.89.209.50 using custom protocol on port 7777. Also C2 at download-virus.ga:8080. Beacons every 60 seconds."},
    {"source": "twitter", "content": "Warning: CVE-2022-30190 (Follina) being used in targeted attacks. Macro document delivered via email. MD5 reported: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8"},
    {"source": "blog", "content": "Analysis of recent ransomware: encrypts files and appends .encrypted extension. Communication with C2 at ransomware-pay.work. Uses RSA-4096 encryption."},
    {"source": "twitter", "content": "Watering hole attack on tech news site. Malicious script loads from 104.248.50.85/js/analytics.js. Also connects to credential-harvest.bid for stolen data."},
    {"source": "reddit", "content": "r/netsec - Anyone seeing scanning activity from 31.220.59.11? Port scanning for vulnerable Exchange servers. Likely related to ProxyShell exploitation."},
    {"source": "twitter", "content": "Multiple CVE-2023-23397 (Outlook elevation) attacks detected. Attackers sending specially crafted emails. IoCs: botnet-ctrl.download, C2 at 165.227.88.157."},
]


def extract_iocs(text, source="unknown"):
    """Extract all IoCs from text using regex patterns."""
    results = []
    for ioc_type, pattern in IOC_PATTERNS.items():
        matches = pattern.findall(text)
        for match in set(matches):  # Deduplicate per text
            # Validate URLs (must start with http)
            if ioc_type == "url" and not match.startswith(("http://", "https://")):
                continue
            # Validate domains (must have at least one dot)
            if ioc_type == "domain" and "." not in match:
                continue
            results.append({
                "ioc_type": ioc_type,
                "value": match.lower() if ioc_type != "ip" else match,
                "source": source,
                "extracted_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
    return results


def process_feeds():
    """Process sample social media feeds and extract IoCs."""
    all_iocs = []
    source_counts = Counter()

    print("  Processing sample social media feeds...")
    for feed in tqdm(SAMPLE_FEEDS):
        iocs = extract_iocs(feed["content"], feed["source"])
        for ioc in iocs:
            # Add context
            ioc["context"] = feed["content"][:100]
            all_iocs.append(ioc)
        source_counts[feed["source"]] += 1

    return all_iocs, source_counts


def deduplicate_iocs(iocs):
    """Deduplicate IoCs by type + value."""
    seen = set()
    unique = []
    for ioc in iocs:
        key = (ioc["ioc_type"], ioc["value"])
        if key not in seen:
            seen.add(key)
            unique.append(ioc)
    return unique


def generate_report(iocs, source_counts):
    """Generate IoC mining report."""
    type_counts = Counter(i["ioc_type"] for i in iocs)

    print(f"\n  {'-' * 50}")
    print("  IoC Mining Report")
    print(f"  {'-' * 50}")
    print(f"\n  Sources processed:")
    for src, count in source_counts.most_common():
        print(f"    {src:12s} {count:4d} posts")
    print(f"\n  IoCs extracted by type:")
    for ioc_type, count in type_counts.most_common():
        print(f"    {ioc_type:10s} {count:4d}")
    print(f"\n  Total unique IoCs: {len(iocs)}")

    # Generate context table
    rows = []
    for ioc in iocs[:50]:  # Show top 50
        rows.append({
            "Type": ioc["ioc_type"],
            "Value": ioc["value"],
            "Source": ioc["source"],
            "Context": ioc.get("context", "")[:60],
        })
    df = pd.DataFrame(rows)
    return df


def main():
    DATA_DIR.mkdir(exist_ok=True); RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 50)
    print("  Threat Intelligence — Social Media Mining")
    print("=" * 50)

    # Process feeds
    all_iocs, source_counts = process_feeds()

    # Deduplicate
    print("\n  Deduplicating IoCs...")
    unique_iocs = deduplicate_iocs(all_iocs)
    print(f"  {len(all_iocs)} raw -> {len(unique_iocs)} unique IoCs")

    # Generate report
    report_df = generate_report(unique_iocs, source_counts)

    # Save results
    report_df.to_csv(RESULTS_DIR / "mined_iocs.csv", index=False)
    print(f"\n  Saved to {RESULTS_DIR / 'mined_iocs.csv'}")

    # Save full JSON
    with open(RESULTS_DIR / "mined_iocs.json", "w") as f:
        json.dump({"iocs": unique_iocs, "total": len(unique_iocs),
                    "sources": dict(source_counts)}, f, indent=2)
    print(f"  Saved to {RESULTS_DIR / 'mined_iocs.json'}")

    # Display table
    print(f"\n  {'-' * 50}")
    print("  Sample extracted IoCs:")
    print(f"  {'-' * 50}")
    for ioc in unique_iocs[:20]:
        ctx = ioc.get("context", "")[:55]
        print(f"  [{ioc['ioc_type']:8s}] {ioc['value']:40s}  | {ctx}")


if __name__ == "__main__":
    main()
