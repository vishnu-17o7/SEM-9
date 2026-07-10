# 15 · Threat Intelligence — Social Media Mining

Gather cybersecurity-related information from social media, research feeds, and public threat intelligence sources. Extract and analyze indicators of compromise using data mining and regex-based NLP techniques.

## Quick Start

```bash
# Run the full pipeline (mine IoCs)
python run.py all

# Or run individual steps
python run.py run        # Mine IoCs from social media feeds
```

## Files

| File | Purpose |
|---|---|
| `mine_iocs.py` | Regex-based IoC extraction from bundled social media feeds (20 sample posts) |
| `results/mined_iocs.csv` | Extracted IoCs with source attribution |
| `results/mined_iocs.json` | Full extraction results in JSON |

## Supported IoC Types

- IP addresses (IPv4)
- Domains (including suspicious TLDs)
- URLs
- MD5, SHA1, SHA256 hashes
- CVE IDs
- Email addresses

## Sources (Sample)

- Simulated Twitter/X security researcher posts
- Simulated Reddit r/netsec, r/blueteamsec, r/malware posts
- Simulated security blog posts

## Setup

```bash
cd 15-threat-intel-social-mining
pip install -r requirements.txt
python mine_iocs.py
```
