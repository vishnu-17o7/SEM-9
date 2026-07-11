"""Download and normalize public authentication-security datasets for CTI 05.

The CICIDS2017 source file is a labelled network-flow capture.  Its web
brute-force records are converted to the detector's event schema without
deriving login outcomes from the attack label.  LANL host events are accepted
when supplied after the source's registration step; their official endpoint
does not allow unauthenticated automation.
"""
from __future__ import annotations

import argparse
import bz2
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
REAL_DIR = DATA_DIR / "real"
MANIFEST_PATH = REAL_DIR / "dataset_manifest.json"
CIC_FILENAME = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet"
CIC_RAW_PATH = REAL_DIR / CIC_FILENAME
CIC_EVENT_PATH = REAL_DIR / "cicids2017_web_bruteforce_events.csv"
CIC_FLOW_PATH = REAL_DIR / "cicids2017_web_bruteforce_flows.csv"
CIC_MIRROR_URL = (
    "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/traffic_labels/"
    f"{CIC_FILENAME}?download=true"
)
CIC_OFFICIAL_URL = "https://www.unb.ca/cic/datasets/ids-2017.html"
LANL_OFFICIAL_URL = "https://csr.lanl.gov/data/2017/"
LANL_EVENT_PATH = REAL_DIR / "lanl_authentication_events.csv"


def download_file(url: str, destination: Path) -> None:
    """Download a dataset file atomically with a visible progress counter."""
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    request = urllib.request.Request(url, headers={"User-Agent": "CTI-Lab-05/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, open(temporary, "wb") as output:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  Downloaded {downloaded / 1_000_000:.1f} / {total / 1_000_000:.1f} MB", end="")
    shutil.move(temporary, destination)
    print()


def normalized_column_name(frame: pd.DataFrame, expected: str) -> str:
    """Find a CICIDS2017 column despite whitespace differences in CSV exports."""
    target = expected.strip().lower()
    for column in frame.columns:
        if column.strip().lower() == target:
            return column
    raise KeyError(f"Expected CICIDS2017 column not found: {expected}")


def prepare_cicids_events(max_benign: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert labelled CIC web brute-force flows to a common event schema."""
    source = pd.read_parquet(CIC_RAW_PATH)
    label = normalized_column_name(source, "Label")
    timestamp = normalized_column_name(source, "Timestamp")
    source_ip = normalized_column_name(source, "Source IP")
    destination = normalized_column_name(source, "Destination IP")
    protocol = normalized_column_name(source, "Protocol")
    destination_port = normalized_column_name(source, "Destination Port")
    labels = source[label].astype(str).str.strip()
    attacks = source.loc[labels.str.contains("web attack.*brute force", case=False, regex=True)].copy()
    benign = source.loc[labels.str.upper().eq("BENIGN")].head(max_benign).copy()

    def to_events(rows: pd.DataFrame, is_attack: int) -> pd.DataFrame:
        events = pd.DataFrame({
            "timestamp": pd.to_datetime(rows[timestamp], errors="coerce", utc=True),
            "username": rows[destination].astype(str),
            "source_ip": rows[source_ip].astype(str),
            # CICIDS2017 does not expose an HTTP user agent or login outcome.
            "user_agent": "protocol-" + rows[protocol].astype(str),
            "success": 1,
            "geo_country": "port-" + rows[destination_port].astype(str),
            "is_attack": is_attack,
            "source_dataset": "CICIDS2017",
        })
        return events.dropna(subset=["timestamp"])

    samples = [to_events(benign, 0), to_events(attacks, 1)]
    if not any(not sample.empty for sample in samples):
        raise ValueError("No benign or web brute-force rows were found in the CICIDS2017 CSV.")

    events = pd.concat(samples, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    events.to_csv(CIC_EVENT_PATH, index=False)
    flow_columns = {
        "destination_port": "Destination Port",
        "flow_duration": "Flow Duration",
        "total_fwd_packets": "Total Fwd Packets",
        "total_backward_packets": "Total Backward Packets",
        "total_fwd_bytes": "Total Length of Fwd Packets",
        "total_backward_bytes": "Total Length of Bwd Packets",
        "flow_bytes_per_second": "Flow Bytes/s",
        "flow_packets_per_second": "Flow Packets/s",
        "syn_flag_count": "SYN Flag Count",
        "ack_flag_count": "ACK Flag Count",
        "fwd_iat_mean": "Fwd IAT Mean",
        "bwd_iat_mean": "Bwd IAT Mean",
    }
    flows = pd.concat([benign, attacks], ignore_index=True)
    prepared_flows = pd.DataFrame({
        name: pd.to_numeric(flows[normalized_column_name(flows, source_name)], errors="coerce")
        for name, source_name in flow_columns.items()
    })
    prepared_flows["is_attack"] = [0] * len(benign) + [1] * len(attacks)
    prepared_flows = prepared_flows.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    prepared_flows.to_csv(CIC_FLOW_PATH, index=False)
    return events, {"benign": int((events["is_attack"] == 0).sum()), "web_brute_force": int((events["is_attack"] == 1).sum())}


def import_lanl_events(path: Path, max_events: int) -> dict[str, int]:
    """Normalize a registered LANL daily host-event archive into login events."""
    records: list[dict[str, object]] = []
    opener = bz2.open if path.suffix == ".bz2" else open
    with opener(path, "rt", encoding="utf-8") as source:
        for line in source:
            event = json.loads(line)
            if event.get("EventID") not in {4624, 4625, 4776}:
                continue
            timestamp = event.get("Time")
            username = event.get("UserName")
            source_host = event.get("Source", event.get("LogHost"))
            if timestamp is None or not username or not source_host:
                continue
            records.append({
                "timestamp": datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat(),
                "username": str(username),
                "source_ip": str(source_host),
                "user_agent": str(event.get("AuthenticationPackage", "unknown")),
                "success": int(str(event.get("Status", "0x0")) == "0x0"),
                "geo_country": str(event.get("DomainName", "unknown")),
                "is_attack": 0,
                "source_dataset": "LANL-UHN-2017",
            })
            if len(records) >= max_events:
                break

    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("The LANL archive did not contain usable authentication events.")
    frame.to_csv(LANL_EVENT_PATH, index=False)
    return {"authentication_events": len(frame)}


def write_manifest(cic_counts: dict[str, int], lanl_counts: dict[str, int] | None) -> None:
    """Write source provenance used by training and the generated HTML report."""
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": [
            {
                "name": "CICIDS2017",
                "role": "Labelled web brute-force evaluation and supervised comparison",
                "official_url": CIC_OFFICIAL_URL,
                "local_file": str(CIC_FLOW_PATH.relative_to(PROJECT_DIR)),
                "raw_file": str(CIC_RAW_PATH.relative_to(PROJECT_DIR)),
                "counts": cic_counts,
                "limitations": (
                    "Network-flow labels, not application login outcomes. Protocol and destination port "
                    "are retained as proxy context; no user-agent, country, or credential-success field exists."
                ),
            },
            {
                "name": "LANL Unified Host and Network Dataset (2017)",
                "role": "Optional de-identified enterprise authentication baseline",
                "official_url": LANL_OFFICIAL_URL,
                "local_file": str(LANL_EVENT_PATH.relative_to(PROJECT_DIR)),
                "counts": lanl_counts or {},
                "status": "ready" if lanl_counts else "awaiting registered source download",
                "limitations": (
                    "Official daily files require registration. It supplies real enterprise host-event "
                    "authentication data but does not provide credential-stuffing labels."
                ),
            },
        ],
        "training_note": (
            "Metrics are reported against CICIDS2017 web brute-force labels. They are an intrusion "
            "detection benchmark, not a claim of real-world web credential-stuffing accuracy."
        ),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    """Acquire CICIDS2017 and optionally normalize a registered LANL archive."""
    parser = argparse.ArgumentParser(description="Prepare real-source data for CTI Lab 05")
    parser.add_argument("--lanl-file", type=Path, help="Registered LANL wls_day-XX.bz2 or JSONL file")
    parser.add_argument("--max-benign", type=int, default=30_000)
    parser.add_argument("--max-lanl-events", type=int, default=100_000)
    args = parser.parse_args()

    REAL_DIR.mkdir(parents=True, exist_ok=True)
    if not CIC_RAW_PATH.exists():
        print("  Downloading CICIDS2017 Thursday labelled-flow parquet...")
        download_file(CIC_MIRROR_URL, CIC_RAW_PATH)
    else:
        print(f"  Reusing {CIC_RAW_PATH.name}")

    events, cic_counts = prepare_cicids_events(args.max_benign)
    lanl_counts = None
    if args.lanl_file:
        lanl_counts = import_lanl_events(args.lanl_file, args.max_lanl_events)
    write_manifest(cic_counts, lanl_counts)
    print(f"  Prepared {len(events):,} CICIDS2017 events: {cic_counts}")
    if not args.lanl_file:
        print("  LANL import deferred: the official source requires user registration before download.")
    print(f"  Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
