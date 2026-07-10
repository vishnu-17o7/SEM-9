"""
Download NSL-KDD dataset for network intrusion detection.

NSL-KDD is an improved version of KDD'99 with no duplicate records.
Source: https://www.unb.ca/cic/datasets/nsl.html
Mirror: https://github.com/defcom17/NSL_KDD
"""

from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"

# Mirrors for NSL-KDD dataset
NSL_KDD_URLS = {
    "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
    "KDDTest+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
}

COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]

ATTACK_TYPES = {
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos",
    "udpstorm": "dos", "processtable": "dos", "worm": "dos",
    "mailbomb": "dos",
    "satan": "probe", "ipsweep": "probe", "nmap": "probe",
    "portsweep": "probe", "mscan": "probe", "saint": "probe",
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l",
    "phf": "r2l", "multihop": "r2l", "warezmaster": "r2l",
    "warezclient": "r2l", "spy": "r2l", "xlock": "r2l",
    "xsnoop": "r2l", "snmpguess": "r2l", "snmpgetattack": "r2l",
    "httptunnel": "r2l",
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r", "xterm": "u2r", "ps": "u2r",
    "sqlattack": "u2r",
    "named": "u2r", "sendmail": "u2r",
}


def download_file(url: str, dest: Path) -> bool:
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(desc=dest.name, total=total, unit="B", unit_scale=True) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk); pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  Failed: {e}"); return False


def main():
    DATA_DIR.mkdir(exist_ok=True)
    print("=" * 50); print("  NSL-KDD Dataset Downloader"); print("=" * 50)

    for fname, url in NSL_KDD_URLS.items():
        dest = DATA_DIR / fname
        if dest.exists():
            print(f"  {fname} already exists ({dest.stat().st_size / 1024:.0f} KB)")
            continue
        print(f"\n  Downloading {fname}...")
        if download_file(url, dest):
            print(f"  Saved to {dest} ({dest.stat().st_size / 1024:.0f} KB)")

    # Load and preprocess
    print("\n  Loading and preprocessing...")
    train_path = DATA_DIR / "KDDTrain+.txt"
    test_path = DATA_DIR / "KDDTest+.txt"

    if not train_path.exists() or not test_path.exists():
        print("  Download failed. Generate synthetic fallback.")
        generate_fallback()
        return

    train_df = pd.read_csv(train_path, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_path, names=COLUMN_NAMES)

    # Map labels to binary (normal=0, anomaly=1)
    train_df["label_binary"] = (train_df["label"] != "normal").astype(int)
    test_df["label_binary"] = (test_df["label"] != "normal").astype(int)

    # Map to attack categories
    train_df["attack_category"] = train_df["label"].str.lower().map(ATTACK_TYPES).fillna("normal")
    test_df["attack_category"] = test_df["label"].str.lower().map(ATTACK_TYPES).fillna("normal")

    # Save processed
    train_df.to_csv(DATA_DIR / "KDDTrain_processed.csv", index=False)
    test_df.to_csv(DATA_DIR / "KDDTest_processed.csv", index=False)

    print(f"\n  Training: {len(train_df)} samples ({train_df['label_binary'].mean():.1%} attacks)")
    print(f"  Test:     {len(test_df)} samples ({test_df['label_binary'].mean():.1%} attacks)")
    print(f"  Attack categories: {train_df['attack_category'].value_counts().to_dict()}")


def generate_fallback():
    """Generate synthetic NSL-KDD-like data if download fails."""
    print("\n  Generating synthetic fallback...")
    np = __import__("numpy")
    random = __import__("random")
    n_train, n_test = 25000, 10000

    protocols = ["tcp", "udp", "icmp"]
    services = ["http", "smtp", "ftp", "ssh", "dns", "pop3", "imap", "telnet", "other"]
    flags = ["SF", "S0", "REJ", "RSTO", "RSTOS0", "SH", "S1", "S2", "S3", "OTH"]

    rows = []
    for is_train in [True, False]:
        n = n_train if is_train else n_test
        for _ in range(n):
            is_attack = random.random() < 0.45
            row = {
                "duration": np.random.exponential(100) if not is_attack else np.random.exponential(500),
                "protocol_type": random.choice(protocols),
                "service": random.choice(services),
                "flag": random.choice(flags),
                "src_bytes": np.random.lognormal(6, 2) if not is_attack else np.random.lognormal(8, 3),
                "dst_bytes": np.random.lognormal(5, 2) if not is_attack else np.random.lognormal(7, 3),
                "land": 1 if random.random() < 0.01 else 0,
                "wrong_fragment": np.random.poisson(0.1) if not is_attack else np.random.poisson(2),
                "urgent": np.random.poisson(0.01),
                "hot": np.random.poisson(0.5) if not is_attack else np.random.poisson(5),
                "num_failed_logins": np.random.poisson(0.1) if not is_attack else np.random.poisson(3),
                "logged_in": 1 if random.random() < (0.3 if is_attack else 0.7) else 0,
                "num_compromised": np.random.poisson(0.05) if not is_attack else np.random.poisson(10),
                "root_shell": 1 if is_attack and random.random() < 0.2 else 0,
                "su_attempted": 1 if is_attack and random.random() < 0.15 else 0,
                "num_root": np.random.poisson(0.1) if not is_attack else np.random.poisson(20),
                "num_file_creations": np.random.poisson(0.1) if not is_attack else np.random.poisson(5),
                "num_shells": np.random.poisson(0.05) if not is_attack else np.random.poisson(3),
                "num_access_files": np.random.poisson(0.1) if not is_attack else np.random.poisson(8),
                "num_outbound_cmds": 0,
                "is_host_login": 0,
                "is_guest_login": 1 if random.random() < (0.1 if is_attack else 0.02) else 0,
                "count": np.random.poisson(5) if not is_attack else np.random.poisson(20),
                "srv_count": np.random.poisson(3) if not is_attack else np.random.poisson(15),
                "serror_rate": random.random() * (0.1 if not is_attack else 0.8),
                "srv_serror_rate": random.random() * (0.1 if not is_attack else 0.8),
                "rerror_rate": random.random() * (0.05 if not is_attack else 0.5),
                "srv_rerror_rate": random.random() * (0.05 if not is_attack else 0.5),
                "same_srv_rate": random.uniform(0.5, 1.0),
                "diff_srv_rate": random.uniform(0, 0.5),
                "srv_diff_host_rate": random.uniform(0, 0.5),
                "dst_host_count": np.random.poisson(10) if not is_attack else np.random.poisson(50),
                "dst_host_srv_count": np.random.poisson(5) if not is_attack else np.random.poisson(30),
                "dst_host_same_srv_rate": random.uniform(0.5, 1.0),
                "dst_host_diff_srv_rate": random.uniform(0, 0.5),
                "dst_host_same_src_port_rate": random.uniform(0, 0.5),
                "dst_host_srv_diff_host_rate": random.uniform(0, 0.5),
                "dst_host_serror_rate": random.random() * (0.1 if not is_attack else 0.8),
                "dst_host_srv_serror_rate": random.random() * (0.1 if not is_attack else 0.8),
                "dst_host_rerror_rate": random.random() * (0.05 if not is_attack else 0.5),
                "dst_host_srv_rerror_rate": random.random() * (0.05 if not is_attack else 0.5),
                "label": random.choice(["neptune", "satan", "ipsweep", "normal"]) if is_attack else "normal",
                "difficulty": random.randint(1, 21),
                "label_binary": 1 if is_attack else 0,
                "attack_category": random.choice(["dos", "probe", "r2l", "normal"]) if is_attack else "normal",
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:n_train + n_test]
    train_df.to_csv(DATA_DIR / "KDDTrain_processed.csv", index=False)
    test_df.to_csv(DATA_DIR / "KDDTest_processed.csv", index=False)
    print(f"  Synthetic fallback: {len(train_df)} train, {len(test_df)} test")

if __name__ == "__main__":
    main()
