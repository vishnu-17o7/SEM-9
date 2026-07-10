"""
Generate synthetic crypto-ransomware detection data.
Simulates file encryption patterns and system resource usage.
"""

import random
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
RANDOM_SEED = 42
random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

NUM_SAMPLES = 15000
MALICIOUS_RATIO = 0.20


def generate_normal_encryption_patterns():
    """Normal file system activity (no encryption)."""
    feats = {}
    # File I/O
    feats["files_read_per_sec"] = random.uniform(1, 20)
    feats["files_written_per_sec"] = random.uniform(0.1, 5)
    feats["avg_file_size_kb"] = random.uniform(10, 1000)
    feats["write_read_ratio"] = random.uniform(0.01, 0.3)
    feats["bytes_written_per_sec"] = random.uniform(1000, 50000)

    # Entropy
    feats["avg_entropy_before"] = random.uniform(3.0, 5.5)
    feats["avg_entropy_after"] = random.uniform(3.0, 5.5)
    feats["entropy_delta"] = random.uniform(-0.5, 0.5)
    feats["max_entropy_delta"] = random.uniform(0, 1.0)

    # CPU/Memory
    feats["cpu_usage_pct"] = random.uniform(2, 30)
    feats["memory_usage_mb"] = random.uniform(100, 2000)
    feats["cpu_std"] = random.uniform(1, 10)
    feats["memory_std"] = random.uniform(10, 300)
    feats["cpu_spike_freq"] = random.uniform(0, 0.1)

    # File type changes
    feats["num_doc_files"] = random.randint(10, 200)
    feats["num_media_files"] = random.randint(5, 50)
    feats["num_encrypted_extensions"] = 0
    feats["file_rename_rate"] = random.uniform(0, 0.02)

    # Timing
    feats["encryption_calls"] = 0
    feats["decryption_calls"] = random.randint(0, 5)
    feats["key_operations"] = 0
    feats["compression_api_calls"] = random.randint(0, 5)

    # Network
    feats["data_exfil_bytes"] = random.uniform(0, 10000)
    feats["c2_connections"] = random.randint(0, 5)

    # Read/Write amplification
    feats["write_amplification"] = random.uniform(1, 3)
    feats["read_amplification"] = random.uniform(1, 5)

    # Process anomalies
    feats["num_suspicious_procs"] = 0
    feats["num_crypto_procs"] = 0
    feats["num_powershell_spawns"] = 0

    # File header changes
    feats["file_header_changes"] = 0
    feats["known_headers_ratio"] = random.uniform(0.9, 1.0)
    feats["unexpected_headers"] = 0

    return feats


def generate_crypto_ransomware_patterns():
    """Simulate crypto-ransomware encryption behavior."""
    feats = {}
    # High file I/O — rapid encrypting
    feats["files_read_per_sec"] = random.uniform(50, 500)
    feats["files_written_per_sec"] = random.uniform(40, 400)
    feats["avg_file_size_kb"] = random.uniform(50, 500)
    feats["write_read_ratio"] = random.uniform(0.5, 1.0)
    feats["bytes_written_per_sec"] = random.uniform(50000, 5000000)

    # Entropy spike (encrypted files)
    feats["avg_entropy_before"] = random.uniform(3.0, 5.5)
    feats["avg_entropy_after"] = random.uniform(6.5, 8.0)
    feats["entropy_delta"] = random.uniform(2.0, 4.5)
    feats["max_entropy_delta"] = random.uniform(3.0, 7.0)

    # CPU/Memory spike
    feats["cpu_usage_pct"] = random.uniform(50, 100)
    feats["memory_usage_mb"] = random.uniform(500, 8000)
    feats["cpu_std"] = random.uniform(5, 30)
    feats["memory_std"] = random.uniform(100, 1000)
    feats["cpu_spike_freq"] = random.uniform(0.3, 1.0)

    # File type changes — target docs, change extensions
    feats["num_doc_files"] = random.randint(100, 5000)
    feats["num_media_files"] = random.randint(20, 200)
    feats["num_encrypted_extensions"] = random.randint(1, 5)
    feats["file_rename_rate"] = random.uniform(0.5, 1.0)

    # Encryption API calls
    feats["encryption_calls"] = random.randint(100, 5000)
    feats["decryption_calls"] = 0
    feats["key_operations"] = random.randint(10, 100)
    feats["compression_api_calls"] = random.randint(0, 10)

    # C2 communication
    feats["data_exfil_bytes"] = random.uniform(1000, 500000)
    feats["c2_connections"] = random.randint(3, 20)

    # Write/Read amplification
    feats["write_amplification"] = random.uniform(5, 20)
    feats["read_amplification"] = random.uniform(2, 10)

    # Suspicious processes
    feats["num_suspicious_procs"] = random.randint(1, 5)
    feats["num_crypto_procs"] = random.randint(1, 3)
    feats["num_powershell_spawns"] = random.randint(0, 5)

    # File header changes (encrypted headers)
    feats["file_header_changes"] = random.randint(100, 5000)
    feats["known_headers_ratio"] = random.uniform(0, 0.3)
    feats["unexpected_headers"] = random.randint(100, 5000)

    return feats


FEATURE_NAMES = [
    "files_read_per_sec", "files_written_per_sec", "avg_file_size_kb",
    "write_read_ratio", "bytes_written_per_sec",
    "avg_entropy_before", "avg_entropy_after", "entropy_delta", "max_entropy_delta",
    "cpu_usage_pct", "memory_usage_mb", "cpu_std", "memory_std", "cpu_spike_freq",
    "num_doc_files", "num_media_files", "num_encrypted_extensions", "file_rename_rate",
    "encryption_calls", "decryption_calls", "key_operations", "compression_api_calls",
    "data_exfil_bytes", "c2_connections", "write_amplification", "read_amplification",
    "num_suspicious_procs", "num_crypto_procs", "num_powershell_spawns",
    "file_header_changes", "known_headers_ratio", "unexpected_headers",
]


def main():
    DATA_DIR.mkdir(exist_ok=True)
    print("=" * 50); print("  Crypto-Ransomware — Data Generator"); print("=" * 50)
    n_mal = int(NUM_SAMPLES * MALICIOUS_RATIO); n_norm = NUM_SAMPLES - n_mal
    print(f"\n  Generating {n_norm} normal sessions...")
    normal = [generate_normal_encryption_patterns() for _ in tqdm(range(n_norm))]
    print(f"  Generating {n_mal} ransomware sessions...")
    malware = [generate_crypto_ransomware_patterns() for _ in tqdm(range(n_mal))]
    all_feats = normal + malware; labels = [0]*n_norm + [1]*n_mal
    combined = list(zip(all_feats, labels)); random.shuffle(combined)
    all_feats, labels = zip(*combined)
    df = pd.DataFrame(all_feats); df["label"] = labels
    out_path = DATA_DIR / "crypto_ransomware.csv"
    df.to_csv(out_path, index=False)
    X = df[FEATURE_NAMES].values.astype(np.float64); y = np.array(labels, dtype=np.int64)
    np.savez_compressed(DATA_DIR / "crypto_ransomware.npz", X=X, y=y)
    print(f"\n  Total: {len(df)} samples, {len(FEATURE_NAMES)} features")
    print(f"  Ransomware ratio: {sum(labels)/len(labels):.1%}")

if __name__ == "__main__":
    main()
