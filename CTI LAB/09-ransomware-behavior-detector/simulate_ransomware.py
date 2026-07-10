"""
Simulate system activities including file operations and registry modifications
to generate data for ransomware behavior detection.
"""

import random
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
RANDOM_SEED = 42
random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

NUM_SAMPLES = 20000
RANSOMWARE_RATIO = 0.15

FILE_TYPES = {"doc": 0, "xls": 0, "pdf": 0, "jpg": 0, "png": 0, "txt": 0,
              "zip": 0, "exe": 0, "dll": 0, "db": 0, "bak": 0, "encrypted": 0}
REGISTRY_AREAS = ["HKCU\\Software", "HKLM\\System", "HKCU\\Startup",
                  "HKLM\\Run", "HKCR\\*\\shell", "HKCU\\Network"]


def generate_normal_activity():
    """Simulate normal system activity."""
    feats = {}
    n_files = random.randint(5, 50)
    feats["num_files_accessed"] = n_files
    feats["num_files_modified"] = random.randint(0, 10)
    feats["num_files_renamed"] = random.randint(0, 3)
    feats["num_files_deleted"] = random.randint(0, 2)
    feats["num_files_created"] = random.randint(0, 8)
    feats["num_dirs_created"] = random.randint(0, 3)
    feats["num_dirs_deleted"] = random.randint(0, 1)

    # File type distribution
    feats["doc_ratio"] = random.uniform(0.1, 0.4)
    feats["exe_ratio"] = random.uniform(0.0, 0.1)
    feats["compressed_ratio"] = random.uniform(0.0, 0.05)
    feats["encrypted_ratio"] = random.uniform(0.0, 0.01)

    # Registry operations
    feats["num_registry_reads"] = random.randint(10, 200)
    feats["num_registry_writes"] = random.randint(0, 20)
    feats["num_registry_deletes"] = 0
    feats["startup_registry_writes"] = 0
    feats["run_registry_writes"] = 0

    # Process behavior
    feats["num_processes_created"] = random.randint(0, 5)
    feats["num_processes_terminated"] = random.randint(0, 3)
    feats["num_network_connections"] = random.randint(0, 10)
    feats["num_dns_queries"] = random.randint(0, 20)

    # Timing
    feats["ops_per_second"] = random.uniform(0.5, 5.0)
    feats["file_ops_per_second"] = random.uniform(0.3, 3.0)
    feats["registry_ops_per_second"] = random.uniform(0.1, 2.0)

    # Entropy
    feats["avg_file_entropy"] = random.uniform(3.0, 5.5)
    feats["max_file_entropy"] = random.uniform(4.0, 7.0)
    feats["entropy_change"] = random.uniform(-0.5, 0.5)

    # File extension diversity
    feats["file_ext_diversity"] = random.uniform(0.3, 0.8)

    # Shadow copy operations
    feats["shadow_copy_deletions"] = 0
    feats["backup_file_ops"] = 0

    return feats


def generate_ransomware_activity():
    """Simulate ransomware-like system activity."""
    feats = {}
    n_files = random.randint(50, 2000)
    feats["num_files_accessed"] = n_files
    feats["num_files_modified"] = random.randint(int(n_files * 0.7), n_files)
    feats["num_files_renamed"] = random.randint(int(n_files * 0.3), int(n_files * 0.8))
    feats["num_files_deleted"] = random.randint(0, n_files // 10)
    feats["num_files_created"] = random.randint(int(n_files * 0.2), int(n_files * 0.6))
    feats["num_dirs_created"] = random.randint(0, 10)
    feats["num_dirs_deleted"] = random.randint(0, 5)

    # Ransomware targets documents — high doc/file ratio
    feats["doc_ratio"] = random.uniform(0.5, 0.9)
    feats["exe_ratio"] = random.uniform(0.0, 0.05)
    feats["compressed_ratio"] = random.uniform(0.0, 0.02)
    feats["encrypted_ratio"] = random.uniform(0.6, 1.0)  # High encryption

    # Registry modifications — persistence
    feats["num_registry_reads"] = random.randint(50, 500)
    feats["num_registry_writes"] = random.randint(20, 200)
    feats["num_registry_deletes"] = random.randint(0, 10)
    feats["startup_registry_writes"] = random.randint(1, 5)
    feats["run_registry_writes"] = random.randint(1, 5)

    # Process behavior
    feats["num_processes_created"] = random.randint(5, 50)
    feats["num_processes_terminated"] = random.randint(5, 30)
    feats["num_network_connections"] = random.randint(10, 100)
    feats["num_dns_queries"] = random.randint(5, 50)

    # Timing — rapid operations
    feats["ops_per_second"] = random.uniform(20.0, 500.0)
    feats["file_ops_per_second"] = random.uniform(15.0, 400.0)
    feats["registry_ops_per_second"] = random.uniform(1.0, 30.0)

    # High entropy (encrypted content)
    feats["avg_file_entropy"] = random.uniform(6.5, 8.0)
    feats["max_file_entropy"] = random.uniform(7.5, 8.0)
    feats["entropy_change"] = random.uniform(2.0, 5.0)

    # Low diversity (all files getting same extension)
    feats["file_ext_diversity"] = random.uniform(0.0, 0.2)

    # Shadow copy deletion
    feats["shadow_copy_deletions"] = random.randint(1, 10)
    feats["backup_file_ops"] = random.randint(0, 5)

    return feats


FEATURE_NAMES = [
    "num_files_accessed", "num_files_modified", "num_files_renamed",
    "num_files_deleted", "num_files_created", "num_dirs_created",
    "num_dirs_deleted", "doc_ratio", "exe_ratio", "compressed_ratio",
    "encrypted_ratio", "num_registry_reads", "num_registry_writes",
    "num_registry_deletes", "startup_registry_writes", "run_registry_writes",
    "num_processes_created", "num_processes_terminated",
    "num_network_connections", "num_dns_queries", "ops_per_second",
    "file_ops_per_second", "registry_ops_per_second",
    "avg_file_entropy", "max_file_entropy", "entropy_change",
    "file_ext_diversity", "shadow_copy_deletions", "backup_file_ops",
]


def main():
    DATA_DIR.mkdir(exist_ok=True)
    print("=" * 50); print("  Ransomware Behavior — Activity Simulator"); print("=" * 50)
    n_ransom = int(NUM_SAMPLES * RANSOMWARE_RATIO)
    n_normal = NUM_SAMPLES - n_ransom
    print(f"\n  Generating {n_normal} normal sessions...")
    normal = [generate_normal_activity() for _ in tqdm(range(n_normal))]
    print(f"  Generating {n_ransom} ransomware sessions...")
    ransomware = [generate_ransomware_activity() for _ in tqdm(range(n_ransom))]
    all_feats = normal + ransomware
    labels = [0] * n_normal + [1] * n_ransom
    combined = list(zip(all_feats, labels)); random.shuffle(combined)
    all_feats, labels = zip(*combined)
    df = pd.DataFrame(all_feats); df["label"] = labels
    out_path = DATA_DIR / "ransomware_activity.csv"
    df.to_csv(out_path, index=False)
    X = df[FEATURE_NAMES].values.astype(np.float64)
    y = np.array(labels, dtype=np.int64)
    np.savez_compressed(DATA_DIR / "ransomware_activity.npz", X=X, y=y)
    print(f"\n  Total: {len(df)} samples, {len(FEATURE_NAMES)} features")
    print(f"  Ransomware ratio: {sum(labels)/len(labels):.1%}")
    print(f"  Saved to {out_path}")

if __name__ == "__main__":
    main()
