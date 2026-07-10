"""
Generate synthetic PE (Portable Executable) feature data for malware detection.

Simulates the EMBER v2 feature space with realistic distributions.
Each sample has 2381 features mimicking the actual EMBER feature set:
  - Byte histogram (256)
  - Byte entropy histogram (256)
  - String info (10)
  - General file info (10)
  - Header info (10)
  - Section info (30+)
  - Imports info (1280+)
  - Exports info (128)
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

NUM_SAMPLES = 20000
MALWARE_RATIO = 0.35  # 35% malware


def generate_benign_features():
    """Generate feature vector for a benign executable."""
    feats = np.zeros(2381, dtype=np.float64)

    # Byte histogram (idx 0-255) — smoother, standard compiler output
    hist = np.random.dirichlet(np.ones(256) * 2.0) * 100
    feats[0:256] = hist

    # Byte entropy histogram (idx 256-511) — lower entropy regions
    entropy_hist = np.random.dirichlet(np.ones(256) * 5.0) * 100
    feats[256:512] = entropy_hist * np.random.uniform(0.3, 0.6, 256)

    # String info (idx 512-521)
    feats[512] = random.randint(50, 500)  # numstrings
    feats[513] = np.random.uniform(5, 30)  # avg_length
    feats[514] = np.random.uniform(0.01, 0.1)  # printable_ratio
    feats[515] = random.randint(5, 50)  # num_interesting
    feats[516] = np.random.uniform(0.001, 0.02)  # interesting_ratio
    feats[517] = random.randint(0, 10)  # num_urls
    feats[518] = np.random.uniform(0, 0.01)  # url_ratio
    feats[519] = np.random.uniform(0.1, 0.5)  # path_ratio
    feats[520] = np.random.uniform(0.1, 0.5)  # registry_ratio
    feats[521] = random.randint(0, 5)  # num_MZ

    # General info (idx 522-531)
    feats[522] = np.random.uniform(0.3, 1.0)  # size ratio
    feats[523] = np.random.uniform(1990, 2024)  # compile time
    feats[524] = random.randint(10000, 500000)  # virtual size
    feats[525] = np.random.uniform(0.1, 0.5)  # has_debug
    feats[526] = random.randint(1, 10)  # num_sections
    feats[527] = random.randint(0, 3)  # num_exports
    feats[528] = random.randint(5, 100)  # num_imports
    feats[529] = np.random.uniform(0.01, 0.2)  # export_ratio
    feats[530] = np.random.uniform(0.05, 0.3)  # import_ratio
    feats[531] = np.random.uniform(0.0, 0.5)  # overlay_ratio

    # Section info (idx 532-561)
    for i in range(30):
        feats[532 + i] = np.random.uniform(0, 1)

    # Imports info (idx 562-1841)
    num_imports = random.randint(5, 50)
    import_indices = random.sample(range(562, 1842), min(num_imports, 1280))
    for idx in import_indices:
        feats[idx] = np.random.uniform(0.5, 1.0)

    # Exports info (idx 1842-1969)
    for i in range(128):
        feats[1842 + i] = np.random.uniform(0, 0.3)

    # Remaining (idx 1970-2380) — misc features
    for i in range(1970, 2381):
        feats[i] = np.random.uniform(0, 0.1)

    return feats


def generate_malware_features():
    """Generate feature vector with malicious indicators."""
    feats = np.zeros(2381, dtype=np.float64)

    # Byte histogram — more variance, unusual distributions
    hist = np.random.dirichlet(np.ones(256) * 0.5) * 100
    feats[0:256] = hist * np.random.uniform(0.8, 1.5, 256)

    # Entropy histogram — higher entropy (packed/encrypted)
    entropy_hist = np.random.dirichlet(np.ones(256) * 1.0) * 100
    feats[256:512] = entropy_hist * np.random.uniform(0.6, 1.0, 256)

    # String info — more suspicious strings
    feats[512] = random.randint(10, 200)
    feats[513] = np.random.uniform(3, 20)
    feats[514] = np.random.uniform(0.0, 0.3)
    feats[515] = random.randint(20, 200)
    feats[516] = np.random.uniform(0.1, 0.8)
    feats[517] = random.randint(5, 50)
    feats[518] = np.random.uniform(0.01, 0.1)
    feats[519] = np.random.uniform(0.3, 0.9)
    feats[520] = np.random.uniform(0.2, 0.8)
    feats[521] = random.randint(3, 50)

    # General info — anomalies
    feats[522] = np.random.uniform(0.0, 0.3)
    feats[523] = np.random.uniform(1980, 2010)
    feats[524] = random.randint(5000, 200000)
    feats[525] = np.random.uniform(0.0, 0.1)
    feats[526] = random.randint(2, 8)
    feats[527] = random.randint(0, 5)
    feats[528] = random.randint(0, 30)
    feats[529] = np.random.uniform(0.0, 0.1)
    feats[530] = np.random.uniform(0.0, 0.2)
    feats[531] = np.random.uniform(0.2, 0.8)

    # Section info — unusual section characteristics
    for i in range(30):
        feats[532 + i] = np.random.uniform(0, 1) * np.random.choice([0.1, 1.5, 2.0])

    # Imports — fewer, more suspicious imports
    num_imports = random.randint(0, 20)
    import_indices = random.sample(range(562, 1842), min(num_imports, 1280))
    for idx in import_indices:
        feats[idx] = np.random.uniform(0.8, 1.0)

    # Suspicious API calls (simulated)
    for idx in random.sample(range(562, 600), min(random.randint(1, 10), 38)):
        feats[idx] = np.random.uniform(0.9, 1.0)

    # Exports — unusual export patterns
    for i in range(128):
        feats[1842 + i] = np.random.uniform(0, 0.5)

    # Misc features — noise
    for i in range(1970, 2381):
        feats[i] = np.random.uniform(0, 0.3)

    return feats


def main():
    DATA_DIR.mkdir(exist_ok=True)
    print("=" * 50)
    print("  PE Malware Detection — Feature Generator")
    print("=" * 50)

    n_malware = int(NUM_SAMPLES * MALWARE_RATIO)
    n_benign = NUM_SAMPLES - n_malware

    print(f"\n  Generating {n_benign} benign samples...")
    benign = np.array([generate_benign_features() for _ in tqdm(range(n_benign))])

    print(f"  Generating {n_malware} malware samples...")
    malware = np.array([generate_malware_features() for _ in tqdm(range(n_malware))])

    X = np.vstack([benign, malware])
    y = np.hstack([np.zeros(n_benign), np.ones(n_malware)])

    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    # Save as compressed numpy
    npz_path = DATA_DIR / "pe_features.npz"
    np.savez_compressed(npz_path, X=X, y=y)
    print(f"\n  Saved {len(X)} samples ({len(X)} features each)")
    print(f"  Malware ratio: {y.mean():.1%}")
    print(f"  File: {npz_path}")

    # Also save a smaller CSV preview (first 1000 samples)
    df = pd.DataFrame(X[:1000, :20])  # First 20 features only
    df["label"] = y[:1000].astype(int)
    df.to_csv(DATA_DIR / "pe_features_preview.csv", index=False)
    print(f"  Preview CSV: {DATA_DIR / 'pe_features_preview.csv'}")


if __name__ == "__main__":
    main()
