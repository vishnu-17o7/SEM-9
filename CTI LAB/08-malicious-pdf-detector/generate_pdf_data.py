"""
Generate synthetic PDF metadata for malicious/benign PDF classification.

Simulates features found in malicious PDFs:
- Embedded JavaScript / actions
- /OpenAction, /Launch, /URI actions
- Embedded files
- Object counts, compression ratios
- Entropy analysis
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

NUM_SAMPLES = 15000
MALICIOUS_RATIO = 0.30


def generate_benign_pdf_features():
    """Generate features for a benign PDF."""
    feats = {}
    # Basic structure
    feats["num_objects"] = random.randint(20, 200)
    feats["num_pages"] = random.randint(1, 50)
    feats["file_size_kb"] = np.random.lognormal(mean=3, sigma=1)
    feats["compression_ratio"] = np.random.uniform(0.3, 0.7)
    feats["num_fonts"] = random.randint(1, 15)
    feats["num_images"] = random.randint(0, 30)

    # Structure features
    feats["num_streams"] = random.randint(5, 100)
    feats["stream_to_obj_ratio"] = feats["num_streams"] / max(feats["num_objects"], 1)
    feats["avg_stream_length"] = np.random.lognormal(mean=6, sigma=2)

    # Metadata
    feats["has_metadata"] = 1 if random.random() < 0.8 else 0
    feats["has_outline"] = 1 if random.random() < 0.3 else 0
    feats["num_bookmarks"] = random.randint(0, 20)

    # Security features
    feats["has_js"] = 0  # benign PDFs usually don't have JS
    feats["num_js_actions"] = 0
    feats["has_openaction"] = 0
    feats["has_launch_action"] = 0
    feats["has_uri_action"] = 0
    feats["has_embedded_file"] = 0
    feats["num_embedded_files"] = 0
    feats["has_acroform"] = 1 if random.random() < 0.2 else 0
    feats["has_richmedia"] = 0

    # Suspicious indicators
    feats["num_suspicious_strings"] = 0
    feats["num_encoded_streams"] = random.randint(0, 3)
    feats["max_entropy"] = random.uniform(3.0, 6.0)
    feats["avg_entropy"] = random.uniform(2.0, 4.5)
    feats["entropy_variance"] = random.uniform(0.5, 2.0)

    # Reference patterns
    feats["has_obfuscated_names"] = 0
    feats["has_trailer_after_eof"] = 0
    feats["has_overlapping_objects"] = 0
    feats["has_malformed_crossref"] = 0

    # Font encoding
    feats["has_nonstandard_encoding"] = 0

    return feats


def generate_malicious_pdf_features():
    """Generate features for a malicious PDF with suspicious characteristics."""
    feats = {}
    feats["num_objects"] = random.randint(10, 500)
    feats["num_pages"] = random.randint(1, 20)
    feats["file_size_kb"] = np.random.lognormal(mean=2.5, sigma=1.5)
    feats["compression_ratio"] = np.random.uniform(0.1, 0.5)
    feats["num_fonts"] = random.randint(0, 5)
    feats["num_images"] = random.randint(0, 5)

    # Structure — often more streams, larger ones
    feats["num_streams"] = random.randint(1, 150)
    feats["stream_to_obj_ratio"] = feats["num_streams"] / max(feats["num_objects"], 1)
    feats["avg_stream_length"] = np.random.lognormal(mean=8, sigma=3)

    # Metadata — often missing or minimal
    feats["has_metadata"] = 1 if random.random() < 0.3 else 0
    feats["has_outline"] = 1 if random.random() < 0.1 else 0
    feats["num_bookmarks"] = random.randint(0, 3)

    # Security — malicious indicators
    feats["has_js"] = 1 if random.random() < 0.7 else 0
    feats["num_js_actions"] = random.randint(0, 10) if feats["has_js"] else 0
    feats["has_openaction"] = 1 if random.random() < 0.5 else 0
    feats["has_launch_action"] = 1 if random.random() < 0.3 else 0
    feats["has_uri_action"] = 1 if random.random() < 0.4 else 0
    feats["has_embedded_file"] = 1 if random.random() < 0.35 else 0
    feats["num_embedded_files"] = random.randint(1, 5) if feats["has_embedded_file"] else 0
    feats["has_acroform"] = 1 if random.random() < 0.3 else 0
    feats["has_richmedia"] = 1 if random.random() < 0.2 else 0

    # Suspicious indicators
    feats["num_suspicious_strings"] = random.randint(1, 50) if random.random() < 0.8 else 0
    feats["num_encoded_streams"] = random.randint(1, 30)
    feats["max_entropy"] = random.uniform(5.0, 8.0)
    feats["avg_entropy"] = random.uniform(3.5, 6.5)
    feats["entropy_variance"] = random.uniform(1.0, 4.0)

    # Malformed/obfuscated features
    feats["has_obfuscated_names"] = 1 if random.random() < 0.6 else 0
    feats["has_trailer_after_eof"] = 1 if random.random() < 0.4 else 0
    feats["has_overlapping_objects"] = 1 if random.random() < 0.3 else 0
    feats["has_malformed_crossref"] = 1 if random.random() < 0.25 else 0

    # Non-standard encoding
    feats["has_nonstandard_encoding"] = 1 if random.random() < 0.4 else 0

    return feats


FEATURE_NAMES = [
    "num_objects", "num_pages", "file_size_kb", "compression_ratio",
    "num_fonts", "num_images", "num_streams", "stream_to_obj_ratio",
    "avg_stream_length", "has_metadata", "has_outline", "num_bookmarks",
    "has_js", "num_js_actions", "has_openaction", "has_launch_action",
    "has_uri_action", "has_embedded_file", "num_embedded_files",
    "has_acroform", "has_richmedia", "num_suspicious_strings",
    "num_encoded_streams", "max_entropy", "avg_entropy", "entropy_variance",
    "has_obfuscated_names", "has_trailer_after_eof", "has_overlapping_objects",
    "has_malformed_crossref", "has_nonstandard_encoding",
]


def main():
    DATA_DIR.mkdir(exist_ok=True)
    print("=" * 50)
    print("  Malicious PDF Detection — Data Generator")
    print("=" * 50)

    n_mal = int(NUM_SAMPLES * MALICIOUS_RATIO)
    n_ben = NUM_SAMPLES - n_mal

    print(f"\n  Generating {n_ben} benign PDFs...")
    benign = [generate_benign_pdf_features() for _ in tqdm(range(n_ben))]

    print(f"  Generating {n_mal} malicious PDFs...")
    malicious = [generate_malicious_pdf_features() for _ in tqdm(range(n_mal))]

    all_feats = benign + malicious
    labels = [0] * n_ben + [1] * n_mal

    # Shuffle
    combined = list(zip(all_feats, labels))
    random.shuffle(combined)
    all_feats, labels = zip(*combined)

    df = pd.DataFrame(all_feats)
    df["label"] = labels

    out_path = DATA_DIR / "pdf_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Total: {len(df)} samples, {len(FEATURE_NAMES)} features")
    print(f"  Malicious ratio: {sum(labels) / len(labels):.1%}")
    print(f"  Saved to {out_path}")

    # Also save as npz
    X = df[FEATURE_NAMES].values.astype(np.float64)
    y = np.array(labels, dtype=np.int64)
    np.savez_compressed(DATA_DIR / "pdf_features.npz", X=X, y=y)


if __name__ == "__main__":
    main()
