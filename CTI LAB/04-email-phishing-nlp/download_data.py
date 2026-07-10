"""
Download email datasets for phishing detection.
Primary: SpamAssassin public corpus.
Fallback: OpenML spambase.
"""

import tarfile
import io
from pathlib import Path

import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
SA_DIR = DATA_DIR / "spamassassin"

SPAMASSASSIN_FILES = {
    "20021010_easy_ham.tar.bz2": "https://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2",
    "20021010_spam.tar.bz2": "https://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2",
    "20030228_easy_ham.tar.bz2": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
    "20030228_spam.tar.bz2": "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
    "20030228_hard_ham.tar.bz2": "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
}


def download_file(url: str, dest: Path) -> bool:
    """Download a file with progress bar. Returns True if successful."""
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            desc=dest.name,
            total=total,
            unit="B",
            unit_scale=True,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  Failed to download {url}: {e}")
        return False


def extract_tarbz2(path: Path) -> bool:
    """Extract a .tar.bz2 file. Returns True if successful."""
    try:
        with tarfile.open(path, "r:bz2") as tar:
            tar.extractall(path=SA_DIR)
        return True
    except Exception as e:
        print(f"  Failed to extract {path.name}: {e}")
        return False


def download_spamassassin() -> int:
    """Download and extract SpamAssassin corpus. Returns number of emails."""
    SA_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for fname, url in SPAMASSASSIN_FILES.items():
        dest = SA_DIR / fname
        if dest.exists():
            print(f"  {fname} already exists, skipping")
        else:
            print(f"  Downloading {fname}...")
            if not download_file(url, dest):
                print(f"  Skipping {fname} due to download failure")
                continue
        if extract_tarbz2(dest):
            # Count emails in extracted directories
            for subdir in SA_DIR.iterdir():
                if subdir.is_dir() and not subdir.name.endswith(".tar.bz2"):
                    count += len(list(subdir.glob("*")))
    return count


def download_spambase_from_openml() -> tuple:
    """Fallback: download spambase from OpenML."""
    print("\n  Downloading Spambase from OpenML (fallback)...")
    try:
        from sklearn.datasets import fetch_openml
        spambase = fetch_openml(name="spambase", version=1, as_frame=True)
        return spambase.data, spambase.target.astype(int)
    except Exception as e:
        print(f"  OpenML download failed: {e}")
        return None, None


def main():
    print("=" * 50)
    print("  Email Phishing Dataset Downloader")
    print("=" * 50)

    # Try SpamAssassin first
    print("\n[1/2] SpamAssassin public corpus...")
    email_count = download_spamassassin()
    if email_count > 0:
        print(f"  Extracted ~{email_count} emails to {SA_DIR}")
    else:
        print("  No emails extracted from SpamAssassin.")

    # Always download OpenML spambase as guaranteed fallback
    print("\n[2/2] OpenML Spambase (fallback/alternative)...")
    X, y = download_spambase_from_openml()
    if X is not None:
        import pandas as pd
        df = pd.concat([X, pd.Series(y, name="label")], axis=1)
        df.to_csv(DATA_DIR / "spambase.csv", index=False)
        print(f"  Saved spambase ({len(df)} samples) to data/spambase.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
