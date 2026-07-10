"""
Build a labelled phishing-vs-legitimate URL dataset.

Downloads phishing URLs from open sources, generates legitimate
URLs from top-domain lists, extracts features, and saves the
resulting DataFrame for model training.
"""

import io
import math
import random
import csv
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from url_feature_extractor import dataframe_from_urls

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
DATASET_FILE = DATA_DIR / "phishing_url_dataset.csv"
FEATURES_FILE = DATA_DIR / "phishing_features.csv"
SPLITS_FILE = DATA_DIR / "dataset_splits.npz"

random.seed(42)

# ── Sources for phishing URLs ──────────────────────────────────────────────
PHISHING_SOURCES = [
    {
        "url": "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-domains-ACTIVE.txt",
        "type": "plain_list",
        "max": 15_000,
    },
    {
        "url": "https://openphish.com/feed.txt",
        "type": "plain_list",
        "max": 5_000,
    },
]

# ── Legitimate top domains ────────────────────────────────────────────────
LEGITIMATE_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "twitter.com", "instagram.com",
    "linkedin.com", "whatsapp.com", "amazon.com", "netflix.com", "microsoft.com",
    "apple.com", "github.com", "stackoverflow.com", "wikipedia.org", "reddit.com",
    "yahoo.com", "zoom.us", "maps.google.com", "mail.google.com", "drive.google.com",
    "docs.google.com", "play.google.com", "news.google.com", "bing.com",
    "office.com", "live.com", "msn.com", "adobe.com", "salesforce.com",
    "dropbox.com", "spotify.com", "telegram.org", "medium.com", "quora.com",
    "pinterest.com", "ebay.com", "walmart.com", "bestbuy.com", "target.com",
    "cnn.com", "bbc.com", "nytimes.com", "theguardian.com", "forbes.com",
    "weather.com", "imdb.com", "rottentomatoes.com", "cnet.com", "techcrunch.com",
    "wsj.com", "bloomberg.com", "reuters.com", "npr.org", "nationalgeographic.com",
    "nginx.org", "python.org", "docker.com", "npmjs.com", "stackoverflow.blog",
    "gitlab.com", "bitbucket.org", "atlassian.com", "jira.com", "confluence.atlassian.com",
    "slack.com", "discord.com", "teams.microsoft.com", "trello.com", "asana.com",
    "notion.so", "evernote.com", "canva.com", "figma.com", "dribbble.com",
    "behance.net", "unsplash.com", "pexels.com", "shutterstock.com", "gettyimages.com",
    "archive.org", "academia.edu", "researchgate.net", "scholar.google.com",
    "britannica.com", "merriam-webster.com", "dictionary.com", "thesaurus.com",
    "webmd.com", "mayoclinic.org", "nih.gov", "who.int", "cdc.gov",
    "nasa.gov", "whitehouse.gov", "usa.gov", "state.gov", "loc.gov",
    "harvard.edu", "mit.edu", "stanford.edu", "berkeley.edu", "ox.ac.uk",
    "cam.ac.uk", "ucla.edu", "umich.edu", "yale.edu", "princeton.edu",
    "booking.com", "expedia.com", "airbnb.com", "uber.com", "lyft.com",
    "doordash.com", "grubhub.com", "seamless.com", "yelp.com", "tripadvisor.com",
    "chase.com", "bankofamerica.com", "wellsfargo.com", "citi.com", "capitalone.com",
    "paypal.com", "stripe.com", "square.com", "venmo.com", "mastercard.com",
    "visa.com", "americanexpress.com", "discover.com", "nerdwallet.com", "creditkarma.com",
]


def _random_path(depth: int = None) -> str:
    """Generate a random URL path segment."""
    if depth is None:
        depth = random.randint(0, 2)  # 0=root, 1 or 2=shallow path (33% root)
    segments = []
    words = [
        "about", "products", "services", "blog", "news", "contact",
        "support", "help", "faq", "terms", "privacy", "careers",
        "pricing", "features", "docs", "api", "status", "community",
        "login", "signup", "download", "app", "dashboard", "settings",
        "profile", "account", "billing", "notifications", "messages",
        "search", "explore", "trending", "popular", "recent", "top",
        "images", "videos", "music", "games", "books", "travel",
    ]
    for _ in range(depth):
        word = random.choice(words)
        if random.random() < 0.2:
            word += str(random.randint(1, 999))
        segments.append(word)
    return "/" + "/".join(segments) + ("/" if random.random() < 0.3 else "")


def _random_query() -> str:
    """Generate a random query string."""
    if random.random() < 0.5:
        return ""
    params = []
    keys = ["q", "s", "id", "page", "ref", "utm_source", "utm_medium", "lang", "sort", "filter"]
    for _ in range(random.randint(1, 4)):
        key = random.choice(keys)
        values = ["hello", "test", "search-term", "12345", "abc", "home", "index"]
        params.append(f"{key}={random.choice(values)}")
    return "?" + "&".join(params)


def _generate_legitimate_urls(n: int = 10_000) -> list[str]:
    """Generate plausible legitimate URLs from known top domains."""
    urls = []
    while len(urls) < n:
        domain = random.choice(LEGITIMATE_DOMAINS)
        scheme = "https" if random.random() < 0.95 else "http"
        path = _random_path()
        query = _random_query()

        # Occasionally add subdomain variations
        if random.random() < 0.15 and "." not in domain.split(".")[0]:
            sub = random.choice(["www", "mail", "app", "blog", "shop", "help", "api", "docs"])
            domain = f"{sub}.{domain}"

        url = f"{scheme}://{domain}{path}{query}"
        urls.append(url)

    return urls[:n]


def _generate_phishing_variants(domain: str) -> list[str]:
    """Generate multiple phishing URL variants from a single phishing domain."""
    variants = []
    paths = [
        "/login", "/verify", "/account/update", "/secure", "/signin",
        "/password-reset", "/confirm", "/authenticate", "/banking",
        "/webscr", "/cmd=login", "/payment/verify", "/security-check",
        "/alert/account", "/restricted", "/limited-access",
        "/recover/verify", "/validate/session", "/challenge/auth",
    ]
    for path in paths:
        scheme = "https" if random.random() < 0.6 else "http"
        qs = "?cmd=login" if random.random() < 0.3 else ""
        variants.append(f"{scheme}://{domain}{path}{qs}")
    return variants


# Typosquatting character substitutions for synthetic phishing generation
TYPO_MAP = {
    "a": ["s", "q", "z"], "b": ["v", "n", "g"], "c": ["x", "v", "d"],
    "d": ["s", "f", "c"], "e": ["w", "r", "d"], "f": ["d", "g", "v"],
    "g": ["f", "h", "b"], "h": ["g", "j", "n"], "i": ["u", "o", "k"],
    "j": ["k", "h", "i"], "k": ["j", "l", "i"], "l": ["k", "m", "o"],
    "m": ["n", "l"], "n": ["m", "b"], "o": ["i", "p", "l"],
    "p": ["o", "l"], "q": ["w", "a"], "r": ["t", "e", "n"],
    "s": ["a", "d", "z"], "t": ["y", "r", "f"], "u": ["y", "i", "v"],
    "v": ["c", "b", "u"], "w": ["q", "e", "s"], "x": ["z", "c"],
    "y": ["t", "u", "z"], "z": ["a", "x", "s"],
    "0": ["o", "9"], "1": ["l", "i"], "2": ["z"], "3": ["e"],
    "4": ["a"], "5": ["s"], "6": ["b"], "7": ["t"], "8": ["b"], "9": ["g", "0"],
}


def _typosquat_domain(domain: str) -> str:
    """Generate a typosquatted version of a domain name."""
    idx = random.randint(0, len(domain) - 1)
    char = domain[idx]
    if char in TYPO_MAP and TYPO_MAP[char]:
        replacement = random.choice(TYPO_MAP[char])
        return domain[:idx] + replacement + domain[idx + 1:]
    # Fallback: double a character or remove it
    if random.random() < 0.5:
        return domain[:idx] + char + char + domain[idx:]
    else:
        return domain[:idx] + domain[idx + 1:] if len(domain) > 1 else domain


def _generate_typosquatting_examples(n: int = 1000) -> list[str]:
    """Generate typosquatted versions of known brand domains."""
    brands_with_tlds = [
        ("paypal", "com"), ("ebay", "com"), ("amazon", "com"), ("apple", "com"),
        ("microsoft", "com"), ("google", "com"), ("netflix", "com"),
        ("facebook", "com"), ("instagram", "com"), ("whatsapp", "com"),
        ("linkedin", "com"), ("twitter", "com"), ("github", "com"),
        ("chase", "com"), ("bankofamerica", "com"), ("wellsfargo", "com"),
        ("capitalone", "com"), ("walmart", "com"), ("bestbuy", "com"),
    ]
    urls = []
    for _ in range(n):
        brand, tld = random.choice(brands_with_tlds)
        squatted = _typosquat_domain(brand)
        # Skip if the squatted version is the same as the original
        if squatted == brand:
            continue
        domain = f"{squatted}.{tld}"
        scheme = "https" if random.random() < 0.6 else "http"
        path = _random_path(random.randint(1, 3))
        urls.append(f"{scheme}://{domain}{path}")
    return urls


def download_phishing_urls(max_total: int = 15_000) -> list[str]:
    """Download phishing URLs from open sources. Returns unique URLs."""
    all_urls: list[str] = []
    sources_tried = 0

    for source in PHISHING_SOURCES:
        try:
            resp = requests.get(source["url"], timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            lines = resp.text.strip().splitlines()
            urls = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Some sources list domains only, some list full URLs
                if line.startswith("http://") or line.startswith("https://"):
                    urls.append(line)
                else:
                    # It's a domain — prefix with https
                    urls.append(f"https://{line}")
            urls = list(set(urls))
            print(f"  ✓ {source['url'][:70]:70s} → {len(urls)} URLs")
            all_urls.extend(urls)
            sources_tried += 1
        except Exception as e:
            print(f"  ✗ {source['url'][:70]:70s} → failed ({e})")

    unique = list(set(all_urls))
    print(f"\n  Total unique phishing URLs downloaded: {len(unique)}")

    if not unique:
        print("  ⚠ No online sources available — generating synthetic phishing URLs.")
        # Generate synthetic phishing domains based on common brand impersonations
        brands = [
            "paypal", "ebay", "amazon", "apple", "microsoft", "google",
            "netflix", "facebook", "instagram", "whatsapp", "linkedin",
            "twitter", "bankofamerica", "chase", "wellsfargo", "citi",
        ]
        tlds = [".com", ".org", ".net", ".xyz", ".top", ".club", ".online", ".live"]
        for brand in brands:
            for variant in range(100):
                domain = f"{brand}-secure{random.randint(1,999)}{random.choice(tlds)}"
                all_urls.extend(_generate_phishing_variants(domain))
        # Also generate typosquatting examples
        all_urls.extend(_generate_typosquatting_examples(2000))
        unique = list(set(all_urls))

    return unique[:max_total]


EXTERNAL_DATASET_URL = "https://huggingface.co/datasets/ealvaradob/phishing-dataset/resolve/main/urls.json"


def download_external_dataset(max_per_class: int = 25_000) -> tuple[list[str], list[int]]:
    """Download real-world phishing URL dataset from Hugging Face (835K URLs, Apache 2.0)."""
    import json

    dataset_path = DATA_DIR / "phish_urls.json"

    if not dataset_path.exists():
        print(f"  Downloading external dataset (69 MB)...")
        resp = requests.get(EXTERNAL_DATASET_URL, timeout=300)
        with open(dataset_path, "wb") as f:
            f.write(resp.content)
        print(f"  ✓ Downloaded to {dataset_path}")
    else:
        print(f"  ✓ Using cached dataset")

    print(f"  Loading JSON...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"  Total entries: {len(data):,}")

    # Split by label
    phishing_urls_raw = [d["text"] for d in data if d["label"] == 1]
    legit_urls_raw = [d["text"] for d in data if d["label"] == 0]
    print(f"  Phishing: {len(phishing_urls_raw):,}  Legitimate: {len(legit_urls_raw):,}")

    # Sample balanced subset
    random.shuffle(phishing_urls_raw)
    random.shuffle(legit_urls_raw)
    n = min(max_per_class, len(phishing_urls_raw), len(legit_urls_raw))
    phishing_sample = phishing_urls_raw[:n]

    # Bias in real datasets: legitimate URLs tend to have long paths while
    # phishing URLs are often bare domains. Counteract by augmenting the
    # legitimate set with bare-domain URLs.
    legit_sample = list(legit_urls_raw[:n])
    # Add ~10K root-domain legitimate URLs (one per domain with some repeats)
    # to balance the 80:20 phishing bias on path_depth=0
    import itertools
    root_domain_urls = [f"https://{d}" for d in LEGITIMATE_DOMAINS]
    while len(root_domain_urls) < 10000:
        root_domain_urls.append(f"https://www.{random.choice(LEGITIMATE_DOMAINS)}")
    legit_sample.extend(root_domain_urls[:10000])
    random.shuffle(legit_sample)

    # Normalize: add scheme if missing
    def normalize(url):
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        return url

    phishing_urls = [normalize(u) for u in phishing_sample]
    legit_urls = [normalize(u) for u in legit_sample]

    print(f"  Using {len(phishing_urls)} phishing + {len(legit_urls)} legitimate")

    all_urls = phishing_urls + legit_urls
    labels = [1] * len(phishing_urls) + [0] * len(legit_urls)
    return all_urls, labels


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build phishing URL dataset")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data only (no external download)")
    parser.add_argument("--max-urls", type=int, default=25_000,
                        help="Max URLs per class (default: 25000)")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  Phishing URL Dataset Builder")
    print("=" * 60)

    if args.synthetic:
        # 1. Download phishing URLs
        print("\n[1/4] Downloading phishing URLs...")
        phishing_urls = download_phishing_urls(max_total=args.max_urls)
        print(f"  Phishing URLs: {len(phishing_urls)}")

        # 2. Generate legitimate URLs
        print("\n[2/4] Generating legitimate URLs...")
        n_legit = min(len(phishing_urls), args.max_urls)
        legit_urls = _generate_legitimate_urls(n=n_legit)
        print(f"  Legitimate URLs: {len(legit_urls)}")

        all_urls = phishing_urls + legit_urls
        labels = [1] * len(phishing_urls) + [0] * len(legit_urls)
    else:
        # Use external real-world dataset
        print("\n[1/2] Loading external URL dataset (835K real URLs)...")
        all_urls, labels = download_external_dataset(max_per_class=args.max_urls)
        print(f"  Total URLs: {len(all_urls)} ({sum(labels)} phishing, {len(labels)-sum(labels)} legitimate)")

    # 2/3. Extract features
    print(f"\n[{3 if args.synthetic else 2}/4] Extracting features...")
    df = dataframe_from_urls(all_urls, labels)
    print(f"  Feature matrix: {df.shape[0]} samples × {df.shape[1]} features")
    print(f"  Class distribution:\n    phishing (1): {df['label'].sum()}\n    legitimate (0): {len(df) - df['label'].sum()}")

    # 4. Save
    step = 4 if args.synthetic else 3
    print(f"\n[{step}/4] Saving datasets...")

    # Save raw URLs with labels
    raw_df = pd.DataFrame({"url": all_urls, "label": labels})
    raw_df.to_csv(DATASET_FILE, index=False)
    print(f"  ✓ Raw URLs: {DATASET_FILE}")

    # Save feature matrix
    df.to_csv(FEATURES_FILE, index=False)
    print(f"  ✓ Features: {FEATURES_FILE}")

    print("\n" + "=" * 60)
    print("  Dataset build complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
