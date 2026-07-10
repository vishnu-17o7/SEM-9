"""
URL-based feature extraction for phishing detection.

Extracts 28+ numerical/categorical features from a raw URL
that discriminate between legitimate and phishing websites.
"""

import math
import re
import ipaddress
from collections import Counter
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import numpy as np
import pandas as pd
import tldextract
from typing import Union

SUSPICIOUS_WORDS = {
    "confirm", "banking", "password", "credential", "authenticate",
    "webscr", "phishing", "alert", "warning", "suspicious",
    "activity", "unusual", "blocked", "restricted", "limited",
    "required", "validate", "verification", "recover", "reset",
}

BRAND_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "instagram.com", "whatsapp.com",
    "amazon.com", "microsoft.com", "office.com", "live.com", "outlook.com",
    "apple.com", "icloud.com", "netflix.com", "paypal.com", "ebay.com",
    "linkedin.com", "twitter.com", "x.com", "github.com", "gitlab.com",
    "reddit.com", "dropbox.com", "adobe.com", "spotify.com", "roblox.com",
    "steampowered.com", "steamcommunity.com", "bankofamerica.com", "wellsfargo.com",
    "chase.com", "capitalone.com", "walmart.com", "bestbuy.com", "target.com",
    "fedex.com", "ups.com", "usps.com", "dhl.com", "irs.gov",
    "telegram.org", "discord.com", "slack.com", "zoom.us",
    "wordpress.com", "squarespace.com", "oracle.com", "ibm.com", "salesforce.com",
    "cloudflare.com", "wikipedia.org", "medium.com",
]

BRAND_NAMES = [
    "google", "youtube", "facebook", "instagram", "whatsapp", "amazon",
    "microsoft", "office365", "outlook", "apple", "icloud", "netflix",
    "paypal", "ebay", "linkedin", "twitter", "github", "gitlab", "reddit",
    "dropbox", "adobe", "spotify", "roblox", "steam", "bankofamerica",
    "wellsfargo", "chase", "capitalone", "walmart", "target", "bestbuy",
    "fedex", "ups", "usps", "dhl", "telegram", "discord", "slack", "zoom",
    "wordpress", "squarespace", "oracle", "ibm", "salesforce", "cloudflare",
    "wikipedia", "medium",
]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (c1 != c2)
            curr.append(min(insertions, deletions, substitutions))
        prev = curr
    return prev[-1]


SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "tiny.cc", "tr.im", "rb.gy", "shorturl.at",
    "cli.gs", "snipurl.com", "adf.ly", "cutt.ly", "short.link",
    "2.gp", "v.gd", "s.id", "x.co", "tiny.one", "shorte.st",
}

FEATURE_NAMES = [
    "url_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_question_marks",
    "num_equals",
    "num_at_symbols",
    "num_ampersands",
    "num_hash",
    "num_digits",
    "num_letters",
    "has_ip_address",
    "has_https",
    "path_length",
    "num_subdomains",
    "tld_length",
    "has_port",
    "num_query_params",
    "url_entropy",
    "domain_length",
    "has_double_slash_redirect",
    "special_char_ratio",
    "has_suspicious_words",
    "has_shortener",
    "digit_ratio",
    "path_depth",
    "has_www",
    "has_typosquatting",
    "has_brand_in_domain",
]


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    text = text.lower()
    counts = Counter(text)
    total = len(text)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return round(entropy, 4)


def is_ip_address(hostname: str) -> bool:
    if not hostname:
        return False
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        pass
    # Handle IPv6 in URLs like [::1]
    hostname = hostname.strip("[]")
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def extract_features(url: str) -> dict[str, Union[int, float]]:
    """Extract 28 hand-crafted features from a raw URL."""
    features: dict[str, Union[int, float]] = {}

    try:
        parsed = urlparse(url)
    except Exception:
        parsed = urlparse("")

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    # ---- Length-based features ----
    features["url_length"] = len(url)
    features["path_length"] = len(path)
    features["domain_length"] = len(hostname)

    # ---- Special character counts ----
    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    features["num_underscores"] = url.count("_")
    features["num_slashes"] = url.count("/")
    features["num_question_marks"] = url.count("?")
    features["num_equals"] = url.count("=")
    features["num_at_symbols"] = url.count("@")
    features["num_ampersands"] = url.count("&")
    features["num_hash"] = url.count("#")

    # ---- Digit / letter counts ----
    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    features["num_digits"] = digits
    features["num_letters"] = letters

    # ---- Presence features ----
    features["has_ip_address"] = 1 if is_ip_address(hostname) else 0
    features["has_https"] = 1 if parsed.scheme == "https" else 0
    features["has_port"] = 1 if parsed.port is not None else 0
    features["has_www"] = 1 if hostname.startswith("www.") else 0
    features["has_double_slash_redirect"] = 1 if "//" in path[1:] else 0  # skip leading /

    # ---- Subdomain / TLD features ----
    try:
        extracted = tldextract.extract(url)
        subdomain = extracted.subdomain
        tld = extracted.suffix
        features["num_subdomains"] = len([s for s in subdomain.split(".") if s])
        features["tld_length"] = len(tld)
    except Exception:
        features["num_subdomains"] = 0
        features["tld_length"] = 0

    # ---- Query parameter features ----
    try:
        params = parse_qs(query)
        features["num_query_params"] = len(params)
    except Exception:
        features["num_query_params"] = 0

    # ---- Entropy ----
    features["url_entropy"] = shannon_entropy(url)

    # ---- Ratio features ----
    total_len = len(url)
    special_chars = sum(not (c.isalnum() or c in ".-_/~") for c in url)
    features["special_char_ratio"] = round(special_chars / max(total_len, 1), 4)
    features["digit_ratio"] = round(digits / max(total_len, 1), 4)

    # ---- Suspicious content ----
    url_lower = url.lower()
    features["has_suspicious_words"] = 1 if any(w in url_lower for w in SUSPICIOUS_WORDS) else 0
    features["has_shortener"] = 1 if hostname.lower() in SHORTENER_DOMAINS else 0

    # ---- Path depth ----
    path_segments = [s for s in path.split("/") if s]
    features["path_depth"] = len(path_segments)

    # ---- Typosquatting / brand impersonation ----
    domain_lower = hostname.lower()
    registered_domain = domain_lower  # Will be refined with tldextract data

    try:
        extracted = tldextract.extract(url)
        registered_domain = f"{extracted.domain}.{extracted.suffix}" if extracted.suffix else extracted.domain
        domain_body = extracted.domain.lower()
    except Exception:
        # Fallback: extract the last two dot-separated parts
        parts = domain_lower.split(".")
        if len(parts) >= 2:
            registered_domain = ".".join(parts[-2:])
            domain_body = parts[-2]
        else:
            registered_domain = domain_lower
            domain_body = domain_lower

    # Check typosquatting: is the registered domain 1 edit from a known brand domain?
    typosquatting = 0
    for brand_domain in BRAND_DOMAINS:
        # Skip exact match (that's legitimate)
        if registered_domain == brand_domain:
            continue
        dist = levenshtein_distance(registered_domain, brand_domain)
        if dist <= 1:
            typosquatting = 1
            break
    features["has_typosquatting"] = typosquatting

    # Check brand-in-domain: does the domain body contain a brand name
    # as a hyphen-separated token? (substring match catches legitimate
    # brand-adjacent domains like "microsoft" in "microsoftonline.com")
    brand_in_domain = 0
    domain_tokens = domain_body.replace("-", " ").split()
    for brand_name in BRAND_NAMES:
        # Exact match on the domain body ≡ own domain → skip
        if domain_body == brand_name:
            continue
        # Check if any hyphen-separated token matches the brand name exactly
        if brand_name in domain_tokens:
            brand_in_domain = 1
            break
    features["has_brand_in_domain"] = brand_in_domain

    return features


FEATURE_NAMES = sorted(FEATURE_NAMES)


def feature_vector(url: str) -> np.ndarray:
    """Returns a numpy array of feature values in FEATURE_NAMES order."""
    feats = extract_features(url)
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float64)


def dataframe_from_urls(urls: list[str], labels: list[int] | None = None) -> pd.DataFrame:
    """Build a DataFrame of features from a list of URLs."""
    rows = []
    for url in urls:
        feats = extract_features(url)
        rows.append(feats)

    df = pd.DataFrame(rows)
    # Ensure all FEATURE_NAMES columns exist, fill missing with 0
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_NAMES]

    if labels is not None:
        df["label"] = labels

    return df


if __name__ == "__main__":
    # quick smoke test
    test_urls = [
        "https://www.google.com/search?q=hello",
        "http://192.168.1.1/login.php?cmd=verify",
        "http://bit.ly/3abcde",
        "https://paypal-secure-login.com/update/account/verify",
        "https://github.com/features/actions",
        "https://www.rnicrosoft.com",
        "https://g00gle.com/login",
        "https://amaz0n-secure.com/verify",
    ]
    df = dataframe_from_urls(test_urls)
    print(df.to_string())
    print(f"\nFeature vector shape: {df.shape}")
    for url, row in zip(test_urls, df.itertuples()):
        print(f"\n{url[:70]:70s} → has_typosquatting={row.has_typosquatting} has_brand_in_domain={row.has_brand_in_domain}")
