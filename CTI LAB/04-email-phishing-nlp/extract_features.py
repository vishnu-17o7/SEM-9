"""
Feature extraction from raw email content for phishing detection.

Extracts:
- Header features: anomalies, routing, sender info
- Body features: text statistics, URL analysis, NLP
- Combined feature vector
"""

import re
import hashlib
from pathlib import Path
from email import policy
from email.parser import BytesParser

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

SUSPICIOUS_KEYWORDS = [
    "urgent", "click here", "free", "win", "prize", "congratulations",
    "account suspended", "verify now", "update your account", "limited time",
    "act now", "exclusive offer", "nigerian", "lottery", "inheritance",
    "wire transfer", "western union", "money gram", "password expired",
    "security alert", "login attempt", "unusual activity", "confirm identity",
    "dear customer", "dear valued", "dear user", "dear account holder",
    "risk", "suspicious", "blocked", "restricted", "terminated",
]

SUSPICIOUS_TLD_PATTERNS = re.compile(
    r"\.(tk|ml|ga|cf|gq|xyz|top|work|download|bid|loan|date|racing|accountant|science)",
    re.IGNORECASE,
)


def parse_email(raw_bytes: bytes) -> dict:
    """Parse raw email bytes into a structured dict."""
    parser = BytesParser(policy=policy.default)
    msg = parser.parsebytes(raw_bytes)

    headers = dict(msg.items())
    body_parts = []
    html_parts = []

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                try:
                    body_parts.append(part.get_content())
                except Exception:
                    pass
            elif ctype == "text/html":
                try:
                    html_parts.append(part.get_content())
                except Exception:
                    pass
    else:
        try:
            ctype = msg.get_content_type()
            content = msg.get_content()
            if ctype == "text/plain":
                body_parts.append(content)
            elif ctype == "text/html":
                html_parts.append(content)
            else:
                body_parts.append(content)
        except Exception:
            pass

    return {
        "headers": headers,
        "body": "\n".join(body_parts),
        "html": "\n".join(html_parts),
        "raw": raw_bytes,
    }


def extract_header_features(parsed: dict) -> dict:
    """Extract features from email headers."""
    headers = parsed["headers"]
    features = {}

    # Number of recipients
    to_val = headers.get("To", "")
    cc_val = headers.get("Cc", "")
    features["num_recipients"] = len(to_val.split(",")) if to_val else 0
    features["has_cc"] = 1 if cc_val else 0

    # Reply-To mismatch
    reply_to = headers.get("Reply-To", "")
    from_val = headers.get("From", "")
    if reply_to and from_val:
        reply_domain = reply_to.split("@")[-1] if "@" in reply_to else ""
        from_domain = from_val.split("@")[-1] if "@" in from_val else ""
        features["reply_to_mismatch"] = 1 if reply_domain != from_domain else 0
    else:
        features["reply_to_mismatch"] = 0

    # Return-Path vs From mismatch
    return_path = headers.get("Return-Path", "")
    if return_path and from_val:
        rp_addr = return_path.strip("<>")
        from_addr = from_val.split("<")[-1].strip(">") if "<" in from_val else from_val
        features["return_path_mismatch"] = 1 if rp_addr != from_addr else 0
    else:
        features["return_path_mismatch"] = 0

    # Missing headers
    features["missing_message_id"] = 1 if "Message-ID" not in headers else 0
    features["missing_date"] = 1 if "Date" not in headers else 0
    features["missing_from"] = 1 if "From" not in headers else 0

    # X-Mailer presence
    features["has_x_mailer"] = 1 if "X-Mailer" in headers else 0
    features["has_x_priority"] = 1 if any(k in headers for k in ["X-Priority", "X-MSMail-Priority", "Importance"]) else 0

    # High priority flag
    priority_vals = [headers.get(k, "").lower() for k in ["X-Priority", "X-MSMail-Priority", "Importance"]]
    features["high_priority"] = 1 if any(v in p for p in priority_vals for v in ["high", "1", "urgent"]) else 0

    # Number of received hops (routing complexity)
    received = headers.get_all("Received", [])
    features["received_hops"] = len(received)

    # Subject suspicious indicators
    subject = headers.get("Subject", "")
    subject_lower = subject.lower()
    features["subject_has_money"] = 1 if re.search(r"\$|money|cash|payment|transfer", subject_lower) else 0
    features["subject_has_urgent"] = 1 if re.search(r"urgent|immediate|alert|warning|verify|suspend", subject_lower) else 0
    features["subject_all_caps_ratio"] = sum(1 for c in subject if c.isupper()) / max(len(subject), 1)

    return features


def extract_body_features(parsed: dict) -> dict:
    """Extract features from email body content."""
    body = parsed["body"]
    html = parsed["html"]
    combined = body + " " + html
    features = {}

    # Length features
    features["body_length"] = len(body)
    features["html_length"] = len(html)
    features["html_ratio"] = len(html) / max(len(combined), 1)

    # URL analysis
    urls = re.findall(r"https?://[^\s<>\"']+", combined)
    features["num_urls"] = len(urls)
    features["num_unique_urls"] = len(set(urls))

    # Suspicious TLDs
    sus_tlds = sum(1 for u in urls if SUSPICIOUS_TLD_PATTERNS.search(u))
    features["suspicious_tlds"] = sus_tlds

    # IP-based URLs
    ip_urls = sum(1 for u in urls if re.search(r"https?://\d+\.\d+\.\d+\.\d+", u))
    features["ip_based_urls"] = ip_urls

    # URL shorteners
    shorteners = [
        "bit.ly", "tinyurl", "goo.gl", "ow.ly", "is.gd", "buff.ly",
        "tiny.cc", "tr.im", "cli.gs", "mcaf.ee", "shorturl", "short.link",
    ]
    features["url_shorteners"] = sum(1 for u in urls if any(s in u.lower() for s in shorteners))

    # Suspicious keywords
    body_lower = combined.lower()
    keyword_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in body_lower)
    features["suspicious_keyword_count"] = keyword_count
    features["suspicious_keyword_ratio"] = keyword_count / max(len(SUSPICIOUS_KEYWORDS), 1)

    # Exclamation/question mark counts
    features["exclamation_count"] = body.count("!")
    features["question_count"] = body.count("?")
    features["caps_ratio"] = sum(1 for c in body if c.isupper()) / max(len(body), 1)

    # Dollar sign (currency indicator)
    features["dollar_signs"] = body.count("$")

    # HTML form / input tags (phishing forms)
    features["has_html_form"] = 1 if re.search(r"<form|<input|<select|<textarea", html, re.IGNORECASE) else 0

    # JavaScript presence
    features["has_javascript"] = 1 if re.search(r"<script|javascript:|onclick=|onload=|onmouse", html, re.IGNORECASE) else 0

    # Entropy of body text (high entropy = obfuscation)
    if body:
        char_counts = {}
        for c in body:
            char_counts[c] = char_counts.get(c, 0) + 1
        probs = [c / len(body) for c in char_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        features["text_entropy"] = round(entropy, 4)
    else:
        features["text_entropy"] = 0

    return features


def extract_all_features(email_text: str) -> dict:
    """
    Extract combined features from an email text string.
    
    If the text looks like raw email (has headers), parse it fully.
    Otherwise treat it as plain body text.
    """
    # Check if this looks like a raw email with headers
    has_headers = any(
        line.startswith(h) for line in email_text.split("\n")[:20]
        for h in ["From:", "To:", "Subject:", "Date:", "Message-ID:"]
    )

    if has_headers:
        parsed = parse_email(email_text.encode("utf-8", errors="replace"))
        header_feats = extract_header_features(parsed)
        body_feats = extract_body_features(parsed)
    else:
        # Treat as plain body text
        parsed = {"body": email_text, "html": "", "headers": {}}
        header_feats = {
            "num_recipients": 0, "has_cc": 0, "reply_to_mismatch": 0,
            "return_path_mismatch": 0, "missing_message_id": 0, "missing_date": 0,
            "missing_from": 0, "has_x_mailer": 0, "has_x_priority": 0,
            "high_priority": 0, "received_hops": 0,
            "subject_has_money": 0, "subject_has_urgent": 0, "subject_all_caps_ratio": 0,
        }
        body_feats = extract_body_features(parsed)

    features = {}
    features.update(header_feats)
    features.update(body_feats)
    return features


FEATURE_NAMES = [
    # Header features
    "num_recipients", "has_cc", "reply_to_mismatch", "return_path_mismatch",
    "missing_message_id", "missing_date", "missing_from",
    "has_x_mailer", "has_x_priority", "high_priority", "received_hops",
    "subject_has_money", "subject_has_urgent", "subject_all_caps_ratio",
    # Body features
    "body_length", "html_length", "html_ratio",
    "num_urls", "num_unique_urls", "suspicious_tlds", "ip_based_urls",
    "url_shorteners", "suspicious_keyword_count", "suspicious_keyword_ratio",
    "exclamation_count", "question_count", "caps_ratio", "dollar_signs",
    "has_html_form", "has_javascript", "text_entropy",
]


def features_to_vector(features: dict) -> np.ndarray:
    """Convert feature dict to numpy array in FEATURE_NAMES order."""
    return np.array([features.get(name, 0) for name in FEATURE_NAMES], dtype=np.float64)


if __name__ == "__main__":
    # Quick test
    test_email = """From: attacker@phishing.xyz
To: victim@gmail.com
Subject: URGENT: Your Account Has Been Suspended

Dear customer,
Your account has been suspended due to unusual activity.
Please click here to verify: http://malicious.tk/verify
$$$
This is an urgent matter that requires immediate attention.
"""
    feats = extract_all_features(test_email)
    for k, v in sorted(feats.items()):
        print(f"  {k:30s} = {v}")
