import sys
sys.stdout.reconfigure(encoding='utf-8')

import tldextract
from url_feature_extractor import extract_features, FEATURE_NAMES, BRAND_DOMAINS
import numpy as np
import joblib
from urllib.parse import urlparse

model = joblib.load('results/models/VotingEnsemble.joblib')

urls = [
    'https://google.com',
    'https://www.google.com', 
    'https://www.google.com/search?q=hello',
    'https://mail.google.com',
    'https://google.co.in',
    'https://www.facebook.com',
    'https://www.amazon.com',
    'https://www.microsoft.com',
    'https://www.apple.com',
    'https://github.com/login',
    'https://github.com/features/actions',
    'https://login.microsoftonline.com/common',
    'https://stackoverflow.com/questions/tagged/python',
    'https://netflix.com/login',
    'https://youtube.com/watch?v=dQw4w9WgXcQ',
    'https://paypal.com/signin',
    'https://www.dropbox.com/login',
    # Phishing
    'https://paypal-secure-login.com/update/account/verify',
    'http://bit.ly/3abcde',
    'http://192.168.1.1/login.php?cmd=verify',
]

print(f"{'RESULT':10s} {'PROB':6s}  URL")
print("-" * 100)
for url in urls:
    feats = extract_features(url)
    X = np.array([[feats[n] for n in FEATURE_NAMES]], dtype=np.float64)
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0, 1])

    # apply same whitelist
    try:
        extracted = tldextract.extract(url)
        registered_domain = f"{extracted.domain}.{extracted.suffix}" if extracted.suffix else extracted.domain
        if registered_domain in BRAND_DOMAINS:
            pred = 0
            prob = 0.01
    except Exception:
        pass

    print(f'{"MALICIOUS" if pred == 1 else "LEGIT":10s} ({prob:.0%})  {url[:75]}')
