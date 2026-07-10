import sys
sys.stdout.reconfigure(encoding='utf-8')

from url_feature_extractor import extract_features, FEATURE_NAMES
import numpy as np
import joblib

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
]

for url in urls:
    feats = extract_features(url)
    X = np.array([[feats[n] for n in FEATURE_NAMES]], dtype=np.float64)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0, 1]
    sigs = [k for k, v in feats.items() if v > 0 and k not in ('has_https', 'has_www')]
    print(f'{"MALICIOUS" if pred == 1 else "LEGIT":10s} ({prob:.0%})  {url:45s}  {str(sigs[:6]):50s}')
