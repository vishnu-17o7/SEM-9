import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd

df = pd.read_csv('data/phishing_features.csv')
legit = df[df['label'] == 0]
phish = df[df['label'] == 1]

print("=== Path depth distribution (after balancing) ===")
print(f"{'path_depth':>12s}  {'Legitimate':>12s}  {'Phishing':>12s}  {'Phish %':>8s}")
for d in sorted(df['path_depth'].unique()):
    l = len(legit[legit['path_depth'] == d])
    p = len(phish[phish['path_depth'] == d])
    pct = p / (l + p) * 100 if (l + p) > 0 else 0
    print(f"{d:>12d}  {l:>12d}  {p:>12d}  {pct:>7.1f}%")

print(f"\nRoot domains (path_depth=0): {len(legit[legit['path_depth']==0])} legit, {len(phish[phish['path_depth']==0])} phishing")
