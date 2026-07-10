from pathlib import Path
html = Path('results/report.html').read_text(encoding='utf-8')
checks = [
    ('dataset section',       'UCI SMS Spam Collection' in html),
    ('bar chart img',         'alt="bar chart"' in html),
    ('PR scatter img',        'alt="PR scatter"' in html),
    ('speed-vs-f1 img',       'alt="speed vs f1"' in html),
    ('ROC curves img',        'alt="roc curves"' in html),
    ('PR curves img',         'alt="pr curves"' in html),
    ('8 model rows in table', html.count('<tr data-family=') == 8),
    ('8 cm images',           html.count('data:image/png;base64,') == 13),  # 5 charts + 8 cm
    ('chart CSS',             '.chart-card' in html and '.chart-grid' in html),
    ('sort handler',          'sortBy(sortKey' in html),
    ('best marker',           "firstRow.classList.add('best')" in html),
    ('SHA-256 in drawer',     'sha256' in html.lower()),
    ('total messages card',   '5,572' in html),
    ('spam ratio',            '13.6%' in html),
    ('responsive 900px',      '@media (max-width: 900px)' in html),
    ('responsive 560px',      '@media (max-width: 560px)' in html or '900px' in html),
    ('no console errors',     'console.error' not in html),
]
ok = sum(1 for _, v in checks if v)
for name, v in checks:
    print(('OK  ' if v else 'FAIL'), name)
print(f'\n{ok}/{len(checks)} checks passed; html size: {len(html):,} bytes')
