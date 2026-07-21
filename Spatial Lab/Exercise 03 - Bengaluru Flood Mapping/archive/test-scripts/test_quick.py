"""Quick Overpass API test."""
import json
import requests

headers = {"User-Agent": "Mozilla/5.0", "Accept": "*/*", "Content-Type": "text/plain"}
data = "[out:json];node(12.97,77.59,12.98,77.60)[amenity=school];out;"

r = requests.post(
    "https://overpass-api.de/api/interpreter",
    data=data.encode("utf-8"),
    headers=headers,
    timeout=30,
)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    j = r.json()
    print(f"Elements: {len(j.get('elements', []))}")
else:
    print(f"Text: {r.text[:300]}")
