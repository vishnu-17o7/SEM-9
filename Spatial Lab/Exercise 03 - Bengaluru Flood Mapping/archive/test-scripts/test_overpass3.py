"""Test Overpass API with form-encoded data."""
import json
import urllib.parse

import requests

query = "[out:json];node(12.97,77.59,12.98,77.60)[amenity=school];out;"

# Method 1: form-encoded (like urllib does it)
data = urllib.parse.urlencode({"data": query}).encode()
headers = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}

try:
    r = requests.post(
        "https://overpass-api.de/api/interpreter",
        data=data,
        headers=headers,
        timeout=30,
    )
    print(f"Method 1 (form-encoded) - Status: {r.status_code}")
    if r.status_code == 200:
        j = r.json()
        print(f"  Elements: {len(j.get('elements', []))}")
    else:
        print(f"  Response: {r.text[:200]}")
except Exception as e:
    print(f"  Error: {e}")

# Method 2: use params dict
try:
    r2 = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query},
        timeout=30,
    )
    print(f"Method 2 (dict data) - Status: {r2.status_code}")
    if r2.status_code == 200:
        j2 = r2.json()
        print(f"  Elements: {len(j2.get('elements', []))}")
    else:
        print(f"  Response: {r2.text[:200]}")
except Exception as e:
    print(f"  Error: {e}")
