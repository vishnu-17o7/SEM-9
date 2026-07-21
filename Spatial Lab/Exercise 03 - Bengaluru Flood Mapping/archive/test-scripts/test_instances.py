"""Test multiple Overpass instances."""
import json
import time
import requests

query = "[out:json];node(12.97,77.59,12.98,77.60)[amenity=school];out;"

INSTANCES = [
    ("Main", "https://overpass-api.de/api/interpreter"),
    ("Kumi", "https://overpass.kumi.systems/api/interpreter"),
    ("OSM RU", "https://overpass.openstreetmap.ru/api/interpreter"),
    ("Swiss", "https://overpass.osm.ch/api/interpreter"),
]

for name, url in INSTANCES:
    try:
        headers = {"User-Agent": "Bangalore-Flood/1.0", "Accept": "*/*"}
        r = requests.post(url, data={"data": query}, headers=headers, timeout=20)
        print(f"{name} ({url}): status={r.status_code}")
        if r.status_code == 200:
            j = r.json()
            print(f"  => Elements: {len(j.get('elements', []))}")
            # Found a working one, use it
            print(f"\n>> WORKING: {url}")
            break
        else:
            print(f"  Text: {r.text[:150]}")
    except Exception as e:
        print(f"{name}: Error - {e}")
    time.sleep(1)
