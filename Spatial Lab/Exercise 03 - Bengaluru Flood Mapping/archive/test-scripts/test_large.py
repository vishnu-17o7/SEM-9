"""Test Overpass with larger query to verify it returns data."""
import requests

query = "[out:json];node(12.90,77.50,13.00,77.70)[amenity=school];out;"
headers = {"User-Agent": "Bangalore-Flood/1.0", "Accept": "*/*"}
r = requests.post(
    "https://overpass-api.de/api/interpreter",
    data={"data": query},
    headers=headers,
    timeout=60,
)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    j = r.json()
    print(f"Elements: {len(j.get('elements', []))}")
    # Print first element
    if j["elements"]:
        print(f"First: {j['elements'][0]}")
