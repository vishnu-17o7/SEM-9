"""
Download Bangalore OSM data for flood risk analysis.
Downloads: rivers, roads, schools, hospitals, and city boundary.
"""

from __future__ import annotations

import json
import time

import requests

BANGALORE_BBOX = "12.86,77.45,13.10,77.75"  # south,west,north,east
OVERPass_URL = "https://overpass-api.de/api/interpreter"

DATA_DIR = r"C:\Users\vishn\Desktop\Programs\SEM 9\Spatial Lab\bangalore_data"


def query_overpass(query: str) -> dict:
    """Query Overpass API and return JSON response."""
    resp = requests.post(
        OVERPass_URL,
        data={"data": query},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def geojson_from_overpass(elements: list) -> dict:
    """Convert Overpass elements to a simple GeoJSON FeatureCollection."""
    features = []
    for el in elements:
        geom = el.get("geometry") or el.get("center")
        if not geom:
            continue
        if geom["type"] == "geometry":
            geom_type = geom["geometry"]
            coords = geom["coordinates"]
        elif "lat" in geom and "lon" in geom:
            geom_type = "Point"
            coords = [geom["lon"], geom["lat"]]
        elif geom.get("type") == "Point" and "coordinates" in geom:
            geom_type = "Point"
            coords = geom["coordinates"]
        else:
            continue

        props = {k: v for k, v in el.get("tags", {}).items()}
        props["osm_id"] = el["id"]
        props["osm_type"] = el.get("type", "node")

        features.append({
            "type": "Feature",
            "geometry": {"type": geom_type, "coordinates": coords},
            "properties": props,
        })

    return {"type": "FeatureCollection", "features": features}


def save_geojson(data: dict, filename: str):
    """Save data as GeoJSON file."""
    path = f"{DATA_DIR}/{filename}"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data['features'])} features to {path}")


def main():
    import os
    os.makedirs(DATA_DIR, exist_ok=True)

    bbox = BANGALORE_BBOX

    # 1. Rivers and waterways
    print("Downloading rivers/waterways...")
    q_rivers = f"""
    [out:json][timeout:180];
    (
      way["waterway"~"river|stream|canal|drain"]({bbox});
      relation["waterway"~"river|stream"]({bbox});
    );
    out geom;
    """
    rivers = query_overpass(q_rivers)
    save_geojson(geojson_from_overpass(rivers["elements"]), "bangalore_waterways.geojson")
    time.sleep(2)

    # 2. Major roads (primary, secondary, tertiary, trunk)
    print("Downloading major roads...")
    q_roads = f"""
    [out:json][timeout:180];
    (
      way["highway"~"motorway|trunk|primary|secondary|tertiary"]({bbox});
    );
    out geom;
    """
    roads = query_overpass(q_roads)
    save_geojson(geojson_from_overpass(roads["elements"]), "bangalore_roads.geojson")
    time.sleep(2)

    # 3. Schools
    print("Downloading schools...")
    q_schools = f"""
    [out:json][timeout:180];
    (
      node["amenity"="school"]({bbox});
      way["amenity"="school"]({bbox});
    );
    out center;
    """
    schools = query_overpass(q_schools)
    save_geojson(geojson_from_overpass(schools["elements"]), "bangalore_schools.geojson")
    time.sleep(2)

    # 4. Hospitals
    print("Downloading hospitals...")
    q_hospitals = f"""
    [out:json][timeout:180];
    (
      node["amenity"="hospital"]({bbox});
      way["amenity"="hospital"]({bbox});
    );
    out center;
    """
    hospitals = query_overpass(q_hospitals)
    save_geojson(geojson_from_overpass(hospitals["elements"]), "bangalore_hospitals.geojson")
    time.sleep(2)

    # 5. All streets (for network analysis)
    print("Downloading all streets for network analysis...")
    q_all_roads = f"""
    [out:json][timeout:180];
    (
      way["highway"]({bbox});
    );
    out geom;
    """
    all_roads = query_overpass(q_all_roads)
    save_geojson(geojson_from_overpass(all_roads["elements"]), "bangalore_all_roads.geojson")
    time.sleep(2)

    # 6. Lakes/water bodies
    print("Downloading water bodies...")
    q_water = f"""
    [out:json][timeout:180];
    (
      way["natural"="water"]({bbox});
      relation["natural"="water"]({bbox});
      way["water"="lake"]({bbox});
      way["water"="pond"]({bbox});
    );
    out geom;
    """
    water = query_overpass(q_water)
    save_geojson(geojson_from_overpass(water["elements"]), "bangalore_water_bodies.geojson")
    time.sleep(2)

    print("All downloads complete!")


if __name__ == "__main__":
    main()
