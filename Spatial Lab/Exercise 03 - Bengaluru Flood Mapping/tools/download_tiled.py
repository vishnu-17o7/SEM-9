"""Download Bangalore OSM data via Overpass API using tiled approach."""
from __future__ import annotations

import json
import os
import time

import requests

BBOX = (12.86, 77.45, 13.10, 77.75)  # south, west, north, east
TILE_SIZE = 0.05  # ~5km tiles

DATA_DIR = r"C:\Users\vishn\Desktop\Programs\SEM 9\Spatial Lab\bangalore_data"
os.makedirs(DATA_DIR, exist_ok=True)


def query_overpass(query: str, retries: int = 3) -> dict | None:
    """Query Overpass API with retries."""
    headers = {"User-Agent": "Bangalore-Flood/1.0", "Accept": "*/*"}
    for attempt in range(retries):
        try:
            r = requests.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": query},
                headers=headers,
                timeout=120,
            )
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 504:
                print(f"  504 timeout (attempt {attempt+1}/{retries}), waiting...")
                time.sleep(10)
                continue
            elif r.status_code == 429:
                print(f"  429 rate limit, waiting 30s...")
                time.sleep(30)
                continue
            else:
                print(f"  HTTP {r.status_code}: {r.text[:200]}")
                return None
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(5)
    return None


def geojson_from_overpass(elements: list) -> dict:
    """Convert Overpass elements to GeoJSON FeatureCollection.
    
    Handles two Overpass geometry formats:
    1. GeoJSON-style: {"type": "LineString", "coordinates": [[lon, lat], ...]}
    2. Node-array style: [{"lat": ..., "lon": ...}, ...]
    """
    features = []
    for el in elements:
        raw_geom = el.get("geometry")
        if not raw_geom:
            continue

        # Format 1: GeoJSON geometry object
        if isinstance(raw_geom, dict) and "coordinates" in raw_geom:
            coords = raw_geom["coordinates"]
            geom_type = raw_geom.get("type")
        # Format 2: List of {lat, lon} dicts (way nodes)
        elif isinstance(raw_geom, list) and len(raw_geom) >= 2:
            coords = [[pt["lon"], pt["lat"]] for pt in raw_geom]
            geom_type = "LineString"
        # Format 3: Single point from node center
        elif "lat" in el and "lon" in el:
            coords = [el["lon"], el["lat"]]
            geom_type = "Point"
        else:
            continue

        # Validate geometry
        if geom_type == "LineString" and len(coords) < 2:
            continue
        if not geom_type:
            continue

        props = dict(el.get("tags", {}))
        props["osm_id"] = el["id"]
        props["osm_type"] = el.get("type", "node")

        features.append({
            "type": "Feature",
            "geometry": {"type": geom_type, "coordinates": coords},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}


def dedup_features(features: list) -> list:
    """Remove duplicate features by osm_id."""
    seen = set()
    result = []
    for f in features:
        oid = f["properties"]["osm_id"]
        if oid not in seen:
            seen.add(oid)
            result.append(f)
    return result


def download_layer(name: str, osm_filter: str, geom_type: str) -> list:
    """Download one layer using tiled Overpass queries."""
    all_features = []
    south, west, north, east = BBOX

    lat = south
    tile_count = 0
    while lat < north:
        lon = west
        while lon < east:
            lat2 = min(lat + TILE_SIZE, north)
            lon2 = min(lon + TILE_SIZE, east)

            bbox_str = f"{lat},{lon},{lat2},{lon2}"
            query = f"[out:json][timeout:120];({geom_type}{osm_filter}({bbox_str}););out geom;"

            tile_count += 1
            print(f"  [{name}] Tile {tile_count}: {bbox_str}...", end=" ")

            result = query_overpass(query)
            if result:
                feats = geojson_from_overpass(result.get("elements", []))["features"]
                all_features.extend(feats)
                print(f"{len(feats)} features")
            else:
                print("FAILED")

            lon = lon2
            time.sleep(1)
        lat = lat2

    return dedup_features(all_features)


def save_geojson(features: list, filename: str):
    """Save features to GeoJSON file."""
    path = os.path.join(DATA_DIR, filename)
    fc = {"type": "FeatureCollection", "features": features}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)
    print(f"  Saved {len(features)} features to {filename}")


def main():
    total_tiles = ((13.10 - 12.86) / TILE_SIZE) * ((77.75 - 77.45) / TILE_SIZE)
    print(f"Bangalore OSM Download: ~{int(total_tiles)} tiles per layer")
    print()

    # 1. Waterways
    print("=== Layer 1/6: Waterways ===")
    ww = download_layer("waterways", '["waterway"~"river|stream|canal|drain"]', "way")
    ww += download_layer("waterways-rel", '["waterway"~"river|stream"]', "relation")
    save_geojson(dedup_features(ww), "bangalore_waterways.geojson")

    # 2. Major roads
    print("\n=== Layer 2/6: Major Roads ===")
    roads = download_layer("roads", '["highway"~"motorway|trunk|primary|secondary|tertiary"]', "way")
    save_geojson(roads, "bangalore_roads.geojson")

    # 3. All roads (for network analysis)
    print("\n=== Layer 3/6: All Roads ===")
    all_roads = download_layer("all_roads", '["highway"]', "way")
    save_geojson(all_roads, "bangalore_all_roads.geojson")

    # 4. Schools
    print("\n=== Layer 4/6: Schools ===")
    schools = download_layer("schools", '["amenity"="school"]', "node;way")
    save_geojson(schools, "bangalore_schools.geojson")

    # 5. Hospitals
    print("\n=== Layer 5/6: Hospitals ===")
    hospitals = download_layer("hospitals", '["amenity"="hospital"]', "node;way")
    save_geojson(hospitals, "bangalore_hospitals.geojson")

    # 6. Water bodies
    print("\n=== Layer 6/6: Water Bodies ===")
    wb = download_layer("water_bodies", '["natural"="water"]', "way;relation")
    save_geojson(wb, "bangalore_water_bodies.geojson")

    # Summary
    print("\n=== Download Complete ===")
    for name in ["bangalore_waterways.geojson", "bangalore_roads.geojson",
                  "bangalore_all_roads.geojson", "bangalore_schools.geojson",
                  "bangalore_hospitals.geojson", "bangalore_water_bodies.geojson"]:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                fc = json.load(f)
                print(f"  {name}: {len(fc['features'])} features")
        else:
            print(f"  {name}: MISSING")


if __name__ == "__main__":
    main()
