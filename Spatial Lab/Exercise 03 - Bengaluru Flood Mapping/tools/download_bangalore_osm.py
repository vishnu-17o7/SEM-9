"""
Download Bangalore OSM data from OSM API (not Overpass).
Splits area into grid tiles to stay within 50000-node limit.
Extracts: waterways, roads, schools, hospitals, water bodies.
"""

from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET

import requests

# Bangalore bounding box
BBOX = (77.45, 12.86, 77.75, 13.10)  # min_lon, min_lat, max_lon, max_lat
TILE_SIZE = 0.025  # degrees per tile side

DATA_DIR = r"C:\Users\vishn\Desktop\Programs\SEM 9\Spatial Lab\bangalore_data"

OSM_API = "https://api.openstreetmap.org/api/0.6/map"


def tile_bboxes():
    """Generate all tile bboxes within the main bbox."""
    min_lon, min_lat, max_lon, max_lat = BBOX
    tiles = []
    lon = min_lon
    while lon < max_lon:
        lat = min_lat
        while lat < max_lat:
            lon2 = min(lon + TILE_SIZE, max_lon)
            lat2 = min(lat + TILE_SIZE, max_lat)
            tiles.append((lon, lat, lon2, lat2))
            lat = lat2
        lon = lon2
    return tiles


def download_tile(lon, lat, lon2, lat2) -> bytes | None:
    """Download OSM data for a tile, return bytes or None on failure."""
    bbox_str = f"{lon},{lat},{lon2},{lat2}"
    url = f"{OSM_API}?bbox={bbox_str}"
    headers = {"User-Agent": "QGIS-Bangalore-Flood-Analysis/1.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=60)
        if resp.status_code == 200:
            return resp.content
        elif "too many nodes" in resp.text:
            print(f"  Tile too large at {bbox_str}, splitting...")
            return None
        else:
            print(f"  HTTP {resp.status_code} for tile {bbox_str}")
            return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def parse_osm_features(xml_data: bytes) -> dict:
    """Parse OSM XML and extract nodes, ways, relations as GeoJSON-like features."""
    root = ET.fromstring(xml_data)

    nodes: dict[int, dict] = {}
    ways: dict[int, dict] = {}
    relations: dict[int, dict] = {}

    # Parse nodes
    for node in root.findall("node"):
        nid = int(node.get("id"))
        lat = float(node.get("lat"))
        lon = float(node.get("lon"))
        tags = {}
        for tag in node.findall("tag"):
            tags[tag.get("k")] = tag.get("v")
        nodes[nid] = {"id": nid, "lat": lat, "lon": lon, "tags": tags, "type": "node"}

    # Parse ways
    for way in root.findall("way"):
        wid = int(way.get("id"))
        refs = [int(nd.get("ref")) for nd in way.findall("nd")]
        tags = {}
        for tag in way.findall("tag"):
            tags[tag.get("k")] = tag.get("v")
        ways[wid] = {"id": wid, "nodes": refs, "tags": tags, "type": "way"}

    # Parse relations
    for rel in root.findall("relation"):
        rid = int(rel.get("id"))
        members = []
        for member in rel.findall("member"):
            members.append({
                "type": member.get("type"),
                "ref": int(member.get("ref")),
                "role": member.get("role"),
            })
        tags = {}
        for tag in rel.findall("tag"):
            tags[tag.get("k")] = tag.get("v")
        relations[rid] = {"id": rid, "members": members, "tags": tags, "type": "relation"}

    return {"nodes": nodes, "ways": ways, "relations": relations}


def extract_waterways(data: dict) -> list[dict]:
    """Extract waterway features (river, stream, canal, drain)."""
    features = []
    for wid, way in data["ways"].items():
        tags = way["tags"]
        if "waterway" in tags and tags["waterway"] in ("river", "stream", "canal", "drain", "ditch"):
            coords = []
            for nid in way["nodes"]:
                if nid in data["nodes"]:
                    n = data["nodes"][nid]
                    coords.append([n["lon"], n["lat"]])
            if len(coords) >= 2:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {"osm_id": wid, "waterway": tags["waterway"], "name": tags.get("name", "")},
                })
    return features


def extract_water_bodies(data: dict) -> list[dict]:
    """Extract water body features (natural=water, water=lake)."""
    features = []
    for wid, way in data["ways"].items():
        tags = way["tags"]
        if tags.get("natural") == "water" or tags.get("water") in ("lake", "pond", "reservoir"):
            coords = []
            for nid in way["nodes"]:
                if nid in data["nodes"]:
                    n = data["nodes"][nid]
                    coords.append([n["lon"], n["lat"]])
            if len(coords) >= 3:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                    "properties": {"osm_id": wid, "type": tags.get("natural", tags.get("water", "")), "name": tags.get("name", "")},
                })
    return features


def extract_roads(data: dict) -> list[dict]:
    """Extract major road features."""
    important_highways = {"motorway", "trunk", "primary", "secondary", "tertiary"}
    features = []
    for wid, way in data["ways"].items():
        tags = way["tags"]
        highway = tags.get("highway")
        if highway and highway in important_highways:
            coords = []
            prev_nid = None
            for nid in way["nodes"]:
                if nid in data["nodes"]:
                    n = data["nodes"][nid]
                    coords.append([n["lon"], n["lat"]])
                    prev_nid = nid
            if len(coords) >= 2:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {
                        "osm_id": wid,
                        "highway": highway,
                        "name": tags.get("name", ""),
                        "oneway": tags.get("oneway", "no"),
                    },
                })
    return features


def extract_all_roads(data: dict) -> list[dict]:
    """Extract ALL road features (for network analysis)."""
    features = []
    for wid, way in data["ways"].items():
        tags = way["tags"]
        highway = tags.get("highway")
        if highway:
            coords = []
            for nid in way["nodes"]:
                if nid in data["nodes"]:
                    n = data["nodes"][nid]
                    coords.append([n["lon"], n["lat"]])
            if len(coords) >= 2:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {
                        "osm_id": wid,
                        "highway": highway,
                        "name": tags.get("name", ""),
                        "oneway": tags.get("oneway", "no"),
                    },
                })
    return features


def extract_schools(data: dict) -> list[dict]:
    """Extract school features."""
    features = []
    # From nodes
    for nid, node in data["nodes"].items():
        if node["tags"].get("amenity") == "school":
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [node["lon"], node["lat"]]},
                "properties": {"osm_id": nid, "type": "node", "amenity": "school", "name": node["tags"].get("name", "")},
            })
    # From ways (use centroid)
    for wid, way in data["ways"].items():
        if way["tags"].get("amenity") == "school":
            coords = []
            for nid in way["nodes"]:
                if nid in data["nodes"]:
                    n = data["nodes"][nid]
                    coords.append([n["lon"], n["lat"]])
            if coords:
                # centroid
                cx = sum(c[0] for c in coords) / len(coords)
                cy = sum(c[1] for c in coords) / len(coords)
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [cx, cy]},
                    "properties": {"osm_id": wid, "type": "way", "amenity": "school", "name": way["tags"].get("name", "")},
                })
    return features


def extract_hospitals(data: dict) -> list[dict]:
    """Extract hospital features."""
    features = []
    for nid, node in data["nodes"].items():
        if node["tags"].get("amenity") == "hospital":
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [node["lon"], node["lat"]]},
                "properties": {"osm_id": nid, "type": "node", "amenity": "hospital", "name": node["tags"].get("name", "")},
            })
    for wid, way in data["ways"].items():
        if way["tags"].get("amenity") == "hospital":
            coords = []
            for nid in way["nodes"]:
                if nid in data["nodes"]:
                    n = data["nodes"][nid]
                    coords.append([n["lon"], n["lat"]])
            if coords:
                cx = sum(c[0] for c in coords) / len(coords)
                cy = sum(c[1] for c in coords) / len(coords)
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [cx, cy]},
                    "properties": {"osm_id": wid, "type": "way", "amenity": "hospital", "name": way["tags"].get("name", "")},
                })
    return features


def merge_feature_lists(lists: list[list[dict]]) -> list[dict]:
    """Merge multiple feature lists, deduplicating by osm_id."""
    seen: set[int] = set()
    merged = []
    for lst in lists:
        for feat in lst:
            oid = feat["properties"]["osm_id"]
            if oid not in seen:
                seen.add(oid)
                merged.append(feat)
    return merged


def save_geojson(features: list[dict], filename: str):
    """Save features as GeoJSON."""
    path = os.path.join(DATA_DIR, filename)
    fc = {"type": "FeatureCollection", "features": features}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(features)} features to {filename}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    tiles = tile_bboxes()
    print(f"Downloading {len(tiles)} tiles...")

    # Accumulators
    all_waterways = []
    all_water_bodies = []
    all_roads = []
    all_all_roads = []
    all_schools = []
    all_hospitals = []

    tiles_needing_split = []

    for i, (lon, lat, lon2, lat2) in enumerate(tiles):
        print(f"[{i+1}/{len(tiles)}] Downloading {lon},{lat},{lon2},{lat2}...")
        data_bytes = download_tile(lon, lat, lon2, lat2)
        if data_bytes is None:
            tiles_needing_split.append((lon, lat, lon2, lat2))
            continue

        parsed = parse_osm_features(data_bytes)

        all_waterways.extend(extract_waterways(parsed))
        all_water_bodies.extend(extract_water_bodies(parsed))
        all_roads.extend(extract_roads(parsed))
        all_all_roads.extend(extract_all_roads(parsed))
        all_schools.extend(extract_schools(parsed))
        all_hospitals.extend(extract_hospitals(parsed))

        print(f"  Waterways: {len(all_waterways)}, Roads: {len(all_roads)}, Schools: {len(all_schools)}, Hospitals: {len(all_hospitals)}")
        time.sleep(1)  # Be nice to the API

    # Handle tiles that were too large - split into smaller pieces
    if tiles_needing_split:
        print(f"\nSplitting {len(tiles_needing_split)} oversized tiles...")
        for lon, lat, lon2, lat2 in tiles_needing_split:
            sub_tile_size = TILE_SIZE / 2
            sub_lon = lon
            while sub_lon < lon2:
                sub_lat = lat
                while sub_lat < lat2:
                    slon2 = min(sub_lon + sub_tile_size, lon2)
                    slat2 = min(sub_lat + sub_tile_size, lat2)
                    print(f"  Sub-tile {sub_lon},{sub_lat},{slon2},{slat2}...")
                    data_bytes = download_tile(sub_lon, sub_lat, slon2, slat2)
                    if data_bytes:
                        parsed = parse_osm_features(data_bytes)
                        all_waterways.extend(extract_waterways(parsed))
                        all_water_bodies.extend(extract_water_bodies(parsed))
                        all_roads.extend(extract_roads(parsed))
                        all_all_roads.extend(extract_all_roads(parsed))
                        all_schools.extend(extract_schools(parsed))
                        all_hospitals.extend(extract_hospitals(parsed))
                    time.sleep(1)
                    sub_lat = slat2
                sub_lon = slon2

    # Save all data
    print("\nSaving data files...")
    save_geojson(merge_feature_lists([all_waterways]), "bangalore_waterways.geojson")
    save_geojson(merge_feature_lists([all_water_bodies]), "bangalore_water_bodies.geojson")
    save_geojson(merge_feature_lists([all_roads]), "bangalore_roads.geojson")
    save_geojson(merge_feature_lists([all_all_roads]), "bangalore_all_roads.geojson")
    save_geojson(merge_feature_lists([all_schools]), "bangalore_schools.geojson")
    save_geojson(merge_feature_lists([all_hospitals]), "bangalore_hospitals.geojson")

    print("\nDone! Summary:")
    print(f"  Waterways: {len(merge_feature_lists([all_waterways]))}")
    print(f"  Water bodies: {len(merge_feature_lists([all_water_bodies]))}")
    print(f"  Major roads: {len(merge_feature_lists([all_roads]))}")
    print(f"  All roads: {len(merge_feature_lists([all_all_roads]))}")
    print(f"  Schools: {len(merge_feature_lists([all_schools]))}")
    print(f"  Hospitals: {len(merge_feature_lists([all_hospitals]))}")


if __name__ == "__main__":
    main()
