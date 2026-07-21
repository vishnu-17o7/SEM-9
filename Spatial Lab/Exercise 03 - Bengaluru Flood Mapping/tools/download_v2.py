"""
Download Bangalore data using Overpass API.
Smaller area, better error handling, intermediate saves.
"""

from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET

import requests

# Central Bangalore - urban core (smaller area for manageable downloads)
BBOX = (77.50, 12.90, 77.70, 13.05)
TILE_SIZE = 0.02

DATA_DIR = r"C:\Users\vishn\Desktop\Programs\SEM 9\Spatial Lab\bangalore_data"

OSM_API = "https://api.openstreetmap.org/api/0.6/map"


def tile_bboxes():
    tiles = []
    lon = BBOX[0]
    while lon < BBOX[2]:
        lat = BBOX[1]
        while lat < BBOX[3]:
            lon2 = min(lon + TILE_SIZE, BBOX[2])
            lat2 = min(lat + TILE_SIZE, BBOX[3])
            tiles.append((lon, lat, lon2, lat2))
            lat = lat2
        lon = lon2
    return tiles


def download_tile(lon, lat, lon2, lat2, retries=3) -> bytes | None:
    bbox_str = f"{lon},{lat},{lon2},{lat2}"
    url = f"{OSM_API}?bbox={bbox_str}"
    headers = {"User-Agent": "Bangalore-Flood-Analysis/1.0"}
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=120)
            if resp.status_code == 200:
                return resp.content
            elif "too many nodes" in resp.text:
                print(f"  Too large at {bbox_str}, splitting...")
                return None
            elif resp.status_code == 509:
                print(f"  509 (rate limit) on {bbox_str}, waiting 30s...")
                time.sleep(30)
                continue
            else:
                print(f"  HTTP {resp.status_code} for {bbox_str}")
                return None
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < retries - 1:
                time.sleep(5)
    return None


def parse_osm(xml_bytes: bytes) -> tuple[dict, dict, dict]:
    root = ET.fromstring(xml_bytes)
    nodes, ways, relations = {}, {}, {}
    for node in root.findall("node"):
        nid = int(node.get("id"))
        nodes[nid] = {"id": nid, "lat": float(node.get("lat")), "lon": float(node.get("lon")),
                      "tags": {t.get("k"): t.get("v") for t in node.findall("tag")}}
    for way in root.findall("way"):
        wid = int(way.get("id"))
        ways[wid] = {"id": wid, "nodes": [int(nd.get("ref")) for nd in way.findall("nd")],
                     "tags": {t.get("k"): t.get("v") for t in way.findall("tag")}}
    for rel in root.findall("relation"):
        rid = int(rel.get("id"))
        relations[rid] = {"id": rid, "members": [{"type": m.get("type"), "ref": int(m.get("ref")),
                          "role": m.get("role")} for m in rel.findall("member")],
                          "tags": {t.get("k"): t.get("v") for t in rel.findall("tag")}}
    return nodes, ways, relations


def way_coords(way, nodes):
    coords = []
    for nid in way["nodes"]:
        if nid in nodes:
            coords.append([nodes[nid]["lon"], nodes[nid]["lat"]])
    return coords


def centroid(coords):
    if not coords:
        return None
    return [sum(c[0] for c in coords) / len(coords), sum(c[1] for c in coords) / len(coords)]


def extract_features(nodes, ways, relations):
    waterways, water_bodies, roads, all_roads, schools_pts, hospitals_pts = [], [], [], [], [], []

    for wid, way in ways.items():
        t = way["tags"]
        coords = way_coords(way, nodes)

        # Waterways
        if "waterway" in t and t["waterway"] in ("river", "stream", "canal", "drain"):
            if len(coords) >= 2:
                waterways.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                                  "properties": {"osm_id": wid, "waterway": t["waterway"], "name": t.get("name", "")}})

        # Water bodies
        if t.get("natural") == "water" or t.get("water") in ("lake", "pond", "reservoir"):
            if len(coords) >= 3:
                water_bodies.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [coords]},
                                     "properties": {"osm_id": wid, "type": t.get("natural", t.get("water", "")), "name": t.get("name", "")}})

        # Roads (all)
        if "highway" in t and len(coords) >= 2:
            f = {"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                 "properties": {"osm_id": wid, "highway": t["highway"], "name": t.get("name", ""),
                                "oneway": t.get("oneway", "no")}}
            all_roads.append(f)
            if t["highway"] in ("motorway", "trunk", "primary", "secondary", "tertiary"):
                roads.append(f)

        # Schools from ways
        if t.get("amenity") == "school" and len(coords) >= 1:
            c = centroid(coords)
            if c:
                schools_pts.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": c},
                                    "properties": {"osm_id": wid, "amenity": "school", "name": t.get("name", "")}})

        # Hospitals from ways
        if t.get("amenity") == "hospital" and len(coords) >= 1:
            c = centroid(coords)
            if c:
                hospitals_pts.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": c},
                                      "properties": {"osm_id": wid, "amenity": "hospital", "name": t.get("name", "")}})

    # Points from nodes
    for nid, node in nodes.items():
        t = node["tags"]
        if t.get("amenity") == "school":
            schools_pts.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [node["lon"], node["lat"]]},
                                "properties": {"osm_id": nid, "amenity": "school", "name": t.get("name", "")}})
        if t.get("amenity") == "hospital":
            hospitals_pts.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [node["lon"], node["lat"]]},
                                  "properties": {"osm_id": nid, "amenity": "hospital", "name": t.get("name", "")}})

    return waterways, water_bodies, roads, all_roads, schools_pts, hospitals_pts


def dedup(features):
    seen = set()
    result = []
    for f in features:
        oid = f["properties"]["osm_id"]
        if oid not in seen:
            seen.add(oid)
            result.append(f)
    return result


def save(name, features):
    path = os.path.join(DATA_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": dedup(features)}, f, ensure_ascii=False)
    print(f"Saved {len(dedup(features))} to {name}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Initialise accumulators
    all_w, all_wb, all_r, all_ar, all_s, all_h = [], [], [], [], [], []

    # Process tiles
    tiles = tile_bboxes()
    print(f"Processing {len(tiles)} tiles for central Bangalore...")

    for i, (lon, lat, lon2, lat2) in enumerate(tiles):
        print(f"[{i+1}/{len(tiles)}] {lon:.4f},{lat:.4f},{lon2:.4f},{lat2:.4f}...")
        data = download_tile(lon, lat, lon2, lat2)
        if data is None:
            # Try splitting
            half = (lon2 - lon) / 2
            if half >= 0.005:
                for slon in [lon, lon + half]:
                    for slat in [lat, lat + half]:
                        slon2 = min(slon + half, lon2)
                        slat2 = min(slat + half, lat2)
                        print(f"  Sub-tile {slon:.4f},{slat:.4f},{slon2:.4f},{slat2:.4f}...")
                        sub = download_tile(slon, slat, slon2, slat2)
                        if sub:
                            n, w, r = parse_osm(sub)
                            ww, wb, rr, ar, ss, hh = extract_features(n, w, r)
                            all_w.extend(ww); all_wb.extend(wb); all_r.extend(rr)
                            all_ar.extend(ar); all_s.extend(ss); all_h.extend(hh)
                        time.sleep(1.5)
            continue

        n, w, r = parse_osm(data)
        ww, wb, rr, ar, ss, hh = extract_features(n, w, r)
        all_w.extend(ww); all_wb.extend(wb); all_r.extend(rr)
        all_ar.extend(ar); all_s.extend(ss); all_h.extend(hh)

        print(f"  -> WW:{len(ww)} WB:{len(wb)} R:{len(rr)} AR:{len(ar)} S:{len(ss)} H:{len(hh)}")
        time.sleep(1)

    # Save
    print("\n=== Saving ===")
    save("bangalore_waterways.geojson", all_w)
    save("bangalore_water_bodies.geojson", all_wb)
    save("bangalore_roads.geojson", all_r)
    save("bangalore_all_roads.geojson", all_ar)
    save("bangalore_schools.geojson", all_s)
    save("bangalore_hospitals.geojson", all_h)

    print("\n=== Summary ===")
    print(f"Waterways:  {len(dedup(all_w))}")
    print(f"Water bodies: {len(dedup(all_wb))}")
    print(f"Major roads: {len(dedup(all_r))}")
    print(f"All roads:   {len(dedup(all_ar))}")
    print(f"Schools:     {len(dedup(all_s))}")
    print(f"Hospitals:   {len(dedup(all_h))}")


if __name__ == "__main__":
    main()
