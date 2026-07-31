"""Convert an Overpass hospital response to a simple point GeoJSON."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "source" / "osm_hospitals_raw.json"
OUTPUT = ROOT / "data" / "processed" / "osm_hospitals.geojson"


def capacity_weight(tags: dict[str, str]) -> int:
    """Return a small classroom capacity weight from available OSM tags."""
    name = tags.get("name", "").lower()
    beds_text = tags.get("beds", "").replace(",", "")
    if beds_text.isdigit():
        return max(1, min(6, round(int(beds_text) / 50)))
    if any(term in name for term in ("medical college", "government", "general hospital")):
        return 4
    if tags.get("emergency") == "yes":
        return 3
    return 1


def main() -> None:
    """Write one point feature for each OSM hospital element."""
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    features: list[dict] = []

    for element in payload.get("elements", []):
        point = element if "lat" in element else element.get("center", {})
        if "lat" not in point or "lon" not in point:
            continue
        tags = element.get("tags", {})
        osm_type = element.get("type", "")
        osm_id = element.get("id")
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [point["lon"], point["lat"]],
                },
                "properties": {
                    "osm_key": f"{osm_type}/{osm_id}",
                    "name": tags.get("name") or f"Hospital {osm_id}",
                    "operator": tags.get("operator", ""),
                    "emergency": tags.get("emergency", ""),
                    "beds": tags.get("beds", ""),
                    "capacity_wt": capacity_weight(tags),
                    "osm_type": osm_type,
                },
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {"type": "FeatureCollection", "name": "osm_hospitals", "features": features},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(features)} hospital points to {OUTPUT}")


if __name__ == "__main__":
    main()
