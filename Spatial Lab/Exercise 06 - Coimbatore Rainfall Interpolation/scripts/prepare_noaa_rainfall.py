from __future__ import annotations

import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "source"
PROCESSED = ROOT / "data" / "processed"

STATIONS_PATH = SOURCE / "ghcnd-stations.txt"
OBSERVATIONS_PATH = SOURCE / "ghcnd_1965.csv.gz"
OUTPUT_CSV = PROCESSED / "coimbatore_rainfall_october_1965_all.csv"
OUTPUT_GEOJSON = PROCESSED / "coimbatore_rainfall_october_1965_all.geojson"
SELECTED_CSV = PROCESSED / "coimbatore_rainfall_october_1965.csv"
SELECTED_GEOJSON = PROCESSED / "coimbatore_rainfall_october_1965.geojson"

# Broad surroundings used to retain stations that reduce interpolation edge effects.
MIN_LAT, MAX_LAT = 9.0, 12.8
MIN_LON, MAX_LON = 75.0, 79.0
COIMBATORE_CENTER = (11.0168, 76.9558)
VALIDATION_NAMES = {"COIMBATORE", "SULUR", "POLLACHI", "ANNUR", "METTUPALAYAM"}


def read_station_metadata() -> dict[str, dict[str, object]]:
    """Read fixed-width GHCN-Daily station metadata inside the study region."""
    stations: dict[str, dict[str, object]] = {}
    with STATIONS_PATH.open("r", encoding="ascii", errors="replace") as handle:
        for line in handle:
            station_id = line[0:11].strip()
            try:
                latitude = float(line[12:20])
                longitude = float(line[21:30])
                elevation = float(line[31:37])
            except ValueError:
                continue
            if not (MIN_LAT <= latitude <= MAX_LAT and MIN_LON <= longitude <= MAX_LON):
                continue
            stations[station_id] = {
                "Station_ID": station_id,
                "Station_Name": line[41:71].strip(),
                "Latitude": latitude,
                "Longitude": longitude,
                "Elevation_m": elevation,
            }
    return stations


def aggregate_october_precipitation(
    stations: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate valid daily PRCP observations for October 1965."""
    totals_tenths_mm: defaultdict[str, int] = defaultdict(int)
    valid_days: defaultdict[str, int] = defaultdict(int)

    with gzip.open(OBSERVATIONS_PATH, "rt", encoding="ascii", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 7:
                continue
            station_id, date, element, value = row[0], row[1], row[2], row[3]
            quality_flag = row[5].strip()
            if (
                station_id not in stations
                or not date.startswith("196510")
                or element != "PRCP"
                or quality_flag
            ):
                continue
            totals_tenths_mm[station_id] += int(value)
            valid_days[station_id] += 1

    rows: list[dict[str, object]] = []
    for station_id, total in totals_tenths_mm.items():
        item = dict(stations[station_id])
        item["Rainfall_mm"] = round(total / 10.0, 1)
        item["Days_Observed"] = valid_days[station_id]
        rows.append(item)
    rows.sort(key=lambda row: (-int(row["Days_Observed"]), str(row["Station_Name"])))
    return rows


def write_outputs(rows: list[dict[str, object]]) -> None:
    """Write tabular and GeoJSON versions of the aggregated observations."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    fields = [
        "Station_ID",
        "Station_Name",
        "Longitude",
        "Latitude",
        "Elevation_m",
        "Rainfall_mm",
        "Days_Observed",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    features = []
    for row in rows:
        properties = {field: row[field] for field in fields if field not in {"Longitude", "Latitude"}}
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {
                    "type": "Point",
                    "coordinates": [row["Longitude"], row["Latitude"]],
                },
            }
        )
    collection = {
        "type": "FeatureCollection",
        "name": "coimbatore_rainfall_october_1965_all",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
    OUTPUT_GEOJSON.write_text(json.dumps(collection, indent=2), encoding="utf-8")


def distance_km(latitude: float, longitude: float) -> float:
    """Return great-circle distance from central Coimbatore."""
    center_latitude, center_longitude = COIMBATORE_CENTER
    phi1 = math.radians(center_latitude)
    phi2 = math.radians(latitude)
    delta_phi = math.radians(latitude - center_latitude)
    delta_lambda = math.radians(longitude - center_longitude)
    haversine = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * 6371.0088 * math.asin(math.sqrt(haversine))


def select_analysis_stations(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Select 22 nearby complete stations and assign a fixed validation subset."""
    complete = [
        dict(row)
        for row in rows
        if int(row["Days_Observed"]) == 31 and float(row["Rainfall_mm"]) > 0
    ]
    for row in complete:
        row["Distance_km"] = round(
            distance_km(float(row["Latitude"]), float(row["Longitude"])),
            1,
        )
    complete.sort(key=lambda row: float(row["Distance_km"]))
    selected = complete[:22]
    for row in selected:
        normalized_name = str(row["Station_Name"]).strip()
        row["Subset"] = "Validation" if normalized_name in VALIDATION_NAMES else "Training"
        row["Period"] = "October 1965"
    return selected


def write_selected_outputs(rows: list[dict[str, object]]) -> None:
    """Write the selected reproducible training/validation dataset."""
    fields = [
        "Station_ID",
        "Station_Name",
        "Longitude",
        "Latitude",
        "Elevation_m",
        "Rainfall_mm",
        "Days_Observed",
        "Distance_km",
        "Subset",
        "Period",
    ]
    with SELECTED_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    features = []
    for row in rows:
        properties = {field: row[field] for field in fields if field not in {"Longitude", "Latitude"}}
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {
                    "type": "Point",
                    "coordinates": [row["Longitude"], row["Latitude"]],
                },
            }
        )
    collection = {
        "type": "FeatureCollection",
        "name": "coimbatore_rainfall_october_1965",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
    SELECTED_GEOJSON.write_text(json.dumps(collection, indent=2), encoding="utf-8")


def main() -> None:
    """Create real-source October 1965 station totals for the study region."""
    stations = read_station_metadata()
    rows = aggregate_october_precipitation(stations)
    write_outputs(rows)
    selected = select_analysis_stations(rows)
    write_selected_outputs(selected)

    counts = {
        threshold: sum(int(row["Days_Observed"]) >= threshold for row in rows)
        for threshold in (1, 10, 20, 25, 28, 31)
    }
    print(f"Regional station metadata records: {len(stations)}")
    print(f"Stations with October 1965 PRCP: {len(rows)}")
    print(f"Completeness counts: {counts}")
    if rows:
        print(
            "Rainfall range (all records): "
            f"{min(float(row['Rainfall_mm']) for row in rows):.1f}–"
            f"{max(float(row['Rainfall_mm']) for row in rows):.1f} mm"
        )
    training_count = sum(row["Subset"] == "Training" for row in selected)
    validation_count = sum(row["Subset"] == "Validation" for row in selected)
    print(
        f"Selected complete stations: {len(selected)} "
        f"({training_count} training, {validation_count} validation)"
    )


if __name__ == "__main__":
    main()
