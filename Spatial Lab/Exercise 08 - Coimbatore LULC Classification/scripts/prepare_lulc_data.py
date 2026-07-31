from __future__ import annotations

import csv
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from osgeo import gdal
from osgeo import ogr
from osgeo import osr


EXERCISE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = EXERCISE_ROOT / "data" / "source"
PROCESSED_DIR = EXERCISE_ROOT / "data" / "processed"
OUTPUT_DIR = EXERCISE_ROOT / "outputs"
BOUNDARY = SOURCE_DIR / "coimbatore_boundary_utm.gpkg"

STAC_ROOT = "https://earth-search.aws.element84.com/v1"
STAC_COLLECTION = "sentinel-2-l2a"
SENTINEL_ITEMS = (
    "S2A_43PFM_20260310_0_L2A",
    "S2A_43PGM_20260310_0_L2A",
    "S2A_43PFN_20260310_0_L2A",
    "S2A_43PGN_20260310_0_L2A",
)
ASSETS = {
    "B02": ("blue", "bilinear"),
    "B03": ("green", "bilinear"),
    "B04": ("red", "bilinear"),
    "B08": ("nir", "bilinear"),
    "B11": ("swir16", "bilinear"),
    "B12": ("swir22", "bilinear"),
    "SCL": ("scl", "near"),
}
WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_N09E075_Map.tif"
)
WORLDCOVER_SOURCE = SOURCE_DIR / "ESA_WorldCover_10m_2021_v200_N09E075_Map.tif"
TARGET_CRS = "EPSG:32643"
PIXEL_SIZE = 20.0
RANDOM_SEED = 42
TRAIN_PER_CLASS = 800
VALIDATION_PER_CLASS = 200

CLASS_NAMES = {
    1: "Water",
    2: "Forest and vegetation",
    3: "Agriculture",
    4: "Built-up",
    5: "Barren and open land",
}
WORLDCOVER_TO_CLASS = {
    80: 1,
    10: 2,
    20: 2,
    30: 2,
    40: 3,
    50: 4,
    60: 5,
}


def log(message: str) -> None:
    """Write a progress message to the console and preparation log."""
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "preparation.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


def fetch_json(url: str, attempts: int = 5) -> dict:
    """Fetch JSON with a bounded retry count."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "SEM9-QGIS-LULC-Lab/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except Exception as exc:
            last_error = exc
            log(f"JSON attempt {attempt}/{attempts} failed: {exc}")
            time.sleep(min(attempt * 2, 8))
    raise RuntimeError(f"Unable to fetch {url} after {attempts} attempts") from last_error


def download_file(url: str, destination: Path, attempts: int = 5) -> None:
    """Download a file with resume-safe temporary output and bounded retries."""
    if destination.exists() and destination.stat().st_size > 1_000_000:
        log(f"Using existing source: {destination.name}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            if temporary.exists():
                temporary.unlink()
            log(f"Downloading {destination.name}, attempt {attempt}/{attempts}")
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "SEM9-QGIS-LULC-Lab/1.0"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                expected = int(response.headers.get("Content-Length", "0"))
                received = 0
                with temporary.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        received += len(chunk)
            if expected and received != expected:
                raise RuntimeError(f"incomplete download: {received} of {expected} bytes")
            temporary.replace(destination)
            log(f"Downloaded {destination.name} ({received / 1_000_000:.1f} MB)")
            return
        except Exception as exc:
            last_error = exc
            log(f"Download attempt {attempt}/{attempts} failed: {exc}")
            time.sleep(min(attempt * 2, 8))
    raise RuntimeError(
        f"Unable to download {destination.name} after {attempts} attempts"
    ) from last_error


def configure_gdal() -> None:
    """Configure reliable HTTP access and compressed GeoTIFF output."""
    gdal.UseExceptions()
    gdal.SetConfigOption("GDAL_HTTP_MULTIRANGE", "YES")
    gdal.SetConfigOption("GDAL_HTTP_RETRY_CODES", "429,500,502,503,504")
    gdal.SetConfigOption("GDAL_HTTP_MAX_RETRY", "5")
    gdal.SetConfigOption("GDAL_HTTP_RETRY_DELAY", "2")
    gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF")
    gdal.SetConfigOption("GDAL_CACHEMAX", "1024")


def load_stac_items() -> list[dict]:
    """Load the selected Sentinel-2 item metadata."""
    items: list[dict] = []
    for item_id in SENTINEL_ITEMS:
        url = f"{STAC_ROOT}/collections/{STAC_COLLECTION}/items/{item_id}"
        item = fetch_json(url)
        items.append(item)
        cloud = item.get("properties", {}).get("eo:cloud_cover")
        log(f"STAC item ready: {item_id}, cloud cover {cloud}%")
    return items


def warp_with_retries(
    destination: Path,
    sources: list[str],
    resample_alg: str,
    output_type: int,
    attempts: int = 5,
    cutline: Path | None = BOUNDARY,
    crop_to_cutline: bool = True,
    src_nodata: int | None = 0,
) -> None:
    """Warp and mosaic one raster product with a bounded retry count."""
    if destination.exists() and destination.stat().st_size > 10_000:
        dataset = gdal.Open(str(destination))
        if dataset is not None:
            log(f"Using existing processed raster: {destination.name}")
            return

    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            if destination.exists():
                destination.unlink()
            log(f"Preparing {destination.name}, attempt {attempt}/{attempts}")
            options = gdal.WarpOptions(
                format="GTiff",
                dstSRS=TARGET_CRS,
                xRes=PIXEL_SIZE,
                yRes=PIXEL_SIZE,
                targetAlignedPixels=True,
                cutlineDSName=str(cutline) if cutline else None,
                cropToCutline=crop_to_cutline,
                srcNodata=src_nodata,
                dstNodata=0,
                resampleAlg=resample_alg,
                outputType=output_type,
                multithread=True,
                warpOptions=["NUM_THREADS=ALL_CPUS"],
                creationOptions=[
                    "TILED=YES",
                    "COMPRESS=DEFLATE",
                    "PREDICTOR=2",
                    "BIGTIFF=IF_SAFER",
                ],
            )
            result = gdal.Warp(str(destination), sources, options=options)
            if result is None:
                raise RuntimeError("GDAL returned no output dataset")
            result.FlushCache()
            result = None
            check = gdal.Open(str(destination))
            if check is None or check.RasterXSize <= 0 or check.RasterYSize <= 0:
                raise RuntimeError("output raster failed validation")
            log(
                f"Created {destination.name}: "
                f"{check.RasterXSize} x {check.RasterYSize} pixels"
            )
            check = None
            return
        except Exception as exc:
            last_error = exc
            log(f"Raster attempt {attempt}/{attempts} failed: {exc}")
            time.sleep(min(attempt * 3, 12))
    raise RuntimeError(
        f"Unable to create {destination.name} after {attempts} attempts"
    ) from last_error


def prepare_sentinel(items: list[dict]) -> dict[str, Path]:
    """Mosaic and clip the required Sentinel-2 bands at 20 metres."""
    outputs: dict[str, Path] = {}
    for band_name, (asset_name, resampling) in ASSETS.items():
        hrefs = [
            f"/vsicurl/{item['assets'][asset_name]['href']}"
            for item in items
        ]
        destination = PROCESSED_DIR / f"{band_name}_20m.tif"
        output_type = gdal.GDT_Byte if band_name == "SCL" else gdal.GDT_UInt16
        warp_with_retries(destination, hrefs, resampling, output_type)
        outputs[band_name] = destination
    return outputs


def prepare_worldcover() -> Path:
    """Download and align ESA WorldCover to the Sentinel target grid."""
    download_file(WORLDCOVER_URL, WORLDCOVER_SOURCE)
    output = PROCESSED_DIR / "worldcover_2021_20m.tif"
    warp_with_retries(
        output,
        [str(WORLDCOVER_SOURCE)],
        "near",
        gdal.GDT_Byte,
    )
    return output


def create_feature_stack(bands: dict[str, Path]) -> Path:
    """Create a nine-band feature stack and mask clouds and NoData."""
    output = PROCESSED_DIR / "lulc_feature_stack_20m.tif"
    if output.exists() and output.stat().st_size > 100_000:
        dataset = gdal.Open(str(output))
        if dataset is not None and dataset.RasterCount == 9:
            log(f"Using existing feature stack: {output.name}")
            return output

    sources = {name: gdal.Open(str(path)) for name, path in bands.items()}
    if any(dataset is None for dataset in sources.values()):
        raise RuntimeError("One or more prepared Sentinel rasters cannot be opened")

    reference = sources["B02"]
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        str(output),
        reference.RasterXSize,
        reference.RasterYSize,
        9,
        gdal.GDT_Float32,
        options=[
            "TILED=YES",
            "COMPRESS=DEFLATE",
            "PREDICTOR=3",
            "BIGTIFF=IF_SAFER",
        ],
    )
    dataset.SetGeoTransform(reference.GetGeoTransform())
    dataset.SetProjection(reference.GetProjection())
    names = ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "MNDWI", "NDBI"]
    for index, name in enumerate(names, start=1):
        band = dataset.GetRasterBand(index)
        band.SetNoDataValue(-9999.0)
        band.SetDescription(name)

    rows = reference.RasterYSize
    columns = reference.RasterXSize
    block_rows = 256
    for row in range(0, rows, block_rows):
        height = min(block_rows, rows - row)
        arrays = {
            name: sources[name].GetRasterBand(1).ReadAsArray(0, row, columns, height)
            for name in ("B02", "B03", "B04", "B08", "B11", "B12")
        }
        scl = sources["SCL"].GetRasterBand(1).ReadAsArray(0, row, columns, height)
        valid = np.isin(scl, (4, 5, 6, 7))
        for array in arrays.values():
            valid &= array > 0

        reflectance = {
            name: array.astype(np.float32) / 10_000.0
            for name, array in arrays.items()
        }

        def normalized_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
            denominator = first + second
            result = np.zeros_like(first, dtype=np.float32)
            np.divide(first - second, denominator, out=result, where=denominator != 0)
            return result

        ndvi = normalized_difference(reflectance["B08"], reflectance["B04"])
        mndwi = normalized_difference(reflectance["B03"], reflectance["B11"])
        ndbi = normalized_difference(reflectance["B11"], reflectance["B08"])
        output_arrays = [
            reflectance["B02"],
            reflectance["B03"],
            reflectance["B04"],
            reflectance["B08"],
            reflectance["B11"],
            reflectance["B12"],
            ndvi,
            mndwi,
            ndbi,
        ]
        for index, array in enumerate(output_arrays, start=1):
            array[~valid] = -9999.0
            dataset.GetRasterBand(index).WriteArray(array, 0, row)
        log(f"Feature stack rows {row + height}/{rows}")

    dataset.FlushCache()
    dataset = None
    for key in list(sources):
        sources[key] = None
    log(f"Created {output.name}")
    return output


def eroded_class_mask(classes: np.ndarray, class_id: int) -> np.ndarray:
    """Return class pixels whose eight neighbours have the same label."""
    mask = classes == class_id
    interior = mask.copy()
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    for row_shift in (-1, 0, 1):
        for column_shift in (-1, 0, 1):
            if row_shift == 0 and column_shift == 0:
                continue
            shifted = np.roll(mask, (row_shift, column_shift), axis=(0, 1))
            interior &= shifted
    return interior


def create_vector_dataset(
    path: Path,
    layer_name: str,
    geometry_type: int,
    samples: list[tuple[int, int, int]],
    geotransform: tuple[float, ...],
    projection: str,
    sample_set: str,
) -> None:
    """Write labelled sample features to a GeoPackage."""
    if path.exists():
        path.unlink()
    driver = ogr.GetDriverByName("GPKG")
    dataset = driver.CreateDataSource(str(path))
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromWkt(projection)
    layer = dataset.CreateLayer(
        layer_name,
        spatial_reference,
        geom_type=geometry_type,
    )
    layer.CreateField(ogr.FieldDefn("sample_id", ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn("class_id", ogr.OFTInteger))
    class_field = ogr.FieldDefn("class_name", ogr.OFTString)
    class_field.SetWidth(40)
    layer.CreateField(class_field)
    set_field = ogr.FieldDefn("sample_set", ogr.OFTString)
    set_field.SetWidth(12)
    layer.CreateField(set_field)

    pixel_width = geotransform[1]
    pixel_height = abs(geotransform[5])
    for sample_id, (row, column, class_id) in enumerate(samples, start=1):
        left = geotransform[0] + column * pixel_width
        top = geotransform[3] - row * pixel_height
        center_x = left + pixel_width / 2
        center_y = top - pixel_height / 2
        if geometry_type == ogr.wkbPoint:
            geometry = ogr.Geometry(ogr.wkbPoint)
            geometry.AddPoint(center_x, center_y)
        else:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(left, top)
            ring.AddPoint(left + pixel_width, top)
            ring.AddPoint(left + pixel_width, top - pixel_height)
            ring.AddPoint(left, top - pixel_height)
            ring.AddPoint(left, top)
            geometry = ogr.Geometry(ogr.wkbPolygon)
            geometry.AddGeometry(ring)

        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("sample_id", sample_id)
        feature.SetField("class_id", class_id)
        feature.SetField("class_name", CLASS_NAMES[class_id])
        feature.SetField("sample_set", sample_set)
        feature.SetGeometry(geometry)
        layer.CreateFeature(feature)
        feature = None

    layer.SyncToDisk()
    dataset = None


def create_samples(worldcover: Path, feature_stack: Path) -> tuple[Path, Path]:
    """Create deterministic, stratified training polygons and validation points."""
    worldcover_dataset = gdal.Open(str(worldcover))
    stack_dataset = gdal.Open(str(feature_stack))
    worldcover_array = worldcover_dataset.GetRasterBand(1).ReadAsArray()
    valid = stack_dataset.GetRasterBand(1).ReadAsArray() != -9999.0

    classes = np.zeros(worldcover_array.shape, dtype=np.uint8)
    for worldcover_value, class_id in WORLDCOVER_TO_CLASS.items():
        classes[worldcover_array == worldcover_value] = class_id
    classes[~valid] = 0

    rng = np.random.default_rng(RANDOM_SEED)
    training: list[tuple[int, int, int]] = []
    validation: list[tuple[int, int, int]] = []
    summary: list[dict[str, int | str]] = []
    for class_id, class_name in CLASS_NAMES.items():
        candidate_rows, candidate_columns = np.where(
            eroded_class_mask(classes, class_id)
        )
        candidate_count = len(candidate_rows)
        required = TRAIN_PER_CLASS + VALIDATION_PER_CLASS
        selected_count = min(candidate_count, required)
        if selected_count < 100:
            raise RuntimeError(
                f"Class {class_name} has only {selected_count} safe sample pixels"
            )
        selected = rng.choice(candidate_count, size=selected_count, replace=False)
        train_count = min(TRAIN_PER_CLASS, max(1, selected_count - VALIDATION_PER_CLASS))
        validation_count = min(VALIDATION_PER_CLASS, selected_count - train_count)
        for index in selected[:train_count]:
            training.append(
                (
                    int(candidate_rows[index]),
                    int(candidate_columns[index]),
                    class_id,
                )
            )
        for index in selected[train_count : train_count + validation_count]:
            validation.append(
                (
                    int(candidate_rows[index]),
                    int(candidate_columns[index]),
                    class_id,
                )
            )
        summary.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "available_safe_pixels": candidate_count,
                "training_samples": train_count,
                "validation_samples": validation_count,
            }
        )
        log(
            f"{class_name}: {train_count} training and "
            f"{validation_count} validation samples"
        )

    geotransform = stack_dataset.GetGeoTransform()
    projection = stack_dataset.GetProjection()
    training_path = PROCESSED_DIR / "training_samples.gpkg"
    validation_path = PROCESSED_DIR / "validation_samples.gpkg"
    create_vector_dataset(
        training_path,
        "training_samples",
        ogr.wkbPolygon,
        training,
        geotransform,
        projection,
        "train",
    )
    create_vector_dataset(
        validation_path,
        "validation_samples",
        ogr.wkbPoint,
        validation,
        geotransform,
        projection,
        "validation",
    )
    with (OUTPUT_DIR / "sampling_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    worldcover_dataset = None
    stack_dataset = None
    log(f"Created {training_path.name} and {validation_path.name}")
    return training_path, validation_path


def write_source_manifest(items: list[dict]) -> None:
    """Write a compact provenance manifest for the downloaded and remote inputs."""
    manifest = {
        "study_area": "Coimbatore district, Tamil Nadu, India",
        "target_crs": TARGET_CRS,
        "pixel_size_metres": PIXEL_SIZE,
        "sentinel_collection": STAC_COLLECTION,
        "sentinel_items": [
            {
                "id": item["id"],
                "datetime": item["properties"].get("datetime"),
                "cloud_cover_percent": item["properties"].get("eo:cloud_cover"),
                "assets": {
                    name: item["assets"][asset_name]["href"]
                    for name, (asset_name, _) in ASSETS.items()
                },
            }
            for item in items
        ],
        "worldcover": {
            "product": "ESA WorldCover 2021 v200",
            "url": WORLDCOVER_URL,
        },
        "class_mapping": {
            str(class_id): class_name
            for class_id, class_name in CLASS_NAMES.items()
        },
        "worldcover_to_class": {
            str(source_class): target_class
            for source_class, target_class in WORLDCOVER_TO_CLASS.items()
        },
        "random_seed": RANDOM_SEED,
    }
    with (SOURCE_DIR / "source_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> int:
    """Prepare all source, raster-feature, and sample inputs."""
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "preparation.log").write_text("", encoding="utf-8")
    configure_gdal()
    log("Starting Exercise 08 data preparation")
    items = load_stac_items()
    write_source_manifest(items)
    sentinel_bands = prepare_sentinel(items)
    worldcover = prepare_worldcover()
    feature_stack = create_feature_stack(sentinel_bands)
    create_samples(worldcover, feature_stack)
    log("Exercise 08 data preparation complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        log(f"FAILED: {error}")
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
