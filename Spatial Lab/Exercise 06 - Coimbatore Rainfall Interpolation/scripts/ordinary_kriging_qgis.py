from __future__ import annotations

from pathlib import Path

import numpy as np
from osgeo import gdal
from osgeo import ogr


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def read_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read point coordinates and rainfall observations from a GeoPackage."""
    dataset = ogr.Open(str(path))
    if dataset is None:
        raise RuntimeError(f"Could not open {path}")
    layer = dataset.GetLayer(0)
    points: list[tuple[float, float]] = []
    values: list[float] = []
    for feature in layer:
        geometry = feature.GetGeometryRef()
        points.append((geometry.GetX(), geometry.GetY()))
        values.append(float(feature.GetField("Rainfall_mm")))
    dataset = None
    return np.asarray(points, dtype=float), np.asarray(values, dtype=float)


def spherical_semivariogram(
    distance: np.ndarray,
    sill: float,
    range_m: float,
) -> np.ndarray:
    """Evaluate a zero-nugget spherical semivariogram."""
    ratio = distance / range_m
    return np.where(
        distance <= range_m,
        sill * (1.5 * ratio - 0.5 * ratio**3),
        sill,
    )


def generate_surface(
    points_path: Path,
    reference_raster: Path,
    prediction_path: Path,
    uncertainty_path: Path,
) -> None:
    """Generate Ordinary Kriging prediction and standard-error rasters."""
    points, values = read_points(points_path)
    pairwise = np.sqrt(
        ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
    )
    sill = float(np.var(values, ddof=1))
    range_m = float(np.max(pairwise) * 0.75)
    station_count = len(values)

    system = np.zeros((station_count + 1, station_count + 1), dtype=float)
    system[:station_count, :station_count] = spherical_semivariogram(
        pairwise,
        sill,
        range_m,
    )
    system[:station_count, station_count] = 1.0
    system[station_count, :station_count] = 1.0
    system_inverse = np.linalg.pinv(system)

    reference = gdal.Open(str(reference_raster))
    if reference is None:
        raise RuntimeError(f"Could not open {reference_raster}")
    width = reference.RasterXSize
    height = reference.RasterYSize
    geotransform = reference.GetGeoTransform()
    projection = reference.GetProjection()
    x_coordinates = (
        geotransform[0]
        + (np.arange(width) + 0.5) * geotransform[1]
    )
    y_coordinates = (
        geotransform[3]
        + (np.arange(height) + 0.5) * geotransform[5]
    )

    prediction = np.empty((height, width), dtype=np.float32)
    uncertainty = np.empty((height, width), dtype=np.float32)
    for row, y_coordinate in enumerate(y_coordinates):
        targets = np.column_stack(
            (x_coordinates, np.full(width, y_coordinate))
        )
        distances = np.sqrt(
            ((targets[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
        )
        gamma = spherical_semivariogram(distances, sill, range_m)
        right_hand_side = np.vstack((gamma.T, np.ones(width)))
        weights = system_inverse @ right_hand_side
        prediction[row, :] = values @ weights[:station_count, :]
        variance = (
            np.sum(weights[:station_count, :].T * gamma, axis=1)
            + weights[station_count, :]
        )
        uncertainty[row, :] = np.sqrt(np.maximum(variance, 0.0))

    driver = gdal.GetDriverByName("GTiff")
    options = ["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"]
    for path, array, description in (
        (prediction_path, prediction, "Ordinary Kriging rainfall prediction (mm)"),
        (uncertainty_path, uncertainty, "Ordinary Kriging standard error (mm)"),
    ):
        output = driver.Create(
            str(path),
            width,
            height,
            1,
            gdal.GDT_Float32,
            options=options,
        )
        output.SetGeoTransform(geotransform)
        output.SetProjection(projection)
        band = output.GetRasterBand(1)
        band.WriteArray(array)
        band.SetNoDataValue(-9999.0)
        band.SetDescription(description)
        band.FlushCache()
        output.FlushCache()
        output = None

    print(
        f"{points_path.name}: {station_count} stations; "
        f"sill={sill:.3f} mm^2; range={range_m:.1f} m"
    )


def main() -> None:
    """Regenerate the training and final Kriging surfaces."""
    generate_surface(
        PROCESSED / "rainfall_training.gpkg",
        PROCESSED / "idw_training.tif",
        PROCESSED / "kriging_training.tif",
        PROCESSED / "kriging_uncertainty.tif",
    )
    generate_surface(
        PROCESSED / "rainfall_stations_utm.gpkg",
        PROCESSED / "rainfall_idw_final.tif",
        PROCESSED / "rainfall_kriging_final.tif",
        PROCESSED / "rainfall_kriging_uncertainty_final.tif",
    )


if __name__ == "__main__":
    main()
