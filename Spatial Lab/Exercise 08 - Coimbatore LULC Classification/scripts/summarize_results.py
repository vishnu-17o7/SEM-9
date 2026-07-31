from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from osgeo import gdal
from osgeo import ogr


EXERCISE_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = EXERCISE_ROOT / "data" / "processed"
OUTPUT_DIR = EXERCISE_ROOT / "outputs"
CLASS_NAMES = {
    1: "Water",
    2: "Forest and vegetation",
    3: "Agriculture",
    4: "Built-up",
    5: "Barren and open land",
}
WORLDCOVER_TO_CLASS = {
    10: 2,
    20: 2,
    30: 2,
    40: 3,
    50: 4,
    60: 5,
    80: 1,
    90: 2,
    95: 2,
    100: 2,
}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write a list of dictionaries to a UTF-8 CSV file."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_validation_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Read reference and predicted classes from the sampled validation layer."""
    dataset = ogr.Open(str(PROCESSED_DIR / "validation_predictions.gpkg"))
    if dataset is None:
        raise RuntimeError("validation_predictions.gpkg cannot be opened")
    layer = dataset.GetLayer(0)
    fields = [
        layer.GetLayerDefn().GetFieldDefn(index).GetName()
        for index in range(layer.GetLayerDefn().GetFieldCount())
    ]
    prediction_fields = [name for name in fields if name.lower().startswith("pred_")]
    if not prediction_fields:
        raise RuntimeError(f"No prediction field found in {fields}")
    prediction_field = prediction_fields[0]
    references: list[int] = []
    predictions: list[int] = []
    for feature in layer:
        reference = feature.GetField("class_id")
        prediction = feature.GetField(prediction_field)
        if reference is None or prediction is None:
            continue
        references.append(int(reference))
        predictions.append(int(round(float(prediction))))
    dataset = None
    return np.asarray(references), np.asarray(predictions)


def calculate_accuracy(references: np.ndarray, predictions: np.ndarray) -> None:
    """Calculate independent confusion-matrix and per-class accuracy metrics."""
    labels = np.asarray(sorted(CLASS_NAMES))
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for reference, prediction in zip(references, predictions, strict=True):
        if reference in CLASS_NAMES and prediction in CLASS_NAMES:
            matrix[reference - 1, prediction - 1] += 1

    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    overall = correct / total if total else 0.0
    row_totals = matrix.sum(axis=1)
    column_totals = matrix.sum(axis=0)
    expected = (
        float(np.dot(row_totals, column_totals)) / float(total * total)
        if total
        else 0.0
    )
    kappa = (overall - expected) / (1.0 - expected) if expected != 1.0 else 0.0

    confusion_rows: list[dict] = []
    for row_index, class_id in enumerate(labels):
        row = {
            "reference_class": CLASS_NAMES[int(class_id)],
            **{
                f"predicted_{CLASS_NAMES[int(predicted_id)].lower().replace(' ', '_')}": int(
                    matrix[row_index, predicted_id - 1]
                )
                for predicted_id in labels
            },
            "total": int(row_totals[row_index]),
        }
        confusion_rows.append(row)
    write_csv(
        OUTPUT_DIR / "external_confusion_matrix.csv",
        list(confusion_rows[0]),
        confusion_rows,
    )

    class_rows: list[dict] = []
    for index, class_id in enumerate(labels):
        true_positive = int(matrix[index, index])
        producer = true_positive / row_totals[index] if row_totals[index] else 0.0
        user = true_positive / column_totals[index] if column_totals[index] else 0.0
        class_rows.append(
            {
                "class_id": int(class_id),
                "class_name": CLASS_NAMES[int(class_id)],
                "reference_samples": int(row_totals[index]),
                "predicted_samples": int(column_totals[index]),
                "correct_samples": true_positive,
                "producer_accuracy_percent": round(producer * 100.0, 2),
                "user_accuracy_percent": round(user * 100.0, 2),
            }
        )
    write_csv(
        OUTPUT_DIR / "class_accuracy.csv",
        list(class_rows[0]),
        class_rows,
    )

    summary_rows = [
        {"metric": "Validation samples used", "value": total},
        {"metric": "Correct predictions", "value": correct},
        {"metric": "Overall accuracy percent", "value": round(overall * 100.0, 2)},
        {"metric": "Cohen kappa", "value": round(kappa, 4)},
    ]
    write_csv(
        OUTPUT_DIR / "accuracy_summary.csv",
        ["metric", "value"],
        summary_rows,
    )


def create_presentation_raster() -> Path:
    """Fill only cloud and NoData gaps with the aligned WorldCover reference."""
    classification = gdal.Open(str(PROCESSED_DIR / "lulc_rf_final_20m.tif"))
    worldcover = gdal.Open(str(PROCESSED_DIR / "worldcover_2021_20m.tif"))
    if classification is None or worldcover is None:
        raise RuntimeError("Classification or WorldCover raster cannot be opened")
    output_path = PROCESSED_DIR / "lulc_rf_presented_20m.tif"
    driver = gdal.GetDriverByName("GTiff")
    output = driver.Create(
        str(output_path),
        classification.RasterXSize,
        classification.RasterYSize,
        1,
        gdal.GDT_Byte,
        options=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
    )
    output.SetGeoTransform(classification.GetGeoTransform())
    output.SetProjection(classification.GetProjection())
    output_band = output.GetRasterBand(1)
    output_band.SetNoDataValue(0)
    block_rows = 512
    for row in range(0, classification.RasterYSize, block_rows):
        height = min(block_rows, classification.RasterYSize - row)
        classes = classification.GetRasterBand(1).ReadAsArray(
            0,
            row,
            classification.RasterXSize,
            height,
        ).astype(np.uint8)
        reference = worldcover.GetRasterBand(1).ReadAsArray(
            0,
            row,
            worldcover.RasterXSize,
            height,
        )
        fallback = np.zeros(reference.shape, dtype=np.uint8)
        for worldcover_class, class_id in WORLDCOVER_TO_CLASS.items():
            fallback[reference == worldcover_class] = class_id
        fill = (classes == 0) & (fallback > 0)
        classes[fill] = fallback[fill]
        output_band.WriteArray(classes, 0, row)
    output_band.FlushCache()
    output = None
    classification = None
    worldcover = None
    return output_path


def calculate_class_areas(raster_path: Path) -> None:
    """Calculate classified area and percentage for every land-cover class."""
    dataset = gdal.Open(str(raster_path))
    if dataset is None:
        raise RuntimeError("lulc_rf_final_20m.tif cannot be opened")
    band = dataset.GetRasterBand(1)
    counts = {class_id: 0 for class_id in CLASS_NAMES}
    block_rows = 512
    for row in range(0, dataset.RasterYSize, block_rows):
        height = min(block_rows, dataset.RasterYSize - row)
        array = band.ReadAsArray(0, row, dataset.RasterXSize, height)
        for class_id in counts:
            counts[class_id] += int(np.count_nonzero(array == class_id))
    pixel_area_m2 = abs(dataset.GetGeoTransform()[1] * dataset.GetGeoTransform()[5])
    total_classified = sum(counts.values())
    rows: list[dict] = []
    for class_id, class_name in CLASS_NAMES.items():
        count = counts[class_id]
        area_km2 = count * pixel_area_m2 / 1_000_000.0
        percentage = count / total_classified * 100.0 if total_classified else 0.0
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "pixel_count": count,
                "area_km2": round(area_km2, 2),
                "percentage": round(percentage, 2),
            }
        )
    write_csv(OUTPUT_DIR / "class_areas.csv", list(rows[0]), rows)
    dataset = None


def main() -> None:
    """Create the accuracy and class-area deliverables."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    references, predictions = load_validation_predictions()
    calculate_accuracy(references, predictions)
    presentation_raster = create_presentation_raster()
    calculate_class_areas(presentation_raster)
    print(f"Validation records: {len(references)}")
    print(f"Accuracy outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
