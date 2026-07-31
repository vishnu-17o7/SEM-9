# Exercise 08 - Coimbatore LULC Classification

This exercise classifies Coimbatore District into five land-use/land-cover classes using Sentinel-2 Level-2A imagery and the Dzetsaka Random Forest classifier in QGIS.

## Open the project

Open the presentation project:

`project/exercise-08-coimbatore-lulc.qgz`

The full working-layer project is retained as:

`project/exercise-08-analysis.qgz`

Both projects use `EPSG:32643 - WGS 84 / UTM zone 43N`. Data are stored inside the exercise directory so the projects do not depend on Downloads, temporary processing folders, or attachment paths.

## Simple procedure

1. Use four low-cloud Sentinel-2 L2A tiles from 10 March 2026.
2. Mosaic and clip B02, B03, B04, B08, B11, B12, and SCL to Coimbatore.
3. Align all rasters to a 20 m grid in `EPSG:32643`.
4. Calculate NDVI, MNDWI, and NDBI.
5. Stack the six reflectance bands and three indices into a nine-band raster.
6. Align ESA WorldCover 2021 to the same grid.
7. Convert WorldCover classes into five simple lab classes.
8. Create 800 automated training polygons and 200 held-out validation points per class.
9. Train **Dzetsaka Random Forest** with 100 trees and a 20% internal validation split.
10. Predict the complete district, create a confidence raster, and apply a four-pixel, eight-connected sieve.
11. Sample the final raster at the 1,000 held-out points and calculate the confusion matrix, accuracy, and class areas.
12. Fill only cloud/NoData gaps in the presentation copy using the aligned WorldCover reference.
13. Export the final A4 landscape layout as PNG and PDF.

## Main results

- Held-out validation samples: **1,000**
- Correct predictions: **817**
- Overall agreement: **81.7%**
- Cohen's kappa: **0.7712**

| Class | Area (km²) | Share |
|---|---:|---:|
| Water | 44.23 | 0.94% |
| Forest and vegetation | 2,665.86 | 56.77% |
| Agriculture | 1,301.73 | 27.72% |
| Built-up | 564.78 | 12.03% |
| Barren and open land | 119.40 | 2.54% |

## Main outputs

- `data/processed/lulc_feature_stack_20m.tif`
- `data/processed/training_samples.gpkg`
- `data/processed/validation_samples.gpkg`
- `data/processed/random_forest_model.pkl`
- `data/processed/lulc_rf_raw_20m.tif`
- `data/processed/lulc_rf_confidence_20m.tif`
- `data/processed/lulc_rf_final_20m.tif`
- `data/processed/lulc_rf_presented_20m.tif`
- `outputs/accuracy_summary.csv`
- `outputs/external_confusion_matrix.csv`
- `outputs/class_accuracy.csv`
- `outputs/class_areas.csv`
- `outputs/Coimbatore_LULC_Random_Forest.png`
- `outputs/Coimbatore_LULC_Random_Forest.pdf`

## Reproduce the preparation

Run the scripts with the QGIS Python environment:

```powershell
& "C:\Program Files\QGIS 4.0.3\bin\python-qgis.bat" "scripts\prepare_lulc_data.py"
& "C:\Program Files\QGIS 4.0.3\bin\python-qgis.bat" "scripts\summarize_results.py"
```

The preparation script retries each network input at most five times. Sentinel source URLs, scene IDs, cloud percentages, class mapping, resolution, and random seed are recorded in `data/source/source_manifest.json`.

## Data sources and limitation

- Sentinel-2 L2A cloud-optimized GeoTIFFs: Element 84 Earth Search / Copernicus Sentinel-2
- Reference land cover: ESA WorldCover 2021 v200
- Boundary: the existing Coimbatore Spatial Lab boundary

The training labels and held-out validation labels were both derived from ESA WorldCover, although their sample locations do not overlap. Therefore, **81.7% measures agreement against held-out WorldCover-derived labels, not field-survey accuracy**. The presentation raster uses WorldCover only to fill cloud/NoData gaps; the untouched Random Forest and sieved rasters are retained separately.
