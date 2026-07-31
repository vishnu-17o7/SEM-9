# Exercise 06 - Coimbatore Rainfall Interpolation

This QGIS exercise compares IDW and Ordinary Kriging rainfall interpolation for Coimbatore district using real October 1965 precipitation observations.

## Open the project

Open `project/exercise-06-coimbatore-rainfall.qgz` in QGIS. The project uses **EPSG:32643 — WGS 84 / UTM Zone 43N**.

The project contains:

- **Final Maps** — clipped IDW, Ordinary Kriging, Kriging uncertainty, 20 mm isohyets, stations, and the district boundary.
- **Accuracy and Test Surfaces** — the fixed 17-station training and 5-station validation split.
- **Inputs** — the 30 km analysis buffer and original source layers.
- Three map themes: `IDW Final`, `Kriging Final`, and `Kriging Uncertainty`.
- Two A4 landscape layouts: comparison and uncertainty.

## Data

- Rainfall: NOAA Global Historical Climatology Network Daily.
- Period: October 1965.
- Selected observations: 22 nearby stations with all 31 daily precipitation observations.
- Boundary: geoBoundaries India ADM2, representing 2021 district boundaries.
- Rainfall units: millimetres.

Source downloads:

- `https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt`
- `https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/1965.csv.gz`
- `https://www.geoboundaries.org/api/current/gbOpen/IND/ADM2/`

## Procedure

1. Aggregate valid NOAA daily `PRCP` values for October 1965.
2. Select the 22 nearest complete stations around Coimbatore.
3. Reproject stations and the district boundary to EPSG:32643.
4. Assign 17 stations to training and five to independent validation.
5. Generate a 1 km IDW surface with power 2.
6. Generate a matching 1 km Ordinary Kriging surface using a zero-nugget spherical semivariogram.
7. Sample both test surfaces at the five validation stations.
8. Calculate residual, absolute error, squared error, MAE, RMSE, and bias.
9. Rerun both methods using all 22 stations.
10. Clip the final rasters to Coimbatore district and generate 20 mm isohyets from the preferred IDW surface.

## Accuracy result

| Method | MAE (mm) | RMSE (mm) | Bias (mm) |
|---|---:|---:|---:|
| IDW | 41.332 | 56.704 | +31.569 |
| Ordinary Kriging | 41.628 | 61.342 | +29.387 |

IDW has the lower validation RMSE for this fixed split and is therefore the preferred surface for this exercise.

## Kriging model

- Semivariogram: spherical
- Nugget: 0
- Final sill: 5,437.152 mm²
- Final range: 76.335 km
- Prediction grid: matched to the IDW raster

SAGA is not installed in the current QGIS 4 environment. The reproducible Ordinary Kriging implementation is stored in `scripts/ordinary_kriging_qgis.py` and uses the QGIS-bundled NumPy and GDAL runtime.

Run the script from the QGIS Python environment. Remove the existing Kriging rasters from the project before regenerating them, because Windows prevents GDAL from overwriting raster files that QGIS currently has open.

## Final outputs

- `outputs/rainfall-idw-vs-kriging-comparison.pdf`
- `outputs/rainfall-idw-vs-kriging-comparison.png`
- `outputs/kriging-uncertainty-coimbatore.pdf`
- `outputs/kriging-uncertainty-coimbatore.png`

The maps are for an academic interpolation exercise. The historical observations and model outputs should not be used for operational rainfall or hazard decisions.
