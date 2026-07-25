# Exercise 05 - Nilgiris Check-Dam Suitability

This is a compact version of the supplied workflow. It demonstrates the required terrain and hydrology ideas without adding unnecessary datasets or layouts.

## Open the exercise

Open `project/exercise-05-nilgiris-check-dam.qgz` in QGIS. The project CRS is **EPSG:32643 — WGS 84 / UTM Zone 43N**.

The project contains three groups:

- **Final Map** — study DEM, hillshade, 20 m contours, derived streams, and six preliminary candidate sites.
- **Terrain Models** — TIN, IDW GRID, TIN edges, slope, and flow accumulation.
- **Inputs** — the source Copernicus DEM and 1 km elevation samples.

## Simplified method

1. Download one [Copernicus DEM GLO-30](https://registry.opendata.aws/copernicus-dem/) tile.
2. Clip and reproject a small Kotagiri-Coonoor study area to EPSG:32643 at 30 m resolution.
3. Generate 20 m contours and sample elevation at a 1 km point grid.
4. Create a TIN surface and an IDW GRID from the same sample points.
5. Derive slope and hillshade from the clipped reference DEM.
6. Run GRASS `r.watershed` with a 1,000-cell stream threshold.
7. Sample stream points and retain six separated examples with slope between 2° and 15° and upstream accumulation above 1,000 cells.
8. Export one final presentation map as PDF and PNG.

## Main outputs

- `outputs/exercise-05-final-check-dam-map.pdf`
- `outputs/exercise-05-final-check-dam-map.png`
- `data/processed/preliminary_checkdam_sites.gpkg`

The candidates are for an academic GIS exercise only. They are not construction recommendations. Field survey, ownership checks, environmental review, runoff estimation, foundation investigation, and engineering design are required before any real check dam is proposed.
