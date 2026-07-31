# Exercise 07 - Coimbatore Healthcare Service Areas

This QGIS exercise uses a small, presentation-ready workflow to:

1. Create Thiessen polygons around all 381 mapped hospitals.
2. Sum WorldPop population inside each service region.
3. Compare population served with a simple hospital-capacity weight.
4. Mark high-overload regions and propose 50 classroom candidate sites.
5. Estimate population within 5 km of any mapped hospital.

## Open the project

Open:

`project/exercise-07-healthcare-service-areas.qgz`

The project CRS is `EPSG:32643 - WGS 84 / UTM zone 43N`, and its data sources are stored as relative paths.

## Simple procedure

1. Download hospital features from OpenStreetMap and convert them to points.
2. Clip the hospitals to Coimbatore and reproject all 381 features to `EPSG:32643`.
3. Run **Voronoi polygons** with a 200% buffer, then clip the result to the district.
4. Clip the WorldPop 2020 1 km population raster to Coimbatore.
5. Run **Zonal statistics** with `Sum` to calculate population in every Thiessen region.
6. Calculate:
   - `target_pop = total population * capacity_wt / total capacity`
   - `load_ratio = pop_sum / target_pop`
7. Classify the regions as low load, balanced, overloaded, or high overload.
8. Create a point on surface for each high-overload region as a simple proposed-site layer.
9. Buffer all hospitals by 5 km, dissolve, clip to the district, and sum population inside the coverage area.
10. Export the final layout as PNG and PDF.

## Main results

- Estimated district population: about **4.20 million**
- Population within 5 km of a mapped hospital: about **77.4%**
- Hospitals used: **381**
- Proposed classroom candidate sites: **50**

## Outputs

- `outputs/coimbatore-healthcare-service-areas.png`
- `outputs/coimbatore-healthcare-service-areas.pdf`

## Data sources and limitation

- Hospitals: OpenStreetMap via Overpass API, accessed 2026-07-28
- Population: WorldPop India 2020, 1 km population counts
- Boundary: geoBoundaries India ADM2

This is a classroom planning model. Capacity weights are inferred from available OpenStreetMap tags, candidate sites are representative points inside overloaded regions, and accessibility is straight-line distance rather than road travel time.
