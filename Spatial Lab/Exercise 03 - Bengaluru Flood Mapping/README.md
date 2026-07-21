# Exercise 03 - Bengaluru Flood Mapping

Open `project/exercise-03-bengaluru-flood-mapping.qgz`.

The project combines the BBMP boundary, stormwater drains, Bengaluru OSM features, and the processed analysis GeoPackage. The GeoPackage contains the boundary, lakes, drains, flood-risk zones, schools, health facilities, roads, and shelter-selection layers.

The expected baseline checks are:

- `bengaluru_boundary_utm`: 1 feature
- `drains_clipped`: 6,838 features
- `roads_routable`: approximately 130,454 features

Download utilities are in `tools`. Failed downloads, test scripts, the earlier empty project, and Python caches are retained in `archive` for provenance and are not part of the active workflow.
