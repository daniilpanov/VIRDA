# VIRDA Electrode Localization

Software for localizing VIRDA electrodes in a patient's MRI coordinate system using a three-stage pipeline: MRI scalp mesh generation, virtual Electrode Surface Equivalent (ESE) construction, and real electrode localization from external measurements.

## Pipeline

### Stage 1 — MRI Head Surface Mesh Generation

Loads a patient MRI, segments the external head surface, and generates a triangular scalp mesh. Anatomical fiducials (nasion, preauricular points, inion) are marked and stored. An electrode offset parameter defines the distance from the scalp to the ESE.

**Output:** scalp mesh, fiducial coordinates, mesh connectivity, ESE offset settings.

### Stage 2 — Electrode Surface Equivalent (ESE)

For each scalp mesh vertex, a local PCA neighborhood is used to estimate the outward surface normal. The ESE point is computed by moving each vertex outward along the normal by the electrode offset distance.

Key steps per vertex:
1. Collect neighboring mesh points (by count or radius).
2. Build and center the local coordinate matrix.
3. Compute the covariance matrix and its eigenvectors.
4. The eigenvector with the smallest eigenvalue is the surface normal.
5. Orient the normal outward (toward the head centroid).
6. Compute the ESE point: `p_ESE = p_0 + d * n`.

**Output:** scalp-to-ESE point pairs, normal vectors, PCA quality metric.

### Stage 3 — Real Electrode Localization

Searches the ESE for the point whose Euclidean distances to the anatomical fiducials best match the externally measured distances (from photogrammetry or calipers). The best-match ESE point is mapped back to the corresponding scalp coordinate.

**Output:** MRI coordinates of each electrode, residual errors, CSV/JSON export, 3D visualization.

## Software Modules

| Module | Role |
|---|---|
| MRI Loader | Read DICOM/NIfTI with spatial metadata |
| Head Segmenter | Separate head from background |
| Surface Extractor | Generate triangular mesh |
| Mesh Cleaner | Remove artifacts, repair defects |
| Fiducial Manager | Create, edit, store landmarks |
| ESE Configuration | Store ESE definition and offset |
| PCA Normal Estimator | Local surface normal via PCA |
| ESE Generator | Build scalp-to-ESE point pairs |
| Measurement Import | Load fiducial distances |
| Electrode Localizer | Match measurements to ESE |
| 3D Viewer | Visualize mesh, ESE, electrodes |
| Exporter | Save results in standard formats |

## Development Order

1. Prototype on a sphere.
2. Prototype on a realistic head mesh.
3. Prototype on a patient MRI.
4. Test with real VIRDA measurements.

## Documentation

Detailed specifications are in [`docs/task/`](docs/task/):

- [Project Overview](docs/task/VIRDA_Project_Overview.md)
- [Stage 1 — Mesh Generation](docs/task/VIRDA_Stage1_Head_Surface_Mesh_Generation.md)
- [Stage 2 — ESE Construction](docs/task/VIRDA_Stage2_Electrode_Surface_Equivalent_Constraction.md)
- [Stage 2 — Radius-Defined Neighborhood](docs/task/Radius_Defined_Neighborhood_Stage2.md)
- [Stage 3 — Electrode Localization](docs/task/VIRDA_Stage3_Real_Electrode_Locations.md)
