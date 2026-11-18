![Python 3.14](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![Package Manager](https://img.shields.io/badge/Package%20Manager-uv-DE5FE9?logo=python)
![CI Status](https://github.com/daniilpanov/VIRDA/actions/workflows/ci.yml/badge.svg)

# VIRDA Electrode Localization

Software for localizing VIRDA electrodes in a patient's MRI coordinate system using a three-stage pipeline: MRI scalp mesh generation, virtual Electrode Surface Equivalent (ESE) construction, and real electrode localization from external measurements.

## Getting started

Requires **Python 3.14** and [uv](https://docs.astral.sh/uv/). Install the whole
workspace (the core package and the GUI) and launch the GUI:

```bash
uv sync --all-extras --all-packages
uv run virda-gui-new
```

The interactive GUI is the supported interface: it opens a project picker and
then the main window with the 3D scene, the steps navigator and the step
popups. The old batch CLI has been removed (`python -m virda` only prints a
notice).

Run the checks with `uv run pre-commit run --all-files` and `uv run pytest -n 6`.

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

## GUI workflow

The GUI turns the pipeline into five ordered steps. Each step stores its
settings in `config/<step>.json` of the project and persists its artifacts, so
reopening a project continues where it stopped. A step needs the artifacts of
the previous one; the steps menu marks each one as `[done]`, `[current]` or
`[needs: ...]` and disables those that cannot run yet.

| Step | What it does |
|---|---|
| **NIfTI import** | Select the source MRI; a byte-for-byte copy goes into `input/`. |
| **Base mesh** | Segment the head and extract the scalp surface, with an optional mask seal. |
| **Mesh postprocessing** | Clean, decimate and smooth the scalp mesh. |
| **ESE mesh** | Build the offset electrode surface and estimate outward normals with local PCA. |
| **Localization** | Enter fiducials and measured distances, then localize electrodes live; the results table, the 3D view and the frame selectors all update together. |

Fiducials, measurements and electrodes can be imported from JSON/CSV. Every
number stays stored in world (scanner RAS) millimetres; the coordinate-frame
selectors of the tables and of the export dialogs only change how numbers are
shown and written. The **Voxel** frame shows NIfTI grid indices and requires
the imported volume; PLY mesh exports do not offer it because the format carries
no frame metadata.

See [GUI widget parameters](docs/GUI_WIDGET_PARAMETERS.md) for every option in
the step popups, tables and viewer.

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

Reference documentation:

- [GUI widget parameters](docs/GUI_WIDGET_PARAMETERS.md) — every option in the GUI, its default and range
- [Coordinate systems](docs/COORDINATE_SYSTEMS.md) — the world, head, fiducials and voxel frames
- [Patient project format](docs/PATIENT_PROJECT_FORMAT.md) — project layout and artifact file formats

Detailed specifications are in [`docs/task/`](docs/task/):

- [Project Overview](docs/task/VIRDA_Project_Overview.md)
- [Stage 1 — Mesh Generation](docs/task/VIRDA_Stage1_Head_Surface_Mesh_Generation.md)
- [Stage 2 — ESE Construction](docs/task/VIRDA_Stage2_Electrode_Surface_Equivalent_Constraction.md)
- [Stage 2 — Radius-Defined Neighborhood](docs/task/Radius_Defined_Neighborhood_Stage2.md)
- [Stage 3 — Electrode Localization](docs/task/VIRDA_Stage3_Real_Electrode_Locations.md)
