# Patient Project Format

Layout and file formats of a **patient project** directory written by the VIRDA
pipeline. The pipeline writes every artifact into a dedicated subfolder so that
any stage can be reproduced exactly from the project alone:

- **Stage 1** (segmentation/mesh/fiducials) always runs;
- **Stage 2** (`ese/`) runs when an ESE config is supplied;
- **Stage 3** (`localization/`) runs when both ESE and measurements are available.

```
<project_dir>/
├── input/              # source MRI NIfTI copy + merged pipeline configuration
├── segmentation/       # head segmentation mask
├── mesh/               # final scalp mesh, arrays and per-step versions
├── fiducials/          # fiducial table
├── config/             # ESE configuration
├── ese/                # Stage 2 output: electrode-skin-entrance surface
├── localization/       # Stage 3 output: localized electrodes
├── quality_control/    # automatic QC report
└── logs/               # pipeline log

viewer.html             # optional: self-contained HTML viewer exported by virda-gui
```

## `input/`

| File | Description |
|---|---|
| `<source_mri>.nii.gz` | Byte-for-byte copy of the source MRI NIfTI. |
| `pipeline_config.json` | Full merged `Config` dump (`model_dump(mode="json")`); written on every run. |

Example `pipeline_config.json` (values follow the `Config` defaults):

```json
{
  "nifti_path": null,
  "project_dir": null,
  "fiducials_path": null,
  "auto_detect_fiducials": false,
  "closing_radius": 5,
  "otsu_scope": "all",
  "otsu_threshold_scale": 0.6,
  "seal_enabled": true,
  "seal_radius": 4,
  "cleaner_min_vertices": 100,
  "cleaner_merge_digits": 7,
  "smoother_type": "laplacian",
  "smoother_iterations": 5,
  "smoother_lamb": 0.5,
  "smoother_nu": -0.53,
  "n_electrodes": null,
  "ese_offset_mm": null,
  "ese_reference": null,
  "neighborhood_radius_mm": 10.0,
  "k_neighbors": null,
  "use_weighted_pca": false,
  "pca_sigma_mm": 5.0,
  "min_neighbors": 5,
  "coordsystem": null,
  "residual_threshold_mm": 10.0,
  "calibrate_ese_offset": false
}
```

When an MNE ``coordsystem.json`` was loaded as an input config file, its parsed
contents are embedded here under `"coordsystem"` (fiducial positions, electrode
count / offset / reference) instead of `null`.

## `segmentation/`

| File | Description |
|---|---|
| `head_mask.nii.gz` | Binary head segmentation mask as a uint8 NIfTI written with the MRI affine, so voxel coordinates map 1:1 to the source MRI. |

## `mesh/`

| File | Description |
|---|---|
| `final_mesh.ply` | Final scalp mesh (trimesh PLY export). |
| `scalp_vertices.npy` | `(N, 3)` `float64` array of vertex coordinates in world millimeters. |
| `scalp_faces.npy` | `(M, 3)` `int64` array of triangular faces referencing vertices. |
| `scalp_face_adjacency.npy` | `(E, 2)` `int64` array of face-index pairs sharing an edge. |
| `n_adjacency_edges.json` | `{"n_adjacency_edges": <E>}`. |
| `versions/mesh-<n>.ply` | One PLY per `ScalpMesh` update produced during the pipeline (extraction, cleaning, smoothing, ...); `n` counts from 1. |

## `fiducials/`

`fiducials.json` — fiducial table:

```json
{
  "fiducials": [
    {
      "fiducial_id": "NAS",
      "name": "Nasion",
      "coordinates": [0.0, 88.0, -10.0],
      "coordinate_system": "world",
      "definition_method": "manual",
      "weight": 1.5
    }
  ]
}
```

- `coordinates` are world millimeters (scanner RAS of the source NIfTI);
  `coordinate_system` mirrors the model value (`"world"` on standard runs,
  `"voxel"` is also valid in the schema).
- `weight` is an optional positive float (default `1.0`) that scales the
  fiducial's influence during Stage 3 localization; files written by earlier
  versions without the field load with unit weight.
- `definition_method` is one of `"manual"` (manual file), `"auto"`
  (auto-detection) or `"imported"` (taken from an MNE `coordsystem.json`).

## `config/`

`ese.json` — ESE configuration; written only when an ESE config is supplied:

```json
{
  "ese": {
    "n_electrodes": 32,
    "ese_offset_mm": 2.5,
    "ese_reference": "electrode_body_center"
  }
}
```

## `ese/`

Stage 2 output (electrode-skin-entrance surface offset outward from the scalp
along local normals); written only when Stage 2 runs.

| File | Description |
|---|---|
| `ese_mesh.ply` | The ESE surface as a trimesh PLY export. |
| `ese_vertices.npy` | `(N, 3)` `float64` array of ESE vertex coordinates (world mm). |
| `ese_faces.npy` | `(M, 3)` `int64` triangular faces of the ESE mesh. |
| `normals.npy` | `(N, 3)` unit normal at each scalp vertex used for the offset. |
| `quality.npy` | `(N,)` normal-estimation quality score per vertex. |
| `point_pairs.json` | Scalp↔ESE correspondence dump (see below). |

`point_pairs.json` maps every scalp vertex to its offset ESE vertex:

```json
{
  "n_points": 232205,
  "scalp_vertices": [[...], ...],
  "ese_vertices": [[...], ...],
  "normals": [[...], ...],
  "quality": [...]
}
```

The arrays are row-aligned: `ese_vertices[i]`, `normals[i]` and `quality[i]`
belong to `scalp_vertices[i]`.

## `localization/`

Stage 3 output; written only when both the ESE surface and measurements are
available.

| File | Description |
|---|---|
| `electrodes.json` | Full localization result for every electrode (see below). |
| `electrodes_scalp.json` | Same electrodes with only `coords` = scalp contact points (scanner RAS). |
| `electrodes_ese.json` | Same electrodes with only `coords` = ESE body-center points (scanner RAS). |
| `electrode_coords.csv` | Tabular copy: `electrode_id, x, y, z, residual_error, confidence, flagged`; `x/y/z` are the ESE coordinates. Empty `residual_error`/`confidence` mark non-localized electrodes. |
| `localization_summary.json` | Aggregate statistics (see below). |

`electrodes.json` — one entry per electrode:

```json
[
  {
    "electrode_id": "EEG 001",
    "measured_distances": {"NAS": 62.28, "LPA": 111.68, "RPA": 158.42},
    "ese_coords": [-35.24, 61.01, 68.10],
    "scalp_coords": [-35.19, 60.96, 68.02],
    "residual_error": 1.27,
    "confidence": 0.0012,
    "flagged": false
  }
]
```

- `measured_distances` are the input distances to each fiducial (mm).
- `ese_coords` / `scalp_coords` are `null` for electrodes that could not be
  localized (`is_localized == false`); `residual_error` and `confidence`
  are then `null` too. All coordinates are scanner RAS millimeters.
- `flagged` marks electrodes whose residual exceeds
  `residual_threshold_mm`.

`localization_summary.json`:

```json
{
  "n_electrodes": 60,
  "n_localized": 60,
  "n_flagged": 0,
  "median_residual_mm": 1.27,
  "residual_threshold_mm": 10.0,
  "calibrated_ese_offset_shift_mm": 0.0
}
```

`calibrated_ese_offset_shift_mm` is present only when
`calibrate_ese_offset` was enabled (`null` otherwise).

## `quality_control/`

`report.json` — automatic QC report written by the final Stage 1 step:

```json
{
  "status": "warn",
  "checks": [
    {
      "name": "mri_metadata",
      "status": "ok",
      "message": "affine, spacing and orientation are valid",
      "affine_shape": [4, 4],
      "spacing": [1.0, 1.0, 1.0],
      "orientation": ["R", "A", "S"]
    }
  ],
  "fiducials": {
    "name": "fiducials_on_surface",
    "status": "ok",
    "message": "all fiducials lie on the scalp surface",
    "checks": [
      {"fiducial_id": "NAS", "name": "Nasion", "distance_to_surface_mm": 1.4}
    ],
    "tolerance_mm": 3.0,
    "warnings": []
  },
  "warnings": []
}
```

- `status` is one of `ok`, `warn`, `fail`; checks with status `skip` are ignored
  when aggregating the overall status (`fail` > `warn` > `ok`).
- Each check entry is `{name, status, message, ...details}`.
- `fiducials.checks` lists the per-fiducial distance to the nearest mesh vertex
  for every fiducial.

### Checks

| Check (`name`) | Spec | Status |
|---|---|---|
| `mri_metadata` | §13.1 "MRI affine is present and valid"; §16 "report missing or inconsistent MRI spatial metadata" | **fail** if the affine is not 4×4, its bottom row is not `[0,0,0,1]`, the spacing is not 3 positive values, or the orientation lacks 3 axis codes (`R`/`L`/`A`/`P`/`S`/`I`). |
| `coordinates_mm` | §13.1 "mesh coordinates are in millimeters"; §16 "do not silently continue if the transform is invalid" | **fail** if the mesh bounding box falls outside the MRI world bounding box (margin = 2×spacing) — a sign of non-millimeter units or a broken transform. |
| `mesh` | §13.1 "mesh contains vertices and triangles"; §16 "warn if the mesh is empty, non-manifold or extremely sparse" | **fail** if there are no vertices/faces or face indices reference missing vertices; **warn** if fewer than 100 vertices. |
| `components` | §16 "warn if segmentation produces multiple large components" | **fail** if the segmentation is empty; **warn** if more than one component of ≥ 100 voxels. |
| `holes_over_scalp` | §13.1 "no major holes over the electrode-bearing scalp" | **warn** if a boundary loop other than the largest (assumed neck opening) exceeds 15 mm in diameter. |
| `fiducials_on_surface` | §13.1 / §16 "fiducial lies far from the head surface" | **warn** if a fiducial is farther than 3 mm from the nearest mesh vertex; **warn** if no fiducials are present. |
| `ese_offset` | §13.1 "ESE offset is positive and recorded" | **fail** if the ESE offset is not positive; **skip** when no ESE config is supplied. |
| `nifti_mask` | §14 mask export | **fail** if the mask file is missing, is not a NIfTI image, or its shape/affine/voxel count differ from the in-memory segmentation. |

### Known gaps

- **Non-manifold detection is not implemented.** Spec §16 requires a warning
  when "the mesh is empty, non-manifold or extremely sparse". `check_mesh`
  covers emptiness and sparsity but does **not** detect non-manifold edges
  (edges shared by more than two faces).

## `logs/`

`pipeline.log` — pipeline log (one line per event):

```
21:55:25 | INFO     | virda.stage_3 | Store 'Electrodes' updated.
```

Line format: `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`.
