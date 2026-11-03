# Patient Project Format

Layout and file formats of a **patient project** directory written by the VIRDA
pipeline. The pipeline writes every artifact into a dedicated subfolder so that
any stage can be reproduced exactly from the project alone:

- **Stage 1** (segmentation/mesh/fiducials) always runs;
- **Stage 2** (`ese/`) runs when an ESE config is supplied;
- **Stage 3** (`localization/`) runs when both ESE and measurements are available.

```
<project_dir>/
├── input/              # canonical input copies: MRI, config, measurements, fiducials
├── segmentation/       # head segmentation mask
├── mesh/               # final scalp mesh and arrays
├── ese/                # Stage 2 output: electrode-skin-entrance surface
├── localization/       # Stage 3 output: localized electrodes
└── logs/               # pipeline log
```

## `input/`

Canonical input files live here. `config.json`/`pipeline_config.json` and `measurements.json` are the
locations the GUI prefills the *Run Pipeline* form from and role-imports into.
`pipeline_config.json` is written by the pipeline on every run.

| File                                 | Description                                                                  |
|--------------------------------------|------------------------------------------------------------------------------|
| `<source_mri>.nii.gz`                | Byte-for-byte copy of the source MRI NIfTI.                                  |
| `config.json`/`pipeline_config.json` | Optional user-supplied pipeline configuration (a partial `Config` JSON). This is where the GUI loads the run *Config file* from. |
| `measurements.json`                  | Optional Stage 3 measurements input (see format below). This is where the GUI loads the run *Measurements* from. |
| `fiducials.json`                     | List of fiducial coordinates; the GUI *Fiducials* input.                     |
| `pipeline_config.json`               | Full merged `Config` dump (`model_dump(mode="json")`); written on every run. |

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
  "ese_offset_mm": null,
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

Example `fiducials.json` — fiducial table:

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

When an MNE ``coordsystem.json`` was loaded as an input config file, its parsed
contents are embedded here under `"coordsystem"` (fiducial positions, electrode
count / offset / reference) instead of `null`.

Example `measurements.json` (the Stage 3 input):

```json
{
  "electrodes": [
    {"electrode_id": "Fz", "measured_distances": {"NAS": 120.5, "LPA": 131.2, "RPA": 142.8}}
  ],
  "fiducial_weights": {"NAS": 1.5}
}
```

- `electrode_id` is optional; sequential ``E001`` ids are assigned when absent.
- `fiducial_weights` is optional; when present it overrides the weights of the
  fiducials store for the localization solver.

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

## `ese/`

Stage 2 output (electrode-skin-entrance surface offset outward from the scalp
along local normals); written only when Stage 2 runs.

| File | Description |
|---|---|
| `ese_mesh.ply` | The ESE surface as a trimesh PLY export. |
| `ese_vertices.npy` | `(N, 3)` `float64` array of ESE vertex coordinates (world mm). |
| `ese_faces.npy` | `(M, 3)` `int64` triangular faces of the ESE mesh. |
| `normals.npy` | `(N, 3)` unit normal at each scalp vertex used for the offset. |

The arrays are row-aligned: `ese_vertices[i]` and `normals[i]` belong to `scalp_vertices[i]`.

## `localization/`

Stage 3 output; written only when both the ESE surface and measurements are
available.

| File | Description |
|---|---|
| `electrodes.json` | Full localization result for every electrode (see below). |
| `electrodes_scalp.json` | Same electrodes with only `coords` = scalp contact points (scanner RAS). |
| `electrodes_ese.json` | Same electrodes with only `coords` = ESE body-center points (scanner RAS). |
| `electrode_coords.csv` | Tabular copy: `electrode_id, x, y, z, residual_error, confidence, flagged`; `x/y/z` are the ESE coordinates. Empty `residual_error`/`confidence` mark non-localized electrodes. |

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

`calibrated_ese_offset_shift_mm` is present only when
`calibrate_ese_offset` was enabled (`null` otherwise).

## `logs/`

`stageN.log` (`stage1.log`, `stage2.log` and `stage3.log`) — pipeline logs:

Line format: `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`,
where `%(name)s` like 'virda.stage<number>'
