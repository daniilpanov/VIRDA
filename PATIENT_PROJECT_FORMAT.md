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
| `measurements.json`                  | Byte-for-byte copy of the measurements file used by Stage 3; written by the pipeline on every Stage 3 run. This is where the GUI loads the run *Measurements* from. |
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

## `ese/`

Stage 2 output (electrode-skin-entrance surface offset outward from the scalp
along local normals); written only when Stage 2 runs.

| File | Description |
|---|---|
| `ese_mesh.ply` | The ESE surface as a trimesh PLY export. |
| `ese_vertices.npy` | `(N, 3)` `float64` array of ESE vertex coordinates (world mm). |
| `ese_faces.npy` | `(M, 3)` `int64` triangular faces of the ESE mesh. |

Row `i` of every array corresponds to scalp vertex `i` of the Stage 1 scalp
mesh (`mesh/scalp_vertices.npy`, `mesh/scalp_faces.npy`): the two surfaces
share the triangulation, and the scalp has no copies here.
`ese_vertices[i]` = `scalp_vertices[i] + ese_offset_mm * normals[i]`, where
`normals[i]` is the outward scalp normal computed during Stage 2 and kept in
memory only — no normals or scalp copies are written into `ese/`.

## `localization/`

Stage 3 output; written only when both the ESE surface and measurements are
available.

| File | Description |
|---|---|
| `electrodes.csv` | Full localization result: `electrode_id, ese_x, ese_y, ese_z, scalp_x, scalp_y, scalp_z, [one column per fiducial, lowercased], residual_mm, flagged`. Coordinates are scanner RAS millimeters; the fiducial columns carry the measured distances (mm). Empty coordinates and residual mark non-localized electrodes. |
| `electrodes_scalp.csv` | Same electrodes with only `x, y, z` = scalp contact points (scanner RAS). |
| `electrodes_ese.csv` | Same electrodes with only `x, y, z` = ESE body-center points (scanner RAS). |

The coordinate tables (`electrodes_scalp.csv`, `electrodes_ese.csv`) carry columns
`electrode_id, x, y, z, residual_mm, flagged`.  Empty `x/y/z` and
`residual_mm` mark electrodes that could not be localized
(`is_localized == false`).  `flagged` (`yes`/`no`) marks electrodes whose
residual exceeds `residual_threshold_mm`.

## GUI step outputs and scanner rules

The interactive GUI (`virda-gui-new`) persists the artifacts of every step
run it completes, using the layout below; the names differ from the batch
pipeline's Stage 1-3 output where noted.

| Path (GUI writes)                      | Produced by step    | Content |
|----------------------------------------|---------------------|---------|
| `input/<source>.nii.gz`                | NIfTI import        | Byte-for-byte copy of the imported source MRI, so the project stays self-contained. |
| `segmentation/head_mask.nii.gz`        | Base mesh           | uint8 NIfTI head mask with the MRI affine (as above). |
| `scalp/base_mesh.ply`                  | Base mesh           | Raw extracted scalp surface (trimesh PLY). |
| `scalp/final_mesh.ply`                 | Mesh postprocessing | Cleaned, decimated and smoothed mesh (trimesh PLY). |
| `ese/mesh.ply`                         | ESE mesh            | ESE surface (trimesh PLY); normals stay in memory only. |
| `localization/fiducials.json`          | Localization        | Fiducial table in the `input/fiducials.json` format. |
| `localization/measurements.json`       | Localization        | Measured distances in the Stage 3 `measurements.json` format (no `fiducial_weights`). |
| `localization/electrodes_scalp.csv`    | Localization        | Localized scalp contact points, in the pipeline's `electrodes_scalp.csv` format (see `localization/` above). |
| `localization/electrodes_ese.csv`      | Localization        | Localized ESE body-center points, in the pipeline's `electrodes_ese.csv` format. |

A point CSV is written only when at least one electrode carries that
coordinate set; electrodes without it keep empty coordinate cells, and
readers skip such rows.

### Step chain rule

The GUI loads a project as a chain: every step needs the artifacts of
the step before it (`volume` -> `scalp_surface` -> `mesh` -> `ese_mesh`).
The first step whose artifact is missing is where the pipeline stops:
the steps after it are not loaded, and the import dialog says so. A
project therefore needs the source MRI (or its copy in `input/`) to be
usable, and a mesh postprocessing result to reach the ESE step.

Scanner rules (the GUI's single source of truth): rules run in order and
the first match fills its slot; every variant name is a first-class rule
with no canonical-folder preference and no fallbacks. Most names are
searched anywhere in the project (`fiducials.json`, `measurements.json`,
`coordsystem.json`, `pipeline_config.json`, `base_mesh.ply`,
`final_mesh.ply`, `ese_mesh.ply`, `electrodes_scalp.csv`,
`electrodes_ese.csv`, `head_mask.nii.gz`, `*.nii.gz` before `*.nii`);
the head mask is excluded from the NIfTI globs by name; `mesh.ply`
counts as the ESE mesh only under `ese/` and outranks an
`ese_mesh.ply` there. When several files match one rule, the first in
sorted path order wins. Dot-directories are never searched. Precedence
between different artifact kinds: `fiducials.json` wins over
`coordsystem.json` as the fiducial input (a lone `coordsystem.json` is
converted through its MRI fiducial coordinates), and `final_mesh.ply`
wins over `base_mesh.ply` as the scene and postprocessing mesh.

Electrode point files are CSV only: the GUI-invented JSON point-row
files (`electrodes_scalp.json` / `electrodes_ese.json`) are no longer
recognized, so projects carrying only those files do not seed electrode
points (no legacy support).

## `logs/`

`stageN.log` (`stage1.log`, `stage2.log` and `stage3.log`) — pipeline logs:

Line format: `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`,
where `%(name)s` like 'virda.stage<number>'
