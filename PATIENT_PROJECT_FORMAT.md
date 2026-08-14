# Patient Project Format

Layout and file formats of a Stage 1 **patient project** (output directory).
The pipeline writes every artifact into a dedicated subfolder so that Stage 2
can be reproduced exactly from the project alone.

```
<project_dir>/
├── input/              # source MRI NIfTI copy + pipeline configuration
├── segmentation/       # head segmentation mask
├── mesh/               # final scalp mesh, arrays and per-step versions
├── fiducials/          # fiducial table
├── config/             # ESE configuration
├── quality_control/    # automatic QC report
└── logs/               # pipeline log
```

## `input/`

| File | Description |
|---|---|
| `<source_mri>.nii.gz` | Byte-for-byte copy of the source MRI NIfTI. |
| `pipeline_config.json` | Full `VirdaSettings` dump (`model_dump()`); written only when settings are supplied. |

Example `pipeline_config.json` (values follow the `VirdaSettings` defaults):

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
  "ese_reference": null
}
```

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
      "definition_method": "manual"
    }
  ]
}
```

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
    "message": "all fiducials lie on the scalp surface"
  },
  "warnings": []
}
```

- `status` is one of `ok`, `warn`, `fail`; checks with status `skip` are ignored
  when aggregating the overall status (`fail` > `warn` > `ok`).
- Each check entry is `{name, status, message, ...details}`.

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
19:14:33,120 | INFO     | virda.stage1 | ...
```

Line format: `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`.
