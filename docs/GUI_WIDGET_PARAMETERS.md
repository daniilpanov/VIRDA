# GUI widget parameters

Reference for every option a user can set in the interactive GUI
(`virda-gui-new`). All labels are shown in English. Numbers are stored in the
project as they are; the coordinate-frame selectors only change how they are
displayed or exported (see [Coordinate systems](COORDINATE_SYSTEMS.md)).

Step settings are persisted in `config/<step>.json` and restored when the
project is reopened:

| Step | Config file |
|---|---|
| Base mesh | `config/base_mesh.json` |
| Mesh postprocessing | `config/mesh_postprocessing.json` |
| ESE mesh | `config/ese_mesh.json` |
| Localization | `config/localization.json` |

## Project start and navigation

### Startup dialog

Title: `VIRDA: Electrode Localization System`.

| Control | Type | Behavior |
|---|---|---|
| `Open project...` | button | Opens the `Open project` folder dialog. |
| `Create project...` | button | Opens `Create new project`; an existing project in the folder asks `Project already exists` / `Open it?`. |
| `Recent projects` | list | Shown when recent projects exist; first row preselected. |
| `Open selected` | button | Opens the highlighted recent project; stale paths are removed with a `Project missing` warning. |
| `Cancel` | button | Closes the dialog. |

Recent paths are kept by a `QSettings` service (last project plus up to eight
recent projects). There is no preferences dialog with editable fields.

### Main window

- Central 3D scene with a **Layers** panel (see below) and a `Reset view` button.
- Separate `Qt.Tool` windows: `Steps`, `File tree`, the four step popups and the
  export popups. Floating pills `Steps` and `Files` toggle the two navigators.
- `&File` menu: `&Open project...`, `&New project...`, `&Recent` and `&Export`
  with `&Measurements`, `&Electrodes`, `&Fiducials`, `&Mesh` and `Mesh &arrays`.
  Export actions are always enabled; a missing source is reported when `Export`
  is clicked.

### Steps navigator

One row per step: `0. NIfTI import`, `1. Base mesh`, `2. Mesh postprocessing`,
`3. ESE mesh`, `4. Localization`. Rows are suffixed with `[done]`, `[current]` or
`[needs: ...]`; later-than-current rows are disabled. Clicking a step that was
already run asks `Re-run <step>?` because it discards later results. Clicking
step 0 opens `Import NIfTI...` with the filter `NIfTI files (*.nii.gz *.nii)`.

### 3D viewer — Layers

| Control | Type | Default | Meaning |
|---|---|---|---|
| `Show NIfTI` | checkbox | off | Show the imported volume. |
| `Show fiducials` | checkbox | on | Show NAS/LPA/RPA points. |
| `Show electrodes` | checkbox | on | Show localized electrode points. |
| `Show labels` | checkbox | on | Show point labels. |
| `Show mesh` | checkbox | on | Show the selected surface. |
| `Show mesh edges` | checkbox | off | Wireframe overlay. |
| `Mesh opacity` | slider, 0–100 | 60 | Surface opacity in percent. |
| `Surface` | radio buttons | `Scalp` | Choose `Scalp` or `ESE`; shown when both meshes exist. |

During fiducial picking a hint reads `Picking: <slot> - click the scalp`;
`Escape` cancels it.

## Common step-popup controls

Every step popup (1–4) shows the step name, a gray status hint (`Running...`,
`Done.`) and a `Run` button, which is hidden in the localization popup.
The first three steps also have a collapsible `Advanced settings` group,
unchecked by default; its contents are hidden until it is checked.

## Step 1 — Base mesh

Requires the step-0 volume; without it the hint `Step 0 has no NIfTI yet.` is
shown and `Run` is disabled.

| Label | Type | Config key | Default | Range / behavior |
|---|---|---|---|---|
| `Voxel size (mm)` | double spinbox | `voxel_size_mm` | `0.0` = `native` | 0.0–10.0, step 0.1; the special value reads `native` and uses the volume's spacing. |
| `Seal mask` | checkbox (Advanced) | `seal_enabled` | on | Fills internal cavities before surface extraction. |
| `Seal radius` | spinbox (Advanced) | `seal_radius` | `4` | 0–50; enabled only while `Seal mask` is checked. |

## Step 2 — Mesh postprocessing

Requires an in-memory mesh; otherwise `No in-memory mesh yet.` is shown and
`Run` is disabled. Running applies cleaning, decimation and smoothing in that
order.

| Label | Type | Config key | Default | Range / behavior |
|---|---|---|---|---|
| `Density percent` | double spinbox | `density_percent` | `100.0` | 0.1–100.0, step 1.0. |
| `Min component vertices` | spinbox (Advanced, `Clean`) | `min_component_vertices` | `100` | 1–100000; smaller components are removed. |
| `Merge digits` | spinbox (Advanced, `Clean`) | `merge_digits` | `7` | 1–12; decimal digits used when merging close vertices. |
| `Smoother` | combobox (Advanced, `Smooth`) | `smoother` | `taubin` | `none`, `laplacian`, `taubin`. |
| `Iterations` | spinbox (Advanced, `Smooth`) | `iterations` | `5` | 0–50; disabled when `Smoother` is `none`. |
| `Lambda` | double spinbox (Advanced, `Smooth`) | `lamb` | `0.5` | 0.05–5.0, 3 decimals; disabled when `Smoother` is `none`. |
| `Nu` | double spinbox (Advanced, `Smooth`) | `nu` | `-0.53` | -5.0–5.0, 3 decimals; disabled when `Smoother` is `none`. |

## Step 3 — ESE mesh

Requires an in-memory mesh; otherwise `No in-memory mesh yet.` is shown and
`Run` is disabled.

| Label | Type | Config key | Default | Range / behavior |
|---|---|---|---|---|
| `ESE offset (mm)` | double spinbox | `ese_offset_mm` | `1.0` | 0.0–10.0, step 0.5; distance from the scalp to the ESE. |
| `Neighborhood radius (mm)` | double spinbox (Advanced) | `neighborhood_radius_mm` | `10.0` | 0.1–100.0, step 0.5; radius of the local PCA neighborhood. |
| `Auto K neighbors` | checkbox (Advanced) | `use_auto_k` | on | When on, `K neighbors` is disabled and the neighbor count is derived from the radius. |
| `K neighbors` | spinbox (Advanced) | `k_neighbors` | `10` | 3–100; enabled only when `Auto K neighbors` is off. |
| `Weighted PCA` | checkbox (Advanced) | `use_weighted_pca` | off | Weight neighbors by distance in the PCA. |
| `PCA sigma (mm)` | double spinbox (Advanced) | `pca_sigma_mm` | `5.0` | 0.1–50.0, step 0.5; width of the weighting kernel. |
| `Min neighbors` | spinbox (Advanced) | `min_neighbors` | `5` | 1–50; minimum neighbors for a valid PCA estimate. |

## Step 4 — Localization

### Options

| Label | Type | Config key | Default | Range / behavior |
|---|---|---|---|---|
| `Calibrate ESE offset` | checkbox | `calibrate_ese_offset` | on | Shift the ESE surface along scalp normals to fit the measured distances. |
| *(unnamed double spinbox)* | double spinbox | `residual_threshold_mm` | `10.0` | 0.1–100.0, 2 decimals; localizations with a larger residual are flagged. |
| `Fiducial tolerance (mm)` | double spinbox | `fiducial_tolerance_mm` | `2.0` | 0.0–50.0, 2 decimals; also marks fiducials off the scalp with a red dot. |
| `Re-localization` | button | — | — | Recomputes all electrodes; disabled while a run is in progress. |

Edits are debounced (400 ms) and trigger a live localization. The status hint
reports when inputs are missing, e.g. `not enough fiducials (need 3)`,
`N electrode(s) need all three measured distances` or `no measurements`.

### Tables and frame selectors

| Section | Control | Meaning |
|---|---|---|
| `Fiducials` | frame selector | `Head`, `World`; `Fiducials` and `Voxel` are not offered here. |
| `Measurements` | — | Distances from each electrode to the three fiducials; no coordinates. |
| `Electrodes` | frame selector | `Head`, `World`, `Fiducials`, `Voxel (i, j, k)`; start on `Head`. |

The frame selectors show numbers only: stored values stay in world coordinates.
`Head` needs a scalp mesh, `Fiducials` needs all three fiducials and `Voxel`
needs the imported NIfTI; unavailable items are greyed out and an invalid
selection falls back to `Head` (or `World` without a scalp mesh). `Voxel` shows
integer grid indices, the other frames millimetres.

**Fiducials table** — fixed rows `NAS`, `LPA`, `RPA` with columns `""`, `Name`,
`X`, `Y`, `Z`, `Pick`. `Name` is read-only, the coordinate cells are editable.
Buttons: `Pick` (choose the point on the scalp), `From file` (import a JSON
fiducials/coordsystem file) and `Empty` (clear all rows). A red `●` marks a
fiducial farther than `Fiducial tolerance (mm)` from the surface, with the
distance in its tooltip.

**Measurements table** — columns `Electrode`, `NAS`, `LPA`, `RPA`, `Predict`.
Buttons: `From file` (import a measurements JSON), `+ Add`, `- Remove selected`
and a per-row `Predict`. `Predict` is enabled only when exactly one distance is
missing and the ESE mesh plus three fiducials are available; predicted values
appear in italics with a residual note in the tooltip. A missing third distance
can also be predicted after a file import.

**Electrodes table** — read-only columns `Electrode`, `ESE X/Y/Z`,
`Scalp X/Y/Z`, `Residual (mm)`, `Flagged`. Coordinates use the selected frame
and two decimals (integer indices for `Voxel`); `Flagged` reads `yes`/`no`.

## Export popups

Every popup has a `Coordinate frame` selector (except measurements), an output
path field, a `Browse` button and an `Export` button. The mesh arrays popup
selects an output directory; the other popups select a file. A successful
export closes the popup and shows `Export successful`; a failure shows
`Export error` with `Close` and `Retry`.

| Popup | Frames offered | Default output |
|---|---|---|
| `Export measurements` | none (no coordinates) | `localization/measurements.json` |
| `Export electrodes` | Head, World, Fiducials, Voxel | `localization/electrodes.csv` |
| `Export fiducials` | World, Head, Voxel | `localization/fiducials.json` |
| `Export mesh` | World, Head, Fiducials | `scalp/base_mesh.ply`, `scalp/final_mesh.ply`, `ese/mesh.ply` |
| `Export mesh arrays` | World, Head, Fiducials, Voxel | `exports/scalp` or `exports/ese` |

The mesh popup adds a `Mesh type` combobox with `Base`, `Postprocessed` and
`ESE`; changing it resets the path to that source's default. The voxel frame is
not offered for meshes because PLY carries no coordinate-frame metadata.
The mesh arrays popup offers the same mesh types and writes `<scalp|ese>_vertices.npy`
and `<scalp|ese>_faces.npy` into the selected directory. Vertices are converted
from the stored world coordinates into the selected frame; face indices do not
change. The arrays do not include a coordinate-frame metadata file.
The fiducials frame is never offered for fiducials themselves, which define it.

The electrodes CSV tags each row with `coordinate_system`, and the fiducials
JSON tags each point with `coordinate_system`, so their numbers are
unambiguous. Exporting without an open project leaves the path field empty.

## Confirmation and warning dialogs

| Dialog | Buttons | When |
|---|---|---|
| `Partial import` | OK / Cancel | Some project artifacts failed to load or the step chain is incomplete. |
| `Re-run <step>?` | Yes / No | Re-running a step would discard later results. |
| `Project already exists` | Yes / Cancel | Creating a project in a folder that already holds one. |
| `Project missing` | OK | A recent project path no longer exists. |
| `Prediction` | OK | One or more measurements could not be predicted. |
| `Recalculate?` | Yes / No | Predicted measurements became stale after a fiducial edit. |
| `Export successful` / `Export error` | Close / (Retry) | After an export finishes or fails. |
