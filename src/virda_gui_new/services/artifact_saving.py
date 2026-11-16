"""Saving produced step artifacts into the project directory.

Each saver mirrors one pipeline step's outputs into the file names the
project scanner knows (:mod:`virda_gui_new.services.project_scan`). The
GUI side only chooses the canonical project paths; every file format is
written by a :mod:`virda.io.exporters` writer. All savers raise on
failure, except :func:`save_volume_copy`: a copy of the source MRI is
a convenience, so it reports a failure by returning ``None``.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from virda.io.exporters.electrodes import export_electrodes_ese, export_electrodes_scalp
from virda.io.exporters.ese_mesh import export_ese_mesh, export_ese_normals
from virda.io.exporters.fiducials import export_fiducials
from virda.io.exporters.measurements import export_measurements
from virda.io.exporters.nifti_copy import export_nifti_copy
from virda.io.exporters.scalp_mesh import export_scalp_mesh
from virda.io.exporters.segmentation import export_head_mask
from virda.models.electrode import Electrodes
from virda.models.mri_volume import MRIVolume
from virda_gui_new.dto.steps import BASE_MESH, ESE_MESH, LOCALIZATION, MESH_POSTPROCESSING, STEPS
from virda_gui_new.services import artifact_paths
from virda_gui_new.services.localization import (
    FiducialRows,
    MeasurementTable,
    build_electrodes,
    build_fiducials,
)
from virda_gui_new.services.project_scan import artifact_files

if TYPE_CHECKING:
    from virda.models.mesh import Mesh, Normals
    from virda.models.scalp_surface import ScalpSurface


def save_volume_copy(project: Path, source: Path) -> Path | None:
    """Copy the source NIfTI into ``project/input/`` under its own name.

    The copy keeps the project self-contained: the scanner finds it on
    the next open, so step 0 (NIfTI import) is restored and the pipeline
    can start where it stopped. A copy failure is not fatal -- the import
    itself already succeeded -- so the written path is returned, or
    ``None`` when the copy failed.
    """
    return export_nifti_copy(project / artifact_paths.INPUT_DIR / source.name, source)


def save_scalp_surface(project: Path, volume: MRIVolume, scalp_surface: ScalpSurface) -> None:
    """Write the step-1 outputs: the segmentation mask and the base mesh.

    The mask is saved as ``segmentation/head_mask.nii.gz`` (uint8 NIfTI
    with the MRI affine, so voxels map 1:1 onto the source volume), the
    mesh as ``scalp/base_mesh.ply``.
    """
    export_head_mask(artifact_paths.head_mask(project), scalp_surface.mask, volume)
    export_scalp_mesh(artifact_paths.base_mesh(project), scalp_surface.mesh)


def save_final_mesh(project: Path, mesh: Mesh) -> None:
    """Write the step-2 output: ``scalp/final_mesh.ply``."""
    export_scalp_mesh(artifact_paths.final_mesh(project), mesh)


def save_ese_mesh(project: Path, mesh: Mesh, normals: Normals) -> None:
    """Write the step-3 output: ``ese/mesh.ply`` and ``ese/normals.npy``.

    The per-vertex normals are the byproduct of the step, and the
    localization needs them: they are stored next to the surface, so a
    reopened project does not have to estimate them again.
    """
    export_ese_mesh(artifact_paths.ese_mesh(project), mesh)
    export_ese_normals(artifact_paths.ese_normals(project), normals)


def save_localization_inputs(
    project: Path, fiducial_rows: FiducialRows, table: MeasurementTable
) -> None:
    """Write the project inputs of the localization step.

    The tables are the inputs of the step, so they are stored even when
    the localization cannot run yet (no fiducials, no measurements): the
    project then keeps what the user entered or imported. An empty
    measurement table removes its file: the project then holds no
    measurements at all, which is what an emptied table means. An empty
    fiducial table keeps its file, so a half-filled project holds the
    other table.
    """
    if fiducial_rows:
        export_fiducials(artifact_paths.fiducials(project), build_fiducials(fiducial_rows))
    if table:
        export_measurements(artifact_paths.measurements(project), build_electrodes(table))
    else:
        artifact_paths.measurements(project).unlink(missing_ok=True)


def save_localization(
    project: Path,
    fiducial_rows: FiducialRows,
    measurements: MeasurementTable,
    electrodes: Electrodes,
) -> None:
    """Write the step-4 outputs into ``localization/``.

    ``fiducials.json`` holds the current fiducial table rows and
    ``measurements.json`` the measured distances. The localized electrode
    coordinates become the point CSVs the scanner reads back: scalp
    contact points to ``electrodes_scalp.csv`` and ESE body-center points
    to ``electrodes_ese.csv``. A point CSV is written only when at least
    one electrode carries that coordinate set.
    """
    if not fiducial_rows:
        raise ValueError("no fiducial rows; refusing to write an empty fiducials.json")
    export_fiducials(artifact_paths.fiducials(project), build_fiducials(fiducial_rows))
    export_measurements(artifact_paths.measurements(project), build_electrodes(measurements))
    if any(electrode.scalp_coords is not None for electrode in electrodes.items):
        export_electrodes_scalp(artifact_paths.scalp_points(project), electrodes)
    if any(electrode.ese_coords is not None for electrode in electrodes.items):
        export_electrodes_ese(artifact_paths.ese_points(project), electrodes)


def remove_artifacts_after_step(project: Path, step_n: int) -> tuple[Path, ...]:
    """Remove persisted outputs of steps later than ``step_n``.

    Fiducials and measurements are global-store inputs, so their files
    are retained. Removing downstream context artifacts prevents the
    next project load from restoring results invalidated in memory.
    """
    if not 0 <= step_n < len(STEPS):
        raise ValueError(f"step index {step_n} is out of range")

    outputs = {
        BASE_MESH.index: ("head_mask", "base_mesh"),
        MESH_POSTPROCESSING.index: ("scalp_mesh",),
        ESE_MESH.index: ("ese_mesh", "ese_normals"),
        LOCALIZATION.index: ("scalp_points", "ese_points"),
    }
    removed: list[Path] = []
    for owner_step, slots in outputs.items():
        if owner_step <= step_n:
            continue
        for slot in slots:
            for path in artifact_files(project, slot):
                path.unlink(missing_ok=True)
                removed.append(path)
    return tuple(removed)
