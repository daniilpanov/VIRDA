"""Saving produced step artifacts into the project directory.

Each saver mirrors one pipeline step's outputs into the file names the
project scanner knows (:mod:`virda_gui_new.services.project_scan`). The
GUI side is a thin orchestration only: it chooses the canonical project
paths, while every file format is written by a :mod:`virda.io.exporters`
writer. All savers raise on failure; popups surface the error through
their hint label and ``failed`` signal.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from virda.io.exporters.electrodes import export_electrodes_ese, export_electrodes_scalp
from virda.io.exporters.ese_mesh import export_ese_mesh
from virda.io.exporters.fiducials import export_fiducials
from virda.io.exporters.measurements import export_measurements
from virda.io.exporters.scalp_mesh import export_scalp_mesh
from virda.io.exporters.segmentation import export_head_mask
from virda.models.electrode import Electrodes
from virda.models.mri_volume import MRIVolume
from virda_gui_new.services.localization import (
    FiducialRows,
    MeasurementTable,
    build_electrodes,
    build_fiducials,
)

if TYPE_CHECKING:
    from virda.models.mesh import Mesh
    from virda_gui_new.api.ops import ScalpSurface

#: Project subdirectory that holds the localization step's outputs.
_LOCALIZATION_DIR = "localization"


def save_scalp_surface(project: Path, volume: MRIVolume, scalp_surface: ScalpSurface) -> None:
    """Write the step-1 outputs: the segmentation mask and the base mesh.

    The mask is saved as ``segmentation/head_mask.nii.gz`` (uint8 NIfTI
    with the MRI affine, so voxels map 1:1 onto the source volume), the
    mesh as ``scalp/base_mesh.ply``.
    """
    export_head_mask(project / "segmentation" / "head_mask.nii.gz", scalp_surface.mask, volume)
    export_scalp_mesh(project / "scalp" / "base_mesh.ply", scalp_surface.mesh)


def save_final_mesh(project: Path, mesh: Mesh) -> None:
    """Write the step-2 output: ``scalp/final_mesh.ply``."""
    export_scalp_mesh(project / "scalp" / "final_mesh.ply", mesh)


def save_ese_mesh(project: Path, mesh: Mesh) -> None:
    """Write the step-3 output: ``ese/mesh.ply``.

    The per-vertex normals stay in-memory only, matching the seeding
    contract.
    """
    export_ese_mesh(project / "ese" / "mesh.ply", mesh)


def save_localization(
    project: Path,
    fiducial_rows: FiducialRows,
    measurements: MeasurementTable,
    electrodes: Electrodes,
) -> None:
    """Write the step-4 outputs into ``localization/``.

    ``fiducials.json`` holds the current fiducial table rows, so the
    saved file matches what the popup shows and what the global store
    holds. ``measurements.json`` follows the project measurements format
    (electrode ids with their measured distances), built from the popup's
    measurement table. The localized electrode coordinates become the
    point CSVs the scanner reads back, in the batch pipeline's format:
    scalp contact points to ``electrodes_scalp.csv`` and ESE body-center
    points to ``electrodes_ese.csv``. A point CSV is written only when at
    least one electrode carries that coordinate set (matching the
    previous behavior of not writing the file at all); electrodes
    without that coordinate set keep empty coordinate cells in the rows,
    and the CSV reader skips such rows on load.
    """
    if not fiducial_rows:
        raise ValueError("no fiducial rows; refusing to write an empty fiducials.json")
    localization = project / _LOCALIZATION_DIR
    export_fiducials(localization / "fiducials.json", build_fiducials(fiducial_rows))
    export_measurements(localization / "measurements.json", build_electrodes(measurements))
    if any(electrode.scalp_coords is not None for electrode in electrodes.items):
        export_electrodes_scalp(localization / "electrodes_scalp.csv", electrodes)
    if any(electrode.ese_coords is not None for electrode in electrodes.items):
        export_electrodes_ese(localization / "electrodes_ese.csv", electrodes)
