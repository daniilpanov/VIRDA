"""User-driven export: thin orchestration over the :mod:`virda.io` exporters.

The GUI side resolves the export source from the live
:class:`~virda_gui_new.services.steps_manager.StepsManager` (everything
is in memory), converts coordinates into the requested target frame and
wires the user-chosen path; every file format is written by a
:mod:`virda.io.exporters` writer, never by this module.

Sources:

- ``measurements``: the step-4 context key ``measurements`` (an
  :class:`~virda.models.electrode.Electrodes` model seeded from
  ``measurements.json`` and mirrored there by the localization popup
  after each successful run). An edited-but-not-yet-run popup table is
  NOT exported: the export always writes the last successfully run or
  seeded state.
- ``electrodes``: the step-4 context key ``electrodes``. The export is
  the FULL pipeline CSV written by
  :func:`virda.io.exporters.electrodes.export_electrodes` (ids,
  ESE/scalp coordinates, measured distances, residual, flag); the point
  CSVs (``electrodes_scalp.csv`` / ``electrodes_ese.csv``) remain
  step-run artifacts written by the localization step itself and are not
  the export target.
- ``fiducials``: the manager's global store (``fiducials``,
  :class:`~virda_gui_new.dto.viewer3d.LabeledPoints`).
- ``mesh``: the chosen mesh type's context value (``base`` -> step-1
  ``scalp_surface``, ``postprocessed`` -> step-2 ``mesh``, ``ese`` ->
  step-3 ``ese_mesh``).

Coordinates: the canonical world values are never touched -- conversion
happens on the exported copy only. Fiducials can be exported in the
``head`` or ``world`` frame (world -> head is a centroid translation;
the ``fiducials`` frame is deliberately not offered for fiducials
themselves). Electrodes and meshes accept ``head`` and ``fiducials``
too: the full frame transform of
:mod:`virda_gui_new.services.coordinates` applies. Measured distances
and residuals are frame-independent scalars and stay.

Missing sources raise ``ValueError`` with a user-facing reason; export
popups surface it in the error dialog.
"""

from enum import Enum
from pathlib import Path

import numpy as np

from virda.io.exporters.electrodes import export_electrodes
from virda.io.exporters.ese_mesh import export_ese_mesh
from virda.io.exporters.fiducials import export_fiducials
from virda.io.exporters.measurements import export_measurements
from virda.io.exporters.scalp_mesh import export_scalp_mesh
from virda.models.electrode import Electrode, Electrodes
from virda.models.mesh import Mesh
from virda_gui_new.dto.steps import BASE_MESH, ESE_MESH, LOCALIZATION, MESH_POSTPROCESSING
from virda_gui_new.dto.viewer3d import LabeledPoints
from virda_gui_new.services.coordinates import CoordinateFrame, CoordinateService
from virda_gui_new.services.localization import FiducialRows, build_fiducials
from virda_gui_new.services.steps_manager import StepsManager

__all__ = [
    "ExportKind",
    "MeshExportKind",
    "default_export_path",
    "export_artifact",
]


class ExportKind(Enum):
    """The four user-facing export kinds of the File -> Export menu."""

    MEASUREMENTS = "measurements"
    ELECTRODES = "electrodes"
    FIDUCIALS = "fiducials"
    MESH = "mesh"


class MeshExportKind(Enum):
    """The mesh source a mesh export can choose from."""

    BASE = "base"
    POSTPROCESSED = "postprocessed"
    ESE = "ese"


#: Step-4 context key of the localization step.
_ELECTRODES_KEY = "electrodes"

#: The manager's global store keys of the project inputs.
_FIDUCIALS_STORE_KEY = "fiducials"
_MEASUREMENTS_STORE_KEY = "measurements"

#: Canonical project-relative default paths, one per mesh kind: the same
#: file names :mod:`virda_gui_new.services.artifact_saving` writes.
_MESH_DEFAULT_PATHS = {
    MeshExportKind.BASE: Path("scalp") / "base_mesh.ply",
    MeshExportKind.POSTPROCESSED: Path("scalp") / "final_mesh.ply",
    MeshExportKind.ESE: Path("ese") / "mesh.ply",
}

#: Context locations of the three mesh kinds (step index, context key).
_MESH_CONTEXTS = {
    MeshExportKind.BASE: (BASE_MESH.index, "scalp_surface"),
    MeshExportKind.POSTPROCESSED: (MESH_POSTPROCESSING.index, "mesh"),
    MeshExportKind.ESE: (ESE_MESH.index, "ese_mesh"),
}


def default_export_path(
    project: Path | None,
    kind: ExportKind,
    mesh_kind: MeshExportKind = MeshExportKind.BASE,
) -> Path | None:
    """The canonical default path of an export kind inside ``project``.

    The same file names the artifact savers write:
    ``localization/measurements.json``, ``localization/electrodes.csv``,
    ``localization/fiducials.json`` and, for meshes,
    ``scalp/base_mesh.ply`` / ``scalp/final_mesh.ply`` / ``ese/mesh.ply``
    depending on the mesh kind. ``None`` without an open project (the
    popup then leaves the path field empty).
    """
    if project is None:
        return None
    if kind is ExportKind.MEASUREMENTS:
        return project / "localization" / "measurements.json"
    if kind is ExportKind.ELECTRODES:
        return project / "localization" / "electrodes.csv"
    if kind is ExportKind.FIDUCIALS:
        return project / "localization" / "fiducials.json"
    return project / _MESH_DEFAULT_PATHS[mesh_kind]


def export_artifact(
    manager: StepsManager,
    kind: ExportKind,
    path: Path | str,
    *,
    frame: CoordinateFrame = CoordinateFrame.WORLD,
    mesh_kind: MeshExportKind = MeshExportKind.BASE,
) -> Path:
    """Export one in-memory artifact to ``path`` and return it.

    ``frame`` is the target coordinate frame of the exported numbers
    (ignored by ``measurements``, which carries no coordinates);
    ``mesh_kind`` selects the mesh source for the ``mesh`` kind.
    Missing sources, an unavailable frame or a not-offered frame/target
    combination raise ``ValueError`` with a user-facing reason.
    """
    coordinates = CoordinateService(manager)
    target = Path(path)
    if kind is ExportKind.MEASUREMENTS:
        return _export_measurements(manager, target)
    if kind is ExportKind.ELECTRODES:
        return _export_electrodes(manager, target, frame, coordinates)
    if kind is ExportKind.FIDUCIALS:
        return _export_fiducials(manager, target, frame, coordinates)
    return _export_mesh(manager, target, mesh_kind, frame, coordinates)


def _export_measurements(manager: StepsManager, path: Path) -> Path:
    """Write the stored measurements model through ``export_measurements``."""
    stored = manager.get_global_item(_MEASUREMENTS_STORE_KEY)
    if not isinstance(stored, Electrodes):
        raise ValueError(
            "No measurements to export: run the localization step first "
            "(or open a project with a measurements file)."
        )
    return export_measurements(path, stored)


def _export_electrodes(
    manager: StepsManager,
    path: Path,
    frame: CoordinateFrame,
    coordinates: CoordinateService,
) -> Path:
    """Write the full electrodes CSV for the frame-converted copy."""
    stored = _context_item(manager, LOCALIZATION.index, _ELECTRODES_KEY)
    if not isinstance(stored, Electrodes):
        raise ValueError("No localized electrodes to export: run the localization step first.")
    converted = Electrodes(
        [_convert_electrode(electrode, frame, coordinates) for electrode in stored.items],
        calibrated_offset_shift_mm=stored.calibrated_offset_shift_mm,
    )
    return export_electrodes(path, converted)


def _export_fiducials(
    manager: StepsManager,
    path: Path,
    frame: CoordinateFrame,
    coordinates: CoordinateService,
) -> Path:
    """Write the store's fiducials in the target frame.

    The ``fiducials`` frame is not offered for fiducials themselves;
    head frame export is a plain centroid translation of the store's
    world coordinates.
    """
    if frame is CoordinateFrame.FIDUCIALS:
        raise ValueError(
            "Fiducials cannot be exported in the fiducials frame; choose head or world."
        )
    points = manager.get_global_item(_FIDUCIALS_STORE_KEY)
    if not isinstance(points, LabeledPoints) or points.labels is None or len(points) == 0:
        raise ValueError(
            "No fiducials to export: enter or import fiducials in the localization popup first."
        )
    rows: FiducialRows = [
        (label, coordinates.from_world(xyz, frame))
        for label, xyz in zip(points.labels, points.points, strict=True)
    ]
    return export_fiducials(path, build_fiducials(rows))


def _export_mesh(
    manager: StepsManager,
    path: Path,
    mesh_kind: MeshExportKind,
    frame: CoordinateFrame,
    coordinates: CoordinateService,
) -> Path:
    """Write the chosen mesh type in the target frame.

    ``base``/``postprocessed`` meshes go through ``export_scalp_mesh``
    (plain PLY, like the step savers), the ESE mesh through
    ``export_ese_mesh``; the exported copy keeps the source faces and
    only its vertices are converted.
    """
    step_n, key = _MESH_CONTEXTS[mesh_kind]
    mesh = _context_item(manager, step_n, key)
    if not isinstance(mesh, Mesh):
        raise ValueError(
            f"No {mesh_kind.value} mesh to export: run the pipeline up to that step first."
        )
    vertices = np.asarray(
        coordinates.from_world(np.asarray(mesh.vertices, dtype=np.float64), frame),
        dtype=np.float64,
    )
    converted = Mesh(vertices, mesh.faces)
    writer = export_ese_mesh if mesh_kind is MeshExportKind.ESE else export_scalp_mesh
    return writer(path, converted)


def _convert_electrode(
    electrode: Electrode,
    frame: CoordinateFrame,
    coordinates: CoordinateService,
) -> Electrode:
    """A copy of ``electrode`` with its coordinate arrays in ``frame``.

    Measured distances, the residual and the flag are frame-independent
    scalars and stay; electrodes without a coordinate set keep it unset.
    """
    return Electrode(
        electrode_id=electrode.electrode_id,
        measured_distances=dict(electrode.measured_distances),
        ese_coords=_convert_points(electrode.ese_coords, frame, coordinates),
        scalp_coords=_convert_points(electrode.scalp_coords, frame, coordinates),
        residual_error=electrode.residual_error,
        flagged=electrode.flagged,
    )


def _convert_points(
    points: np.ndarray | None,
    frame: CoordinateFrame,
    coordinates: CoordinateService,
) -> np.ndarray | None:
    """World points -> ``frame`` display coordinates; ``None`` stays ``None``."""
    if points is None:
        return None
    return np.asarray(
        coordinates.from_world(np.asarray(points, dtype=np.float64), frame),
        dtype=np.float64,
    )


def _context_item(manager: StepsManager, step_n: int, key: str) -> object | None:
    """The step context value, or ``None`` when the step is out of range."""
    try:
        return manager.get_step_context_item(step_n, key)
    except ValueError:
        return None
