"""Export the in-memory artifacts of a project to files.

The source is read from the steps manager, the coordinates are converted
into the requested target frame and the file is written by the matching
``virda.io.exporters`` writer. ``measurements`` and ``electrodes`` come
from the localization step's context, ``fiducials`` from the global
store and the meshes from their own step; a missing source raises
``ValueError``.
"""

from enum import Enum
from pathlib import Path

import numpy as np

from virda.geometry import Geometry
from virda.io.exporters.electrodes import export_electrodes
from virda.io.exporters.ese_mesh import export_ese_mesh
from virda.io.exporters.fiducials import export_fiducials
from virda.io.exporters.measurements import export_measurements
from virda.io.exporters.scalp_mesh import export_scalp_mesh
from virda.models.coordinates import CoordinateSystem
from virda.models.electrode import Electrode, Electrodes
from virda.models.mesh import Mesh, Normals
from virda.models.stage3_config import Stage3Config
from virda.ops.localize_electrodes import localize_electrodes
from virda_gui_new.dto.steps import (
    BASE_MESH,
    ELECTRODES_KEY,
    ESE_MESH,
    FIDUCIALS_KEY,
    LOCALIZATION,
    MEASUREMENTS_KEY,
    MESH_POSTPROCESSING,
)
from virda_gui_new.dto.viewer3d import LabeledPoints
from virda_gui_new.services import artifact_paths
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

#: The manager's global store keys of the project inputs.

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
        return artifact_paths.measurements(project)
    if kind is ExportKind.ELECTRODES:
        return artifact_paths.electrodes(project)
    if kind is ExportKind.FIDUCIALS:
        return artifact_paths.fiducials(project)
    if mesh_kind is MeshExportKind.BASE:
        return artifact_paths.base_mesh(project)
    if mesh_kind is MeshExportKind.ESE:
        return artifact_paths.ese_mesh(project)
    return artifact_paths.final_mesh(project)


def export_artifact(
    manager: StepsManager,
    kind: ExportKind,
    path: Path | str,
    *,
    frame: CoordinateSystem = CoordinateSystem.WORLD,
    mesh_kind: MeshExportKind = MeshExportKind.BASE,
    options: Stage3Config | None = None,
) -> Path:
    """Export one in-memory artifact to ``path`` and return it.

    ``frame`` is the target coordinate frame of the exported numbers
    (ignored by ``measurements``, which carries no coordinates);
    ``mesh_kind`` selects the mesh source for the ``mesh`` kind;
    ``options`` are the localization settings the electrodes are
    localized with when the project holds no results yet.
    Missing sources, an unavailable frame or a not-offered frame/target
    combination raise ``ValueError`` with a user-facing reason.
    """
    coordinates = manager.geometry
    target = Path(path)
    if kind is ExportKind.MEASUREMENTS:
        return _export_measurements(manager, target)
    if kind is ExportKind.ELECTRODES:
        return _export_electrodes(manager, target, frame, coordinates, options)
    if kind is ExportKind.FIDUCIALS:
        return _export_fiducials(manager, target, frame, coordinates)
    return _export_mesh(manager, target, mesh_kind, frame, coordinates)


def _export_measurements(manager: StepsManager, path: Path) -> Path:
    """Write the stored measurements model through ``export_measurements``."""
    stored = manager.get_global_item(MEASUREMENTS_KEY)
    if not isinstance(stored, Electrodes):
        raise ValueError(
            "No measurements to export: run the localization step first "
            "(or open a project with a measurements file)."
        )
    return export_measurements(path, stored)


def _export_electrodes(
    manager: StepsManager,
    path: Path,
    frame: CoordinateSystem,
    coordinates: Geometry,
    options: Stage3Config | None,
) -> Path:
    """Write the full electrodes CSV for the frame-converted copy.

    The results live in the step-4 context. A project that holds the
    measured distances but no results yet gets them computed here, the
    way the localization step computes them, so the export writes the
    numbers of the current state.
    """
    stored = _context_item(manager, LOCALIZATION.index, ELECTRODES_KEY)
    if not isinstance(stored, Electrodes):
        stored = _compute_results(manager, options)
    if not isinstance(stored, Electrodes):
        raise ValueError("No localized electrodes to export: run the localization step first.")
    converted = Electrodes(
        [_convert_electrode(electrode, frame, coordinates) for electrode in stored.items],
        calibrated_offset_shift_mm=stored.calibrated_offset_shift_mm,
    )
    return export_electrodes(path, converted)


def _compute_results(manager: StepsManager, options: Stage3Config | None) -> Electrodes | None:
    """The localized electrodes of the current state.

    The measured distances of the project and the geometry of the steps
    are enough: the electrodes are localized the way the step does it,
    with the options the step holds. ``None`` when the project holds
    neither.
    """
    ese = _context_item(manager, ESE_MESH.index, "ese_mesh")
    normals = _context_item(manager, ESE_MESH.index, "normals")
    scalp = manager.active_scalp_mesh()
    points = manager.get_global_item(FIDUCIALS_KEY)
    measurements = manager.get_global_item(MEASUREMENTS_KEY)
    if not isinstance(ese, Mesh) or not isinstance(normals, Normals) or not isinstance(scalp, Mesh):
        return None
    if not isinstance(points, LabeledPoints) or not isinstance(measurements, Electrodes):
        return None
    try:
        fiducials = build_fiducials(_fiducial_rows(points), CoordinateSystem.WORLD)
    except ValueError:
        return None
    if options is None:
        options = Stage3Config()
    return localize_electrodes(
        ese,
        normals,
        scalp,
        fiducials,
        measurements,
        config=options,
    )


def _fiducial_rows(points: LabeledPoints) -> FiducialRows:
    """The labeled points of the store as fiducial table rows."""
    labels = points.labels or ()
    return [
        (label, np.asarray(xyz, dtype=np.float64))
        for label, xyz in zip(labels, points.points, strict=True)
    ]


def _export_fiducials(
    manager: StepsManager,
    path: Path,
    frame: CoordinateSystem,
    coordinates: Geometry,
) -> Path:
    """Write the store's fiducials in the target frame.

    The ``fiducials`` frame is not offered for fiducials themselves;
    head frame export is a plain centroid translation of the store's
    world coordinates. The written ``coordinate_system`` tag always
    matches the numbers that were written: a head-frame export is tagged
    ``head`` and is meant for external tools, not for re-import as a
    project fiducials file (those accept only world and voxel).
    """
    if frame is CoordinateSystem.FIDUCIALS:
        raise ValueError(
            "Fiducials cannot be exported in the fiducials frame; choose head or world."
        )
    points = manager.get_global_item(FIDUCIALS_KEY)
    if not isinstance(points, LabeledPoints) or points.labels is None or len(points) == 0:
        raise ValueError(
            "No fiducials to export: enter or import fiducials in the localization popup first."
        )
    rows: FiducialRows = [
        (label, coordinates.from_world(xyz, frame))
        for label, xyz in zip(points.labels, points.points, strict=True)
    ]
    return export_fiducials(path, build_fiducials(rows, frame))


def _export_mesh(
    manager: StepsManager,
    path: Path,
    mesh_kind: MeshExportKind,
    frame: CoordinateSystem,
    coordinates: Geometry,
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
    frame: CoordinateSystem,
    coordinates: Geometry,
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
    frame: CoordinateSystem,
    coordinates: Geometry,
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
