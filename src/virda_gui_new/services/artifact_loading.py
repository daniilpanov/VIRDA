"""Artifact file loading: step-context file paths -> ready viewer sources.

Electrode point files (``electrodes_scalp.json`` / ``electrodes_ese.json``)
hold a JSON array of point objects ``{"name": str, "x": mm, "y": mm,
"z": mm}``; extra keys are tolerated and rows with missing or null
coordinates are skipped.
"""

import json
from pathlib import Path

import numpy as np

from virda.io.fiducial_helpers import load_fiducials
from virda.io.importers.nifti import import_nifti
from virda.io.importers.scalp_mesh import import_scalp_mesh
from virda.models.mesh import Mesh
from virda.models.mri_volume import MRIVolume
from virda_gui_new.dto.viewer3d import LabeledPoints


def load_volume(path: Path) -> MRIVolume:
    """Read a NIfTI file into an :class:`MRIVolume`."""
    return import_nifti(path)


def load_mesh(path: Path) -> Mesh:
    """Read a PLY/OBJ/STL surface file into a :class:`Mesh`."""
    return import_scalp_mesh(path)


def load_fiducial_points(path: Path) -> LabeledPoints:
    """Read a fiducials JSON file into labeled points."""
    fiducials = load_fiducials(path)
    points = (
        np.stack([fiducial.coordinates for fiducial in fiducials.items])
        if fiducials.items
        else np.empty((0, 3))
    )
    labels = tuple(fiducial.name or fiducial.fiducial_id for fiducial in fiducials.items)
    return LabeledPoints(np.asarray(points, dtype=np.float64), labels)


def load_point_rows(path: Path) -> LabeledPoints:
    """Read an electrode point file into labeled points."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    points: list[tuple[float, float, float]] = []
    labels: list[str] = []
    for row in rows:
        x, y, z = row.get("x"), row.get("y"), row.get("z")
        if x is None or y is None or z is None:
            continue
        points.append((x, y, z))
        labels.append(str(row.get("name", f"electrode {len(labels)}")))
    return LabeledPoints(np.asarray(points, dtype=np.float64), tuple(labels))
