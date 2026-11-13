"""Artifact file loading: step-context file paths -> ready viewer sources.

Electrode point files (``electrodes_scalp.csv`` / ``electrodes_ese.csv``)
are the pipeline's coords CSV tables (``electrode_id, x, y, z,
residual_mm, flagged``); they are read through the virda point importer,
which skips the rows whose coordinate cells are empty.
"""

from pathlib import Path

import numpy as np

from virda.io.fiducial_helpers import load_fiducials
from virda.io.importers.electrode_points import import_electrode_points
from virda.io.importers.measurements import import_measurements
from virda.io.importers.nifti import import_nifti
from virda.io.importers.scalp_mesh import import_scalp_mesh
from virda.models.electrode import Electrodes
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
    """Read a fiducials JSON file into labeled points.

    Also accepts an MNE-style ``coordsystem.json``: its MRI fiducial
    coordinates are converted to millimetres through
    :meth:`~virda.models.coordsystem.Coordsystem.to_fiducials`.
    """
    fiducials = load_fiducials(path)
    if not fiducials.items:
        raise ValueError("no fiducials found in file")
    points = np.stack([fiducial.coordinates for fiducial in fiducials.items])
    labels = tuple(fiducial.name or fiducial.fiducial_id for fiducial in fiducials.items)
    return LabeledPoints(np.asarray(points, dtype=np.float64), labels)


def load_measurements(path: Path) -> Electrodes:
    """Read a measurements JSON file into an :class:`Electrodes` model."""
    return import_measurements(path)


def load_electrode_points(path: Path) -> LabeledPoints:
    """Read an electrode point CSV (pipeline format) into labeled points."""
    rows = import_electrode_points(path)
    points = np.asarray([row.coordinates for row in rows], dtype=np.float64)
    labels = tuple(row.electrode_id for row in rows)
    return LabeledPoints(points, labels)
