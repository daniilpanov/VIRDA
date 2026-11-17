"""Coordinate helpers shared across modules."""

import numpy as np

from virda.models.fiducial import Fiducial


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine ``transform`` to an (N, 3) array of points."""
    return np.asarray(points @ transform[:3, :3].T + transform[:3, 3], dtype=np.float64)


def fiducials_world_coordinates(fiducials: list[Fiducial], affine: np.ndarray) -> np.ndarray:
    """Fiducial coordinates in world space, converting voxel coords when needed."""
    rows: list[np.ndarray] = []
    for fiducial in fiducials:
        coords = np.asarray(fiducial.coordinates, dtype=np.float64)
        if fiducial.coordinate_system == "voxel":
            coords = apply_transform(coords.reshape(1, 3), affine)[0]
        rows.append(coords)
    if not rows:
        return np.zeros((0, 3))
    return np.asarray(rows, dtype=np.float64)


def mesh_voxel_coordinates(vertices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Mesh vertices transformed into voxel (index) space."""
    return apply_transform(vertices, np.linalg.inv(affine))
