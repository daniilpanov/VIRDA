"""Coordinate helpers shared by QC artifact generators."""

import numpy as np

from virda.models.fiducial import Fiducial


def fiducials_world_coordinates(fiducials: list[Fiducial], affine: np.ndarray) -> np.ndarray:
    """Fiducial coordinates in world space, converting voxel coords when needed."""
    rows: list[np.ndarray] = []
    for fiducial in fiducials:
        coords = np.asarray(fiducial.coordinates, dtype=np.float64)
        if fiducial.coordinate_system == "voxel":
            coords = coords @ affine[:3, :3].T + affine[:3, 3]
        rows.append(coords)
    if not rows:
        return np.zeros((0, 3))
    return np.asarray(rows, dtype=np.float64)


def mesh_voxel_coordinates(vertices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Mesh vertices transformed into voxel (index) space."""
    inverse = np.linalg.inv(affine)
    return vertices @ inverse[:3, :3].T + inverse[:3, 3]
