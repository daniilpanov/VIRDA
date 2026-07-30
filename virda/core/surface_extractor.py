"""Triangular surface mesh extraction from binary volumes."""

from __future__ import annotations

import logging

import numpy as np
from skimage.measure import marching_cubes

from .types import MeshData

logger = logging.getLogger(__name__)


def extract_surface(
    mask: np.ndarray,
    voxel_size: np.ndarray,
    affine: np.ndarray | None = None,
    level: float = 0.5,
) -> MeshData:
    """Extract a triangular surface mesh using marching cubes.

    Parameters
    ----------
    mask : np.ndarray
        Binary volume (3D array).
    voxel_size : np.ndarray
        Voxel dimensions in (row, col, slice) order, in mm.
    affine : np.ndarray, optional
        Voxel-to-world transformation matrix. If None, an identity matrix is used.
    level : float
        Isovalue for marching cubes. Default 0.5 for binary masks.

    Returns
    -------
    MeshData
        Triangular mesh with vertices in MRI world coordinates (mm).
    """
    verts, faces, normals, _ = marching_cubes(
        mask.astype(np.float64),
        level=level,
        spacing=tuple(voxel_size),
    )

    if affine is not None:
        verts_h = np.hstack([verts, np.ones((len(verts), 1))])
        verts = (affine @ verts_h.T).T[:, :3]

    adjacency = _build_adjacency(faces, len(verts))

    transform = affine if affine is not None else np.eye(4)

    logger.info(
        "Extracted mesh: %d vertices, %d faces",
        len(verts),
        len(faces),
    )

    return MeshData(
        vertices=verts.astype(np.float64),
        faces=faces.astype(np.int64),
        adjacency=adjacency,
        coordinate_system="MRI_world_mm",
        transform=transform,
    )


def _build_adjacency(faces: np.ndarray, num_vertices: int) -> list[list[int]]:
    """Build vertex-to-vertex adjacency list from face array."""
    adjacency: list[set[int]] = [set() for _ in range(num_vertices)]
    for tri in faces:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        adjacency[i].update([j, k])
        adjacency[j].update([i, k])
        adjacency[k].update([i, j])
    return [sorted(s) for s in adjacency]
