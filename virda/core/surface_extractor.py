"""Surface extraction module — marching cubes from binary segmentation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MeshData:
    """Triangular mesh data container."""

    vertices: np.ndarray
    faces: np.ndarray
    vertex_normals: Optional[np.ndarray] = None
    face_normals: Optional[np.ndarray] = None
    adjacency: Optional[dict[int, list[int]]] = None

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    def get_adjacency(self) -> dict[int, list[int]]:
        """Compute vertex adjacency if not already cached."""
        if self.adjacency is not None:
            return self.adjacency
        self.adjacency = _compute_adjacency(self.faces, self.num_vertices)
        return self.adjacency

    def compute_vertex_normals(self) -> np.ndarray:
        """Compute per-vertex normals as averaged face normals."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        face_normals /= norms

        vertex_normals = np.zeros_like(self.vertices)
        for i in range(3):
            np.add.at(vertex_normals, self.faces[:, i], face_normals)

        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vertex_normals /= norms

        self.vertex_normals = vertex_normals
        self.face_normals = face_normals
        return vertex_normals


def _compute_adjacency(faces: np.ndarray, num_vertices: int) -> dict[int, list[int]]:
    """Build vertex adjacency list from face array."""
    adj = {i: set() for i in range(num_vertices)}
    for face in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adj[face[i]].add(face[j])
    return {k: sorted(v) for k, v in adj.items()}


def extract_surface(
    mask: np.ndarray,
    voxel_size: np.ndarray,
    affine: Optional[np.ndarray] = None,
    step: int = 1,
    method: str = "marching_cubes",
) -> MeshData:
    """Extract triangular surface mesh from binary volume.

    Parameters
    ----------
    mask : np.ndarray
        Binary segmentation mask (3D).
    voxel_size : np.ndarray
        Voxel dimensions in mm.
    affine : np.ndarray, optional
        Voxel-to-world transformation matrix. If provided, vertices are
        transformed to world coordinates.
    step : int
        Step size for marching cubes (1 = full resolution).
    method : str
        Surface extraction method. Currently only 'marching_cubes'.

    Returns
    -------
    MeshData
        Extracted triangular mesh.
    """
    if method == "marching_cubes":
        return _marching_cubes(mask, voxel_size, affine, step)
    raise ValueError(f"Unknown surface extraction method: {method}")


def _marching_cubes(
    mask: np.ndarray,
    voxel_size: np.ndarray,
    affine: Optional[np.ndarray],
    step: int,
) -> MeshData:
    """Run marching cubes on a binary volume."""
    from skimage.measure import marching_cubes

    if mask.sum() == 0:
        logger.warning("Empty mask — returning empty mesh")
        return MeshData(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=int))

    spacing = tuple(float(v) for v in voxel_size)

    verts, faces, normals, _ = marching_cubes(
        mask.astype(np.float64),
        level=0.5,
        spacing=spacing,
        step_size=step,
        allow_degenerate=False,
    )

    if affine is not None and not np.allclose(affine, np.eye(4)):
        ones = np.ones((len(verts), 1))
        verts_h = np.hstack([verts, ones])
        verts = (affine @ verts_h.T).T[:, :3]

    mesh = MeshData(
        vertices=verts.astype(np.float64),
        faces=faces.astype(np.int64),
        face_normals=normals,
    )

    logger.info(
        "Marching cubes: %d vertices, %d faces, step=%d",
        mesh.num_vertices,
        mesh.num_faces,
        step,
    )

    return mesh
