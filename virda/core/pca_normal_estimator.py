"""PCA-based surface normal estimation for scalp mesh vertices."""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import cKDTree

from .types import MeshData, NormalResult

logger = logging.getLogger(__name__)


def estimate_normals_pca(
    mesh: MeshData,
    radius_mm: float = 10.0,
    min_neighbors: int = 5,
) -> NormalResult:
    """Estimate outward surface normals using local PCA.

    Parameters
    ----------
    mesh : MeshData
        Triangular scalp mesh.
    radius_mm : float
        Neighborhood radius in mm for PCA.
    min_neighbors : int
        Minimum neighbors required for a reliable estimate.

    Returns
    -------
    NormalResult
        Normals, quality metric, and eigenvalues for each vertex.
    """
    vertices = mesh.vertices
    n_vertices = len(vertices)
    normals = np.zeros((n_vertices, 3), dtype=np.float64)
    quality = np.zeros(n_vertices, dtype=np.float64)
    eigenvalues = np.zeros((n_vertices, 3), dtype=np.float64)

    tree = cKDTree(vertices)
    head_centroid = vertices.mean(axis=0)

    unreliable_count = 0
    for i in range(n_vertices):
        neighbor_indices = tree.query_ball_point(vertices[i], radius_mm)
        if len(neighbor_indices) < min_neighbors:
            radial = vertices[i] - head_centroid
            nrm = np.linalg.norm(radial)
            normals[i] = radial / nrm if nrm > 1e-10 else np.array([0.0, 0.0, 1.0])
            quality[i] = 1.0
            unreliable_count += 1
            continue

        neighbor_verts = vertices[neighbor_indices]
        n_i, q_i, ev_i = _compute_local_pca(neighbor_verts)
        normals[i] = n_i
        quality[i] = q_i
        eigenvalues[i] = ev_i

    normals = _orient_normals_outward(normals, vertices, head_centroid)

    logger.info(
        "PCA normals estimated: %d vertices, %d unreliable (radius=%.1f mm)",
        n_vertices,
        unreliable_count,
        radius_mm,
    )

    return NormalResult(
        normals=normals,
        quality=quality,
        eigenvalues=eigenvalues,
    )


def _compute_local_pca(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Compute local PCA for a set of 3D points.

    Parameters
    ----------
    points : np.ndarray
        Neighbor coordinates (K,3).

    Returns
    -------
    tuple[np.ndarray, float, np.ndarray]
        Normal vector (3,), quality metric, and sorted eigenvalues (3,).
    """
    centroid = points.mean(axis=0)
    X = points - centroid
    k = len(X)
    C = (X.T @ X) / max(k - 1, 1)

    eigenvalues_full, eigenvectors = np.linalg.eigh(C)

    sorted_idx = np.argsort(eigenvalues_full)[::-1]
    eigenvalues_sorted = eigenvalues_full[sorted_idx]
    eigenvectors_sorted = eigenvectors[:, sorted_idx]

    normal = eigenvectors_sorted[:, 2]

    ev_sum = eigenvalues_sorted.sum()
    q = float(eigenvalues_sorted[2] / ev_sum) if ev_sum > 1e-15 else 1.0

    return normal, q, eigenvalues_sorted


def _orient_normals_outward(
    normals: np.ndarray,
    vertices: np.ndarray,
    head_centroid: np.ndarray,
) -> np.ndarray:
    """Orient normals outward from head centroid.

    Parameters
    ----------
    normals : np.ndarray
        Normal vectors (N,3).
    vertices : np.ndarray
        Vertex coordinates (N,3).
    head_centroid : np.ndarray
        Approximate head center (3,).

    Returns
    -------
    np.ndarray
        Outward-oriented normals (N,3).
    """
    radial = vertices - head_centroid
    dots = np.sum(normals * radial, axis=1)
    flip_mask = dots < 0
    normals[flip_mask] *= -1
    return normals
