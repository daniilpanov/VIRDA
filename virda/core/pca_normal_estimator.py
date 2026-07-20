"""PCA-based surface normal estimator for mesh vertices."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .surface_extractor import MeshData

logger = logging.getLogger(__name__)


@dataclass
class NormalResult:
    """Result of PCA normal estimation."""

    normals: np.ndarray
    eigenvalues: np.ndarray
    quality: np.ndarray
    head_centroid: np.ndarray


def estimate_normals_pca(
    mesh: MeshData,
    radius_mm: float = 10.0,
    k_neighbors: Optional[int] = None,
    min_neighbors: int = 5,
    weighted: bool = False,
    weight_sigma: float = 5.0,
) -> NormalResult:
    """Estimate outward surface normals using local PCA.

    Parameters
    ----------
    mesh : MeshData
        Triangular scalp mesh.
    radius_mm : float
        Radius in mm for neighborhood search.
    k_neighbors : int, optional
        If provided, use k nearest neighbors instead of radius search.
    min_neighbors : int
        Minimum neighbors required for valid PCA.
    weighted : bool
        Use distance-weighted PCA.
    weight_sigma : float
        Gaussian sigma for distance weighting.

    Returns
    -------
    NormalResult
        Normals, eigenvalues, quality metric, and head centroid.
    """
    verts = mesh.vertices
    n = len(verts)

    head_centroid = verts.mean(axis=0)

    if k_neighbors is not None:
        tree = cKDTree(verts)
        _, neighbor_idx = tree.query(verts, k=k_neighbors + 1)
        neighbor_idx = neighbor_idx[:, 1:]
        use_radius = False
    else:
        tree = cKDTree(verts)
        use_radius = True

    normals = np.zeros((n, 3))
    eigenvalues = np.zeros((n, 3))
    quality = np.zeros(n)

    for i in range(n):
        if use_radius:
            idx = tree.query_ball_point(verts[i], r=radius_mm)
            idx = [j for j in idx if j != i]
        else:
            idx = neighbor_idx[i].tolist()

        if len(idx) < min_neighbors:
            dir_vec = verts[i] - head_centroid
            norm = np.linalg.norm(dir_vec)
            if norm > 0:
                normals[i] = dir_vec / norm
            else:
                normals[i] = np.array([0.0, 0.0, 1.0])
            eigenvalues[i] = np.inf
            quality[i] = 1.0
            continue

        neighbor_coords = verts[idx]

        if weighted:
            dists = np.linalg.norm(neighbor_coords - verts[i], axis=1)
            weights = np.exp(-dists ** 2 / (2 * weight_sigma ** 2))
            weights /= weights.sum()
            centroid = np.average(neighbor_coords, axis=0, weights=weights)
            centered = neighbor_coords - centroid
            W = np.diag(weights)
            cov = centered.T @ W @ centered
        else:
            centroid = neighbor_coords.mean(axis=0)
            centered = neighbor_coords - centroid
            cov = (centered.T @ centered) / len(idx)

        eigvals, eigvecs = np.linalg.eigh(cov)

        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        normal = eigvecs[:, 0]

        vec_to_centroid = verts[i] - head_centroid
        if np.dot(normal, vec_to_centroid) < 0:
            normal = -normal

        normals[i] = normal
        eigenvalues[i] = eigvals

        total_var = eigvals.sum()
        if total_var > 0:
            quality[i] = eigvals[0] / total_var
        else:
            quality[i] = 1.0

    logger.info(
        "PCA normals estimated: %d vertices, radius=%.1f mm, median quality=%.4f",
        n,
        radius_mm,
        float(np.median(quality)),
    )

    return NormalResult(
        normals=normals,
        eigenvalues=eigenvalues,
        quality=quality,
        head_centroid=head_centroid,
    )
