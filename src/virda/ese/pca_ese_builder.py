import logging
from typing import cast

import numpy as np
from scipy.spatial import cKDTree

from virda.ese.contracts import ESEBuilder
from virda.models.ese_mesh import ESEMesh
from virda.models.scalp_mesh import ScalpMesh
from virda.models.stage2_config import Stage2Config

logger = logging.getLogger(__name__)


def _local_pca(neighbors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Estimate normals and quality via batched PCA over k-NN neighborhoods."""
    centroids = neighbors.mean(axis=1, keepdims=True)
    centered = neighbors - centroids
    covariance = np.einsum("nki,nkj->nij", centered, centered) / (k - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    total = eigenvalues.sum(axis=1)
    quality = np.where(total > 1e-15, eigenvalues[:, 0] / total, 1.0)
    return cast(np.ndarray, eigenvectors[:, :, 0]), quality


def _orient_outward(normals: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    head_centroid = vertices.mean(axis=0)
    flip_mask = np.sum(normals * (vertices - head_centroid), axis=1) < 0
    normals[flip_mask] *= -1
    return normals


class PCAESEBuilder(ESEBuilder):
    def __init__(self, config: Stage2Config, ese_offset_mm: float) -> None:
        self._config = config
        self._ese_offset_mm = ese_offset_mm

    def _process(self, scalp_mesh: ScalpMesh) -> ESEMesh:
        k = self._config.k_neighbors
        if k is None:
            raise NotImplementedError("radius mode is not implemented yet")
        if self._config.use_weighted_pca:
            raise NotImplementedError("weighted PCA is not implemented yet")

        vertices = scalp_mesh.vertices
        if k >= vertices.shape[0]:
            raise ValueError(
                "k_neighbors must be less than the number of vertices: "
                f"k={k}, n_vertices={vertices.shape[0]}"
            )
        normals, quality = self._estimate_normals_knn(vertices, k)
        normals = _orient_outward(normals, vertices)
        ese_vertices = vertices + self._ese_offset_mm * normals

        logger.info(
            "ESE estimated: %d vertices, k-NN k=%d, median quality=%.4f",
            vertices.shape[0],
            k,
            float(np.median(quality)),
        )

        return ESEMesh(
            vertices=ese_vertices,
            faces=scalp_mesh.faces,
            scalp_vertices=vertices,
            normals=normals,
            quality=quality,
        )

    def _estimate_normals_knn(
        self, vertices: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        tree = cKDTree(vertices)
        _, neighbor_indices = tree.query(vertices, k=k + 1)
        neighbors = vertices[neighbor_indices[:, 1:]]
        return _local_pca(neighbors, k)
