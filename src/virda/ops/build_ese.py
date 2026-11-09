"""Step: build the ESE surface offset from the scalp mesh."""

from logging import Logger

import numpy as np
from scipy.spatial import cKDTree

from virda.models.ese_mesh import ESEMesh, Normals
from virda.models.scalp_mesh import ScalpMesh
from virda.models.stage2_config import Stage2Config

_FALLBACK_K_NEIGHBORS = 20


def _normals_from_covariance(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    total = eigenvalues.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        quality = np.where(total > 1e-15, eigenvalues[:, 0] / total, 1.0)
    return eigenvectors[:, :, 0], quality


def _local_pca(neighbors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Estimate normals and quality via batched PCA over k-NN neighborhoods."""
    centroids = neighbors.mean(axis=1, keepdims=True)
    centered = neighbors - centroids
    covariance = np.einsum("nki,nkj->nij", centered, centered) / (k - 1)
    return _normals_from_covariance(covariance)


def _local_pca_weighted(
    neighbors: np.ndarray, distances: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate normals via batched weighted PCA, down-weighting distant points."""
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    weight_sum = weights.sum(axis=1, keepdims=True)
    centroids = (weights[:, :, None] * neighbors).sum(axis=1) / weight_sum
    centered = neighbors - centroids[:, None, :]
    covariance = np.einsum("nk,nki,nkj->nij", weights, centered, centered) / weight_sum[:, :, None]
    return _normals_from_covariance(covariance)


def _orient_outward(normals: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    head_centroid = vertices.mean(axis=0)
    flip_mask = np.sum(normals * (vertices - head_centroid), axis=1) < 0
    normals[flip_mask] *= -1
    return normals


def build_ese(
    scalp_mesh: ScalpMesh,
    *,
    config: Stage2Config,
    ese_offset_mm: float,
    logger: Logger | None = None,
) -> tuple[ESEMesh, Normals]:
    """Estimate outward normals for the scalp mesh and offset them by ``ese_offset_mm``.

    ``config.k_neighbors`` switches between a fixed k-NN neighborhood and a
    radius-based one with a k-NN fallback for sparse regions.
    """
    vertices = scalp_mesh.vertices
    k = config.k_neighbors
    if k is not None:
        if k >= vertices.shape[0]:
            raise ValueError(
                "k_neighbors must be less than the number of vertices: "
                f"k={k}, n_vertices={vertices.shape[0]}"
            )
        normals, quality = _estimate_normals_knn(
            vertices, k, use_weighted_pca=config.use_weighted_pca, sigma=config.pca_sigma_mm
        )
        mode = f"k-NN k={k}"
    else:
        normals, quality = _estimate_normals_radius(vertices, config)
        mode = f"radius r={config.neighborhood_radius_mm} mm"

    normals = _orient_outward(normals, vertices)
    ese_vertices = vertices + ese_offset_mm * normals

    if logger:
        logger.info(
            "ESE estimated: %d vertices, %s, median quality=%.4f",
            vertices.shape[0],
            mode,
            float(np.median(quality)),
        )

    ese_mesh = ESEMesh(
        vertices=ese_vertices,
        faces=scalp_mesh.faces,
        quality=quality,
    )
    return ese_mesh, normals


def _estimate_normals_knn(
    vertices: np.ndarray,
    k: int,
    *,
    use_weighted_pca: bool,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(vertices)
    distances, neighbor_indices = tree.query(vertices, k=k + 1)
    distances = np.asarray(distances)
    neighbor_indices = np.asarray(neighbor_indices)
    neighbors = vertices[neighbor_indices[:, 1:]]
    if use_weighted_pca:
        return _local_pca_weighted(neighbors, distances[:, 1:], sigma)
    return _local_pca(neighbors, k)


def _estimate_normals_radius(
    vertices: np.ndarray,
    config: Stage2Config,
) -> tuple[np.ndarray, np.ndarray]:
    radius = config.neighborhood_radius_mm
    min_neighbors = config.min_neighbors
    weighted = config.use_weighted_pca
    sigma = config.pca_sigma_mm
    normals = np.zeros_like(vertices)
    quality = np.zeros(vertices.shape[0], dtype=np.float64)
    tree = cKDTree(vertices)

    for i in range(vertices.shape[0]):
        neighbor_indices = tree.query_ball_point(vertices[i], radius)
        if len(neighbor_indices) < min_neighbors:
            fallback_k = min(_FALLBACK_K_NEIGHBORS, vertices.shape[0] - 1)
            distances, knn_indices = tree.query(vertices[i], k=fallback_k + 1)
            distances = np.asarray(distances)
            knn_indices = np.asarray(knn_indices)
            neighbors = vertices[knn_indices[1:]]
            if weighted:
                normals_i, quality_i = _local_pca_weighted(
                    neighbors[None, :, :], distances[1:][None, :], sigma
                )
            else:
                normals_i, quality_i = _local_pca(neighbors[None, :, :], fallback_k)
        else:
            neighbors = vertices[neighbor_indices]
            if weighted:
                distances = np.linalg.norm(neighbors - vertices[i], axis=1)
                normals_i, quality_i = _local_pca_weighted(
                    neighbors[None, :, :], distances[None, :], sigma
                )
            else:
                normals_i, quality_i = _local_pca(neighbors[None, :, :], len(neighbor_indices))
        normals[i] = normals_i[0]
        quality[i] = quality_i[0]

    return normals, quality
