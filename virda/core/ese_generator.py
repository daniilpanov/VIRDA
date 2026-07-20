"""ESE generator — create scalp-to-ESE point pairs."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .surface_extractor import MeshData
from .ese_config import ESEConfig
from .pca_normal_estimator import NormalResult

logger = logging.getLogger(__name__)


@dataclass
class ESEPoint:
    """Single ESE point with associated scalp point."""

    point_id: int
    scalp_coords: np.ndarray
    ese_coords: np.ndarray
    normal_vector: np.ndarray
    pca_quality: float
    neighborhood_radius: float


@dataclass
class ESEResult:
    """Complete ESE generation result."""

    ese_points: list[ESEPoint]
    scalp_vertices: np.ndarray
    ese_vertices: np.ndarray
    normals: np.ndarray
    quality: np.ndarray
    head_centroid: np.ndarray

    @property
    def num_points(self) -> int:
        return len(self.ese_points)

    def get_ese_point_cloud(self) -> np.ndarray:
        """Return ESE vertices as (N, 3) array."""
        return self.ese_vertices.copy()

    def get_scalp_point_cloud(self) -> np.ndarray:
        """Return scalp vertices as (N, 3) array."""
        return self.scalp_vertices.copy()

    def get_point_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return scalp-to-ESE point pair arrays."""
        return self.scalp_vertices.copy(), self.ese_vertices.copy()


def generate_ese(
    mesh: MeshData,
    normal_result: NormalResult,
    config: ESEConfig,
) -> ESEResult:
    """Generate Electrode Surface Equivalent from scalp mesh."""
    verts = mesh.vertices
    normals = normal_result.normals
    quality = normal_result.quality
    offset = config.offset_mm

    ese_vertices = verts + offset * normals

    ese_points = []
    for i in range(len(verts)):
        ep = ESEPoint(
            point_id=i,
            scalp_coords=verts[i].copy(),
            ese_coords=ese_vertices[i].copy(),
            normal_vector=normals[i].copy(),
            pca_quality=float(quality[i]),
            neighborhood_radius=offset,
        )
        ese_points.append(ep)

    logger.info(
        "ESE generated: %d points, offset=%.2f mm",
        len(ese_points),
        offset,
    )

    return ESEResult(
        ese_points=ese_points,
        scalp_vertices=verts.copy(),
        ese_vertices=ese_vertices,
        normals=normals.copy(),
        quality=quality.copy(),
        head_centroid=normal_result.head_centroid.copy(),
    )
