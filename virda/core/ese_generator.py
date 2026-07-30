"""Electrode Surface Equivalent (ESE) generation."""

from __future__ import annotations

import logging

import numpy as np

from .ese_config import ESEConfig
from .types import ESEResult, MeshData, NormalResult

logger = logging.getLogger(__name__)


def generate_ese(
    mesh: MeshData,
    normal_result: NormalResult,
    config: ESEConfig,
) -> ESEResult:
    """Generate the ESE surface by offsetting scalp vertices along normals.

    Parameters
    ----------
    mesh : MeshData
        Triangular scalp mesh.
    normal_result : NormalResult
        PCA-estimated normals and quality.
    config : ESEConfig
        ESE configuration with offset distance.

    Returns
    -------
    ESEResult
        Scalp-to-ESE point pairs and metadata.
    """
    vertices = mesh.vertices
    normals = normal_result.normals
    quality = normal_result.quality
    offset = config.offset_mm

    ese_vertices = vertices + offset * normals
    head_centroid = vertices.mean(axis=0)

    result = ESEResult(
        scalp_vertices=vertices.copy(),
        ese_vertices=ese_vertices,
        normals=normals.copy(),
        quality=quality.copy(),
        head_centroid=head_centroid,
        num_points=len(vertices),
    )

    logger.info(
        "ESE generated: %d points, offset=%.1f mm",
        result.num_points,
        offset,
    )

    return result
