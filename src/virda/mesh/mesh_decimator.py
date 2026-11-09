"""Reduce the vertex count of a scalp mesh with quadric-metric decimation."""

import logging

import numpy as np
import trimesh

from virda.models.mesh import Mesh

_logger = logging.getLogger(__name__)


def decimate_mesh(
    mesh: Mesh,
    *,
    density_percent: float = 100.0,
    logger: logging.Logger = _logger,
) -> Mesh:
    """Return ``mesh`` reduced to ``density_percent`` of its original density.

    ``density_percent`` expresses how much of the mesh is kept: ``100`` is a
    no-op (full mesh as extracted), ``50`` removes roughly half the vertices,
    values are rejected outside ``[1, 100]``.  The reduction is a
    quadric-metric edge-collapse decimation built on ``trimesh`` /
    ``fast-simplification``.
    """
    if not 1.0 <= density_percent <= 100.0:
        raise ValueError(f"density_percent must be within [1, 100], got {density_percent}")
    if density_percent < 100.0:
        try:
            import fast_simplification  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "Mesh density below 100% requires the 'fast-simplification' package: "
                "install it with `uv add fast-simplification`"
                " (or `pip install fast-simplification`)"
            ) from None

    if mesh.faces.shape[0] == 0:
        logger.info("Mesh has no faces, nothing to decimate")
        return mesh
    if density_percent >= 100.0:
        logger.info("Mesh density 100%%: keeping the full mesh (%d vertices)", len(mesh.vertices))
        return mesh

    target_faces = max(1, round(mesh.faces.shape[0] * density_percent / 100.0))
    if target_faces >= mesh.faces.shape[0]:
        return mesh

    trimesh_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    decimated = trimesh_mesh.simplify_quadric_decimation(face_count=target_faces)
    logger.info(
        "Mesh density %.1f%%: decimating %d -> %d vertices",
        density_percent,
        len(mesh.vertices),
        len(decimated.vertices),
    )
    return Mesh(
        vertices=np.asarray(decimated.vertices, dtype=np.float64),
        faces=np.asarray(decimated.faces, dtype=np.int64),
    )
