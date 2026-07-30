"""Mesh cleaning, artifact removal, and smoothing."""

from __future__ import annotations

import logging

import numpy as np
import trimesh

from .types import MeshData

logger = logging.getLogger(__name__)


def clean_mesh(
    mesh: MeshData,
    remove_degenerate: bool = True,
    fill_holes: bool = True,
    smooth_iterations: int = 0,
    min_component_faces: int = 100,
) -> tuple[MeshData, dict]:
    """Clean a triangular mesh: remove artifacts, fix defects, optionally smooth.

    Parameters
    ----------
    mesh : MeshData
        Input mesh.
    remove_degenerate : bool
        Remove degenerate (zero-area) triangles.
    fill_holes : bool
        Attempt to fill small holes.
    smooth_iterations : int
        Number of Laplacian smoothing iterations (0 = no smoothing).
    min_component_faces : int
        Minimum faces for a connected component to be kept.

    Returns
    -------
    tuple[MeshData, dict]
        Cleaned mesh and statistics dictionary.
    """
    stats: dict = {
        "original_vertices": mesh.num_vertices,
        "original_faces": mesh.num_faces,
    }

    tm = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        process=False,
    )

    if remove_degenerate:
        n_before = len(tm.faces)
        mask = tm.nondegenerate_faces()
        tm.update_faces(mask)
        stats["degenerate_removed"] = n_before - len(tm.faces)

    n_before = len(tm.vertices)
    tm.merge_vertices(merge_tex=True, merge_norm=True)
    stats["duplicates_removed"] = n_before - len(tm.vertices)

    if min_component_faces > 0 and len(tm.faces) > 0:
        bodies = tm.split(only_watertight=False)
        if len(bodies) > 1:
            kept = max(bodies, key=lambda b: len(b.faces))
            kept_faces = len(kept.faces)
            if kept_faces < len(tm.faces):
                tm = kept
                stats["small_components_removed"] = len(bodies) - 1

    if fill_holes and len(tm.faces) > 0:
        try:
            trimesh.repair.fill_holes(tm)
        except Exception:
            pass

    if smooth_iterations > 0 and len(tm.vertices) > 0:
        trimesh.smoothing.filter_laplacian(tm, iterations=smooth_iterations)
        stats["smooth_iterations"] = smooth_iterations

    adjacency = _build_adjacency(tm.faces, len(tm.vertices))

    result = MeshData(
        vertices=tm.vertices.astype(np.float64),
        faces=tm.faces.astype(np.int64),
        adjacency=adjacency,
        coordinate_system=mesh.coordinate_system,
        transform=mesh.transform,
    )

    stats["final_vertices"] = result.num_vertices
    stats["final_faces"] = result.num_faces

    logger.info(
        "Mesh cleaned: %d -> %d vertices, %d -> %d faces",
        stats["original_vertices"],
        stats["final_vertices"],
        stats["original_faces"],
        stats["final_faces"],
    )

    return result, stats


def _build_adjacency(faces: np.ndarray, num_vertices: int) -> list[list[int]]:
    """Build vertex-to-vertex adjacency list from face array."""
    adjacency: list[set[int]] = [set() for _ in range(num_vertices)]
    for tri in faces:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        adjacency[i].update([j, k])
        adjacency[j].update([i, k])
        adjacency[k].update([i, j])
    return [sorted(s) for s in adjacency]
