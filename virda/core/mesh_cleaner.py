"""Mesh cleaning module — remove artifacts, fill holes, smooth."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .surface_extractor import MeshData

logger = logging.getLogger(__name__)


@dataclass
class CleaningReport:
    """Report of mesh cleaning operations."""

    original_vertices: int
    original_faces: int
    removed_degenerate_faces: int
    removed_duplicate_vertices: int
    removed_small_components: int
    filled_holes: int
    final_vertices: int
    final_faces: int
    smoothed: bool
    smoothing_iterations: int


def clean_mesh(
    mesh: MeshData,
    remove_degenerate: bool = True,
    remove_duplicates: bool = True,
    remove_small_components: bool = True,
    min_component_fraction: float = 0.1,
    fix_winding: bool = True,
    smooth_iterations: int = 0,
    smooth_lambda: float = 0.5,
) -> tuple[MeshData, CleaningReport]:
    """Clean a triangular mesh.

    Parameters
    ----------
    mesh : MeshData
        Input mesh.
    remove_degenerate : bool
        Remove faces with duplicate vertex indices.
    remove_duplicates : bool
        Remove duplicate vertices and remap faces.
    remove_small_components : bool
        Remove small disconnected components.
    min_component_fraction : float
        Minimum fraction of total faces for a component to be kept.
    fix_winding : bool
        Ensure consistent triangle orientation.
    smooth_iterations : int
        Number of Laplacian smoothing iterations (0 = no smoothing).
    smooth_lambda : float
        Smoothing weight per iteration.

    Returns
    -------
    MeshData
        Cleaned mesh.
    CleaningReport
        Summary of operations performed.
    """
    import trimesh

    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()

    report = CleaningReport(
        original_vertices=len(verts),
        original_faces=len(faces),
        removed_degenerate_faces=0,
        removed_duplicate_vertices=0,
        removed_small_components=0,
        filled_holes=0,
        final_vertices=0,
        final_faces=0,
        smoothed=False,
        smoothing_iterations=smooth_iterations,
    )

    if remove_degenerate:
        valid = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (
            faces[:, 0] != faces[:, 2]
        )
        degenerate_count = int((~valid).sum())
        faces = faces[valid]
        report.removed_degenerate_faces = degenerate_count
        if degenerate_count > 0:
            logger.info("Removed %d degenerate faces", degenerate_count)

    if remove_duplicates:
        unique_verts, inverse = np.unique(
            verts, axis=0, return_inverse=True
        )
        if len(unique_verts) < len(verts):
            report.removed_duplicate_vertices = len(verts) - len(unique_verts)
            faces = inverse[faces]
            verts = unique_verts
            logger.info(
                "Removed %d duplicate vertices",
                report.removed_duplicate_vertices,
            )

    if remove_small_components and len(faces) > 0:
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        bodies = tm.split(only_watertight=False)

        if len(bodies) > 1:
            face_counts = [len(b.faces) for b in bodies]
            max_faces = max(face_counts)
            threshold = max_faces * min_component_fraction

            keep_bodies = [b for b, c in zip(bodies, face_counts) if c >= threshold]
            removed = len(bodies) - len(keep_bodies)

            if removed > 0:
                verts_list = []
                faces_list = []
                offset = 0
                for b in keep_bodies:
                    verts_list.append(b.vertices)
                    faces_list.append(b.faces + offset)
                    offset += len(b.vertices)

                verts = np.vstack(verts_list)
                faces = np.vstack(faces_list)
                report.removed_small_components = removed
                logger.info("Removed %d small components", removed)

    if fix_winding and len(faces) > 0:
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        tm.fix_normals()
        verts = tm.vertices
        faces = tm.faces

    if smooth_iterations > 0 and len(verts) > 0:
        verts, faces = _laplacian_smooth(verts, faces, smooth_iterations, smooth_lambda)
        report.smoothed = True

    report.final_vertices = len(verts)
    report.final_faces = len(faces)

    return MeshData(vertices=verts, faces=faces), report


def _laplacian_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int,
    lambd: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Laplacian smoothing."""
    from scipy.sparse import lil_matrix

    n = len(vertices)
    adj = lil_matrix((n, n), dtype=np.float64)

    for face in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adj[face[i], face[j]] = 1.0

    adj = adj.tocsr()

    degree = np.array(adj.sum(axis=1)).flatten()
    degree[degree == 0] = 1.0

    verts = vertices.copy()

    for _ in range(iterations):
        neighbor_sum = adj @ verts
        mean_neighbor = neighbor_sum / degree[:, np.newaxis]
        verts = verts + lambd * (mean_neighbor - verts)

    return verts, faces


def smooth_mesh(
    mesh: MeshData,
    iterations: int = 5,
    lambd: float = 0.5,
) -> MeshData:
    """Apply Laplacian smoothing to a mesh."""
    verts, faces = _laplacian_smooth(mesh.vertices, mesh.faces, iterations, lambd)
    return MeshData(vertices=verts, faces=faces)
