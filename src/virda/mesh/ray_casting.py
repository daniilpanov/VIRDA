"""Cavity-wall detection by ray casting along outward face normals."""

import numpy as np
import trimesh

from virda.mesh.air_depth import connected_components_containing_seeds

# Region in world mm where internal (cavity) walls can occur: the facial block.
# Excludes the ears (|x| > 55) and the open neck rim (z < -58).
DEFAULT_FACE_REGION: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
    (-55.0, 55.0),
    (-75.0, 75.0),
    (-58.0, 40.0),
)

_SELF_HIT_SKIP_MM = 2.0


def ray_internal_face_mask(
    mesh: trimesh.Trimesh,
    seed_mm: float = 30.0,
    flood_mm: float = 8.0,
    ray_length_mm: float = 90.0,
    region: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
) -> np.ndarray:
    """Return a boolean mask of internal (cavity-wall) faces to remove.

    Ray-casting strategy: cast a ray from every face centroid along the
    outward normal and record the distance to the first real mesh intersection
    (self/neighbour hits below ``_SELF_HIT_SKIP_MM`` are ignored). Deep faces
    (>= ``seed_mm``) are guaranteed internal; a face is then removed if it
    belongs to a connected component of faces (via edge adjacency) whose
    members all have a real hit depth in [``flood_mm``, ``ray_length_mm``] and
    that contains a seed face.
    """
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return np.zeros(0, dtype=bool)
    centers = mesh.triangles_center
    normals = _outward_normals(mesh)
    in_region = _in_region(centers, region) if region is not None else np.ones(n_faces, dtype=bool)

    target = np.flatnonzero(in_region)
    depths = np.full(n_faces, np.inf)
    if len(target):
        depths[target] = _ray_depths(mesh, centers[target], normals[target], ray_length_mm)

    eligible = (depths >= flood_mm) & (depths <= ray_length_mm) & in_region
    seeds = (depths >= seed_mm) & in_region
    return connected_components_containing_seeds(mesh, eligible, seeds)


def _outward_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    normals = mesh.face_normals.astype(np.float64).copy()
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    centers = mesh.triangles_center
    head_centre = mesh.vertices.mean(axis=0)
    flip = (normals * (centers - head_centre)).sum(axis=1) < 0
    normals[flip] *= -1.0
    return normals


def _in_region(
    centers: np.ndarray,
    region: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    (x0, x1), (y0, y1), (z0, z1) = region
    return (
        (centers[:, 0] >= x0)
        & (centers[:, 0] <= x1)
        & (centers[:, 1] >= y0)
        & (centers[:, 1] <= y1)
        & (centers[:, 2] >= z0)
        & (centers[:, 2] <= z1)
    )


def _ray_depths(
    mesh: trimesh.Trimesh,
    centers: np.ndarray,
    normals: np.ndarray,
    ray_length_mm: float,
) -> np.ndarray:
    depths = np.full(len(centers), np.inf)
    batch = 8000
    for start in range(0, len(centers), batch):
        stop = min(start + batch, len(centers))
        origins = centers[start:stop]
        directions = normals[start:stop]
        locations, ray_idx, _ = mesh.ray.intersects_location(
            origins, origins + directions * ray_length_mm
        )
        if not len(locations):
            continue
        hit_dist = np.linalg.norm(locations - origins[ray_idx], axis=1)
        real = hit_dist >= _SELF_HIT_SKIP_MM
        if not real.any():
            continue
        uniq, inverse = np.unique(ray_idx[real], return_inverse=True)
        first_hit = np.full(len(uniq), np.inf)
        np.minimum.at(first_hit, inverse, hit_dist[real])
        depths[start + uniq] = first_hit
    return depths
