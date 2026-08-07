from typing import cast

import numpy as np
import trimesh

from virda.mesh.air_depth import connected_components_containing_seeds, internal_face_mask
from virda.mesh.hole_fill import fill_small_boundary_holes
from virda.models.scalp_mesh import ScalpMesh

# Region in world mm where internal (cavity) walls can occur: the facial block.
# Excludes the ears (|x| > 55) and the open neck rim (z < -58).
DEFAULT_FACE_REGION: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
    (-55.0, 55.0),
    (-75.0, 75.0),
    (-58.0, 40.0),
)

_SELF_HIT_SKIP_MM = 2.0

_INTERNAL_FACE_METHODS = ("geodesic", "ray")


class TrimeshCleaner:
    def __init__(
        self,
        min_component_vertices: int = 100,
        merge_digits: int = 6,
        remove_internal_faces: bool = True,
        internal_face_method: str = "geodesic",
        internal_face_wide_mm: float = 10.0,
        internal_face_seed_mm: float = 20.0,
        internal_face_flood_mm: float = 12.0,
        internal_face_seed_depth_mm: float = 30.0,
        internal_face_flood_depth_mm: float = 8.0,
        internal_face_ray_length_mm: float = 90.0,
        internal_face_region: (
            tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None
        ) = DEFAULT_FACE_REGION,
        fill_small_holes: bool = True,
        fill_small_holes_max_mm: float = 15.0,
        subdivide_max_edge: float | None = None,
    ) -> None:
        if internal_face_method not in _INTERNAL_FACE_METHODS:
            raise ValueError(
                f"internal_face_method must be one of {_INTERNAL_FACE_METHODS}, "
                f"got {internal_face_method!r}"
            )
        self._min_component_vertices = min_component_vertices
        self._merge_digits = merge_digits
        self._remove_internal_faces = remove_internal_faces
        self._internal_face_method = internal_face_method
        self._internal_face_wide_mm = internal_face_wide_mm
        self._internal_face_seed_mm = internal_face_seed_mm
        self._internal_face_flood_mm = internal_face_flood_mm
        self._internal_face_seed_depth_mm = internal_face_seed_depth_mm
        self._internal_face_flood_depth_mm = internal_face_flood_depth_mm
        self._internal_face_ray_length_mm = internal_face_ray_length_mm
        self._internal_face_region = internal_face_region
        self._fill_small_holes = fill_small_holes
        self._fill_small_holes_max_mm = fill_small_holes_max_mm
        self._subdivide_max_edge = subdivide_max_edge

    def clean(
        self,
        mesh: ScalpMesh,
        *,
        mask: np.ndarray | None = None,
        affine: np.ndarray | None = None,
    ) -> ScalpMesh:
        trimesh_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        trimesh_mesh.merge_vertices(
            merge_tex=True, merge_norm=True, digits_vertex=self._merge_digits
        )
        trimesh_mesh.process(validate=True)
        if self._remove_internal_faces:
            remove_mask = self._internal_face_mask(trimesh_mesh, mask, affine)
            if remove_mask.any():
                trimesh_mesh.update_faces(~remove_mask)
                trimesh_mesh.process(validate=True)
                if self._fill_small_holes:
                    fill_small_boundary_holes(trimesh_mesh, self._fill_small_holes_max_mm)
                    trimesh_mesh.process(validate=True)
        if self._subdivide_max_edge is not None:
            trimesh_mesh = cast(
                trimesh.Trimesh,
                trimesh_mesh.subdivide_to_size(max_edge=self._subdivide_max_edge),
            )
        components = trimesh_mesh.split(only_watertight=False)
        if len(components) > 1:
            main_body = _keep_largest_component(components)
            return _from_trimesh(main_body)
        return _from_trimesh(trimesh_mesh)

    def _internal_face_mask(
        self,
        mesh: trimesh.Trimesh,
        mask: np.ndarray | None,
        affine: np.ndarray | None,
    ) -> np.ndarray:
        if self._internal_face_method == "geodesic":
            if mask is None or affine is None:
                raise ValueError(
                    "geodesic internal-face removal requires the segmentation mask "
                    "and affine; pass mask= and affine= to clean(), or use "
                    "internal_face_method='ray'"
                )
            return internal_face_mask(
                mesh,
                mask,
                affine,
                wide_mm=self._internal_face_wide_mm,
                seed_mm=self._internal_face_seed_mm,
                flood_mm=self._internal_face_flood_mm,
            )
        return _ray_internal_face_mask(
            mesh,
            seed_depth_mm=self._internal_face_seed_depth_mm,
            flood_depth_mm=self._internal_face_flood_depth_mm,
            ray_length_mm=self._internal_face_ray_length_mm,
            region=self._internal_face_region,
        )


def _keep_largest_component(components: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    return max(components, key=lambda component: len(component.vertices))


def _from_trimesh(mesh: trimesh.Trimesh) -> ScalpMesh:
    return ScalpMesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        face_adjacency=np.asarray(mesh.face_adjacency, dtype=np.int64),
        coordinate_system="world",
    )


def _ray_internal_face_mask(
    mesh: trimesh.Trimesh,
    seed_depth_mm: float = 30.0,
    flood_depth_mm: float = 8.0,
    ray_length_mm: float = 90.0,
    region: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
) -> np.ndarray:
    """Return a boolean mask of internal (cavity-wall) faces to remove.

    Legacy ray-casting strategy: cast a ray from every face centroid along the
    outward normal and record the distance to the first real mesh intersection
    (self/neighbour hits below ``_SELF_HIT_SKIP_MM`` are ignored). Deep faces
    (>= ``seed_depth_mm``) are guaranteed internal; a face is then removed if
    it belongs to a connected component of faces (via edge adjacency) whose
    members all have a real hit depth in [``flood_depth_mm``, ``ray_length_mm``]
    and that contains a seed face.
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

    eligible = (depths >= flood_depth_mm) & (depths <= ray_length_mm) & in_region
    seeds = (depths >= seed_depth_mm) & in_region
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
