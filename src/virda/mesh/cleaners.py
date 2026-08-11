from typing import cast

import numpy as np
import trimesh

from virda.mesh.air_depth import from_trimesh, to_trimesh
from virda.mesh.ray_casting import DEFAULT_FACE_REGION, ray_internal_face_mask
from virda.models.scalp_mesh import ScalpMesh


class MergeCleaner:
    def __init__(self, merge_digits: int = 7) -> None:
        self._merge_digits = merge_digits

    def clean(
        self,
        mesh: ScalpMesh,
        *,
        mask: np.ndarray | None = None,
        affine: np.ndarray | None = None,
    ) -> ScalpMesh:
        trimesh_mesh = to_trimesh(mesh)
        trimesh_mesh.merge_vertices(
            merge_tex=True, merge_norm=True, digits_vertex=self._merge_digits
        )
        trimesh_mesh.process(validate=True)
        return from_trimesh(trimesh_mesh)


class RayCastCleaner:
    def __init__(
        self,
        seed_mm: float = 30.0,
        flood_mm: float = 8.0,
        ray_length_mm: float = 90.0,
        region: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = (
            DEFAULT_FACE_REGION
        ),
    ) -> None:
        self._seed_mm = seed_mm
        self._flood_mm = flood_mm
        self._ray_length_mm = ray_length_mm
        self._region = region

    def clean(
        self,
        mesh: ScalpMesh,
        *,
        mask: np.ndarray | None = None,
        affine: np.ndarray | None = None,
    ) -> ScalpMesh:
        trimesh_mesh = to_trimesh(mesh)
        remove_mask = ray_internal_face_mask(
            trimesh_mesh,
            seed_mm=self._seed_mm,
            flood_mm=self._flood_mm,
            ray_length_mm=self._ray_length_mm,
            region=self._region,
        )
        if remove_mask.any():
            trimesh_mesh.update_faces(~remove_mask)
            trimesh_mesh.process(validate=True)
        return from_trimesh(trimesh_mesh)


class SubdivideCleaner:
    def __init__(self, max_edge: float) -> None:
        self._max_edge = max_edge

    def clean(
        self,
        mesh: ScalpMesh,
        *,
        mask: np.ndarray | None = None,
        affine: np.ndarray | None = None,
    ) -> ScalpMesh:
        trimesh_mesh = to_trimesh(mesh)
        subdivided = cast(trimesh.Trimesh, trimesh_mesh.subdivide_to_size(max_edge=self._max_edge))
        return from_trimesh(subdivided)


class LargestComponentCleaner:
    def __init__(self, min_vertices: int = 100) -> None:
        self._min_vertices = min_vertices

    def clean(
        self,
        mesh: ScalpMesh,
        *,
        mask: np.ndarray | None = None,
        affine: np.ndarray | None = None,
    ) -> ScalpMesh:
        trimesh_mesh = to_trimesh(mesh)
        components = trimesh_mesh.split(only_watertight=False)
        if len(components) > 1:
            return from_trimesh(_keep_largest_component(components, self._min_vertices))
        return from_trimesh(trimesh_mesh)


def _keep_largest_component(
    components: list[trimesh.Trimesh], min_vertices: int
) -> trimesh.Trimesh:
    qualified = [component for component in components if len(component.vertices) >= min_vertices]
    return max(qualified or components, key=lambda component: len(component.vertices))
