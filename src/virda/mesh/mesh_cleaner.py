from typing import cast

import numpy as np
import trimesh

from virda.mesh.air_depth import internal_face_mask
from virda.models.scalp_mesh import ScalpMesh


class TrimeshCleaner:
    def __init__(
        self,
        min_component_vertices: int = 100,
        merge_digits: int = 6,
        remove_internal_faces: bool = True,
        internal_face_wide_mm: float = 10.0,
        internal_face_seed_mm: float = 20.0,
        internal_face_flood_mm: float = 12.0,
    ) -> None:
        self._min_component_vertices = min_component_vertices
        self._merge_digits = merge_digits
        self._remove_internal_faces = remove_internal_faces
        self._internal_face_wide_mm = internal_face_wide_mm
        self._internal_face_seed_mm = internal_face_seed_mm
        self._internal_face_flood_mm = internal_face_flood_mm

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
            remove_mask = self._geodesic_internal_face_mask(trimesh_mesh, mask, affine)
            if remove_mask.any():
                trimesh_mesh.update_faces(~remove_mask)
                trimesh_mesh.process(validate=True)
        trimesh_mesh = cast(trimesh.Trimesh, trimesh_mesh.subdivide_to_size(max_edge=5.0))
        components = trimesh_mesh.split(only_watertight=False)
        if len(components) > 1:
            main_body = _keep_largest_component(components)
            return ScalpMesh(
                vertices=np.asarray(main_body.vertices, dtype=np.float64),
                faces=np.asarray(main_body.faces, dtype=np.int64),
            )
        return ScalpMesh(
            vertices=np.asarray(trimesh_mesh.vertices, dtype=np.float64),
            faces=np.asarray(trimesh_mesh.faces, dtype=np.int64),
        )

    def _geodesic_internal_face_mask(
        self,
        mesh: trimesh.Trimesh,
        mask: np.ndarray | None,
        affine: np.ndarray | None,
    ) -> np.ndarray:
        if mask is None or affine is None:
            raise ValueError(
                "geodesic internal-face removal requires the segmentation mask "
                "and affine; pass mask= and affine= to clean()"
            )
        return internal_face_mask(
            mesh,
            mask,
            affine,
            wide_mm=self._internal_face_wide_mm,
            seed_mm=self._internal_face_seed_mm,
            flood_mm=self._internal_face_flood_mm,
        )


def _keep_largest_component(components: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    return max(components, key=lambda component: len(component.vertices))
