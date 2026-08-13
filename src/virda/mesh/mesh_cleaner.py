from typing import cast

import numpy as np
import trimesh

from virda.mesh.adjacency import build_scalp_mesh
from virda.mesh.contracts import MeshPostprocessor
from virda.models.scalp_mesh import ScalpMesh


class TrimeshCleaner(MeshPostprocessor):
    def __init__(self, min_component_vertices: int = 100, merge_digits: int = 6) -> None:
        self._min_component_vertices = min_component_vertices
        self._merge_digits = merge_digits

    def _process(self, mesh: ScalpMesh) -> ScalpMesh:
        trimesh_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        trimesh_mesh.merge_vertices(
            merge_tex=True, merge_norm=True, digits_vertex=self._merge_digits
        )
        trimesh_mesh.process(validate=True)
        trimesh_mesh = cast(trimesh.Trimesh, trimesh_mesh.subdivide_to_size(max_edge=5.0))
        components = trimesh_mesh.split(only_watertight=False)
        if len(components) > 1:
            main_body = _keep_largest_component(components)
            return build_scalp_mesh(
                vertices=np.asarray(main_body.vertices, dtype=np.float64),
                faces=np.asarray(main_body.faces, dtype=np.int64),
            )
        return build_scalp_mesh(
            vertices=np.asarray(trimesh_mesh.vertices, dtype=np.float64),
            faces=np.asarray(trimesh_mesh.faces, dtype=np.int64),
        )


def _keep_largest_component(components: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    return max(components, key=lambda component: len(component.vertices))
