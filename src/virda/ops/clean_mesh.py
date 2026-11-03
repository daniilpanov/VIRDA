"""Step: clean a scalp mesh (merge vertices, drop small islands)."""

from typing import cast

import numpy as np
import trimesh

from virda.mesh.adjacency import build_scalp_mesh
from virda.models.scalp_mesh import ScalpMesh


def clean_mesh(
    mesh: ScalpMesh,
    *,
    min_component_vertices: int = 100,
    merge_digits: int = 6,
) -> ScalpMesh:
    """Merge near-coincident vertices, split into components and keep the main body."""
    trimesh_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    trimesh_mesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=merge_digits)
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
