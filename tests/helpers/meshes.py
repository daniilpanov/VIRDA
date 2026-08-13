import numpy as np
import trimesh
from scipy.spatial import Delaunay

from virda.mesh.adjacency import build_scalp_mesh
from virda.models.scalp_mesh import ScalpMesh


def make_sphere(radius: float = 30.0, subdivisions: int = 2) -> ScalpMesh:
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return build_scalp_mesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
    )


def make_plane(n_points: int = 64, seed: int = 42) -> ScalpMesh:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-10, 10, (n_points, 2))
    vertices = np.column_stack([xy, np.zeros(n_points)])
    faces = Delaunay(xy).simplices
    return build_scalp_mesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
    )
