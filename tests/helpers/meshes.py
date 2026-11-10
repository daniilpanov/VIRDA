import numpy as np
import trimesh
from scipy.spatial import Delaunay

from virda.models.mesh import Mesh


def make_sphere(radius: float = 30.0, subdivisions: int = 2) -> Mesh:
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return Mesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
    )


def make_plane(n_points: int = 64, seed: int = 42) -> Mesh:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-10, 10, (n_points, 2))
    vertices = np.column_stack([xy, np.zeros(n_points)])
    faces = Delaunay(xy).simplices
    return Mesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
    )
