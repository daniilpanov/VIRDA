import numpy as np
import trimesh

from virda.models.scalp_mesh import ScalpMesh


def compute_face_adjacency(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return pairs of faces sharing an edge, as an (E, 2) int array."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return np.asarray(mesh.face_adjacency, dtype=np.int64)


def build_scalp_mesh(vertices: np.ndarray, faces: np.ndarray) -> ScalpMesh:
    """Construct a ScalpMesh together with its face adjacency."""
    return ScalpMesh(
        vertices=vertices,
        faces=faces,
        face_adjacency=compute_face_adjacency(vertices, faces),
    )
