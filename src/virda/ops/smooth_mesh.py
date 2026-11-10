"""Step: smooth a scalp mesh (Taubin or Laplacian filter)."""

import trimesh

from virda.models.mesh import Mesh


def smooth_mesh(
    mesh: Mesh,
    *,
    smoother_type: str = "laplacian",
    iterations: int = 10,
    lamb: float = 0.5,
    nu: float = -0.53,
) -> Mesh:
    """Apply the selected trimesh smoothing filter to the mesh.

    ``smoother_type="taubin"`` additionally uses ``nu`` (the Mu parameter must
    be negative and unscale the ``lamb`` shrinkage); ``"laplacian"`` ignores it.
    """
    trimesh_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    if smoother_type == "taubin":
        trimesh.smoothing.filter_taubin(trimesh_mesh, lamb=lamb, nu=nu, iterations=iterations)
    else:
        trimesh.smoothing.filter_laplacian(trimesh_mesh, lamb=lamb, iterations=iterations)
    return Mesh(
        vertices=trimesh_mesh.vertices.copy(),
        faces=trimesh_mesh.faces.copy(),
    )
