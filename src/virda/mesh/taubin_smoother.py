import trimesh

from virda.mesh.contracts import MeshPostprocessor
from virda.models.scalp_mesh import ScalpMesh


class TaubinSmoother(MeshPostprocessor):
    def __init__(self, iterations: int = 10, lamb: float = 0.5, nu: float = -0.53) -> None:
        self._iterations = iterations
        self._lamb = lamb
        self._nu = nu

    def _process(self, mesh: ScalpMesh) -> ScalpMesh:
        trimesh_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        trimesh.smoothing.filter_taubin(
            trimesh_mesh, lamb=self._lamb, nu=self._nu, iterations=self._iterations
        )
        return ScalpMesh(
            vertices=trimesh_mesh.vertices.copy(),
            faces=trimesh_mesh.faces.copy(),
        )
