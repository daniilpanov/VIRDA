import trimesh

from virda.mesh.contracts import MeshPostprocessor
from virda.models.scalp_mesh import ScalpMesh


class LaplacianSmoother(MeshPostprocessor):
    def __init__(self, iterations: int = 10, lamb: float = 0.5) -> None:
        self._iterations = iterations
        self._lamb = lamb

    def _process(self, mesh: ScalpMesh) -> ScalpMesh:
        trimesh_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        trimesh.smoothing.filter_laplacian(
            trimesh_mesh, lamb=self._lamb, iterations=self._iterations
        )
        return ScalpMesh(
            vertices=trimesh_mesh.vertices.copy(),
            faces=trimesh_mesh.faces.copy(),
        )
