import numpy as np
import pytest
import trimesh

from virda.mesh.mesh_decimator import decimate_mesh
from virda.models.mesh import Mesh


@pytest.fixture
def icosphere_mesh() -> Mesh:
    mesh = trimesh.creation.icosphere(subdivisions=3)
    return Mesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
    )


class TestDecimateMesh:
    def test_density_100_is_a_noop(self, icosphere_mesh: Mesh) -> None:
        decimated = decimate_mesh(icosphere_mesh, density_percent=100.0)

        assert decimated.vertices is icosphere_mesh.vertices
        assert decimated.faces is icosphere_mesh.faces

    def test_density_50_removes_half_the_vertices(self, icosphere_mesh: Mesh) -> None:
        decimated = decimate_mesh(icosphere_mesh, density_percent=50.0)

        original = icosphere_mesh.vertices.shape[0]
        ratio = decimated.vertices.shape[0] / original
        assert 0.35 <= ratio <= 0.65
        assert decimated.faces.shape[0] < icosphere_mesh.faces.shape[0]

    def test_density_low_keeps_valid_mesh(self, icosphere_mesh: Mesh) -> None:
        decimated = decimate_mesh(icosphere_mesh, density_percent=15.0)

        assert isinstance(decimated, Mesh)
        assert decimated.vertices.shape[1] == 3
        assert decimated.faces.shape[1] == 3
        assert decimated.faces.min() >= 0
        assert decimated.faces.max() < decimated.vertices.shape[0]
        assert decimated.vertices.shape[0] < icosphere_mesh.vertices.shape[0]

    @pytest.mark.parametrize("density_percent", [0.0, 0.5, 150.0])
    def test_density_out_of_range_raises(
        self, icosphere_mesh: Mesh, density_percent: float
    ) -> None:
        with pytest.raises(ValueError, match="density_percent"):
            decimate_mesh(icosphere_mesh, density_percent=density_percent)
