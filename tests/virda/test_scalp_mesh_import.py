import numpy as np
import pytest
import trimesh

from virda.io.importers.scalp_mesh import import_scalp_mesh
from virda.models.mesh import Mesh


@pytest.fixture
def icosphere_path(tmp_path):
    mesh = trimesh.creation.icosphere(subdivisions=2)
    path = tmp_path / "scalp.ply"
    mesh.export(path)
    return path


class TestImportMesh:
    def test_reads_back_geometry(self, icosphere_path) -> None:
        mesh = import_scalp_mesh(icosphere_path)

        assert isinstance(mesh, Mesh)
        expected = trimesh.creation.icosphere(subdivisions=2)
        np.testing.assert_array_almost_equal(mesh.vertices, expected.vertices)
        np.testing.assert_array_equal(mesh.faces, expected.faces)


class TestImportMeshErrors:
    def test_point_cloud_rejected(self, tmp_path) -> None:
        path = tmp_path / "points.ply"
        trimesh.PointCloud(vertices=np.zeros((10, 3))).export(path)

        with pytest.raises(ValueError, match="Not a triangular surface mesh"):
            import_scalp_mesh(path)

    def test_scene_rejected(self, tmp_path) -> None:
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.creation.icosphere(subdivisions=1))
        second = trimesh.creation.icosphere(subdivisions=1)
        second.apply_translation([10.0, 0.0, 0.0])
        scene.add_geometry(second)
        path = tmp_path / "scene.glb"
        scene.export(path)

        with pytest.raises(ValueError, match="Not a triangular surface mesh"):
            import_scalp_mesh(path)

    def test_missing_file_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="not a file"):
            import_scalp_mesh(tmp_path / "missing.ply")
