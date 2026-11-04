from pathlib import Path

import numpy as np
import trimesh

from tests.helpers.measurements import make_ese
from virda.io.exporters.ese_mesh import export_ese_mesh
from virda.io.importers.scalp_mesh import import_scalp_mesh
from virda.models.ese_mesh import ESEMesh


class TestExportEseMesh:
    def test_round_trip_through_surface_reader(self, tmp_path: Path) -> None:
        ese = make_ese()

        path = export_ese_mesh(tmp_path / "out" / "mesh.ply", ese)
        loaded = import_scalp_mesh(path)

        np.testing.assert_array_almost_equal(loaded.faces, ese.faces)
        np.testing.assert_array_almost_equal(loaded.vertices, ese.vertices)

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        path = export_ese_mesh(tmp_path / "deep" / "ese" / "mesh.ply", make_ese())

        assert path == tmp_path / "deep" / "ese" / "mesh.ply"
        assert path.is_file()

    def test_ply_carries_triangular_surface(self, tmp_path: Path) -> None:
        ese = make_ese()

        path = export_ese_mesh(tmp_path / "mesh.ply", ese)

        raw = trimesh.load(path)
        assert isinstance(raw, trimesh.Trimesh)
        assert raw.faces.shape == ese.faces.shape

    def test_custom_mesh(self, tmp_path: Path) -> None:
        ese = ESEMesh(
            vertices=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
            scalp_vertices=np.array(
                [[0.0, 0.0, -2.0], [1.0, 0.0, -2.0], [0.0, 1.0, -2.0]], dtype=np.float64
            ),
            normals=np.tile(np.array([0.0, 0.0, 1.0]), (3, 1)),
            quality=np.ones(3),
        )

        path = export_ese_mesh(tmp_path / "mesh.ply", ese)

        raw = trimesh.load(path)
        assert isinstance(raw, trimesh.Trimesh)
        np.testing.assert_array_almost_equal(raw.vertices, ese.vertices)
