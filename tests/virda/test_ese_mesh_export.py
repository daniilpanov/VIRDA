from pathlib import Path

import numpy as np
import pytest
import trimesh

from tests.helpers.measurements import make_ese
from virda.io.exporters.ese_mesh import (
    export_ese_companions,
    export_ese_mesh,
    export_ese_normals,
)
from virda.io.importers.ese_mesh import import_ese_mesh, import_ese_normals
from virda.io.importers.scalp_mesh import import_scalp_mesh
from virda.models.mesh import Mesh


class TestExportEseNormals:
    def test_the_normals_are_written_to_the_given_file(self, tmp_path: Path) -> None:
        normals = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

        path = export_ese_normals(tmp_path / "ese" / "normals.npy", normals)

        assert path == tmp_path / "ese" / "normals.npy"
        assert path.is_file()

    def test_the_normals_round_trip(self, tmp_path: Path) -> None:
        normals = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        export_ese_normals(tmp_path / "ese" / "normals.npy", normals)

        loaded = import_ese_normals(tmp_path / "ese" / "normals.npy")

        np.testing.assert_array_almost_equal(loaded, normals)

    def test_the_file_that_is_named_is_the_file_that_is_read(self, tmp_path: Path) -> None:
        """The importer reads the file it is given, not a sibling of it."""
        normals = np.array([[0.0, 0.0, 1.0]])
        target = tmp_path / "ese" / "custom.npy"
        export_ese_normals(target, normals)

        loaded = import_ese_normals(target)

        np.testing.assert_array_almost_equal(loaded, normals)

    def test_the_missing_normals_are_reported(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Cannot import the ESE normals"):
            import_ese_normals(tmp_path / "ese" / "normals.npy")


class TestExportEseMesh:
    def test_round_trip_through_surface_reader(self, tmp_path: Path) -> None:
        ese, _, _ = make_ese()

        path = export_ese_mesh(tmp_path / "out" / "mesh.ply", ese)
        loaded = import_scalp_mesh(path)

        np.testing.assert_array_almost_equal(loaded.faces, ese.faces)
        np.testing.assert_array_almost_equal(loaded.vertices, ese.vertices)

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        path = export_ese_mesh(tmp_path / "deep" / "ese" / "mesh.ply", make_ese()[0])

        assert path == tmp_path / "deep" / "ese" / "mesh.ply"
        assert path.is_file()

    def test_ply_carries_triangular_surface(self, tmp_path: Path) -> None:
        ese, _, _ = make_ese()

        path = export_ese_mesh(tmp_path / "mesh.ply", ese)

        raw = trimesh.load(path)
        assert isinstance(raw, trimesh.Trimesh)
        assert raw.faces.shape == ese.faces.shape

    def test_custom_mesh(self, tmp_path: Path) -> None:
        ese = Mesh(
            vertices=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )

        path = export_ese_mesh(tmp_path / "mesh.ply", ese)

        raw = trimesh.load(path)
        assert isinstance(raw, trimesh.Trimesh)
        np.testing.assert_array_almost_equal(raw.vertices, ese.vertices)


class TestExportEseCompanions:
    def test_writes_the_expected_files(self, tmp_path: Path) -> None:
        ese, _, _ = make_ese()

        root = export_ese_companions(tmp_path / "ese" / "mesh.ply", ese)

        assert root == tmp_path / "ese"
        for name in (
            "ese_vertices.npy",
            "ese_faces.npy",
        ):
            assert (root / name).is_file()

    def test_full_round_trip_with_importer(self, tmp_path: Path) -> None:
        ese = make_ese(ese_offset_mm=1.5)[0]

        export_ese_companions(tmp_path / "ese" / "mesh.ply", ese)
        (tmp_path / "ese" / "mesh.ply").touch()
        restored = import_ese_mesh(tmp_path / "ese" / "mesh.ply")

        np.testing.assert_array_almost_equal(restored.vertices, ese.vertices)
        np.testing.assert_array_equal(restored.faces, ese.faces)
