import numpy as np
import pytest

from virda.io.exporters.mesh_arrays import export_mesh_faces, export_mesh_vertices


class TestExportMeshVertices:
    def test_writes_float64_array(self, tmp_path) -> None:
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])

        path = export_mesh_vertices(tmp_path / "vertices.npy", vertices)

        loaded = np.load(path)
        np.testing.assert_array_almost_equal(loaded, vertices)
        assert loaded.dtype == np.float64

    def test_appends_npy_suffix(self, tmp_path) -> None:
        path = export_mesh_vertices(tmp_path / "vertices", np.zeros((1, 3)))

        assert path.name == "vertices.npy"
        assert path.is_file()

    def test_creates_parent_directory(self, tmp_path) -> None:
        path = export_mesh_vertices(tmp_path / "deep" / "dir" / "vertices.npy", np.zeros((1, 3)))

        assert path.is_file()

    def test_rejects_wrong_shapes(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="vertices"):
            export_mesh_vertices(tmp_path / "flat.npy", np.zeros(3))
        with pytest.raises(ValueError, match="vertices"):
            export_mesh_vertices(tmp_path / "four.npy", np.zeros((4, 4)))


class TestExportMeshFaces:
    def test_writes_int64_array(self, tmp_path) -> None:
        faces = np.array([[0, 1, 2], [1, 3, 2]])

        path = export_mesh_faces(tmp_path / "faces.npy", faces)

        loaded = np.load(path)
        np.testing.assert_array_equal(loaded, faces)
        assert loaded.dtype == np.int64

    def test_float_indices_cast_to_int(self, tmp_path) -> None:
        path = export_mesh_faces(tmp_path / "faces.npy", np.array([[0.0, 1.0, 2.0]]))

        loaded = np.load(path)
        assert loaded.dtype == np.int64
        np.testing.assert_array_equal(loaded, [[0, 1, 2]])

    def test_non_integer_float_indices_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="integers"):
            export_mesh_faces(tmp_path / "faces.npy", np.array([[0.5, 1.0, 2.0]]))

    def test_appends_npy_suffix_and_creates_parents(self, tmp_path) -> None:
        path = export_mesh_faces(tmp_path / "deep" / "faces", np.array([[0, 1, 2]]))

        assert path.name == "faces.npy"
        assert path.is_file()

    def test_rejects_wrong_shapes(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="faces"):
            export_mesh_faces(tmp_path / "flat.npy", np.zeros(3))
        with pytest.raises(ValueError, match="faces"):
            export_mesh_faces(tmp_path / "quad.npy", np.zeros((2, 4)))
