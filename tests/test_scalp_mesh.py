import numpy as np
import pytest

from virda.models.scalp_mesh import ScalpMesh


class TestScalpMesh:
    def test_default_fields(self) -> None:
        mesh = ScalpMesh(
            vertices=np.zeros((3, 3), dtype=np.float64),
            faces=np.zeros((1, 3), dtype=np.int64),
        )
        assert mesh.face_adjacency is None
        assert mesh.coordinate_system == "world"
        assert mesh.metadata == {}

    def test_accepts_face_adjacency(self) -> None:
        mesh = ScalpMesh(
            vertices=np.zeros((3, 3), dtype=np.float64),
            faces=np.zeros((1, 3), dtype=np.int64),
            face_adjacency=np.zeros((0, 2), dtype=np.int64),
        )
        assert mesh.face_adjacency is not None
        assert mesh.face_adjacency.shape == (0, 2)

    def test_invalid_face_adjacency_raises(self) -> None:
        with pytest.raises(ValueError, match="Face adjacency"):
            ScalpMesh(
                vertices=np.zeros((3, 3), dtype=np.float64),
                faces=np.zeros((1, 3), dtype=np.int64),
                face_adjacency=np.zeros((4, 3), dtype=np.int64),
            )

    def test_invalid_vertices_raises(self) -> None:
        with pytest.raises(ValueError, match="Vertices"):
            ScalpMesh(
                vertices=np.zeros((3, 2), dtype=np.float64),
                faces=np.zeros((1, 3), dtype=np.int64),
            )
