import numpy as np
import pytest

from virda.mesh.cleaners import MergeCleaner
from virda.mesh.mesh_extractor import MarchingCubesExtractor
from virda.models.scalp_mesh import ScalpMesh


def _sphere_mesh() -> ScalpMesh:
    grid = np.indices((20, 20, 20))
    center = np.array([10, 10, 10])
    squared_distance = np.sum((grid - center.reshape(-1, 1, 1, 1)) ** 2, axis=0)
    mask = squared_distance <= 8**2
    return MarchingCubesExtractor().extract(mask, np.eye(4))


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

    def test_cleaner_populates_face_adjacency(self) -> None:
        cleaned = MergeCleaner().clean(_sphere_mesh())

        assert cleaned.face_adjacency is not None
        assert cleaned.face_adjacency.ndim == 2
        assert cleaned.face_adjacency.shape[1] == 2
        assert cleaned.face_adjacency.max() < cleaned.faces.shape[0]
        assert cleaned.coordinate_system == "world"
