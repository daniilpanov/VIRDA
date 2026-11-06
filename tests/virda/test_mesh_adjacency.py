import numpy as np
import pytest

from virda.mesh.adjacency import build_scalp_mesh, compute_face_adjacency
from virda.models.scalp_mesh import ScalpMesh


def tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [0, 1, 3],
            [1, 2, 3],
        ]
    )
    return vertices, faces


class TestComputeFaceAdjacency:
    def test_tetrahedron_has_six_edges(self) -> None:
        vertices, faces = tetrahedron()
        adjacency = compute_face_adjacency(vertices, faces)

        assert adjacency.shape == (6, 2)
        assert adjacency.dtype == np.int64
        assert set(map(frozenset, adjacency)) == {
            frozenset((0, 1)),
            frozenset((0, 2)),
            frozenset((0, 3)),
            frozenset((1, 2)),
            frozenset((1, 3)),
            frozenset((2, 3)),
        }

    def test_disconnected_triangles_have_no_adjacency(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [5.0, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]])

        adjacency = compute_face_adjacency(vertices, faces)

        assert adjacency.shape == (0, 2)


class TestBuildScalpMesh:
    def test_builds_mesh_with_matching_adjacency(self) -> None:
        vertices, faces = tetrahedron()
        mesh = build_scalp_mesh(vertices, faces)

        assert isinstance(mesh, ScalpMesh)
        assert np.array_equal(mesh.vertices, vertices)
        assert np.array_equal(mesh.faces, faces)
        assert mesh.face_adjacency.shape == (6, 2)

    def test_rejects_invalid_adjacency(self) -> None:
        vertices, faces = tetrahedron()
        with pytest.raises(ValueError, match="indices"):
            ScalpMesh(
                vertices=vertices,
                faces=faces,
                face_adjacency=np.array([[0, 99]]),
            )

    def test_rejects_bad_adjacency_shape(self) -> None:
        vertices, faces = tetrahedron()
        with pytest.raises(ValueError, match=r"\(E, 2\)"):
            ScalpMesh(
                vertices=vertices,
                faces=faces,
                face_adjacency=np.array([0, 1]),
            )
