import numpy as np
import pytest

from virda.mesh.cleaners import (
    LargestComponentCleaner,
    MergeCleaner,
    RayCastCleaner,
    SubdivideCleaner,
)
from virda.mesh.contracts import MeshCleaner
from virda.mesh.mesh_extractor import MarchingCubesExtractor
from virda.models.scalp_mesh import ScalpMesh


@pytest.fixture
def sphere_mesh() -> ScalpMesh:
    grid = np.indices((20, 20, 20))
    center = np.array([10, 10, 10])
    squared_distance = np.sum((grid - center.reshape(-1, 1, 1, 1)) ** 2, axis=0)
    mask = squared_distance <= 8**2
    return MarchingCubesExtractor().extract(mask, np.eye(4))


def _box_with_inner_cavity_mesh() -> ScalpMesh:
    """Solid 64^3 box with a spherical cavity open to the exterior via a tunnel."""
    n = 64
    center = np.array([32.0, 32.0, 32.0])
    grid = np.indices((n, n, n)).astype(float)
    radius = np.linalg.norm(grid - center.reshape(-1, 1, 1, 1), axis=0)
    cavity = radius <= 16.0
    tunnel = (
        (np.abs(grid[0] - 32) <= 4)
        & (np.abs(grid[1] - 32) <= 4)
        & (grid[2] >= 32)
        & (grid[2] <= 56)
    )
    mask = np.ones((n, n, n), dtype=bool)
    mask &= ~(cavity | tunnel)
    mask[:8, :, :] = False
    mask[-8:, :, :] = False
    mask[:, :8, :] = False
    mask[:, -8:, :] = False
    mask[:, :, :8] = False
    mask[:, :, -8:] = False
    return MarchingCubesExtractor().extract(mask, np.eye(4))


def _cavity_shell_faces(mesh: ScalpMesh, center: np.ndarray) -> np.ndarray:
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    distance = np.linalg.norm(centroids - center, axis=1)
    return np.asarray((distance >= 14.0) & (distance <= 20.0))


class TestCleanersImplementProtocol:
    @pytest.mark.parametrize(
        "cleaner",
        [
            MergeCleaner(),
            RayCastCleaner(),
            SubdivideCleaner(max_edge=5.0),
            LargestComponentCleaner(),
        ],
    )
    def test_is_mesh_cleaner(self, cleaner: object) -> None:
        assert isinstance(cleaner, MeshCleaner)


class TestMergeCleaner:
    def test_merges_duplicate_vertices(self, sphere_mesh: ScalpMesh) -> None:
        duplicated = ScalpMesh(
            vertices=np.vstack([sphere_mesh.vertices, sphere_mesh.vertices]),
            faces=np.vstack([sphere_mesh.faces, sphere_mesh.faces + len(sphere_mesh.vertices)]),
        )

        cleaned = MergeCleaner().clean(duplicated)

        assert cleaned.faces.shape[1] == 3
        assert cleaned.faces.min() >= 0
        assert cleaned.faces.max() < cleaned.vertices.shape[0]
        assert cleaned.face_adjacency is not None
        assert cleaned.coordinate_system == "world"

    def test_removes_degenerate_faces(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        mesh = ScalpMesh(vertices=vertices, faces=faces)

        cleaned = MergeCleaner().clean(mesh)

        assert cleaned.faces.shape[0] <= 2


class TestLargestComponentCleaner:
    def test_removes_small_island(self, sphere_mesh: ScalpMesh) -> None:
        island_vertices = np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.0, 0.0]])
        island_faces = np.array([[0, 1, 2]])
        combined_vertices = np.vstack([sphere_mesh.vertices, island_vertices])
        combined_faces = np.vstack(
            [sphere_mesh.faces, island_faces + sphere_mesh.vertices.shape[0]]
        )
        corrupted_mesh = ScalpMesh(vertices=combined_vertices, faces=combined_faces)

        cleaned = LargestComponentCleaner(min_vertices=50).clean(corrupted_mesh)

        assert cleaned.vertices.shape[0] <= sphere_mesh.vertices.shape[0] + 3
        assert cleaned.faces.shape[0] <= sphere_mesh.faces.shape[0]


class TestRayCastCleaner:
    def test_removes_cavity_walls(self) -> None:
        mesh = _box_with_inner_cavity_mesh()
        center = np.array([32.0, 32.0, 32.0])
        assert _cavity_shell_faces(mesh, center).sum() > 1000

        cleaned = RayCastCleaner(region=None).clean(mesh)

        assert _cavity_shell_faces(cleaned, center).sum() < 50
        assert cleaned.vertices.shape[0] > 1000

    def test_preserves_outer_surface(self) -> None:
        mesh = _box_with_inner_cavity_mesh()
        center = np.array([32.0, 32.0, 32.0])
        outer_before = int(
            (np.linalg.norm(mesh.vertices[mesh.faces].mean(axis=1) - center, axis=1) > 20).sum()
        )

        cleaned = RayCastCleaner(region=None).clean(mesh)
        face_centers = cleaned.vertices[cleaned.faces].mean(axis=1)
        outer_after = int((np.linalg.norm(face_centers - center, axis=1) > 20).sum())

        assert outer_after > outer_before * 0.9


class TestSubdivideCleaner:
    def test_subdivides_to_max_edge(self, sphere_mesh: ScalpMesh) -> None:
        cleaned = SubdivideCleaner(max_edge=1.0).clean(sphere_mesh)

        assert cleaned.vertices.shape[0] > sphere_mesh.vertices.shape[0]
        assert cleaned.faces.shape[0] > sphere_mesh.faces.shape[0]
        assert cleaned.face_adjacency is not None
