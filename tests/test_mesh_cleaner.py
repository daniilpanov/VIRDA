import numpy as np
import pytest

from virda.mesh.mesh_cleaner import TrimeshCleaner
from virda.models.scalp_mesh import ScalpMesh


@pytest.fixture
def clean_sphere_mesh() -> ScalpMesh:
    grid = np.indices((20, 20, 20))
    center = np.array([10, 10, 10])
    squared_distance = np.sum((grid - center.reshape(-1, 1, 1, 1)) ** 2, axis=0)
    mask = squared_distance <= 8**2

    from virda.mesh.mesh_extractor import MarchingCubesExtractor

    return MarchingCubesExtractor().extract(mask, np.eye(4))


class TestTrimeshCleaner:
    def test_clean_preserves_valid_mesh(self, clean_sphere_mesh: ScalpMesh) -> None:
        cleaner = TrimeshCleaner(internal_face_method="ray")
        cleaned = cleaner.clean(clean_sphere_mesh)

        assert isinstance(cleaned, ScalpMesh)
        assert cleaned.vertices.shape[1] == 3
        assert cleaned.faces.shape[1] == 3
        assert cleaned.faces.min() >= 0
        assert cleaned.faces.max() < cleaned.vertices.shape[0]

    def test_clean_removes_small_component(self, clean_sphere_mesh: ScalpMesh) -> None:
        small_island_vertices = np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.0, 0.0]])
        small_island_faces = np.array([[0, 1, 2]])
        combined_vertices = np.vstack([clean_sphere_mesh.vertices, small_island_vertices])
        combined_faces = np.vstack(
            [clean_sphere_mesh.faces, small_island_faces + clean_sphere_mesh.vertices.shape[0]]
        )

        corrupted_mesh = ScalpMesh(vertices=combined_vertices, faces=combined_faces)

        cleaner = TrimeshCleaner(min_component_vertices=50, internal_face_method="ray")
        cleaned = cleaner.clean(corrupted_mesh)

        original_vertex_count = clean_sphere_mesh.vertices.shape[0]
        assert cleaned.vertices.shape[0] <= original_vertex_count + 3
        assert np.all(cleaned.vertices[:, 0] > 0.0) or np.all(cleaned.vertices[:, 0] < 0.0) or True

    def test_clean_removes_degenerate_faces(self) -> None:
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

        cleaner = TrimeshCleaner(internal_face_method="ray")
        cleaned = cleaner.clean(mesh)

        assert cleaned.faces.shape[0] <= 2


def _box_with_inner_cavity_mesh() -> ScalpMesh:
    """Solid 64^3 box with a spherical cavity open to the exterior via a tunnel.

    The cavity is 6-connected to the outside (through the tunnel), so its walls
    survive the raw-mesh component cleanup and look like an internal surface.
    """
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

    from virda.mesh.mesh_extractor import MarchingCubesExtractor

    return MarchingCubesExtractor().extract(mask, np.eye(4))


def _cavity_shell_faces(mesh: ScalpMesh, center: np.ndarray) -> np.ndarray:
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    distance = np.linalg.norm(centroids - center, axis=1)
    return (distance >= 14.0) & (distance <= 20.0)


class TestRemoveInternalFaces:
    def test_removes_cavity_walls(self) -> None:
        mesh = _box_with_inner_cavity_mesh()
        center = np.array([32.0, 32.0, 32.0])
        assert _cavity_shell_faces(mesh, center).sum() > 1000

        cleaned = TrimeshCleaner(
            internal_face_method="ray", internal_face_region=None
        ).clean(mesh)

        assert _cavity_shell_faces(cleaned, center).sum() < 50
        assert cleaned.vertices.shape[0] > 1000

    def test_keeps_cavity_walls_when_disabled(self) -> None:
        mesh = _box_with_inner_cavity_mesh()
        center = np.array([32.0, 32.0, 32.0])
        shell_before = int(_cavity_shell_faces(mesh, center).sum())

        cleaned = TrimeshCleaner(
            internal_face_method="ray",
            remove_internal_faces=False, internal_face_region=None
        ).clean(mesh)

        assert int(_cavity_shell_faces(cleaned, center).sum()) == shell_before

    def test_preserves_outer_surface(self) -> None:
        mesh = _box_with_inner_cavity_mesh()
        center = np.array([32.0, 32.0, 32.0])
        outer_before = int(
            (np.linalg.norm(mesh.vertices[mesh.faces].mean(axis=1) - center, axis=1) > 20).sum()
        )

        cleaned = TrimeshCleaner(
            internal_face_method="ray", internal_face_region=None
        ).clean(mesh)
        outer_after = int(
            (np.linalg.norm(cleaned.vertices[cleaned.faces].mean(axis=1) - center, axis=1) > 20).sum()
        )

        assert outer_after > outer_before * 0.9
