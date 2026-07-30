import numpy as np
import pytest

from virda.mesh.mesh_extractor import MarchingCubesExtractor
from virda.models.scalp_mesh import ScalpMesh


@pytest.fixture
def sphere_mask() -> np.ndarray:
    volume_shape = (30, 30, 30)
    center = np.array([15, 15, 15])
    sphere_radius = 10
    grid_indices = np.indices(volume_shape)
    squared_distance = np.sum((grid_indices - center.reshape(-1, 1, 1, 1)) ** 2, axis=0)

    return squared_distance <= sphere_radius**2


class TestMeshExtractor:
    @pytest.fixture(autouse=True)
    def setup_extractor(self) -> None:
        self.extractor = MarchingCubesExtractor()

    def test_extract_surface_returns_valid_mesh(self, sphere_mask: np.ndarray) -> None:
        affine = np.eye(4)
        mesh = self.extractor.extract(sphere_mask, affine)

        assert isinstance(mesh, ScalpMesh)
        assert mesh.vertices.ndim == 2
        assert mesh.vertices.shape[1] == 3
        assert mesh.faces.ndim == 2
        assert mesh.faces.shape[1] == 3
        assert mesh.faces.min() >= 0
        assert mesh.faces.max() < mesh.vertices.shape[0]

    def test_extract_surface_produces_reasonable_vertex_count(
        self, sphere_mask: np.ndarray
    ) -> None:
        affine = np.eye(4)
        mesh = self.extractor.extract(sphere_mask, affine)

        sphere_surface_area_pixels = 4 * np.pi * 10**2
        expected_vertex_range = (
            int(sphere_surface_area_pixels * 0.3),
            int(sphere_surface_area_pixels * 2),
        )

        assert expected_vertex_range[0] < mesh.vertices.shape[0] < expected_vertex_range[1]

    def test_extract_surface_transforms_vertices_with_affine(self, sphere_mask: np.ndarray) -> None:
        voxel_to_world = np.array(
            [
                [1.5, 0.0, 0.0, 10.0],
                [0.0, 1.5, 0.0, 20.0],
                [0.0, 0.0, 2.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        identity_affine = np.eye(4)

        mesh_world = self.extractor.extract(sphere_mask, voxel_to_world)
        mesh_voxel = self.extractor.extract(sphere_mask, identity_affine)

        expected_world = mesh_voxel.vertices @ voxel_to_world[:3, :3].T + voxel_to_world[:3, 3]
        np.testing.assert_array_almost_equal(mesh_world.vertices, expected_world)

    def test_extract_surface_raises_on_invalid_mask(self) -> None:
        empty_mask = np.zeros((10, 10, 10), dtype=bool)
        affine = np.eye(4)

        with pytest.raises(ValueError):
            self.extractor.extract(empty_mask, affine)
