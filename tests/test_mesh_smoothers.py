import numpy as np
import pytest

from tests.helpers.pipelines import build_context
from virda.mesh.laplacian_smoother import LaplacianSmoother
from virda.mesh.taubin_smoother import TaubinSmoother
from virda.models.mri_volume import MRIVolume
from virda.models.scalp_mesh import ScalpMesh
from virda.models.segmentation_mask import SegmentationMask


@pytest.fixture
def sphere_volume() -> MRIVolume:
    volume_shape = (30, 30, 30)
    center = np.array([15, 15, 15])
    sphere_radius = 10

    grid_indices = np.indices(volume_shape)
    squared_distance_from_center = np.sum((grid_indices - center.reshape(-1, 1, 1, 1)) ** 2, axis=0)
    is_inside_sphere = squared_distance_from_center <= sphere_radius**2

    image_data = np.zeros(volume_shape, dtype=np.float32)
    image_data[is_inside_sphere] = 100.0

    zero_affine = np.eye(4)
    voxel_spacing = (1.0, 1.0, 1.0)
    orientation = ("R", "A", "S")

    return MRIVolume(
        data=image_data,
        affine=zero_affine,
        spacing=voxel_spacing,
        orientation=orientation,
    )


@pytest.fixture
def sphere_mesh(sphere_volume: MRIVolume) -> ScalpMesh:
    grid = np.indices((20, 20, 20))
    center = np.array([10, 10, 10])
    squared_distance = np.sum((grid - center.reshape(-1, 1, 1, 1)) ** 2, axis=0)
    mask = SegmentationMask(mask=squared_distance <= 8**2)

    from virda.mesh.mesh_extractor import MarchingCubesExtractor

    return MarchingCubesExtractor().run(
        build_context(SegmentationMask=mask, MRIVolume=sphere_volume)
    )


class TestLaplacianSmoother:
    def test_smooth_preserves_mesh_structure(self, sphere_mesh: ScalpMesh) -> None:
        smoother = LaplacianSmoother(iterations=3, lamb=0.5)
        smoothed = smoother.run(build_context(ScalpMesh=sphere_mesh))

        assert isinstance(smoothed, ScalpMesh)
        assert smoothed.vertices.shape == sphere_mesh.vertices.shape
        assert smoothed.faces.shape == sphere_mesh.faces.shape
        assert np.array_equal(smoothed.faces, sphere_mesh.faces)

    def test_smooth_moves_vertices(self, sphere_mesh: ScalpMesh) -> None:
        smoother = LaplacianSmoother(iterations=5, lamb=0.5)
        smoothed = smoother.run(build_context(ScalpMesh=sphere_mesh))

        vertex_displacement = np.abs(smoothed.vertices - sphere_mesh.vertices).max()
        assert vertex_displacement > 1e-7
        assert vertex_displacement < 5.0


class TestTaubinSmoother:
    def test_smooth_preserves_mesh_structure(self, sphere_mesh: ScalpMesh) -> None:
        smoother = TaubinSmoother(iterations=3, lamb=0.5, nu=-0.53)
        smoothed = smoother.run(build_context(ScalpMesh=sphere_mesh))

        assert isinstance(smoothed, ScalpMesh)
        assert smoothed.vertices.shape == sphere_mesh.vertices.shape
        assert smoothed.faces.shape == sphere_mesh.faces.shape
        assert np.array_equal(smoothed.faces, sphere_mesh.faces)

    def test_smooth_moves_vertices(self, sphere_mesh: ScalpMesh) -> None:
        smoother = TaubinSmoother(iterations=5, lamb=0.5, nu=-0.53)
        smoothed = smoother.run(build_context(ScalpMesh=sphere_mesh))

        vertex_displacement = np.abs(smoothed.vertices - sphere_mesh.vertices).max()
        assert vertex_displacement > 1e-7
        assert vertex_displacement < 5.0
