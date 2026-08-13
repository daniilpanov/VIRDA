import numpy as np
import pytest

from tests.helpers.pipelines import build_context
from virda.models.mri_volume import MRIVolume
from virda.segmentation.head_segmenter import OtsuHeadSegmenter


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

    voxel_to_world_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    voxel_spacing = (1.0, 1.0, 1.0)
    orientation = ("R", "A", "S")

    return MRIVolume(
        data=image_data,
        affine=voxel_to_world_affine,
        spacing=voxel_spacing,
        orientation=orientation,
    )


class TestHeadSegmenter:
    @pytest.fixture(autouse=True)
    def setup_segmenter(self) -> None:
        self.segmenter = OtsuHeadSegmenter(closing_radius=0)

    def test_segment_sphere_returns_bool_mask_with_correct_shape(
        self, sphere_volume: MRIVolume
    ) -> None:
        segmentation_mask = self.segmenter.run(build_context(MRIVolume=sphere_volume))

        assert segmentation_mask.mask.shape == (30, 30, 30)
        assert segmentation_mask.mask.dtype == bool

    def test_segment_sphere_covers_all_foreground_voxels(self, sphere_volume: MRIVolume) -> None:
        sphere_center = np.array([15, 15, 15])
        sphere_radius = 10
        grid_indices = np.indices((30, 30, 30))
        squared_distance = np.sum((grid_indices - sphere_center.reshape(-1, 1, 1, 1)) ** 2, axis=0)
        expected_inside = squared_distance <= sphere_radius**2

        segmentation_mask = self.segmenter.run(build_context(MRIVolume=sphere_volume))

        assert np.all(segmentation_mask.mask[expected_inside])
        assert segmentation_mask.mask.sum() == expected_inside.sum()

    def test_segment_sphere_excludes_background(self, sphere_volume: MRIVolume) -> None:
        sphere_center = np.array([15, 15, 15])
        sphere_radius = 10
        grid_indices = np.indices((30, 30, 30))
        squared_distance = np.sum((grid_indices - sphere_center.reshape(-1, 1, 1, 1)) ** 2, axis=0)
        expected_outside = squared_distance > sphere_radius**2

        segmentation_mask = self.segmenter.run(build_context(MRIVolume=sphere_volume))

        assert not np.any(segmentation_mask.mask[expected_outside])

    def test_all_zero_volume_returns_empty_mask(self) -> None:
        empty_data = np.zeros((10, 10, 10), dtype=np.float32)
        volume = MRIVolume(
            data=empty_data,
            affine=np.eye(4),
            spacing=(1.0, 1.0, 1.0),
            orientation=("R", "A", "S"),
        )

        segmentation_mask = self.segmenter.run(build_context(MRIVolume=volume))

        assert segmentation_mask.mask.shape == (10, 10, 10)
        assert segmentation_mask.mask.dtype == bool
        assert not np.any(segmentation_mask.mask)
