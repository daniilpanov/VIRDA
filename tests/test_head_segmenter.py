import numpy as np
import pytest

from tests.helpers.pipelines import build_context
from virda.models.mri_volume import MRIVolume
from virda.segmentation.head_segmenter import OtsuHeadSegmenter


def _volume_from_flat_data(flat_data: np.ndarray, shape: tuple[int, int, int]) -> MRIVolume:
    return MRIVolume(
        data=flat_data.reshape(shape).astype(np.float32),
        affine=np.eye(4),
        spacing=(1.0, 1.0, 1.0),
        orientation=("R", "A", "S"),
    )


@pytest.fixture
def three_level_volume() -> MRIVolume:
    """Background 0 (dominant), low tissue 40 (majority of foreground),
    high tissue 100 (rare). Reproduces the real-data histogram shape where
    background air drags the global Otsu threshold down to the background edge."""
    total = 60 * 60 * 60
    background = int(total * 0.7)
    low_tissue = int(total * 0.29)
    flat_data = np.zeros(total, dtype=np.float32)
    flat_data[background : background + low_tissue] = 40.0
    flat_data[background + low_tissue :] = 100.0
    return _volume_from_flat_data(flat_data, (60, 60, 60))


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


class TestOtsuScopeAndScale:
    N_LOW = int(60 * 60 * 60 * 0.29)
    N_HIGH = 60 * 60 * 60 - int(60 * 60 * 60 * 0.7) - int(60 * 60 * 60 * 0.29)

    def test_scope_all_keeps_low_intensity_tissue(self, three_level_volume: MRIVolume) -> None:
        segmenter = OtsuHeadSegmenter(closing_radius=0, otsu_scope="all")

        mask = segmenter.run(build_context(MRIVolume=three_level_volume)).mask

        assert mask.sum() == self.N_LOW + self.N_HIGH

    def test_scope_foreground_excludes_low_intensity_tissue(
        self, three_level_volume: MRIVolume
    ) -> None:
        segmenter = OtsuHeadSegmenter(
            closing_radius=0, otsu_scope="foreground", threshold_scale=1.0
        )

        mask = segmenter.run(build_context(MRIVolume=three_level_volume)).mask

        assert mask.sum() == self.N_HIGH

    def test_threshold_scale_brings_low_intensity_tissue_back(
        self, three_level_volume: MRIVolume
    ) -> None:
        segmenter = OtsuHeadSegmenter(
            closing_radius=0, otsu_scope="foreground", threshold_scale=0.4
        )

        mask = segmenter.run(build_context(MRIVolume=three_level_volume)).mask

        assert mask.sum() == self.N_LOW + self.N_HIGH

    def test_foreground_scope_matches_all_scope_on_uniform_sphere(
        self, sphere_volume: MRIVolume
    ) -> None:
        foreground = OtsuHeadSegmenter(
            closing_radius=0, otsu_scope="foreground", threshold_scale=1.0
        )
        all_scope = OtsuHeadSegmenter(closing_radius=0, otsu_scope="all")

        foreground_mask = foreground.run(build_context(MRIVolume=sphere_volume)).mask
        all_mask = all_scope.run(build_context(MRIVolume=sphere_volume)).mask

        np.testing.assert_array_equal(foreground_mask, all_mask)

    def test_invalid_scope_raises(self) -> None:
        with pytest.raises(ValueError, match="otsu_scope"):
            OtsuHeadSegmenter(otsu_scope="bogus")  # type: ignore[arg-type]

    def test_nonpositive_threshold_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold_scale"):
            OtsuHeadSegmenter(threshold_scale=0)
