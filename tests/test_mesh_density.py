import pytest

from virda.mesh.density import step_size_for_voxel_size


class TestStepSizeForVoxelSize:
    def test_native_spacing_yields_step_one(self) -> None:
        step, real = step_size_for_voxel_size(1.0, (1.0, 1.0, 1.0))

        assert step == 1
        assert real == (1.0, 1.0, 1.0)

    def test_mean_spacing_yields_step_one_anisotropic(self) -> None:
        spacing = (1.0, 1.0, 1.3)
        mean = sum(spacing) / 3.0

        step, real = step_size_for_voxel_size(mean, spacing)

        assert step == 1
        assert real == (1.0, 1.0, 1.3)

    def test_larger_desired_size_scales_step(self) -> None:
        step, real = step_size_for_voxel_size(2.0, (1.0, 1.0, 1.0))

        assert step == 2
        assert real == (2.0, 2.0, 2.0)

    def test_real_voxel_is_step_times_spacing(self) -> None:
        spacing = (1.2, 1.4, 2.0)

        step, real = step_size_for_voxel_size(4.0, spacing)

        assert step == 3
        assert real == pytest.approx((3.6, 4.2, 6.0))

    def test_small_desired_size_clamps_to_one(self) -> None:
        step, real = step_size_for_voxel_size(0.1, (1.0, 1.0, 1.0))

        assert step == 1
        assert real == (1.0, 1.0, 1.0)

    def test_rejects_nonpositive_desired_size(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            step_size_for_voxel_size(0.0, (1.0, 1.0, 1.0))
        with pytest.raises(ValueError, match="positive"):
            step_size_for_voxel_size(-2.0, (1.0, 1.0, 1.0))

    def test_rejects_invalid_spacing(self) -> None:
        with pytest.raises(ValueError, match="spacing"):
            step_size_for_voxel_size(1.0, (1.0, 0.0, 1.0))
        with pytest.raises(ValueError, match="spacing"):
            step_size_for_voxel_size(1.0, (1.0, 1.0))
