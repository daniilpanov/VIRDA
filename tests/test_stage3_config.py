import pytest

from virda.models.stage3_config import Stage3Config


class TestStage3Config:
    def test_defaults(self) -> None:
        config = Stage3Config()
        assert config.residual_threshold_mm == 10.0

    @pytest.mark.parametrize("threshold", [0.0, -1.0])
    def test_rejects_non_positive_threshold(self, threshold: float) -> None:
        with pytest.raises(ValueError, match="residual_threshold_mm must be positive"):
            Stage3Config(residual_threshold_mm=threshold)
