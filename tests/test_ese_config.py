import pytest

from virda.models.ese_config import ESEConfig


class TestESEConfig:
    @pytest.mark.parametrize("ese_offset_mm", [0.0, -1.0])
    def test_rejects_non_positive_offset(self, ese_offset_mm: float) -> None:
        with pytest.raises(ValueError, match="ese_offset_mm must be positive"):
            ESEConfig(ese_offset_mm=ese_offset_mm)
