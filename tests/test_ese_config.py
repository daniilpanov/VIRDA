import pytest

from virda.models.ese_config import ESEConfig


class TestESEConfig:
    def test_rejects_unknown_reference(self) -> None:
        with pytest.raises(ValueError, match="ese_reference must be one of"):
            ESEConfig(n_electrodes=32, ese_offset_mm=1.0, ese_reference="scalp_surface")

    @pytest.mark.parametrize("n_electrodes", [0, -5])
    def test_rejects_non_positive_electrode_count(self, n_electrodes: int) -> None:
        with pytest.raises(ValueError, match="n_electrodes must be positive"):
            ESEConfig(
                n_electrodes=n_electrodes, ese_offset_mm=1.0, ese_reference="electrode_body_center"
            )

    @pytest.mark.parametrize("ese_offset_mm", [0.0, -1.0])
    def test_rejects_non_positive_offset(self, ese_offset_mm: float) -> None:
        with pytest.raises(ValueError, match="ese_offset_mm must be positive"):
            ESEConfig(
                n_electrodes=32, ese_offset_mm=ese_offset_mm, ese_reference="electrode_body_center"
            )
