import numpy as np
import pytest

from virda.models.electrode import Electrode, Electrodes


class TestElectrode:
    def test_defaults(self) -> None:
        electrode = Electrode(electrode_id="Fz", measured_distances={"NAS": 120.0})
        assert electrode.ese_coords is None
        assert electrode.scalp_coords is None
        assert electrode.residual_error is None
        assert electrode.confidence is None
        assert electrode.flagged is False
        assert electrode.is_localized is False

    def test_rejects_empty_id(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            Electrode(electrode_id="", measured_distances={"NAS": 120.0})

    def test_rejects_empty_measurements(self) -> None:
        with pytest.raises(ValueError, match="at least one measurement"):
            Electrode(electrode_id="Fz", measured_distances={})

    @pytest.mark.parametrize("distance", [0.0, -1.0])
    def test_rejects_non_positive_distance(self, distance: float) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            Electrode(electrode_id="Fz", measured_distances={"NAS": distance})

    def test_rejects_malformed_coords(self) -> None:
        with pytest.raises(ValueError, match="ese_coords must be a \\(3,\\) array"):
            Electrode(
                electrode_id="Fz",
                measured_distances={"NAS": 120.0},
                ese_coords=np.zeros(2),
            )

    def test_localized_flag(self) -> None:
        electrode = Electrode(
            electrode_id="Fz",
            measured_distances={"NAS": 120.0},
            ese_coords=np.array([1.0, 2.0, 3.0]),
            scalp_coords=np.array([1.0, 2.0, 3.0]),
            residual_error=0.5,
            confidence=0.9,
            flagged=True,
        )
        assert electrode.is_localized is True


class TestElectrodes:
    def test_require_unique_ids(self) -> None:
        with pytest.raises(ValueError, match="must be unique"):
            Electrodes(
                items=[
                    Electrode(electrode_id="Fz", measured_distances={"NAS": 120.0}),
                    Electrode(electrode_id="Fz", measured_distances={"NAS": 121.0}),
                ]
            )

    def test_get_returns_matching_electrode(self) -> None:
        electrodes = Electrodes(
            items=[
                Electrode(electrode_id="Fz", measured_distances={"NAS": 120.0}),
                Electrode(electrode_id="Cz", measured_distances={"NAS": 130.0}),
            ]
        )
        assert electrodes.ids == ["Fz", "Cz"]
        assert electrodes.get("Cz") is not None
        assert electrodes.get("Pz") is None
