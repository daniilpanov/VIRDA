import json
from pathlib import Path

import numpy as np

from virda.io.exporters.electrodes import (
    export_electrodes,
    export_electrodes_ese,
    export_electrodes_scalp,
)
from virda.models.electrode import Electrode, Electrodes


def _localized() -> Electrode:
    return Electrode(
        electrode_id="Fz",
        measured_distances={"NAS": 110.5},
        ese_coords=np.array([1.0, 2.0, 3.0]),
        scalp_coords=np.array([1.0, 2.0, 2.5]),
        residual_error=1.25,
        confidence=0.9,
        flagged=False,
    )


def _unlocalized() -> Electrode:
    return Electrode(
        electrode_id="Cz",
        measured_distances={"NAS": 95.0},
        flagged=True,
    )


class TestExportElectrodes:
    def test_writes_full_entries(self, tmp_path: Path) -> None:
        electrodes = Electrodes(items=[_localized(), _unlocalized()])

        path = export_electrodes(tmp_path / "localization" / "electrodes.json", electrodes)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data[0] == {
            "electrode_id": "Fz",
            "measured_distances": {"NAS": 110.5},
            "ese_coords": [1.0, 2.0, 3.0],
            "scalp_coords": [1.0, 2.0, 2.5],
            "residual_error": 1.25,
            "confidence": 0.9,
            "flagged": False,
        }
        # Non-localized electrodes keep null coordinates and metrics.
        assert data[1]["ese_coords"] is None
        assert data[1]["scalp_coords"] is None
        assert data[1]["residual_error"] is None
        assert data[1]["flagged"] is True

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        path = export_electrodes(tmp_path / "deep" / "electrodes.json", Electrodes(items=[]))

        assert path.is_file()
        assert json.loads(path.read_text(encoding="utf-8")) == []


class TestExportElectrodesScalp:
    def test_coords_are_scalp_contact_points(self, tmp_path: Path) -> None:
        path = export_electrodes_scalp(
            tmp_path / "electrodes_scalp.json", Electrodes(items=[_localized()])
        )

        entry = json.loads(path.read_text(encoding="utf-8"))[0]
        assert entry["coords"] == [1.0, 2.0, 2.5]
        assert "ese_coords" not in entry
        assert "scalp_coords" not in entry
        assert entry["residual_error"] == 1.25
        assert entry["flagged"] is False


class TestExportElectrodesEse:
    def test_coords_are_ese_body_center_points(self, tmp_path: Path) -> None:
        path = export_electrodes_ese(
            tmp_path / "electrodes_ese.json", Electrodes(items=[_localized()])
        )

        entry = json.loads(path.read_text(encoding="utf-8"))[0]
        assert entry["coords"] == [1.0, 2.0, 3.0]
        assert entry["confidence"] == 0.9

    def test_unlocalized_electrode_has_null_coords(self, tmp_path: Path) -> None:
        path = export_electrodes_ese(
            tmp_path / "electrodes_ese.json", Electrodes(items=[_unlocalized()])
        )

        assert json.loads(path.read_text(encoding="utf-8"))[0]["coords"] is None
