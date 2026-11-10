import csv
from pathlib import Path

import numpy as np

from virda.io.exporters.electrodes import (
    export_electrodes,
    export_electrodes_ese,
    export_electrodes_scalp,
)
from virda.models.electrode import Electrode, Electrodes


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def _localized() -> Electrode:
    return Electrode(
        electrode_id="Fz",
        measured_distances={"NAS": 110.5, "LPA": 90.0},
        ese_coords=np.array([1.0, 2.0, 3.0]),
        scalp_coords=np.array([1.0, 2.0, 2.5]),
        residual_error=1.25,
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

        path = export_electrodes(tmp_path / "localization" / "electrodes.csv", electrodes)

        headers, rows = _read_rows(path)
        assert headers == [
            "electrode_id",
            "ese_x",
            "ese_y",
            "ese_z",
            "scalp_x",
            "scalp_y",
            "scalp_z",
            "nas",
            "lpa",
            "residual_mm",
            "flagged",
        ]
        assert rows[0] == {
            "electrode_id": "Fz",
            "ese_x": "1.0",
            "ese_y": "2.0",
            "ese_z": "3.0",
            "scalp_x": "1.0",
            "scalp_y": "2.0",
            "scalp_z": "2.5",
            "nas": "110.5",
            "lpa": "90.0",
            "residual_mm": "1.25",
            "flagged": "no",
        }
        # Non-localized electrodes keep empty coordinates and residual.
        assert rows[1]["electrode_id"] == "Cz"
        assert rows[1]["ese_x"] == ""
        assert rows[1]["scalp_z"] == ""
        assert rows[1]["nas"] == "95.0"
        assert rows[1]["lpa"] == ""
        assert rows[1]["residual_mm"] == ""
        assert rows[1]["flagged"] == "yes"

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        path = export_electrodes(tmp_path / "deep" / "electrodes.csv", Electrodes(items=[]))

        assert path.is_file()
        headers, rows = _read_rows(path)
        assert rows == []
        assert "electrode_id" in headers


class TestExportElectrodesScalp:
    def test_coords_are_scalp_contact_points(self, tmp_path: Path) -> None:
        path = export_electrodes_scalp(
            tmp_path / "electrodes_scalp.csv", Electrodes(items=[_localized()])
        )

        headers, rows = _read_rows(path)
        assert headers == ["electrode_id", "x", "y", "z", "residual_mm", "flagged"]
        assert rows[0]["x"] == "1.0"
        assert rows[0]["residual_mm"] == "1.25"
        assert rows[0]["flagged"] == "no"


class TestExportElectrodesEse:
    def test_coords_are_ese_body_center_points(self, tmp_path: Path) -> None:
        path = export_electrodes_ese(
            tmp_path / "electrodes_ese.csv", Electrodes(items=[_localized()])
        )

        headers, rows = _read_rows(path)
        assert headers == ["electrode_id", "x", "y", "z", "residual_mm", "flagged"]
        assert rows[0]["x"] == "1.0"

    def test_unlocalized_electrode_has_empty_coords(self, tmp_path: Path) -> None:
        path = export_electrodes_ese(
            tmp_path / "electrodes_ese.csv", Electrodes(items=[_unlocalized()])
        )

        _, rows = _read_rows(path)
        assert rows[0]["x"] == ""
        assert rows[0]["residual_mm"] == ""
