import csv
from pathlib import Path

import numpy as np

from virda.io.exporters.localization_table import export_localization_table
from virda.models.electrode import Electrode, Electrodes


def _localized_electrode() -> Electrode:
    return Electrode(
        electrode_id="Fz",
        measured_distances={"NAS": 110.5, "LPA": 120.0, "RPA": 125.0},
        ese_coords=np.array([10.0, 0.0, 120.0]),
        scalp_coords=np.array([10.0, 0.0, 118.0]),
        residual_error=1.25,
        flagged=False,
    )


class TestExportLocalizationTable:
    def test_writes_localized_rows_with_head_frame(self, tmp_path: Path) -> None:
        electrodes = Electrodes(items=[_localized_electrode()])
        cras_offset = np.array([5.0, 0.0, 10.0])

        path = export_localization_table(tmp_path / "table.csv", electrodes, cras_offset)

        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        assert len(rows) == 1
        row = rows[0]
        assert row["name"] == "Fz"
        assert [row[key] for key in ("world_x", "world_y", "world_z")] == [
            "10.0",
            "0.0",
            "120.0",
        ]
        assert [row[key] for key in ("head_x", "head_y", "head_z")] == ["5.0", "0.0", "110.0"]
        assert row["nas"] == "110.5"
        assert row["lpa"] == "120.0"
        assert row["rpa"] == "125.0"
        assert row["residual_mm"] == "1.25"
        assert row["flagged"] == "no"

    def test_without_offset_omits_head_columns(self, tmp_path: Path) -> None:
        electrodes = Electrodes(items=[_localized_electrode()])

        path = export_localization_table(tmp_path / "table.csv", electrodes)

        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        header = rows[0].keys()
        assert "head_x" not in header
        assert "head_y" not in header
        assert "head_z" not in header

    def test_unlocalized_electrodes_left_blank(self, tmp_path: Path) -> None:
        electrodes = Electrodes(
            items=[
                Electrode(
                    electrode_id="Cz",
                    measured_distances={"NAS": 100.0},
                )
            ]
        )

        path = export_localization_table(tmp_path / "table.csv", electrodes)

        row = next(csv.DictReader(path.open(newline="", encoding="utf-8")))
        assert row["world_x"] == ""
        assert row["residual_mm"] == ""
        assert row["flagged"] == "no"

    def test_flagged_electrode_marked(self, tmp_path: Path) -> None:
        electrodes = Electrodes(
            items=[
                Electrode(
                    electrode_id="Cz",
                    measured_distances={"NAS": 100.0},
                    ese_coords=np.array([0.0, 0.0, 100.0]),
                    scalp_coords=np.array([0.0, 0.0, 98.0]),
                    flagged=True,
                )
            ]
        )

        path = export_localization_table(tmp_path / "table.csv", electrodes)

        row = next(csv.DictReader(path.open(newline="", encoding="utf-8")))
        assert row["flagged"] == "yes"

    def test_missing_distances_left_blank(self, tmp_path: Path) -> None:
        electrodes = Electrodes(
            items=[
                Electrode(
                    electrode_id="Cz",
                    measured_distances={"NAS": 100.0},
                    ese_coords=np.array([0.0, 0.0, 100.0]),
                    scalp_coords=np.array([0.0, 0.0, 98.0]),
                )
            ]
        )

        path = export_localization_table(tmp_path / "table.csv", electrodes)

        row = next(csv.DictReader(path.open(newline="", encoding="utf-8")))
        assert row["nas"] == "100.0"
        assert row["lpa"] == ""
