"""Unit tests for :mod:`virda.io.importers.electrode_points`.

The importer must read back the tables the electrode exporters write,
skipping the non-localized rows whose coordinate cells are empty.
"""

from pathlib import Path

import numpy as np
import pytest

from virda.io.exporters.electrodes import export_electrodes_ese, export_electrodes_scalp
from virda.io.importers.electrode_points import import_electrode_points
from virda.models.electrode import Electrode, Electrodes


def _localized(electrode_id: str, coords: tuple[float, float, float]) -> Electrode:
    return Electrode(
        electrode_id=electrode_id,
        measured_distances={"NAS": 100.0},
        ese_coords=np.array([*coords]),
        scalp_coords=np.array([*coords]),
        residual_error=1.5,
        flagged=False,
    )


def _unlocalized(electrode_id: str) -> Electrode:
    return Electrode(electrode_id=electrode_id, measured_distances={"NAS": 100.0})


def test_reads_back_exporter_scalp_table(tmp_path: Path) -> None:
    electrodes = Electrodes(items=[_localized("A", (1.0, 2.0, 3.0)), _unlocalized("B")])
    path = export_electrodes_scalp(tmp_path / "electrodes_scalp.csv", electrodes)

    points = import_electrode_points(path)

    assert len(points) == 1
    assert points[0].electrode_id == "A"
    assert points[0].coordinates.tolist() == [1.0, 2.0, 3.0]
    assert points[0].residual_mm == 1.5
    assert points[0].flagged is False


def test_reads_back_exporter_ese_table(tmp_path: Path) -> None:
    electrodes = Electrodes(items=[_localized("Fz", (4.0, 5.0, 6.0))])
    path = export_electrodes_ese(tmp_path / "electrodes_ese.csv", electrodes)

    points = import_electrode_points(path)

    assert points[0].electrode_id == "Fz"
    assert points[0].coordinates.tolist() == [4.0, 5.0, 6.0]


def test_flagged_rows_are_parsed(tmp_path: Path) -> None:
    electrode = Electrode(
        electrode_id="Fz",
        measured_distances={"NAS": 100.0},
        scalp_coords=np.array([1.0, 2.0, 3.0]),
        residual_error=20.0,
        flagged=True,
    )
    path = export_electrodes_scalp(tmp_path / "electrodes.csv", Electrodes(items=[electrode]))

    points = import_electrode_points(path)

    assert points[0].residual_mm == 20.0
    assert points[0].flagged is True


def test_extra_columns_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "electrodes.csv"
    path.write_text(
        "electrode_id,x,y,z,residual_mm,flagged,nas\nFz,1.0,2.0,3.0,1.5,no,110.5\n",
        encoding="utf-8",
    )

    points = import_electrode_points(path)

    assert points[0].electrode_id == "Fz"
    assert points[0].coordinates.tolist() == [1.0, 2.0, 3.0]


def test_missing_coordinate_columns_raise(tmp_path: Path) -> None:
    path = tmp_path / "electrodes.csv"
    path.write_text("not a table\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must carry the x, y, z columns"):
        import_electrode_points(path)


def test_rows_without_coordinates_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "electrodes.csv"
    path.write_text(
        "electrode_id,x,y,z,residual_mm,flagged\nA,1.0,2.0,3.0,,no\nB,,,,\n",
        encoding="utf-8",
    )

    points = import_electrode_points(path)

    assert [point.electrode_id for point in points] == ["A"]


def test_only_unlocalized_rows_raise(tmp_path: Path) -> None:
    path = export_electrodes_scalp(
        tmp_path / "electrodes.csv", Electrodes(items=[_unlocalized("Cz")])
    )

    with pytest.raises(ValueError, match="no electrode points found"):
        import_electrode_points(path)


def test_invalid_coordinate_value_raises(tmp_path: Path) -> None:
    path = tmp_path / "electrodes.csv"
    path.write_text(
        "electrode_id,x,y,z,residual_mm,flagged\nA,oops,2.0,3.0,,no\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="invalid x coordinate"):
        import_electrode_points(path)


def test_empty_id_gets_sequential_fallback(tmp_path: Path) -> None:
    path = tmp_path / "electrodes.csv"
    path.write_text("electrode_id,x,y,z,residual_mm,flagged\n,1.0,2.0,3.0,,no\n", encoding="utf-8")

    points = import_electrode_points(path)

    assert points[0].electrode_id == "E001"


def test_creates_nothing_but_reads_given_path(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "electrodes.csv"
    path.parent.mkdir()
    path.write_text("electrode_id,x,y,z,residual_mm,flagged\nA,1,2,3,,no\n", encoding="utf-8")

    assert import_electrode_points(path)[0].coordinates.tolist() == [1.0, 2.0, 3.0]
