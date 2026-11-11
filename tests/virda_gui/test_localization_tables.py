"""Unit tests for the localization table widgets offscreen."""

import os

import numpy as np
import pytest

from virda.models.electrode import Electrode, Electrodes
from virda_gui_new.widgets.popup.localization.electrodes_table import ElectrodesTable
from virda_gui_new.widgets.popup.localization.fiducials_table import FiducialsTable
from virda_gui_new.widgets.popup.localization.measurements_table import MeasurementsTable


@pytest.fixture
def qapp():
    """Yield an offscreen ``QApplication``, skipping without a Qt install."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on local Qt install
        pytest.skip(f"Qt platform unavailable: {exc}")

    app = QApplication.instance() or QApplication([])
    return app


def _item(text: str) -> object:
    from PySide6.QtWidgets import QTableWidgetItem

    return QTableWidgetItem(text)


def test_fiducials_table_has_three_fixed_named_rows(qapp) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QPushButton

    widget = FiducialsTable()

    assert widget._table.rowCount() == 3
    assert [widget._table.item(row, 0).text() for row in range(3)] == ["NAS", "LPA", "RPA"]
    for row in range(3):
        name_item = widget._table.item(row, 0)
        assert not (name_item.flags() & Qt.ItemFlag.ItemIsEditable)
    assert widget.collect_fiducial_rows() == []  # coordinates blank
    buttons = widget.findChildren(QPushButton)
    assert [button.text() for button in buttons] == ["Empty"]


def test_fiducials_table_collects_complete_rows_in_fixed_order(qapp) -> None:
    widget = FiducialsTable()
    rows = [
        ("NAS", "0", "0", "1"),
        ("LPA", "1.5", "0", "0"),
        ("RPA", "-1", "  ", "0"),  # incomplete row -> skipped
    ]
    for row, cells in enumerate(rows):
        for column, text in enumerate(cells[1:], start=1):
            widget._table.setItem(row, column, _item(text))

    collected = widget.collect_fiducial_rows()
    assert [name for name, _xyz in collected] == ["NAS", "LPA"]
    assert all(isinstance(name, str) for name in (name for name, _xyz in collected))
    assert all(xyz.shape == (3,) and xyz.dtype.kind == "f" for _name, xyz in collected)
    assert collected[0][1].tolist() == [0.0, 0.0, 1.0]
    assert collected[1][1].tolist() == [1.5, 0.0, 0.0]


def test_fiducials_table_set_rows_matches_fixed_slots(qapp) -> None:
    widget = FiducialsTable()
    events: list[int] = []
    widget.rowsChanged.connect(lambda: events.append(1))

    widget.set_rows(
        [
            ("Nasion", np.array([0.0, 1.0, 2.0])),  # alias of the NAS slot
            ("LPA", np.array([-1.0, 0.0, 0.0])),
            ("Extra", np.array([9.0, 9.0, 9.0])),  # unknown name -> ignored
        ]
    )

    assert len(events) == 1  # one signal for the whole replacement
    collected = widget.collect_fiducial_rows()
    assert [name for name, _xyz in collected] == ["Nasion", "LPA"]
    assert collected[0][1].tolist() == [0.0, 1.0, 2.0]
    assert collected[1][1].tolist() == [-1.0, 0.0, 0.0]
    # The artifact label is displayed and kept for the round trip.
    assert widget._table.item(0, 0).text() == "Nasion"
    assert widget._table.item(2, 1).text() == ""  # unmatched slot cleared

    # A prefilled row is editable like a manually entered one.
    widget._table.item(0, 1).setText("5")
    assert widget.collect_fiducial_rows()[0][1].tolist() == [5.0, 1.0, 2.0]


def test_fiducials_table_set_rows_clears_absent_slots(qapp) -> None:
    widget = FiducialsTable()
    widget.set_rows(
        [
            ("NAS", np.array([0.0, 0.0, 1.0])),
            ("LPA", np.array([1.0, 0.0, 0.0])),
            ("RPA", np.array([-1.0, 0.0, 0.0])),
        ]
    )

    widget.set_rows([("RPA", np.array([7.0, 7.0, 7.0]))])

    collected = widget.collect_fiducial_rows()
    assert [name for name, _xyz in collected] == ["RPA"]
    assert collected[0][1].tolist() == [7.0, 7.0, 7.0]
    assert widget._table.item(0, 0).text() == "NAS"  # name back to canonical


def test_fiducials_table_empty_clears_grid_and_signals(qapp) -> None:
    widget = FiducialsTable()
    widget.set_rows(
        [
            ("NASION", np.array([0.0, 0.0, 1.0])),
            ("LPA", np.array([1.0, 0.0, 0.0])),
            ("RPA", np.array([-1.0, 0.0, 0.0])),
        ]
    )
    events: list[str] = []
    widget.cleared.connect(lambda: events.append("cleared"))
    widget.rowsChanged.connect(lambda: events.append("rows"))

    widget.clear_grid()

    assert events == ["cleared", "rows"]
    assert widget.collect_fiducial_rows() == []
    for row in range(3):
        assert widget._table.item(row, 0).text() == ("NAS", "LPA", "RPA")[row]
        for column in (1, 2, 3):
            assert widget._table.item(row, column).text() == ""


def test_measurements_table_last_row_wins(qapp) -> None:
    widget = MeasurementsTable()
    assert widget.collect_measurement_table() == {}

    for _row in range(3):
        widget.add_row()
    for row, texts in enumerate((("A", "1", "10"), ("B", "1", "8"), ("A", "1", "bad"))):
        for column, text in enumerate(texts):
            widget._table.setItem(row, column, _item(text))
    assert widget.collect_measurement_table() == {"A": {"1": 10.0}, "B": {"1": 8.0}}

    widget._table.setItem(2, 2, _item("12"))
    # Same electrode id + fiducial name appearing twice: the later row wins.
    assert widget.collect_measurement_table() == {"A": {"1": 12.0}, "B": {"1": 8.0}}


def test_electrodes_table_renders_model(qapp) -> None:
    widget = ElectrodesTable()
    electrodes = Electrodes(
        [
            Electrode(
                "A",
                {"LPA": 10.0},
                ese_coords=np.array([1.0, 2.0, 3.0]),
                scalp_coords=np.array([4.0, 5.0, 6.0]),
                residual_error=0.125,
                flagged=True,
            ),
            Electrode("B", {"RPA": 9.0}),
        ]
    )

    widget.show_electrodes(electrodes)

    grid = widget._table
    texts = [
        [grid.item(row, column).text() for column in range(grid.columnCount())]
        for row in range(grid.rowCount())
    ]
    assert texts[0] == ["A", "1.00", "2.00", "3.00", "4.00", "5.00", "6.00", "0.12", "yes"]
    assert texts[1] == ["B", "", "", "", "", "", "", "", "no"]
