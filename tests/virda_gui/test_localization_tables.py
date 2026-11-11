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


def test_fiducials_table_collects_complete_rows(qapp) -> None:
    widget = FiducialsTable()
    assert widget.collect_fiducial_rows() == []

    for _row in range(3):
        widget.add_row()
    rows = [
        ("  Nasion  ", "0", "0", "1"),
        ("LPA", "1.5", "0", "0"),
        ("RPA", "-1", "  ", "0"),  # incomplete row -> skipped
    ]
    for row, cells in enumerate(rows):
        for column, text in enumerate(cells):
            widget._table.setItem(row, column, _item(text))

    collected = widget.collect_fiducial_rows()
    assert len(collected) == 2
    names = [name for name, _xyz in collected]
    xyzs = [xyz for _name, xyz in collected]
    assert names == ["Nasion", "LPA"]
    assert all(isinstance(name, str) for name in names)
    assert all(xyz.shape == (3,) and xyz.dtype.kind == "f" for xyz in xyzs)
    assert collected[0][1][2] == 1.0
    assert collected[1][1][0] == 1.5


def test_fiducials_table_set_rows_fills_editable_grid(qapp) -> None:
    widget = FiducialsTable()
    events: list[int] = []
    widget.rowsChanged.connect(lambda: events.append(1))

    widget.set_rows([("Nasion", np.array([0.0, 1.0, 2.0])), ("LPA", np.array([-1.0, 0.0, 0.0]))])

    assert len(events) == 1  # one signal for the whole replacement
    collected = widget.collect_fiducial_rows()
    assert [name for name, _xyz in collected] == ["Nasion", "LPA"]
    assert collected[0][1].tolist() == [0.0, 1.0, 2.0]
    assert collected[1][1].tolist() == [-1.0, 0.0, 0.0]

    # A prefilled cell is editable like a manual row.
    widget._table.item(0, 0).setText("Nz")
    assert widget.collect_fiducial_rows()[0][0] == "Nz"


def test_fiducials_table_emits_rows_changed(qapp) -> None:
    widget = FiducialsTable()
    events: list[int] = []
    widget.rowsChanged.connect(lambda: events.append(len(events)))

    widget.add_row()
    assert events
    widget._table.setItem(0, 0, _item("Nasion"))
    assert len(events) >= 2
    widget._table.selectRow(0)
    widget.remove_selected_rows()
    assert len(events) >= 3
    count = len(events)
    widget.remove_selected_rows()  # nothing selected: no further emission
    assert len(events) == count


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
