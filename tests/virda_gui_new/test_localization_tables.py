"""Unit tests for the localization table widgets offscreen."""

import os
from typing import Any, cast

import numpy as np
import pytest
from PySide6.QtWidgets import QTableWidgetItem

from tests.virda_gui_new.fixtures.steps import advance_to
from virda.models.electrode import Electrode, Electrodes
from virda.models.mesh import Mesh
from virda_gui_new.dto.steps import STEPS
from virda_gui_new.dto.viewer3d import LabeledPoints
from virda_gui_new.services.coordinates import CoordinateFrame, CoordinateService
from virda_gui_new.services.steps_manager import StepsManager
from virda_gui_new.widgets.popup.localization.electrodes_table import ElectrodesTable
from virda_gui_new.widgets.popup.localization.fiducials_table import FiducialsTable
from virda_gui_new.widgets.popup.localization.frame_selector import FrameSelector
from virda_gui_new.widgets.popup.localization.measurements_table import MeasurementsTable

# World fiducials whose frame comes out axis-aligned (NAS at the origin).
_WORLD_FIDUCIALS = np.array([[0.0, 100.0, 0.0], [-70.0, 0.0, 0.0], [70.0, 0.0, 0.0]])
_FIDUCIAL_LABELS = ("NASION", "LPA", "RPA")
# Mesh whose centroid is (7.5, 7.5, 7.5): the head frame shifts by that.
_MESH_VERTICES = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]])
_HEAD_ORIGIN = cast("Any", np.array([7.5, 7.5, 7.5]))


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


def _service(fiducials: bool = True) -> CoordinateService:
    """Coordinate service over a manager with the sample mesh/fiducials."""
    manager = StepsManager(tuple(step.required_keys for step in STEPS))
    advance_to(manager, 4)
    mesh = Mesh(_MESH_VERTICES, np.zeros((1, 3), dtype=int))
    manager.set_step_context_item(1, "scalp_surface", mesh)
    manager.set_step_context_item(2, "mesh", mesh)
    if fiducials:
        manager.set_global_item("fiducials", LabeledPoints(_WORLD_FIDUCIALS, _FIDUCIAL_LABELS))
    return CoordinateService(manager)


def _item(text: str) -> QTableWidgetItem:
    return QTableWidgetItem(text)


def test_fiducials_table_has_three_fixed_named_rows(qapp) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QPushButton

    widget = FiducialsTable()

    assert widget._table.rowCount() == 3
    assert [cast("QTableWidgetItem", widget._table.item(row, 0)).text() for row in range(3)] == [
        "NAS",
        "LPA",
        "RPA",
    ]
    for row in range(3):
        name_item = widget._table.item(row, 0)
        assert not (cast("QTableWidgetItem", name_item).flags() & Qt.ItemFlag.ItemIsEditable)
    assert widget.collect_fiducial_rows() == []  # coordinates blank
    buttons = widget.findChildren(QPushButton)
    assert [button.text() for button in buttons] == ["From file", "Empty"]


def test_fiducials_table_import_button_signals(qapp) -> None:
    from PySide6.QtWidgets import QPushButton

    widget = FiducialsTable()
    events: list[int] = []
    widget.importRequested.connect(lambda: events.append(1))

    from_file = widget.findChildren(QPushButton)[0]
    from_file.click()

    assert events == [1]


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
            ("Nasion", cast("Any", np.array([0.0, 1.0, 2.0]))),  # alias of the NAS slot
            ("LPA", cast("Any", np.array([-1.0, 0.0, 0.0]))),
            ("Extra", cast("Any", np.array([9.0, 9.0, 9.0]))),  # unknown name -> ignored
        ]
    )

    assert len(events) == 1  # one signal for the whole replacement
    collected = widget.collect_fiducial_rows()
    assert [name for name, _xyz in collected] == ["Nasion", "LPA"]
    assert collected[0][1].tolist() == [0.0, 1.0, 2.0]
    assert collected[1][1].tolist() == [-1.0, 0.0, 0.0]
    # The artifact label is displayed and kept for the round trip.
    assert cast("QTableWidgetItem", widget._table.item(0, 0)).text() == "Nasion"
    assert cast("QTableWidgetItem", widget._table.item(2, 1)).text() == ""  # unmatched slot cleared

    # A prefilled row is editable like a manually entered one.
    cast("QTableWidgetItem", widget._table.item(0, 1)).setText("5")
    assert widget.collect_fiducial_rows()[0][1].tolist() == [5.0, 1.0, 2.0]


def test_fiducials_table_set_rows_clears_absent_slots(qapp) -> None:
    widget = FiducialsTable()
    widget.set_rows(
        [
            ("NAS", cast("Any", np.array([0.0, 0.0, 1.0]))),
            ("LPA", cast("Any", np.array([1.0, 0.0, 0.0]))),
            ("RPA", cast("Any", np.array([-1.0, 0.0, 0.0]))),
        ]
    )

    widget.set_rows([("RPA", cast("Any", np.array([7.0, 7.0, 7.0])))])

    collected = widget.collect_fiducial_rows()
    assert [name for name, _xyz in collected] == ["RPA"]
    assert collected[0][1].tolist() == [7.0, 7.0, 7.0]
    assert (
        cast("QTableWidgetItem", widget._table.item(0, 0)).text() == "NAS"
    )  # name back to canonical


def test_fiducials_table_empty_clears_grid_and_signals(qapp) -> None:
    widget = FiducialsTable()
    widget.set_rows(
        [
            ("NASION", cast("Any", np.array([0.0, 0.0, 1.0]))),
            ("LPA", cast("Any", np.array([1.0, 0.0, 0.0]))),
            ("RPA", cast("Any", np.array([-1.0, 0.0, 0.0]))),
        ]
    )
    events: list[str] = []
    widget.cleared.connect(lambda: events.append("cleared"))
    widget.rowsChanged.connect(lambda: events.append("rows"))

    widget.clear_grid()

    assert events == ["cleared", "rows"]
    assert widget.collect_fiducial_rows() == []
    for row in range(3):
        assert (
            cast("QTableWidgetItem", widget._table.item(row, 0)).text()
            == ("NAS", "LPA", "RPA")[row]
        )
        for column in (1, 2, 3):
            assert cast("QTableWidgetItem", widget._table.item(row, column)).text() == ""


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


def test_measurements_table_set_rows_replaces_the_grid(qapp) -> None:
    widget = MeasurementsTable()
    widget.add_row()
    widget._table.setItem(0, 0, _item("Old"))
    events: list[int] = []
    widget.rowsChanged.connect(lambda: events.append(1))

    widget.set_rows({"EEG 001": {"NAS": 62.5, "LPA": 111.25}, "EEG 002": {"NAS": 53.0}})

    assert len(events) == 1
    assert widget._table.rowCount() == 3  # one row per (electrode, fiducial) pair
    assert widget.collect_measurement_table() == {
        "EEG 001": {"NAS": 62.5, "LPA": 111.25},
        "EEG 002": {"NAS": 53.0},
    }
    assert cast("QTableWidgetItem", widget._table.item(0, 0)).text() == "EEG 001"
    assert cast("QTableWidgetItem", widget._table.item(0, 2)).text() == "62.5"


def test_measurements_table_import_button_signals(qapp) -> None:
    from PySide6.QtWidgets import QPushButton

    widget = MeasurementsTable()
    events: list[int] = []
    widget.importRequested.connect(lambda: events.append(1))

    buttons = widget.findChildren(QPushButton)
    assert [button.text() for button in buttons] == [
        "From file",
        "+ Add",
        "- Remove selected",
    ]
    buttons[0].click()

    assert events == [1]


def test_electrodes_table_renders_model(qapp) -> None:
    widget = ElectrodesTable()
    electrodes = Electrodes(
        [
            Electrode(
                "A",
                {"LPA": 10.0},
                ese_coords=cast("Any", np.array([1.0, 2.0, 3.0])),
                scalp_coords=cast("Any", np.array([4.0, 5.0, 6.0])),
                residual_error=0.125,
                flagged=True,
            ),
            Electrode("B", {"RPA": 9.0}),
        ]
    )

    widget.show_electrodes(electrodes)

    grid = widget._table
    texts = [
        [
            cast("QTableWidgetItem", grid.item(row, column)).text()
            for column in range(grid.columnCount())
        ]
        for row in range(grid.rowCount())
    ]
    assert texts[0] == ["A", "1.00", "2.00", "3.00", "4.00", "5.00", "6.00", "0.12", "yes"]
    assert texts[1] == ["B", "", "", "", "", "", "", "", "no"]


def test_frame_selector_defaults_to_head_and_signals(qapp) -> None:
    widget = FrameSelector()
    frames: list[CoordinateFrame] = []
    widget.frameChanged.connect(frames.append)

    assert widget.frame() == CoordinateFrame.HEAD
    widget.setCurrentIndex(widget.findData(CoordinateFrame.WORLD))

    assert frames == [CoordinateFrame.WORLD]
    assert widget.frame() == CoordinateFrame.WORLD


def test_frame_selector_greys_out_unavailable_frames(qapp) -> None:
    from PySide6.QtGui import QStandardItemModel

    widget = FrameSelector()
    fiducials_index = widget.findData(CoordinateFrame.FIDUCIALS)

    widget.set_frame_enabled(CoordinateFrame.FIDUCIALS, False)

    model = widget.model()
    assert isinstance(model, QStandardItemModel)
    assert not model.item(fiducials_index).isEnabled()
    assert model.item(widget.findData(CoordinateFrame.HEAD)).isEnabled()

    widget.set_frame_enabled(CoordinateFrame.FIDUCIALS, True)

    assert model.item(fiducials_index).isEnabled()


def test_fiducials_table_head_frame_display_keeps_world_canonical(qapp) -> None:
    widget = FiducialsTable(_service())
    widget.set_rows([("NASION", _WORLD_FIDUCIALS[0]), ("LPA", _WORLD_FIDUCIALS[1])])

    widget.set_display_frame(CoordinateFrame.HEAD)

    # Displayed values shift by the head origin; collected rows do not.
    head_display = [
        cast("QTableWidgetItem", widget._table.item(0, column)).text() for column in (1, 2, 3)
    ]
    assert head_display == ["-7.5", "92.5", "-7.5"]
    assert widget.collect_fiducial_rows()[0][1].tolist() == [0.0, 100.0, 0.0]
    assert [name for name, _xyz in widget.collect_fiducial_rows()] == ["NASION", "LPA"]

    widget.set_display_frame(CoordinateFrame.WORLD)

    assert [
        cast("QTableWidgetItem", widget._table.item(0, column)).text() for column in (1, 2, 3)
    ] == ["0", "100", "0"]
    assert widget.collect_fiducial_rows()[0][1].tolist() == [0.0, 100.0, 0.0]


def test_fiducials_table_edits_in_head_frame_store_world(qapp) -> None:
    widget = FiducialsTable(_service())
    widget.set_display_frame(CoordinateFrame.HEAD)

    for column, text in enumerate(("1", "2", "3"), start=1):
        widget._table.setItem(0, column, _item(text))

    collected = widget.collect_fiducial_rows()
    assert collected[0][1].tolist() == [1.0 + 7.5, 2.0 + 7.5, 3.0 + 7.5]
    assert (
        cast("QTableWidgetItem", widget._table.item(0, 1)).text() == "1"
    )  # display untouched while typing


def test_fiducials_table_fiducials_frame_shows_nas_at_the_origin(qapp) -> None:
    widget = FiducialsTable(_service())
    widget.set_rows(
        [(label, xyz) for label, xyz in zip(_FIDUCIAL_LABELS, _WORLD_FIDUCIALS, strict=True)]
    )

    widget.set_display_frame(CoordinateFrame.FIDUCIALS)

    assert [
        cast("QTableWidgetItem", widget._table.item(0, column)).text() for column in (1, 2, 3)
    ] == ["0", "0", "0"]
    assert [
        cast("QTableWidgetItem", widget._table.item(1, column)).text() for column in (1, 2, 3)
    ] == ["-70", "-100", "0"]
    # Canonical rows stay world.
    assert widget.collect_fiducial_rows()[0][1].tolist() == [0.0, 100.0, 0.0]


def test_fiducials_table_unavailable_frame_falls_back_to_world(qapp) -> None:
    widget = FiducialsTable(_service(fiducials=False))
    widget.set_rows([("NASION", _WORLD_FIDUCIALS[0])])

    widget.set_display_frame(CoordinateFrame.FIDUCIALS)

    assert widget._frame is CoordinateFrame.WORLD
    assert cast("QTableWidgetItem", widget._table.item(0, 1)).text() == "0"


def test_fiducials_table_frame_change_clears_incomplete_rows(qapp) -> None:
    widget = FiducialsTable(_service())
    widget.set_rows([("NASION", _WORLD_FIDUCIALS[0]), ("LPA", _WORLD_FIDUCIALS[1])])
    cast("QTableWidgetItem", widget._table.item(0, 2)).setText("")  # NAS row becomes incomplete

    widget.set_display_frame(CoordinateFrame.HEAD)

    # Incomplete rows cannot be converted cell-wise; the frame change blanks them.
    assert [name for name, _xyz in widget.collect_fiducial_rows()] == ["LPA"]
    assert cast("QTableWidgetItem", widget._table.item(0, 1)).text() == ""
    # Complete rows keep their values across the switch.
    assert cast("QTableWidgetItem", widget._table.item(1, 1)).text() == f"{-70.0 - 7.5:g}"


def test_electrodes_table_renders_in_display_frame(qapp) -> None:
    widget = ElectrodesTable(_service())
    electrodes = Electrodes(
        [
            Electrode(
                "A",
                {"LPA": 10.0},
                ese_coords=_WORLD_FIDUCIALS[0],
                scalp_coords=np.array([1.0, 2.0, 3.0]),
                residual_error=0.125,
            )
        ]
    )

    widget.show_electrodes(electrodes)
    widget.set_display_frame(CoordinateFrame.HEAD)

    grid = widget._table
    assert (
        cast("QTableWidgetItem", grid.item(0, 1)).text() == f"{0.0 - 7.5:.2f}"
    )  # ESE X in head frame
    assert cast("QTableWidgetItem", grid.item(0, 2)).text() == f"{100.0 - 7.5:.2f}"
    assert (
        cast("QTableWidgetItem", grid.item(0, 7)).text() == "0.12"
    )  # residuals are frame-independent

    widget.set_display_frame(CoordinateFrame.FIDUCIALS)

    # NAS is the fiducials-frame origin.
    assert cast("QTableWidgetItem", grid.item(0, 1)).text() == "0.00"
    assert cast("QTableWidgetItem", grid.item(0, 2)).text() == "0.00"


def test_electrodes_table_without_converter_stays_world(qapp) -> None:
    widget = ElectrodesTable()
    widget.show_electrodes(
        Electrodes([Electrode("A", {"LPA": 1.0}, ese_coords=np.array([1.0, 2.0, 3.0]))])
    )

    widget.set_display_frame(CoordinateFrame.HEAD)

    assert widget._frame is CoordinateFrame.WORLD
    assert cast("QTableWidgetItem", widget._table.item(0, 1)).text() == "1.00"
