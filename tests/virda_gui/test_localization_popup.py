"""Unit tests for :class:`virda_gui_new.widgets.popup.localization.abstract`.

Runs under the offscreen Qt platform; ``ops.localize`` is mocked, so no
heavy mesh work runs; only hint text, context writes and hooks flow.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from PySide6.QtWidgets import QTableWidgetItem

from virda.models.electrode import Electrode, Electrodes
from virda.models.mesh import Mesh
from virda_gui_new.api import ops
from virda_gui_new.dto.steps import LOCALIZATION, STEPS
from virda_gui_new.services import artifact_saving
from virda_gui_new.services.coordinates import CoordinateFrame
from virda_gui_new.services.steps_manager import StepsManager
from virda_gui_new.widgets.popup.localization import LocalizationPopup
from virda_gui_new.widgets.popup.localization.abstract import (
    LocalizationPopupBase,
)
from virda_gui_new.widgets.popup.localization.frame_selector import FrameSelector

_ESE = Mesh(np.zeros((4, 3)), np.zeros((1, 3), dtype=int))
_SCALP = Mesh(np.zeros((3, 3)), np.zeros((1, 3), dtype=int))
# World fiducials whose frame comes out axis-aligned (NAS at the origin).
_WORLD_FIDUCIALS = np.array([[0.0, 100.0, 0.0], [-70.0, 0.0, 0.0], [70.0, 0.0, 0.0]])
_FIDUCIAL_LABELS = ("NASION", "LPA", "RPA")
# Mesh whose centroid is (7.5, 7.5, 7.5): the head frame shifts by that.
_MESH_VERTICES = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]])
_FIDUCIAL_ROWS = [
    ("Nasion", cast("Any", np.array([0.0, 0.0, 1.0]))),
    ("LPA", cast("Any", np.array([1.0, 0.0, 0.0]))),
    ("RPA", cast("Any", np.array([-1.0, 0.0, 0.0]))),
]
_MEASUREMENTS = {"A": {"LPA": 10.0, "RPA": 12.0}}
_NO_GEOMETRY = "Localization needs the ESE mesh and normals (run the ESE mesh step first)."


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


def _make_manager(ese=True, normals=True) -> StepsManager:
    """Manager at step 4 with the geometry contexts of steps 1-3 set."""
    steps = StepsManager(tuple(step.required_keys for step in STEPS))
    steps.next_step()
    steps.set_step_context_item(1, "scalp_surface", _SCALP)
    steps.next_step()
    steps.set_step_context_item(2, "mesh", _SCALP)
    steps.next_step()
    if ese:
        steps.set_step_context_item(3, "ese_mesh", _ESE)
    if normals:
        steps.set_step_context_item(3, "normals", np.ones((4, 3)))
    steps.next_step()
    return steps


class RecordingPopup(LocalizationPopupBase):
    """Concrete popup with canned rows/table and recorded hooks."""

    def __init__(
        self,
        manager: StepsManager,
        fiducial_rows=None,
        table=None,
        parent=None,
        project=None,
    ) -> None:
        self._rows = [] if fiducial_rows is None else fiducial_rows
        self._table = {} if table is None else table
        self.localized_called: list[Any] = []
        self.statuses: list[Any] = []
        super().__init__(LOCALIZATION, manager, parent, project=project)

    def collect_fiducial_rows(self):
        return self._rows

    def collect_measurement_table(self):
        return self._table

    def show_localized(self, electrodes: Electrodes) -> None:
        self.localized_called.append(electrodes)

    def on_status(self, text: str) -> None:
        self.statuses.append(text)


@pytest.fixture
def popups(qapp):
    """Collect popups and stop their runner threads at teardown."""
    created: list[Any] = []
    yield created
    for popup in created:
        popup.shutdown()


def _patch_localize(monkeypatch, calls: list[Any], payload: Electrodes) -> None:
    def fake_localize(ese, normals, scalp, fiducials, electrodes, options):
        calls.append((fiducials, electrodes))
        return payload

    monkeypatch.setattr(ops, "localize", fake_localize)


def _settle(qapp, popup, text: str, timeout_s: float = 5.0) -> None:
    """Pump events until the hint shows ``text`` or the deadline passes."""
    deadline = time.monotonic() + timeout_s
    while popup._hint_label.text() != text and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


def test_missing_geometry_shows_precondition(qapp, popups) -> None:
    popup = RecordingPopup(_make_manager(ese=False))
    popups.append(popup)

    assert popup._localization_runner is None
    assert popup._hint_label.text() == _NO_GEOMETRY
    assert not popup._run_button.isVisibleTo(popup)
    popup._on_inputs_changed()  # no-op without geometry
    assert popup._hint_label.text() == _NO_GEOMETRY


def test_blocked_path_hinting(qapp, popups, monkeypatch) -> None:
    calls: list[Any] = []

    def fail_late(*_args):
        calls.append(())
        return Electrodes([Electrode("A", {"LPA": 1.0})])

    monkeypatch.setattr(ops, "localize", fail_late)
    popup = RecordingPopup(_make_manager(), fiducial_rows=[])
    popups.append(popup)

    assert popup._hint_label.text() == "Blocked: not enough fiducials (need 3)."
    popup._on_inputs_changed()
    assert "Blocked:" in popup._hint_label.text()
    popup._rows = _FIDUCIAL_ROWS
    popup._on_inputs_changed()
    assert popup._hint_label.text() == "Blocked: no measurements."
    _settle(qapp, popup, "no marker")
    assert calls == []


def test_full_path_localizes(qapp, popups, monkeypatch) -> None:
    calls: list[Any] = []
    payload = Electrodes(
        [
            Electrode("A", {"LPA": 10.0}, ese_coords=np.zeros(3), scalp_coords=np.zeros(3)),
            Electrode("B", {"RPA": 1.0}, ese_coords=np.zeros(3), scalp_coords=np.zeros(3)),
        ]
    )
    _patch_localize(monkeypatch, calls, payload)
    manager = _make_manager()
    popup = RecordingPopup(manager, fiducial_rows=_FIDUCIAL_ROWS, table=_MEASUREMENTS)
    popups.append(popup)

    popup._on_inputs_changed()
    assert popup._hint_label.text() == "Running..."
    _settle(qapp, popup, "Localized 2 electrodes.")

    assert popup._hint_label.text() == "Localized 2 electrodes."
    assert popup.localized_called == [payload]
    assert manager.get_step_context_item(LOCALIZATION.index, "electrodes") is payload


def test_localized_run_refreshes_fiducial_store(qapp, popups, monkeypatch) -> None:
    """The global store mirrors the current table rows after each run."""
    from virda_gui_new.dto.viewer3d import LabeledPoints

    payload = Electrodes([Electrode("A", {"LPA": 10.0})])
    _patch_localize(monkeypatch, [], payload)
    manager = _make_manager()
    popup = RecordingPopup(manager, fiducial_rows=_FIDUCIAL_ROWS, table=_MEASUREMENTS)
    popups.append(popup)

    popup._on_inputs_changed()
    _settle(qapp, popup, "Localized 1 electrode.")

    points = cast("LabeledPoints", manager.get_global_item("fiducials"))
    assert isinstance(points, LabeledPoints)
    assert points.labels == ("Nasion", "LPA", "RPA")
    assert points.points.tolist() == [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]


def test_unchanged_fiducials_skip_the_store_write(qapp, popups, monkeypatch) -> None:
    """An equal store is not rewritten, so listeners are not re-triggered."""

    from virda_gui_new.services.localization import labeled_points_from_rows

    payload = Electrodes([Electrode("A", {"LPA": 10.0})])
    _patch_localize(monkeypatch, [], payload)
    manager = _make_manager()
    manager.set_global_item("fiducials", labeled_points_from_rows(_FIDUCIAL_ROWS))
    notifications: list[int] = []
    manager.subscribe(lambda: notifications.append(len(notifications)))
    popup = RecordingPopup(manager, fiducial_rows=_FIDUCIAL_ROWS, table=_MEASUREMENTS)
    popups.append(popup)

    popup._on_inputs_changed()
    _settle(qapp, popup, "Localized 1 electrode.")

    # Only the electrodes context write notified; the store was untouched.
    assert len(notifications) == 1


def test_stale_context_run_keeps_fiducial_store(qapp, popups) -> None:
    """A dropped (stale) result leaves the global store untouched."""
    from virda_gui_new.dto.viewer3d import LabeledPoints

    manager = _make_manager()
    points = LabeledPoints(np.array([[0.0, 0.0, 2.0]]), ("NAS",))
    manager.set_global_item("fiducials", points)
    popup = RecordingPopup(manager, fiducial_rows=_FIDUCIAL_ROWS, table=_MEASUREMENTS)
    popups.append(popup)
    manager.invalidate_steps_after(3)

    popup._on_localized(Electrodes([Electrode("A", {"LPA": 1.0})]))

    assert popup._hint_label.text() == "Pipeline changed; results dropped."
    assert manager.get_global_item("fiducials") is points


def test_localized_run_persists_localization_files(
    qapp, popups, monkeypatch, tmp_path: Path
) -> None:
    """Each successful run writes fiducials, measurements and electrodes."""
    from virda_gui_new.services.artifact_loading import load_fiducial_points

    payload = Electrodes(
        [
            Electrode("A", {"LPA": 10.0}, ese_coords=np.zeros(3), scalp_coords=np.zeros(3)),
        ]
    )
    _patch_localize(monkeypatch, [], payload)
    manager = _make_manager()
    popup = RecordingPopup(
        manager, fiducial_rows=_FIDUCIAL_ROWS, table=_MEASUREMENTS, project=tmp_path
    )
    popups.append(popup)

    popup._on_inputs_changed()
    _settle(qapp, popup, "Localized 1 electrodes.")

    assert popup._hint_label.text() == "Localized 1 electrodes."
    localization = tmp_path / "localization"
    assert (localization / "fiducials.json").is_file()
    assert (localization / "measurements.json").is_file()
    assert (localization / "electrodes_scalp.csv").is_file()
    assert (localization / "electrodes_ese.csv").is_file()
    # The saved fiducials match what the popup shows (and the store holds).
    points = load_fiducial_points(localization / "fiducials.json")
    assert points.labels == ("Nasion", "LPA", "RPA")
    assert points.points.tolist() == [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]


def test_localization_save_failure_is_a_failed_status(
    qapp, popups, monkeypatch, tmp_path: Path
) -> None:

    payload = Electrodes([Electrode("A", {"LPA": 10.0})])
    _patch_localize(monkeypatch, [], payload)
    failures: list[str] = []

    def fail_save(*_args: object) -> None:
        raise RuntimeError("cannot write")

    monkeypatch.setattr(artifact_saving, "save_localization", fail_save)
    manager = _make_manager()
    popup = RecordingPopup(
        manager, fiducial_rows=_FIDUCIAL_ROWS, table=_MEASUREMENTS, project=tmp_path
    )
    popups.append(popup)
    popup.failed.connect(failures.append)

    popup._on_inputs_changed()
    _settle(qapp, popup, "Localized 1 electrode.")

    assert popup._hint_label.text() == "Failed to save localization files: cannot write"
    assert failures == ["cannot write"]
    assert manager.get_step_context_item(LOCALIZATION.index, "electrodes") is payload


def test_stale_context_drop(qapp, popups) -> None:
    manager = _make_manager()
    popup = RecordingPopup(manager)
    popups.append(popup)
    manager.invalidate_steps_after(3)  # step-4 context is gone

    popup._on_localized(Electrodes([Electrode("A", {"LPA": 1.0})]))

    assert popup._hint_label.text() == "Pipeline changed; results dropped."
    assert popup.localized_called == []


def test_shutdown_is_safe(qapp, popups, monkeypatch) -> None:
    monkeypatch.setattr(ops, "localize", lambda *_args: None)
    popup = RecordingPopup(_make_manager(), fiducial_rows=_FIDUCIAL_ROWS, table=_MEASUREMENTS)
    popups.append(popup)

    popup.shutdown()  # idempotent: live runner then the base machinery
    popup.shutdown()


def test_concrete_popup_prefills_fiducials_from_store(qapp, popups) -> None:
    """Seeded global-store fiducials land in the table as editable rows."""
    from virda_gui_new.dto.viewer3d import LabeledPoints

    manager = _make_manager()
    points = LabeledPoints(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]), ("Nasion", "LPA"))
    manager.set_global_item("fiducials", points)
    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)

    rows = popup.collect_fiducial_rows()
    assert [name for name, _xyz in rows] == ["Nasion", "LPA"]
    assert rows[0][1].tolist() == [0.0, 0.0, 1.0]
    assert rows[1][1].tolist() == [1.0, 0.0, 0.0]
    assert cast("QTableWidgetItem", popup._fiducials_table._table.item(0, 0)).text() == "Nasion"


def test_concrete_popup_prefill_absent_keeps_empty_grid(qapp, popups) -> None:
    """Without seeded fiducials the table starts empty as before."""
    popup = LocalizationPopup(LOCALIZATION, _make_manager())
    popups.append(popup)

    assert popup.collect_fiducial_rows() == []


def test_concrete_popup_empty_clears_fiducial_store(qapp, popups) -> None:
    """The Empty action clears the grid and the global store (one write)."""
    from virda_gui_new.dto.viewer3d import LabeledPoints

    manager = _make_manager()
    points = LabeledPoints(
        np.array([[3.0, 94.0, 32.0], [-69.0, 10.0, -25.0], [77.0, 12.0, -30.0]]),
        ("NASION", "LPA", "RPA"),
    )
    manager.set_global_item("fiducials", points)
    notifications: list[int] = []
    manager.subscribe(lambda: notifications.append(1))
    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)

    assert [name for name, _xyz in popup.collect_fiducial_rows()] == ["NASION", "LPA", "RPA"]

    popup._fiducials_table.clear_grid()

    assert popup.collect_fiducial_rows() == []
    assert manager.get_global_item("fiducials") is None
    assert len(notifications) == 1  # only the store write notified


def _stub_file_dialog(monkeypatch: object, path: str | None) -> None:
    """Make every QFileDialog.getOpenFileName return ``path`` (or cancel)."""

    def fake_get_open_file_name(*_args: object, **_kwargs: object) -> tuple[str, str]:
        return ("", "") if path is None else (path, "")

    monkeypatch.setattr(  # type: ignore[attr-defined]
        "virda_gui_new.widgets.popup.localization.popup.QFileDialog.getOpenFileName",
        staticmethod(fake_get_open_file_name),
    )


def test_concrete_popup_imports_fiducials_from_file(
    qapp, popups, monkeypatch, tmp_path: Path
) -> None:
    """Import fills the fixed slots and refreshes the global store."""
    from virda_gui_new.dto.viewer3d import LabeledPoints

    source = tmp_path / "other.json"
    source.write_text(
        json.dumps(
            {
                "fiducials": [
                    {
                        "fiducial_id": "NAS",
                        "name": "NAS",
                        "coordinates": [10.0, 0.0, 0.0],
                        "coordinate_system": "world",
                        "definition_method": "imported",
                    },
                    {
                        "fiducial_id": "LPA",
                        "name": "LPA",
                        "coordinates": [0.0, 10.0, 0.0],
                        "coordinate_system": "world",
                        "definition_method": "imported",
                    },
                    {
                        "fiducial_id": "RPA",
                        "name": "RPA",
                        "coordinates": [0.0, 0.0, 10.0],
                        "coordinate_system": "world",
                        "definition_method": "imported",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    manager = _make_manager()
    points = LabeledPoints(
        np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]),
        ("NAS", "LPA", "RPA"),
    )
    manager.set_global_item("fiducials", points)
    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)
    _stub_file_dialog(monkeypatch, str(source))

    popup._fiducials_table.importRequested.emit()

    rows = popup.collect_fiducial_rows()
    assert [name for name, _xyz in rows] == ["NAS", "LPA", "RPA"]
    assert rows[0][1].tolist() == [10.0, 0.0, 0.0]
    points = cast("LabeledPoints", manager.get_global_item("fiducials"))
    assert isinstance(points, LabeledPoints)
    assert points.points.tolist() == [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]


def test_concrete_popup_fiducials_import_error_is_a_hint(qapp, popups, monkeypatch) -> None:
    """An unreadable import file shows a hint instead of crashing."""
    popup = LocalizationPopup(LOCALIZATION, _make_manager())
    popups.append(popup)
    _stub_file_dialog(monkeypatch, str(Path("missing.json")))

    popup._fiducials_table.importRequested.emit()

    assert popup._hint_label.text().startswith("Failed to import fiducials:")
    assert popup.collect_fiducial_rows() == []


def test_concrete_popup_imports_measurements_from_file(
    qapp, popups, monkeypatch, tmp_path: Path
) -> None:
    """Import fills the measurement grid from the project format."""
    source = tmp_path / "measurements.json"
    source.write_text(
        json.dumps(
            {
                "electrodes": [
                    {"electrode_id": "EEG 001", "measured_distances": {"NAS": 62.5}},
                    {"electrode_id": "EEG 002", "measured_distances": {"LPA": 111.25}},
                ]
            }
        ),
        encoding="utf-8",
    )
    popup = LocalizationPopup(LOCALIZATION, _make_manager())
    popups.append(popup)
    _stub_file_dialog(monkeypatch, str(source))

    popup._measurements_table.importRequested.emit()

    assert popup.collect_measurement_table() == {
        "EEG 001": {"NAS": 62.5},
        "EEG 002": {"LPA": 111.25},
    }


def test_concrete_popup_measurements_import_error_is_a_hint(qapp, popups, monkeypatch) -> None:
    popup = LocalizationPopup(LOCALIZATION, _make_manager())
    popups.append(popup)
    _stub_file_dialog(monkeypatch, str(Path("missing.json")))

    popup._measurements_table.importRequested.emit()

    assert popup._hint_label.text().startswith("Failed to import measurements:")
    assert popup.collect_measurement_table() == {}


def _fiducials_item_enabled(selector: FrameSelector) -> bool:
    """Whether the selector's fiducials item is enabled."""
    from PySide6.QtGui import QStandardItemModel

    index = selector.findData(CoordinateFrame.FIDUCIALS)
    model = selector.model()
    assert isinstance(model, QStandardItemModel)
    return bool(model.item(index).isEnabled())


def test_concrete_popup_frame_selectors_default_to_head(qapp, popups) -> None:
    """Both selectors default to head; head is available (scalp mesh set)."""
    popup = LocalizationPopup(LOCALIZATION, _make_manager())
    popups.append(popup)

    assert popup._fiducials_frames.frame() == CoordinateFrame.HEAD
    assert popup._electrodes_frames.frame() == CoordinateFrame.HEAD
    assert popup._fiducials_table._frame is CoordinateFrame.HEAD
    assert popup._electrodes_table._frame is CoordinateFrame.HEAD


def test_concrete_popup_fiducials_frame_needs_three_fiducials(qapp, popups) -> None:
    """The fiducials frame option is greyed out without all three points."""
    from virda_gui_new.dto.viewer3d import LabeledPoints

    manager = _make_manager()
    manager.set_global_item("fiducials", LabeledPoints(_WORLD_FIDUCIALS, _FIDUCIAL_LABELS))

    bare = LocalizationPopup(LOCALIZATION, _make_manager())
    popups.append(bare)
    assert not _fiducials_item_enabled(bare._fiducials_frames)

    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)
    assert _fiducials_item_enabled(popup._fiducials_frames)

    popup._fiducials_frames.setCurrentIndex(
        popup._fiducials_frames.findData(CoordinateFrame.FIDUCIALS)
    )

    # NAS sits at the fiducials-frame origin; collection stays world.
    assert [cast("QTableWidgetItem", popup._fiducials_table._table.item(0, c)).text() for c in (1, 2, 3)] == ["0", "0", "0"]
    assert popup.collect_fiducial_rows()[0][1].tolist() == _WORLD_FIDUCIALS[0].tolist()


def test_concrete_popup_empty_resets_fiducials_frame(qapp, popups) -> None:
    """Clearing the store makes the fiducials frame unavailable -> fallback."""
    from virda_gui_new.dto.viewer3d import LabeledPoints

    manager = _make_manager()
    manager.set_global_item("fiducials", LabeledPoints(_WORLD_FIDUCIALS, _FIDUCIAL_LABELS))
    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)

    popup._fiducials_frames.setCurrentIndex(
        popup._fiducials_frames.findData(CoordinateFrame.FIDUCIALS)
    )
    popup._fiducials_table.clear_grid()

    assert popup._fiducials_frames.frame() == CoordinateFrame.HEAD  # head fallback
    assert not _fiducials_item_enabled(popup._fiducials_frames)


def test_localized_save_is_world_canonical_in_head_display(
    qapp, popups, monkeypatch, tmp_path: Path
) -> None:
    """A run with head-frame display still saves world-canonical files."""
    from virda_gui_new.dto.viewer3d import LabeledPoints
    from virda_gui_new.services.artifact_loading import load_fiducial_points

    payload = Electrodes(
        [Electrode("A", {"LPA": 10.0}, ese_coords=np.zeros(3), scalp_coords=np.zeros(3))]
    )
    _patch_localize(monkeypatch, [], payload)
    manager = _make_manager()
    mesh = Mesh(_MESH_VERTICES, np.zeros((1, 3), dtype=int))
    manager.set_step_context_item(2, "mesh", mesh)
    manager.set_global_item("fiducials", LabeledPoints(_WORLD_FIDUCIALS, _FIDUCIAL_LABELS))
    popup = LocalizationPopup(LOCALIZATION, manager, project=tmp_path)
    popups.append(popup)

    assert popup._fiducials_frames.frame() == CoordinateFrame.HEAD
    assert [cast("QTableWidgetItem", popup._fiducials_table._table.item(0, c)).text() for c in (1, 2, 3)] == [
        "-7.5",
        "92.5",
        "-7.5",
    ]

    popup._measurements_table.add_row()
    for column, text in enumerate(("A", "LPA", "10")):
        popup._measurements_table._table.setItem(0, column, QTableWidgetItem(text))

    _settle(qapp, popup, "Localized 1 electrode.")

    points = load_fiducial_points(tmp_path / "localization" / "fiducials.json")
    assert points.labels == _FIDUCIAL_LABELS
    assert points.points.tolist() == _WORLD_FIDUCIALS.tolist()


def test_concrete_popup_prefill_precedes_runner_setup(qapp, popups) -> None:
    """No geometry: the prefill emits before the runner exists; no crash."""
    from virda_gui_new.dto.viewer3d import LabeledPoints

    manager = _make_manager(normals=False)
    points = LabeledPoints(np.array([[0.0, 0.0, 1.0]]), ("Nasion",))
    manager.set_global_item("fiducials", points)

    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)

    assert popup._localization_runner is None
    assert [name for name, _xyz in popup.collect_fiducial_rows()] == ["Nasion"]


def test_concrete_popup_blocked_states(qapp, popups, monkeypatch) -> None:
    calls: list[Any] = []

    def fail_late(*_args):
        calls.append(())
        return Electrodes([])

    monkeypatch.setattr(ops, "localize", fail_late)
    popup = LocalizationPopup(LOCALIZATION, _make_manager())
    popups.append(popup)

    assert popup._hint_label.text() == "Blocked: not enough fiducials (need 3)."
    for row, coords in enumerate(("0 0 1", "1 0 0", "-1 0 0")):
        for column, text in enumerate(coords.split(), start=1):
            popup._fiducials_table._table.setItem(row, column, QTableWidgetItem(text))
    assert popup._hint_label.text() == "Blocked: no measurements."
    popup._measurements_table.add_row()
    assert popup._hint_label.text() == "Blocked: no measurements."
    popup._measurements_table._table.setItem(0, 0, QTableWidgetItem("A"))

    time.sleep(0.3)
    qapp.processEvents()
    assert calls == []  # incomplete measurement row: nothing runs
    assert not popup._electrodes_table._table.rowCount()


def test_concrete_popup_live_flow(qapp, popups, monkeypatch) -> None:
    calls: list[Any] = []
    payload = Electrodes(
        [
            Electrode("A", {"LPA": 10.0}, ese_coords=np.zeros(3), scalp_coords=np.zeros(3)),
            Electrode("B", {"RPA": 1.0}, ese_coords=np.zeros(3), scalp_coords=np.zeros(3)),
        ]
    )
    _patch_localize(monkeypatch, calls, payload)
    manager = _make_manager()
    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)

    for row, coords in enumerate(("0 0 1", "1 0 0", "-1 0 0")):
        for column, text in enumerate(coords.split(), start=1):
            popup._fiducials_table._table.setItem(row, column, QTableWidgetItem(text))
    popup._measurements_table.add_row()
    for column, text in enumerate(("A", "LPA", "10")):
        popup._measurements_table._table.setItem(0, column, QTableWidgetItem(text))

    _settle(qapp, popup, "Localized 2 electrodes.")

    assert popup._hint_label.text() == "Localized 2 electrodes."
    assert len(calls) >= 1
    assert manager.get_step_context_item(LOCALIZATION.index, "electrodes") is payload
    grid = popup._electrodes_table._table
    assert cast("QTableWidgetItem", grid.item(0, 0)).text() == "A"
    assert cast("QTableWidgetItem", grid.item(0, 1)).text() == "0.00"
    assert cast("QTableWidgetItem", grid.item(1, 0)).text() == "B"
    assert cast("QTableWidgetItem", grid.item(1, 7)).text() == ""
    assert cast("QTableWidgetItem", grid.item(1, 8)).text() == "no"
