"""Unit tests for :class:`virda_gui_new.widgets.popup.localization.abstract`.

Runs under the offscreen Qt platform; ``localize_electrodes`` is mocked, so no
heavy mesh work runs; only hint text, context writes and hooks flow.
"""

import json
import time
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QVBoxLayout

from tests.virda_gui_new.fixtures.steps import advance
from virda.models.coordinates import CoordinateSystem
from virda.models.electrode import Electrode, Electrodes
from virda.models.fiducial import Fiducials
from virda.models.mesh import Mesh
from virda.models.mri_volume import MRIVolume
from virda_gui_new.dto.steps import LOCALIZATION, NIFTI, STEPS
from virda_gui_new.services import artifact_saving
from virda_gui_new.services.steps_manager import StepsManager
from virda_gui_new.widgets.popup.localization import LocalizationPopup
from virda_gui_new.widgets.popup.localization.abstract import (
    LocalizationPopupBase,
)
from virda_gui_new.widgets.popup.localization.frame_selector import FrameSelector

type _LocalizeCall = tuple[Fiducials, Electrodes]

_ESE = Mesh(np.zeros((4, 3)), np.zeros((1, 3), dtype=int))
_SCALP = Mesh(np.zeros((3, 3)), np.zeros((1, 3), dtype=int))
# World fiducials whose frame comes out axis-aligned (NAS at the origin).
_WORLD_FIDUCIALS = np.array([[0.0, 100.0, 0.0], [-70.0, 0.0, 0.0], [70.0, 0.0, 0.0]])
_FIDUCIAL_LABELS = ("NASION", "LPA", "RPA")
_VOXEL_AFFINE = np.array(
    [
        [0.5, 0.0, 0.0, 10.0],
        [0.0, 0.5, 0.0, -5.0],
        [0.0, 0.0, 0.5, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
# Mesh whose centroid is (7.5, 7.5, 7.5): the head frame shifts by that.
_MESH_VERTICES = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]])
_FIDUCIAL_ROWS = [
    ("Nasion", np.array([0.0, 0.0, 1.0])),
    ("LPA", np.array([1.0, 0.0, 0.0])),
    ("RPA", np.array([-1.0, 0.0, 0.0])),
]
_MEASUREMENTS = {"A": {"NAS": 9.0, "LPA": 10.0, "RPA": 12.0}}
_TWO_ELECTRODES = {
    "A": {"NAS": 9.0, "LPA": 10.0, "RPA": 12.0},
    "B": {"NAS": 1.0, "LPA": 2.0, "RPA": 3.0},
}
_NO_GEOMETRY = (
    "Localization needs the ESE mesh and the scalp surface (run the earlier steps first)."
)
_NEEDS_INPUTS = "Localization needs {} (run the earlier steps first)."


def _cell_text(grid: QTableWidget, row: int, column: int) -> str:
    item = grid.item(row, column)
    assert item is not None
    return item.text()


def _make_manager(ese=True, normals=True) -> StepsManager:
    """Manager at step 4 with the geometry contexts of steps 1-3 set."""
    steps = StepsManager(tuple(step.required_keys for step in STEPS))
    advance(steps)
    steps.set_step_context_item(1, "scalp_surface", _SCALP)
    advance(steps)
    steps.set_step_context_item(2, "mesh", _SCALP)
    advance(steps)
    if ese:
        steps.set_step_context_item(3, "ese_mesh", _ESE)
    else:
        # No ESE mesh: the localization runs without the step-3 surface.
        steps.set_step_context_item(3, "ese_mesh", None)
    if normals:
        steps.set_step_context_item(3, "normals", np.ones((4, 3)))
    advance(steps)
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
        self.localized_called: list[Electrodes] = []
        self.statuses: list[str] = []
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
    created: list[LocalizationPopupBase] = []
    yield created
    for popup in created:
        popup.shutdown()


def _patch_localize(monkeypatch, calls: list[_LocalizeCall], payload: Electrodes) -> None:
    def fake_localize(
        ese, normals, scalp, fiducials, electrodes, *, config, fixed_offset_shift_mm=None
    ):
        calls.append((fiducials, electrodes))
        return payload

    monkeypatch.setattr(
        "virda_gui_new.services.localization.runner.localize_electrodes", fake_localize
    )


def _settle(qapp, popup, text: str, timeout_s: float = 5.0) -> None:
    """Pump events until the hint shows ``text`` or the deadline passes."""
    deadline = time.monotonic() + timeout_s
    while popup._hint_label.text() != text and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


def _run_pending(popup) -> bool:
    """Whether a localization run is scheduled for the popup."""
    runner = popup._localization_runner
    return runner is not None and runner._timer.isActive()


def test_the_geometry_attributes_exist_while_the_form_is_built(qapp, popups) -> None:
    """The form is built before the geometry is read: the attributes must exist."""
    seen: list[object] = []

    class Probe(LocalizationPopupBase):
        def build_form(self, layout) -> None:
            seen.append(self._ese_mesh)
            super().build_form(layout)

    popup = Probe(LOCALIZATION, _make_manager())
    popups.append(popup)

    assert seen == [None]


def test_the_popup_names_the_inputs_the_manager_misses(qapp, popups) -> None:
    """The popup and the steps menu read one answer from the manager."""
    manager = _make_manager()
    manager.set_step_context_item(3, "normals", None)
    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)

    assert manager.missing_inputs(LOCALIZATION.index) == ("normals",)
    assert popup._hint_label.text() == _NEEDS_INPUTS.format("normals")


def test_the_unusable_geometry_shows_the_plain_message(qapp, popups) -> None:
    """A context that does not hold the geometry cannot run the step."""
    manager = _make_manager()
    manager.set_step_context_item(3, "ese_mesh", "ready")
    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)

    assert manager.missing_inputs(LOCALIZATION.index) == ()
    assert popup._localization_runner is None
    assert popup._hint_label.text() == _NO_GEOMETRY


def test_missing_geometry_shows_precondition(qapp, popups) -> None:
    popup = RecordingPopup(_make_manager(ese=False))
    popups.append(popup)

    assert popup._localization_runner is None
    assert popup._hint_label.text() == _NEEDS_INPUTS.format("ese_mesh")
    assert not popup._run_button.isVisibleTo(popup)
    popup._on_inputs_changed()  # no-op without geometry
    assert popup._hint_label.text() == _NEEDS_INPUTS.format("ese_mesh")


def test_blocked_path_hinting(qapp, popups, monkeypatch) -> None:
    calls: list[object] = []

    def fail_late(*_args, **_kwargs):
        calls.append(())
        return Electrodes([Electrode("A", {"LPA": 1.0})])

    monkeypatch.setattr("virda_gui_new.services.localization.runner.localize_electrodes", fail_late)
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
    calls: list[_LocalizeCall] = []
    payload = Electrodes(
        [
            Electrode("A", {"LPA": 10.0}, ese_coords=np.zeros(3), scalp_coords=np.zeros(3)),
            Electrode("B", {"RPA": 1.0}, ese_coords=np.zeros(3), scalp_coords=np.zeros(3)),
        ]
    )
    _patch_localize(monkeypatch, calls, payload)
    manager = _make_manager()
    popup = RecordingPopup(manager, fiducial_rows=_FIDUCIAL_ROWS, table=_TWO_ELECTRODES)
    popups.append(popup)

    popup._on_inputs_changed()
    assert popup._hint_label.text() == "Running..."
    _settle(qapp, popup, "Localized 2 electrodes.")

    assert popup._hint_label.text() == "Localized 2 electrodes."
    assert popup.localized_called == [payload]
    assert manager.get_step_context_item(LOCALIZATION.index, "electrodes") is payload


def test_the_results_live_in_the_step_context(qapp, popups, monkeypatch) -> None:
    """The localized results are the context of the step, not a copy."""
    payload = Electrodes(
        [Electrode("A", {"LPA": 10.0}, ese_coords=np.zeros(3), scalp_coords=np.zeros(3))]
    )
    _patch_localize(monkeypatch, [], payload)
    manager = _make_manager()
    popup = RecordingPopup(manager, fiducial_rows=_FIDUCIAL_ROWS, table=_MEASUREMENTS)
    popups.append(popup)

    popup._on_inputs_changed()
    _settle(qapp, popup, "Localized 1 electrode.")

    # The popup reads the results from the context: there is no second
    # place the results live in.
    assert manager.get_step_context_item(LOCALIZATION.index, "electrodes") is payload
    assert popup._stored_results() is payload


def test_localized_run_mirrors_measurements_into_the_store(qapp, popups, monkeypatch) -> None:
    """The measurement table lands in the global store as a model."""
    from virda.models.electrode import Electrode, Electrodes

    payload = Electrodes(
        [
            Electrode("A", {"LPA": 10.0}, ese_coords=np.zeros(3), scalp_coords=np.zeros(3)),
            Electrode("B", {"RPA": 1.0}, ese_coords=np.zeros(3), scalp_coords=np.zeros(3)),
        ]
    )
    _patch_localize(monkeypatch, [], payload)
    manager = _make_manager()
    popup = RecordingPopup(manager, fiducial_rows=_FIDUCIAL_ROWS, table=_MEASUREMENTS)
    popups.append(popup)

    popup._on_inputs_changed()
    _settle(qapp, popup, "Localized 2 electrodes.")

    stored = manager.get_global_item("measurements")
    assert isinstance(stored, Electrodes)
    assert stored.ids == ["A"]
    stored_a = stored.get("A")
    assert stored_a is not None
    assert stored_a.measured_distances == _MEASUREMENTS["A"]


def test_localized_stale_context_keeps_seeded_measurements(qapp, popups) -> None:
    """A dropped (stale) result leaves the seeded measurements untouched."""
    from virda.models.electrode import Electrode, Electrodes

    manager = _make_manager()
    seeded = Electrodes([Electrode("Fz", {"NAS": 120.5})])
    manager.set_global_item("measurements", seeded)
    popup = RecordingPopup(manager, fiducial_rows=_FIDUCIAL_ROWS, table=_MEASUREMENTS)
    popups.append(popup)
    manager.invalidate_steps_after(3)  # step-4 context is gone

    popup._on_localized(Electrodes([Electrode("A", {"LPA": 1.0})]))

    assert popup._hint_label.text() == "Pipeline changed; results dropped."
    # The measurements live in the global store: the stale step-4 context
    # does not touch them.
    assert manager.get_global_item("measurements") is seeded
    with pytest.raises(ValueError, match="out of bounds"):
        manager.get_step_context_item(LOCALIZATION.index, "electrodes")


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

    points = manager.get_global_item("fiducials")
    assert isinstance(points, LabeledPoints)
    assert points.labels == ("Nasion", "LPA", "RPA")
    assert points.points.tolist() == [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]


def test_unchanged_fiducials_skip_the_store_write(qapp, popups, monkeypatch) -> None:
    """An equal store is not rewritten, so listeners are not re-triggered."""

    from virda_gui_new.services.localization import labeled_points_from_rows

    payload = Electrodes([Electrode("A", {"LPA": 10.0})])
    _patch_localize(monkeypatch, [], payload)
    manager = _make_manager()
    stored_points = labeled_points_from_rows(_FIDUCIAL_ROWS)
    manager.set_global_item("fiducials", stored_points)
    notifications: list[int] = []
    manager.subscribe(lambda: notifications.append(len(notifications)))
    popup = RecordingPopup(manager, fiducial_rows=_FIDUCIAL_ROWS, table=_MEASUREMENTS)
    popups.append(popup)

    popup._on_inputs_changed()
    _settle(qapp, popup, "Localized 1 electrode.")

    # The electrodes and the measurements store notified; the fiducials
    # store was untouched.
    assert len(notifications) <= 2
    assert manager.get_global_item("fiducials") is stored_points


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
    _settle(qapp, popup, "Localized 1 electrode.")

    assert popup._hint_label.text() == "Localized 1 electrode."
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
    _settle(qapp, popup, "Failed to save localization files: cannot write")

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
    monkeypatch.setattr(
        "virda_gui_new.services.localization.runner.localize_electrodes", lambda *_args, **_kw: None
    )
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
    name_item = popup._fiducials_table._table.item(0, 1)
    assert name_item is not None
    assert name_item.text() == "Nasion"


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
    stored = manager.get_global_item("fiducials")
    assert isinstance(stored, LabeledPoints)
    assert stored.points.tolist() == [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]


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

    index = selector.findData(CoordinateSystem.FIDUCIALS)
    if index < 0:
        return False
    model = selector.model()
    assert isinstance(model, QStandardItemModel)
    return bool(model.item(index).isEnabled())


def _frame_item_enabled(selector: FrameSelector, frame: CoordinateSystem) -> bool:
    """Whether a frame item exists and is enabled."""
    from PySide6.QtGui import QStandardItemModel

    index = selector.findData(frame)
    if index < 0:
        return False
    model = selector.model()
    assert isinstance(model, QStandardItemModel)
    return bool(model.item(index).isEnabled())


def test_concrete_popup_fiducials_frame_offers_voxel_when_volume_is_loaded(qapp, popups) -> None:
    """The fiducials table can display voxel coordinates with a NIfTI affine."""
    from virda_gui_new.dto.viewer3d import LabeledPoints

    manager = _make_manager()
    manager.set_step_context_item(
        NIFTI.index,
        "volume",
        MRIVolume(np.zeros((2, 2, 2)), _VOXEL_AFFINE, (0.5, 0.5, 0.5), ("R", "A", "S")),
    )
    manager.set_global_item("fiducials", LabeledPoints(_WORLD_FIDUCIALS, _FIDUCIAL_LABELS))

    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)

    index = popup._fiducials_frames.findData(CoordinateSystem.VOXEL)
    assert index >= 0
    assert _frame_item_enabled(popup._fiducials_frames, CoordinateSystem.VOXEL)
    popup._fiducials_frames.setCurrentIndex(index)

    assert popup._fiducials_table._frame is CoordinateSystem.VOXEL
    assert [_cell_text(popup._fiducials_table._table, 0, c) for c in (2, 3, 4)] == [
        "-20",
        "210",
        "-6",
    ]
    assert popup.collect_fiducial_rows()[0][1].tolist() == _WORLD_FIDUCIALS[0].tolist()


def test_concrete_popup_disables_voxel_frame_without_volume(qapp, popups) -> None:
    """The voxel option remains visible but disabled without an affine."""
    popup = LocalizationPopup(LOCALIZATION, _make_manager())
    popups.append(popup)

    assert popup._fiducials_frames.findData(CoordinateSystem.VOXEL) >= 0
    assert not _frame_item_enabled(popup._fiducials_frames, CoordinateSystem.VOXEL)


def test_concrete_popup_pick_button_asks_the_viewer(qapp, popups) -> None:
    """The Pick button asks the 3D view for a point of that slot."""
    from PySide6.QtWidgets import QPushButton

    popup = LocalizationPopup(LOCALIZATION, _make_manager())
    popups.append(popup)
    events: list[str] = []
    popup.pickRequested.connect(lambda slot: events.append(slot))

    button = popup._fiducials_table._table.cellWidget(0, 5)
    assert isinstance(button, QPushButton)
    button.click()

    assert events == ["NAS"]


def test_concrete_popup_pick_button_stops_the_same_slot(qapp, popups) -> None:
    """Picking the same slot twice leaves the picking mode."""
    from PySide6.QtWidgets import QPushButton

    popup = LocalizationPopup(LOCALIZATION, _make_manager())
    popups.append(popup)
    cancelled: list[int] = []
    popup.pickCancelled.connect(lambda: cancelled.append(1))

    button = popup._fiducials_table._table.cellWidget(0, 5)
    assert isinstance(button, QPushButton)
    button.click()
    button.click()

    assert cancelled == [1]


def test_concrete_popup_writes_the_picked_point(qapp, popups) -> None:
    """A picked point lands in the row of its slot."""
    popup = LocalizationPopup(LOCALIZATION, _make_manager())
    popups.append(popup)

    popup.set_picked_point("NAS", np.array([1.0, 2.0, 3.0]))

    rows = popup.collect_fiducial_rows()
    assert [name for name, _xyz in rows] == ["NAS"]
    assert rows[0][1].tolist() == [1.0, 2.0, 3.0]


def test_concrete_popup_frame_selectors_default_to_head(qapp, popups) -> None:
    """Both selectors default to head; head is available (scalp mesh set)."""
    popup = LocalizationPopup(LOCALIZATION, _make_manager())
    popups.append(popup)

    assert popup._fiducials_frames.frame() == CoordinateSystem.HEAD
    assert popup._electrodes_frames.frame() == CoordinateSystem.HEAD
    assert popup._fiducials_table._frame is CoordinateSystem.HEAD
    assert popup._electrodes_table._frame is CoordinateSystem.HEAD


def test_the_fiducials_table_has_no_fiducials_frame(qapp, popups) -> None:
    """The fiducials frame is defined by the rows the table shows."""
    from virda_gui_new.dto.viewer3d import LabeledPoints

    manager = _make_manager()
    manager.set_global_item("fiducials", LabeledPoints(_WORLD_FIDUCIALS, _FIDUCIAL_LABELS))

    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)

    assert popup._fiducials_frames.findData(CoordinateSystem.FIDUCIALS) == -1
    assert popup._electrodes_frames.findData(CoordinateSystem.FIDUCIALS) >= 0


def test_concrete_popup_fiducials_frame_needs_three_fiducials(qapp, popups) -> None:
    """The fiducials frame option is greyed out without all three points."""
    from virda_gui_new.dto.viewer3d import LabeledPoints

    manager = _make_manager()
    manager.set_global_item("fiducials", LabeledPoints(_WORLD_FIDUCIALS, _FIDUCIAL_LABELS))

    bare = LocalizationPopup(LOCALIZATION, _make_manager())
    popups.append(bare)
    assert not _fiducials_item_enabled(bare._electrodes_frames)

    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)
    assert _fiducials_item_enabled(popup._electrodes_frames)

    popup._electrodes_frames.setCurrentIndex(
        popup._electrodes_frames.findData(CoordinateSystem.FIDUCIALS)
    )

    # The canonical rows stay world, whatever the display frame is.
    assert popup.collect_fiducial_rows()[0][1].tolist() == _WORLD_FIDUCIALS[0].tolist()


def test_concrete_popup_empty_resets_fiducials_frame(qapp, popups) -> None:
    """Clearing the store makes the fiducials frame unavailable -> fallback."""
    from virda_gui_new.dto.viewer3d import LabeledPoints

    manager = _make_manager()
    manager.set_global_item("fiducials", LabeledPoints(_WORLD_FIDUCIALS, _FIDUCIAL_LABELS))
    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)

    popup._electrodes_frames.setCurrentIndex(
        popup._electrodes_frames.findData(CoordinateSystem.FIDUCIALS)
    )
    popup._fiducials_table.clear_grid()

    assert popup._electrodes_frames.frame() == CoordinateSystem.HEAD  # head fallback
    assert not _fiducials_item_enabled(popup._electrodes_frames)


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

    assert popup._fiducials_frames.frame() == CoordinateSystem.HEAD
    assert [
        cast("QTableWidgetItem", popup._fiducials_table._table.item(0, c)).text() for c in (2, 3, 4)
    ] == [
        "-7.5",
        "92.5",
        "-7.5",
    ]

    popup._measurements_table.add_row()
    popup._measurements_table._table.setItem(0, 0, QTableWidgetItem("A"))
    popup._measurements_table._table.setItem(0, 2, QTableWidgetItem("10"))

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
    calls: list[object] = []

    def fail_late(*_args, **_kwargs):
        calls.append(())
        return Electrodes([])

    monkeypatch.setattr("virda_gui_new.services.localization.runner.localize_electrodes", fail_late)
    popup = LocalizationPopup(LOCALIZATION, _make_manager())
    popups.append(popup)

    assert popup._hint_label.text() == "Blocked: not enough fiducials (need 3)."
    for row, coords in enumerate(("0 0 1", "1 0 0", "-1 0 0")):
        for column, text in enumerate(coords.split(), start=2):
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
    calls: list[_LocalizeCall] = []
    payload = Electrodes(
        [Electrode("A", {"LPA": 10.0}, ese_coords=np.zeros(3), scalp_coords=np.zeros(3))]
    )
    _patch_localize(monkeypatch, calls, payload)
    manager = _make_manager()
    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)

    for row, coords in enumerate(("0 0 1", "1 0 0", "-1 0 0")):
        for column, text in enumerate(coords.split(), start=2):
            popup._fiducials_table._table.setItem(row, column, QTableWidgetItem(text))
    popup._measurements_table.add_row()
    popup._measurements_table._table.setItem(0, 0, QTableWidgetItem("A"))
    popup._measurements_table._table.setItem(0, 1, QTableWidgetItem("10"))
    popup._measurements_table._table.setItem(0, 2, QTableWidgetItem("11"))
    popup._measurements_table._table.setItem(0, 3, QTableWidgetItem("12"))

    _settle(qapp, popup, "Localized 1 electrode.")

    assert popup._hint_label.text() == "Localized 1 electrode."
    assert len(calls) >= 1
    results = manager.get_step_context_item(LOCALIZATION.index, "electrodes")
    assert isinstance(results, Electrodes)
    assert results.ids == ["A"]
    grid = popup._electrodes_table._table
    assert _cell_text(grid, 0, 0) == "A"
    assert _cell_text(grid, 0, 1) == "0.00"


def test_popup_prefills_fiducials_from_the_global_store(qapp, popups) -> None:
    """The popup builds and fills the grid when the store holds fiducials.

    ``build_form`` prefills the fiducial table, and the prefill schedules
    an input change: the runner attribute must already exist while the
    form is built.
    """
    from virda_gui_new.dto.viewer3d import LabeledPoints

    class _Probe(LocalizationPopup):
        def build_form(self, layout: QVBoxLayout) -> None:
            super().build_form(layout)
            assert self._localization_runner is None

    manager = _make_manager()
    manager.set_global_item(
        "fiducials",
        LabeledPoints(
            np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            ("NAS", "LPA", "RPA"),
        ),
    )

    popup = _Probe(LOCALIZATION, manager)
    popups.append(popup)

    rows = popup.collect_fiducial_rows()
    assert [name for name, _xyz in rows] == ["NAS", "LPA", "RPA"]


def test_popup_prefills_the_measurements_from_the_global_store(qapp, popups) -> None:
    """The stored measurements are shown in the grid again."""
    manager = _make_manager()
    manager.set_global_item(
        "measurements", Electrodes([Electrode("A", {"LPA": 10.0, "RPA": 12.0})])
    )

    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)

    grid = popup._measurements_table._table
    assert grid.rowCount() == 1
    assert _cell_text(grid, 0, 0) == "A"
    assert _cell_text(grid, 0, 2) == "10"  # the LPA column
    assert _cell_text(grid, 0, 3) == "12"  # the RPA column


def test_popup_prefills_the_electrodes_from_the_seeded_points(qapp, popups, monkeypatch) -> None:
    """The localized coordinates of the project are shown in the grid."""
    from virda_gui_new.dto.viewer3d import LabeledPoints

    _patch_localize(monkeypatch, [], Electrodes([]))
    manager = _make_manager()
    manager.set_global_item("measurements", Electrodes([Electrode("A", {"LPA": 10.0})]))
    manager.set_step_context_item(
        LOCALIZATION.index,
        "scalp_points",
        LabeledPoints(np.array([[1.0, 2.0, 3.0]]), ("A",)),
    )

    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)

    grid = popup._electrodes_table._table
    assert grid.rowCount() == 1
    assert _cell_text(grid, 0, 0) == "A"
    # The seeded scalp point shows in the scalp columns.
    assert _cell_text(grid, 0, 4) == "1.00"


def test_edited_measurement_recomputes_that_electrode(qapp, popups, monkeypatch) -> None:
    """Without the global calibration one edited electrode is enough."""
    calls: list[_LocalizeCall] = []
    _patch_localize(
        monkeypatch,
        calls,
        Electrodes([Electrode("A", {"LPA": 10.0}, scalp_coords=np.zeros(3))]),
    )
    manager = _make_manager()
    popup = RecordingPopup(
        manager,
        fiducial_rows=_FIDUCIAL_ROWS,
        table=_TWO_ELECTRODES,
    )
    popups.append(popup)
    # The calibration shifts every electrode, so it needs the whole set.
    popup._calibrate_box.setChecked(False)
    calls.clear()

    popup._on_measurement_edited("A")
    _settle(qapp, popup, "Localized 1 electrode.")

    assert len(calls) == 1
    _fiducials, electrodes = calls[0]
    assert [item.electrode_id for item in electrodes.items] == ["A"]


def test_edited_measurement_with_the_calibration_recomputes_only_that_electrode(
    qapp, popups, monkeypatch
) -> None:
    """A row edit reuses the current calibration and remains row-scoped."""
    calls: list[_LocalizeCall] = []
    _patch_localize(
        monkeypatch,
        calls,
        Electrodes([Electrode("A", {"LPA": 10.0}, scalp_coords=np.zeros(3))]),
    )
    manager = _make_manager()
    popup = RecordingPopup(
        manager,
        fiducial_rows=_FIDUCIAL_ROWS,
        table=_TWO_ELECTRODES,
    )
    popups.append(popup)
    calls.clear()

    popup._on_measurement_edited("A")
    _settle(qapp, popup, "Localized 1 electrode.")

    assert len(calls) == 1
    _fiducials, electrodes = calls[0]
    assert [item.electrode_id for item in electrodes.items] == ["A"]


def test_incomplete_measurement_edit_does_not_schedule_localization(qapp, popups) -> None:
    popup = RecordingPopup(
        _make_manager(),
        fiducial_rows=_FIDUCIAL_ROWS,
        table={"A": {"NAS": 9.0, "LPA": 10.0}},
    )
    popups.append(popup)

    popup._on_measurement_edited("A")

    assert not _run_pending(popup)


def test_a_changed_cell_recomputes_only_its_electrode(qapp, popups, monkeypatch) -> None:
    """A changed distance moves its own electrode, not the whole set."""
    calls: list[_LocalizeCall] = []
    _patch_localize(
        monkeypatch,
        calls,
        Electrodes([Electrode("A", {"LPA": 10.0}, scalp_coords=np.zeros(3))]),
    )
    manager = _make_manager()
    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)
    popup._fiducials_table.set_rows(_FIDUCIAL_ROWS)
    popup._measurements_table.set_rows(_TWO_ELECTRODES)
    popup._calibrate_box.setChecked(False)
    calls.clear()

    popup._measurements_table._table.setItem(0, 1, QTableWidgetItem("11"))
    _settle(qapp, popup, "Localized 1 electrode.")

    assert len(calls) == 1
    _fiducials, electrodes = calls[0]
    assert [item.electrode_id for item in electrodes.items] == ["A"]


def test_a_removed_row_drops_the_electrode_without_a_recompute(qapp, popups, monkeypatch) -> None:
    """A removed electrode leaves the results, and nothing is recomputed."""
    calls: list[_LocalizeCall] = []
    payload = Electrodes(
        [
            Electrode("A", {"NAS": 10.0}, scalp_coords=np.zeros(3)),
            Electrode("B", {"NAS": 11.0}, scalp_coords=np.ones(3)),
        ]
    )
    _patch_localize(monkeypatch, calls, payload)
    manager = _make_manager()
    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)
    popup._fiducials_table.set_rows(_FIDUCIAL_ROWS)
    popup._measurements_table.set_rows(_TWO_ELECTRODES)
    popup._on_inputs_changed()
    _settle(qapp, popup, "Localized 2 electrodes.")
    calls.clear()

    popup._measurements_table._table.setItem(0, 0, QTableWidgetItem("A"))
    popup._measurements_table._table.selectRow(1)
    popup._measurements_table.remove_selected_rows()

    assert calls == []
    results = manager.get_step_context_item(LOCALIZATION.index, "electrodes")
    assert isinstance(results, Electrodes)
    assert results.ids == ["A"]


def test_relocalize_button_is_disabled_without_fiducials(qapp, popups, monkeypatch) -> None:
    """Without the fiducials the button waits, the table stays editable."""
    _patch_localize(monkeypatch, [], Electrodes([]))
    popup = LocalizationPopup(LOCALIZATION, _make_manager())
    popups.append(popup)

    button = popup._relocalize_button
    assert button is not None
    assert button.isEnabled() is False
    assert popup._measurements_table.isEnabled() is True


def test_relocalize_button_recomputes_every_electrode(qapp, popups, monkeypatch) -> None:
    """The button asks for a full recompute of the current tables."""
    calls: list[_LocalizeCall] = []
    _patch_localize(monkeypatch, calls, Electrodes([]))
    manager = _make_manager()
    popup = RecordingPopup(
        manager,
        fiducial_rows=_FIDUCIAL_ROWS,
        table=_TWO_ELECTRODES,
    )
    popups.append(popup)
    calls.clear()

    button = popup._relocalize_button
    assert button is not None
    button.click()
    _settle(qapp, popup, "Localized 2 electrodes.")

    assert len(calls) == 1
    _fiducials, electrodes = calls[0]
    assert [item.electrode_id for item in electrodes.items] == ["A", "B"]


def _wait_for_the_save(qapp, popup, timeout_s: float = 5.0) -> None:
    """Pump events until the debounced save has run."""
    deadline = time.monotonic() + timeout_s
    while popup._save_timer.isActive() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


def test_the_tables_are_stored_when_they_change(qapp, popups, tmp_path) -> None:
    """An edited cell reaches the project, even when no run follows."""
    manager = _make_manager()
    popup = RecordingPopup(
        manager,
        fiducial_rows=_FIDUCIAL_ROWS,
        table=_MEASUREMENTS,
        project=tmp_path,
    )
    popups.append(popup)

    popup._on_inputs_changed()
    _wait_for_the_save(qapp, popup)

    assert (tmp_path / "localization" / "fiducials.json").is_file()
    assert (tmp_path / "localization" / "measurements.json").is_file()


def test_imported_measurements_are_stored_in_the_project(
    qapp, popups, monkeypatch, tmp_path
) -> None:
    """An import is written to the project, even without a localization run."""
    _patch_localize(monkeypatch, [], Electrodes([]))
    manager = _make_manager()
    popup = LocalizationPopup(LOCALIZATION, manager, project=tmp_path)
    popups.append(popup)

    source = tmp_path / "measurements.json"
    source.write_text(
        json.dumps({"electrodes": [{"electrode_id": "A", "measured_distances": {"LPA": 10.0}}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "virda_gui_new.widgets.popup.localization.popup.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (str(source), "")),
    )

    popup._import_measurements_from_file()

    stored = json.loads(
        (tmp_path / "localization" / "measurements.json").read_text(encoding="utf-8")
    )
    assert stored["electrodes"] == [{"electrode_id": "A", "measured_distances": {"LPA": 10.0}}]


def test_localization_options_are_saved_and_restored(qapp, popups, tmp_path) -> None:
    """The localization settings are stored in the project and read back."""
    manager = _make_manager()
    popup = RecordingPopup(
        manager,
        fiducial_rows=_FIDUCIAL_ROWS,
        table={"A": {"LPA": 10.0}},
        project=tmp_path,
    )
    popups.append(popup)

    popup._calibrate_box.setChecked(False)
    popup._threshold_spin.setValue(12.5)

    stored = json.loads((tmp_path / "config" / "localization.json").read_text(encoding="utf-8"))
    assert stored["calibrate_ese_offset"] is False
    assert stored["residual_threshold_mm"] == 12.5

    restored = RecordingPopup(
        manager,
        fiducial_rows=_FIDUCIAL_ROWS,
        table={"A": {"LPA": 10.0}},
        project=tmp_path,
    )
    popups.append(restored)

    assert restored._calibrate_box.isChecked() is False
    assert restored._threshold_spin.value() == 12.5


def test_an_electrode_with_two_distances_is_not_localized(qapp, popups, monkeypatch) -> None:
    """Only an electrode with all three distances is localized."""
    calls: list[_LocalizeCall] = []
    _patch_localize(monkeypatch, calls, Electrodes([]))
    manager = _make_manager()
    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)

    popup._fiducials_table.set_rows(_FIDUCIAL_ROWS)
    popup._measurements_table.set_rows({"A": {"NAS": 10.0, "LPA": 11.0}})

    assert "no measurements" in popup._hint_label.text()
    assert not _run_pending(popup)
    time.sleep(0.6)
    assert calls == []


def test_the_localized_electrodes_reach_the_scene_context(qapp, popups, monkeypatch) -> None:
    """A localized electrode writes its points into the step context."""
    payload = Electrodes(
        [
            Electrode(
                "A",
                {"NAS": 10.0, "LPA": 11.0, "RPA": 12.0},
                ese_coords=np.zeros(3),
                scalp_coords=np.ones(3),
            )
        ]
    )
    _patch_localize(monkeypatch, [], payload)
    manager = _make_manager()
    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)

    popup._fiducials_table.set_rows(_FIDUCIAL_ROWS)
    popup._measurements_table.set_rows({"A": {"NAS": 10.0, "LPA": 11.0, "RPA": 12.0}})
    popup._on_measurement_edited("A")
    _settle(qapp, popup, "Localized 1 electrode.")

    scalp_points = manager.get_step_context_item(LOCALIZATION.index, "scalp_points")
    ese_points = manager.get_step_context_item(LOCALIZATION.index, "ese_points")
    assert scalp_points is not None
    assert ese_points is not None
    assert scalp_points.labels == ("A",)
    assert ese_points.labels == ("A",)
    assert scalp_points.points[0].tolist() == [1.0, 1.0, 1.0]


def test_opening_with_only_incomplete_measurements_does_not_run(qapp, popups, monkeypatch) -> None:
    """Opening the popup waits until one measurement row is complete."""
    from virda_gui_new.dto.viewer3d import LabeledPoints

    calls: list[_LocalizeCall] = []
    _patch_localize(monkeypatch, calls, Electrodes([]))
    manager = _make_manager()
    manager.set_global_item("fiducials", LabeledPoints(_WORLD_FIDUCIALS, _FIDUCIAL_LABELS))
    manager.set_global_item("measurements", Electrodes([Electrode("A", {"NAS": 10.0})]))
    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)

    assert popup._localization_runner is not None
    assert not _run_pending(popup)
    time.sleep(0.6)
    assert calls == []


def test_the_step_computes_the_results_it_has_no_result_for(qapp, popups, monkeypatch) -> None:
    """The measured distances without results are localized when the step opens."""
    from virda_gui_new.services.localization import labeled_points_from_rows

    calls: list[_LocalizeCall] = []
    payload = Electrodes(
        [Electrode("A", {"NAS": 10.0}, ese_coords=np.zeros(3), scalp_coords=np.zeros(3))]
    )
    _patch_localize(monkeypatch, calls, payload)
    manager = _make_manager()
    manager.set_global_item("fiducials", labeled_points_from_rows(_FIDUCIAL_ROWS))
    manager.set_global_item("measurements", Electrodes([Electrode("A", _MEASUREMENTS["A"])]))
    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)

    _settle(qapp, popup, "Localized 1 electrode.")

    assert len(calls) == 1
    assert manager.get_step_context_item(LOCALIZATION.index, "electrodes") is payload


def test_opening_recomputes_existing_results(qapp, popups, monkeypatch) -> None:
    """Opening the step reruns all complete measurements, even with results."""
    from virda_gui_new.services.localization import labeled_points_from_rows

    calls: list[_LocalizeCall] = []
    _patch_localize(monkeypatch, calls, Electrodes([]))
    manager = _make_manager()
    manager.set_global_item("fiducials", labeled_points_from_rows(_FIDUCIAL_ROWS))
    manager.set_global_item("measurements", Electrodes([Electrode("A", _MEASUREMENTS["A"])]))
    manager.set_step_context_item(LOCALIZATION.index, "electrodes", Electrodes([]))
    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)

    _settle(qapp, popup, "Localized 0 electrodes.")
    assert len(calls) == 1


def test_the_published_results_follow_the_measurement_table(qapp, popups, monkeypatch) -> None:
    """A table without electrodes publishes no results."""
    payload = Electrodes(
        [Electrode("A", {"LPA": 10.0}, ese_coords=np.zeros(3), scalp_coords=np.zeros(3))]
    )
    _patch_localize(monkeypatch, [], payload)
    manager = _make_manager()
    popup = RecordingPopup(manager, fiducial_rows=_FIDUCIAL_ROWS, table=_MEASUREMENTS)
    popups.append(popup)

    popup._on_inputs_changed()
    _settle(qapp, popup, "Localized 1 electrode.")

    published = manager.get_step_context_item(LOCALIZATION.index, "electrodes")
    assert isinstance(published, Electrodes)
    assert published.ids == ["A"]

    # The electrode leaves the table: the results follow it.
    popup._table = {}
    popup._on_inputs_settled()

    published = manager.get_step_context_item(LOCALIZATION.index, "electrodes")
    assert isinstance(published, Electrodes)
    assert published.ids == []


def test_the_measurements_are_mirrored_without_a_run(qapp, popups) -> None:
    """The store follows the table even when no localization ran."""
    manager = _make_manager()
    popup = RecordingPopup(manager, fiducial_rows=_FIDUCIAL_ROWS, table=_MEASUREMENTS)
    popups.append(popup)

    popup._on_inputs_settled()

    stored = manager.get_global_item("measurements")
    assert isinstance(stored, Electrodes)
    assert stored.ids == ["A"]
    stored_a = stored.get("A")
    assert stored_a is not None
    assert stored_a.measured_distances == _MEASUREMENTS["A"]


def test_the_ready_popup_relocalizes_on_open(qapp, popups, monkeypatch) -> None:
    """Opening with complete inputs starts a full localization run."""
    calls: list[_LocalizeCall] = []
    _patch_localize(
        monkeypatch,
        calls,
        Electrodes([Electrode("A", _MEASUREMENTS["A"], scalp_coords=np.zeros(3))]),
    )
    manager = _make_manager()
    popup = RecordingPopup(manager, fiducial_rows=_FIDUCIAL_ROWS, table=_MEASUREMENTS)
    popups.append(popup)

    _settle(qapp, popup, "Localized 1 electrode.")
    assert len(calls) == 1
    assert calls[0][1].ids == ["A"]
