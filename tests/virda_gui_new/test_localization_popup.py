"""Unit tests for :class:`virda_gui_new.widgets.popup.localization.abstract`.

Runs under the offscreen Qt platform; ``localize_electrodes`` is mocked, so no
heavy mesh work runs; only hint text, context writes and hooks flow.
"""

import json
import time

import numpy as np
import pytest
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QVBoxLayout

from tests.virda_gui_new.fixtures.steps import advance
from virda.models.electrode import Electrode, Electrodes
from virda.models.fiducial import Fiducials
from virda.models.mesh import Mesh
from virda_gui_new.dto.steps import LOCALIZATION, STEPS
from virda_gui_new.services.steps_manager import StepsManager
from virda_gui_new.widgets.popup.localization import LocalizationPopup
from virda_gui_new.widgets.popup.localization.abstract import (
    LocalizationPopupBase,
)

type _LocalizeCall = tuple[Fiducials, Electrodes]

_ESE = Mesh(np.zeros((4, 3)), np.zeros((1, 3), dtype=int))
_SCALP = Mesh(np.zeros((3, 3)), np.zeros((1, 3), dtype=int))
_WORLD_FIDUCIALS = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
_FIDUCIAL_LABELS = ("Nasion", "LPA", "RPA")
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
    assert popup._hint_label.text() == _NO_GEOMETRY
    assert not popup._run_button.isVisibleTo(popup)
    popup._on_inputs_changed()  # no-op without geometry
    assert popup._hint_label.text() == _NO_GEOMETRY


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
    manager.set_global_item("fiducials", labeled_points_from_rows(_FIDUCIAL_ROWS))
    notifications: list[int] = []
    manager.subscribe(lambda: notifications.append(len(notifications)))
    popup = RecordingPopup(manager, fiducial_rows=_FIDUCIAL_ROWS, table=_MEASUREMENTS)
    popups.append(popup)

    popup._on_inputs_changed()
    _settle(qapp, popup, "Localized 1 electrode.")

    # The electrodes and the two point sets notified; the store was
    # untouched.
    assert len(notifications) == 3


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
    name_item = popup._fiducials_table._table.item(0, 0)
    assert name_item is not None
    assert name_item.text() == "Nasion"


def test_concrete_popup_prefill_absent_keeps_empty_grid(qapp, popups) -> None:
    """Without seeded fiducials the table starts empty as before."""
    popup = LocalizationPopup(LOCALIZATION, _make_manager())
    popups.append(popup)

    assert popup.collect_fiducial_rows() == []


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
    for _row in range(3):
        popup._fiducials_table.add_row()
    assert popup._hint_label.text() == "Blocked: not enough fiducials (need 3)."
    for row, coords in enumerate(("0 0 1", "1 0 0", "-1 0 0")):
        popup._fiducials_table._table.setItem(row, 0, QTableWidgetItem("Nasion"))
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
    calls: list[_LocalizeCall] = []
    payload = Electrodes(
        [Electrode("A", {"LPA": 10.0}, ese_coords=np.zeros(3), scalp_coords=np.zeros(3))]
    )
    _patch_localize(monkeypatch, calls, payload)
    manager = _make_manager()
    popup = LocalizationPopup(LOCALIZATION, manager)
    popups.append(popup)

    for _row in range(3):
        popup._fiducials_table.add_row()
    for row, (name, coords) in enumerate(
        zip(("Nasion", "LPA", "RPA"), ("0 0 1", "1 0 0", "-1 0 0"), strict=True)
    ):
        popup._fiducials_table._table.setItem(row, 0, QTableWidgetItem(name))
        for column, text in enumerate(coords.split(), start=1):
            popup._fiducials_table._table.setItem(row, column, QTableWidgetItem(text))
    popup._measurements_table.add_row()
    for row, (electrode, fiducial, distance) in enumerate(
        (("A", "NAS", "10"), ("A", "LPA", "11"), ("A", "RPA", "12"))
    ):
        if row:
            popup._measurements_table.add_row()
        for column, text in enumerate((electrode, fiducial, distance)):
            popup._measurements_table._table.setItem(row, column, QTableWidgetItem(text))

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


def test_the_step_does_not_run_on_its_own(qapp, popups, monkeypatch) -> None:
    """Opening the popup starts no localization: the user does."""
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


def test_the_ready_popup_says_so(qapp, popups) -> None:
    """The popup with the fiducials and a measurement row is ready."""
    manager = _make_manager()
    popup = RecordingPopup(manager, fiducial_rows=_FIDUCIAL_ROWS, table=_MEASUREMENTS)
    popups.append(popup)

    assert popup._hint_label.text() == "Ready."
