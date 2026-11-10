"""Unit tests for :class:`virda_gui_new.widgets.popup.localization.abstract`.

Runs under the offscreen Qt platform; ``ops.localize`` is mocked, so no
heavy mesh work runs; only hint text, context writes and hooks flow.
"""

import os
import time
from typing import Any

import numpy as np
import pytest

from virda.models.electrode import Electrode, Electrodes
from virda.models.mesh import Mesh
from virda_gui_new.api import ops
from virda_gui_new.dto.steps import LOCALIZATION, STEPS
from virda_gui_new.services.steps_manager import StepsManager
from virda_gui_new.widgets.popup.localization.abstract import (
    LocalizationPopupBase,
)

_ESE = Mesh(np.zeros((4, 3)), np.zeros((1, 3), dtype=int))
_SCALP = Mesh(np.zeros((3, 3)), np.zeros((1, 3), dtype=int))
_FIDUCIAL_ROWS = [
    ("Nasion", np.array([0.0, 0.0, 1.0])),
    ("LPA", np.array([1.0, 0.0, 0.0])),
    ("RPA", np.array([-1.0, 0.0, 0.0])),
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

    def __init__(self, manager: StepsManager, fiducial_rows=None, table=None, parent=None) -> None:
        self._rows = [] if fiducial_rows is None else fiducial_rows
        self._table = {} if table is None else table
        self.localized_called: list[Any] = []
        self.statuses: list[Any] = []
        super().__init__(LOCALIZATION, manager)

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
