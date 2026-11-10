"""Unit tests for :class:`virda_gui_new.widgets.popup.mesh_postprocessing`.

Runs under the offscreen Qt platform; skips without a Qt install. The
``ops`` clean/decimate/smooth facades are mocked, so no heavy mesh work
runs; only form values flow to the mocks and a tiny real result back.
"""

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from virda.models.mesh import Mesh
from virda_gui.api.options import CleanOptions, DecimateOptions, SmootherKind, SmoothOptions
from virda_gui_new.api import ops
from virda_gui_new.dto.steps import MESH_POSTPROCESSING, STEPS
from virda_gui_new.services.steps_manager import StepsManager
from virda_gui_new.widgets.popup.mesh_postprocessing import MeshPostprocessingPopup

_NO_MESH = "Mesh postprocessing needs an in-memory mesh (run the previous steps first)."
_MESH = Mesh(np.zeros((4, 3)), np.zeros((1, 3), dtype=int))


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


def _make_manager(mesh: Mesh | None) -> StepsManager:
    """Manager at step 2; sets the step-1 scalp surface plus a step-2 mesh."""
    steps = StepsManager(tuple(step.required_keys for step in STEPS))
    steps.next_step()
    steps.set_step_context_item(1, "scalp_surface", _MESH)
    steps.next_step()
    if mesh is not None:
        steps.set_step_context_item(2, "mesh", mesh)
    return steps


@pytest.fixture
def manager(qapp) -> StepsManager:
    """Manager at step 2 with a ready mesh in the step-1 context."""
    return _make_manager(None)


@pytest.fixture
def popups(qapp):
    """Collect popups and stop their runner threads at teardown."""
    created: list[Any] = []
    yield created
    for popup in created:
        popup.shutdown()


def _patch_ops(monkeypatch, calls: list[Any]) -> None:
    def fake_clean(mesh, options):
        calls.append(("clean", mesh, options))
        return _MESH

    def fake_decimate(mesh, options):
        calls.append(("decimate", mesh, options))
        return _MESH

    def fake_smooth(mesh, options):
        calls.append(("smooth", mesh, options))
        return _MESH

    monkeypatch.setattr(ops, "clean", fake_clean)
    monkeypatch.setattr(ops, "decimate", fake_decimate)
    monkeypatch.setattr(ops, "smooth", fake_smooth)


def _settle(qapp, popup, timeout_s: float = 5.0) -> None:
    """Pump events until the run settles (busy flag cleared)."""
    deadline = time.monotonic() + timeout_s
    while popup._busy and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


def test_form_defaults(qapp, manager, popups) -> None:
    popup = MeshPostprocessingPopup(MESH_POSTPROCESSING, manager)
    popups.append(popup)

    assert popup._run_button.isEnabled()
    assert popup._min_spin.value() == 100
    assert popup._merge_spin.value() == 7
    assert popup._density_spin.value() == 100.0
    assert popup._smooth_combo.currentText() == "laplacian"
    assert popup._iterations_spin.value() == 5
    assert popup._lamb_spin.value() == 0.5
    assert popup._nu_spin.value() == -0.53
    assert popup._iterations_spin.isEnabled()
    assert popup._lamb_spin.isEnabled()
    assert popup._nu_spin.isEnabled()


def test_smoothing_fields_follow_combo(qapp, manager, popups) -> None:
    popup = MeshPostprocessingPopup(MESH_POSTPROCESSING, manager)
    popups.append(popup)

    popup._smooth_combo.setCurrentText("none")
    assert not popup._iterations_spin.isEnabled()
    assert not popup._lamb_spin.isEnabled()
    assert not popup._nu_spin.isEnabled()

    popup._smooth_combo.setCurrentText("taubin")
    assert popup._iterations_spin.isEnabled()


def test_input_resolution(qapp, popups, manager) -> None:
    popup = MeshPostprocessingPopup(MESH_POSTPROCESSING, manager)
    popups.append(popup)
    assert popup._resolve_input() is _MESH

    steps = _make_manager(_MESH)
    fallback = Mesh(np.zeros((3, 3)), np.zeros((1, 3), dtype=int))
    steps.set_step_context_item(1, "scalp_surface", fallback)
    popup2 = MeshPostprocessingPopup(MESH_POSTPROCESSING, steps)
    popups.append(popup2)
    assert popup2._resolve_input() is _MESH  # step-2 mesh wins


def test_collect_run_step1_fallback(qapp, manager, popups, monkeypatch) -> None:
    calls: list[Any] = []
    _patch_ops(monkeypatch, calls)
    popup = MeshPostprocessingPopup(MESH_POSTPROCESSING, manager)
    popups.append(popup)

    work = popup._collect_run()
    assert calls == []  # nothing runs until the work callable is invoked

    work()
    assert calls == [
        ("clean", _MESH, CleanOptions(min_component_vertices=100, merge_digits=7)),
        ("decimate", _MESH, DecimateOptions(density_percent=100.0)),
        (
            "smooth",
            _MESH,
            SmoothOptions(smoother="laplacian", iterations=5, lamb=0.5, nu=-0.53),
        ),
    ]


def test_collect_run_maps_form_values(qapp, manager, popups, monkeypatch) -> None:
    calls: list[Any] = []
    _patch_ops(monkeypatch, calls)
    popup = MeshPostprocessingPopup(MESH_POSTPROCESSING, manager)
    popups.append(popup)

    popup._min_spin.setValue(50)
    popup._merge_spin.setValue(3)
    popup._density_spin.setValue(25.5)
    popup._smooth_combo.setCurrentText("taubin")
    popup._iterations_spin.setValue(10)
    popup._lamb_spin.setValue(1.25)
    popup._nu_spin.setValue(-2.5)

    popup._collect_run()()
    smoother: SmootherKind = "taubin"
    assert calls == [
        ("clean", _MESH, CleanOptions(min_component_vertices=50, merge_digits=3)),
        ("decimate", _MESH, DecimateOptions(density_percent=25.5)),
        (
            "smooth",
            _MESH,
            SmoothOptions(smoother=smoother, iterations=10, lamb=1.25, nu=-2.5),
        ),
    ]


def test_collect_run_none_smoother_still_calls_smooth(qapp, manager, popups, monkeypatch) -> None:
    calls: list[Any] = []
    _patch_ops(monkeypatch, calls)
    popup = MeshPostprocessingPopup(MESH_POSTPROCESSING, manager)
    popups.append(popup)

    popup._smooth_combo.setCurrentText("none")
    popup._collect_run()()
    assert calls == [
        ("clean", _MESH, CleanOptions(min_component_vertices=100, merge_digits=7)),
        ("decimate", _MESH, DecimateOptions(density_percent=100.0)),
        (
            "smooth",
            _MESH,
            SmoothOptions(smoother="none", iterations=5, lamb=0.5, nu=-0.53),
        ),
    ]


def test_apply_result_writes_mesh(qapp, manager, popups) -> None:
    popup = MeshPostprocessingPopup(MESH_POSTPROCESSING, manager)
    popups.append(popup)

    popup._apply_result(_MESH)

    assert manager.get_step_context_item(2, "mesh") is _MESH


def test_mocked_run_writes_context_and_advances(qapp, manager, popups, monkeypatch) -> None:
    calls: list[Any] = []
    _patch_ops(monkeypatch, calls)
    popup = MeshPostprocessingPopup(MESH_POSTPROCESSING, manager)
    popups.append(popup)
    completed: list[Any] = []
    popup.completed.connect(completed.append)

    popup._run_button.click()
    _settle(qapp, popup)

    assert popup._hint_label.text() == "Done."
    assert completed == [2]
    assert manager.get_step_context_item(2, "mesh") is _MESH
    assert manager.get_current_step() == 3  # base advanced past step 2


def test_missing_mesh_fails_collect(qapp, popups) -> None:
    steps = StepsManager(tuple(step.required_keys for step in STEPS))
    popup = MeshPostprocessingPopup(MESH_POSTPROCESSING, steps)
    popups.append(popup)

    assert not popup._run_button.isEnabled()
    with pytest.raises(ValueError, match="Mesh postprocessing needs an in-memory mesh"):
        popup._collect_run()


def test_path_mesh_still_fails_collect(qapp, popups) -> None:
    steps = _make_manager(None)
    steps.set_step_context_item(2, "mesh", Path("some.ply"))
    popup = MeshPostprocessingPopup(MESH_POSTPROCESSING, steps)
    popups.append(popup)

    with pytest.raises(ValueError) as excinfo:
        popup._collect_run()
    assert str(excinfo.value) == _NO_MESH
