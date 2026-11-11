"""Unit tests for :class:`virda_gui_new.widgets.popup.ese_mesh`.

Runs under the offscreen Qt platform; skips without a Qt install. The
``ops`` generate_ese facade is mocked, so no heavy mesh work runs; only
form values flow to the mock and a tiny real result back.
"""

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from virda.models.mesh import Mesh
from virda_gui.api.options import EseOptions
from virda_gui_new.api import ops
from virda_gui_new.dto.steps import ESE_MESH, MESH_POSTPROCESSING, STEPS
from virda_gui_new.services.steps_manager import StepsManager
from virda_gui_new.widgets.popup.ese_mesh import EseMeshPopup

_NO_MESH = "ESE mesh needs the postprocessed mesh (run the previous steps first)."
_MESH = Mesh(np.zeros((4, 3)), np.zeros((1, 3), dtype=int))


def _result() -> tuple[Mesh, np.ndarray]:
    """A tiny real (mesh, normals) pair as ``ops.generate_ese`` returns."""
    return Mesh(np.zeros((4, 3)), np.zeros((1, 3), dtype=int)), np.ones((4, 3))


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
    """Manager at step 3; sets the step-1 scalp surface plus a step-2 mesh."""
    steps = StepsManager(tuple(step.required_keys for step in STEPS))
    steps.next_step()
    steps.set_step_context_item(1, "scalp_surface", _MESH)
    steps.next_step()
    if mesh is not None:
        steps.set_step_context_item(2, "mesh", mesh)
    steps.next_step()
    return steps


@pytest.fixture
def manager(qapp) -> StepsManager:
    """Manager at step 3 with a ready mesh in the step-2 context."""
    return _make_manager(_MESH)


@pytest.fixture
def popups(qapp):
    """Collect popups and stop their runner threads at teardown."""
    created: list[Any] = []
    yield created
    for popup in created:
        popup.shutdown()


def _patch_ops(monkeypatch, calls: list[Any]) -> None:
    def fake_generate_ese(mesh, options):
        calls.append(("generate_ese", mesh, options))
        return _result()

    monkeypatch.setattr(ops, "generate_ese", fake_generate_ese)


def _settle(qapp, popup, timeout_s: float = 5.0) -> None:
    """Pump events until the run settles (busy flag cleared)."""
    deadline = time.monotonic() + timeout_s
    while popup._busy and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


def test_form_defaults(qapp, manager, popups) -> None:
    popup = EseMeshPopup(ESE_MESH, manager)
    popups.append(popup)

    assert popup._run_button.isEnabled()
    assert popup._offset_spin.value() == 1.0
    assert popup._radius_spin.value() == 10.0
    assert popup._auto_k_check.isChecked()
    assert not popup._k_spin.isEnabled()
    assert popup._k_spin.value() == 10
    assert not popup._weighted_check.isChecked()
    assert popup._sigma_spin.value() == 5.0
    assert popup._min_neighbors_spin.value() == 5


def test_k_field_follows_auto_check(qapp, manager, popups) -> None:
    popup = EseMeshPopup(ESE_MESH, manager)
    popups.append(popup)

    popup._auto_k_check.setChecked(False)
    assert popup._k_spin.isEnabled()

    popup._auto_k_check.setChecked(True)
    assert not popup._k_spin.isEnabled()


def test_input_resolution(qapp, popups, manager) -> None:
    popup = EseMeshPopup(ESE_MESH, manager)
    popups.append(popup)
    assert popup._resolve_input() is _MESH

    steps = _make_manager(None)
    fallback = Mesh(np.zeros((3, 3)), np.zeros((1, 3), dtype=int))
    steps.set_step_context_item(1, "scalp_surface", fallback)
    popup2 = EseMeshPopup(ESE_MESH, steps)
    popups.append(popup2)
    assert popup2._resolve_input() is fallback  # step-1 fallback without step-2 mesh

    steps.set_step_context_item(2, "mesh", _MESH)
    popup3 = EseMeshPopup(ESE_MESH, steps)
    popups.append(popup3)
    assert popup3._resolve_input() is _MESH  # step-2 mesh wins


def test_collect_run_maps_form_values(qapp, manager, popups, monkeypatch) -> None:
    calls: list[Any] = []
    _patch_ops(monkeypatch, calls)
    popup = EseMeshPopup(ESE_MESH, manager)
    popups.append(popup)

    popup._offset_spin.setValue(2.5)
    popup._radius_spin.setValue(20.0)
    popup._auto_k_check.setChecked(False)
    popup._k_spin.setValue(12)
    popup._weighted_check.setChecked(True)
    popup._sigma_spin.setValue(7.5)
    popup._min_neighbors_spin.setValue(8)

    work = popup._collect_run()
    assert calls == []  # nothing runs until the work callable is invoked

    result = work()
    assert result[0].vertices.shape == (4, 3)
    assert result[1].shape == (4, 3)
    assert calls == [
        (
            "generate_ese",
            _MESH,
            EseOptions(
                ese_offset_mm=2.5,
                neighborhood_radius_mm=20.0,
                k_neighbors=12,
                use_weighted_pca=True,
                pca_sigma_mm=7.5,
                min_neighbors=8,
            ),
        )
    ]


def test_collect_run_auto_k_gives_none(qapp, manager, popups, monkeypatch) -> None:
    calls: list[Any] = []
    _patch_ops(monkeypatch, calls)
    popup = EseMeshPopup(ESE_MESH, manager)
    popups.append(popup)

    popup._collect_run()()
    assert calls == [
        (
            "generate_ese",
            _MESH,
            EseOptions(ese_offset_mm=1.0, neighborhood_radius_mm=10.0),
        )
    ]


def test_apply_result_writes_both_keys(qapp, manager, popups) -> None:
    popup = EseMeshPopup(ESE_MESH, manager)
    popups.append(popup)

    assert not manager.is_step_done(3)
    popup._apply_result(_result())

    ese = manager.get_step_context_item(3, "ese_mesh")
    normals = manager.get_step_context_item(3, "normals")
    assert ese.vertices.shape == (4, 3)
    assert normals.shape == (4, 3)
    assert manager.is_step_done(3)


def test_mocked_run_writes_context_and_advances(qapp, manager, popups, monkeypatch) -> None:
    calls: list[Any] = []
    _patch_ops(monkeypatch, calls)
    popup = EseMeshPopup(ESE_MESH, manager)
    popups.append(popup)
    completed: list[Any] = []
    popup.completed.connect(completed.append)

    popup._run_button.click()
    _settle(qapp, popup)

    assert popup._hint_label.text() == "Done."
    assert completed == [3]
    assert manager.get_step_context_item(3, "ese_mesh").vertices.shape == (4, 3)
    assert manager.get_step_context_item(3, "normals").shape == (4, 3)
    assert manager.is_step_done(3)
    assert manager.get_current_step() == 4  # base advanced past step 3


def test_run_with_project_persists_ese_mesh(
    qapp, manager, popups, monkeypatch, tmp_path: Path
) -> None:
    _patch_ops(monkeypatch, [])
    popup = EseMeshPopup(ESE_MESH, manager, project=tmp_path)
    popups.append(popup)

    popup._run_button.click()
    _settle(qapp, popup)

    assert popup._hint_label.text() == "Done."
    assert (tmp_path / "ese" / "mesh.ply").is_file()


def test_missing_mesh_fails_collect(qapp, popups) -> None:
    steps = StepsManager(tuple(step.required_keys for step in STEPS))
    popup = EseMeshPopup(ESE_MESH, steps)
    popups.append(popup)

    assert not popup._run_button.isEnabled()
    with pytest.raises(ValueError, match="ESE mesh needs the postprocessed mesh"):
        popup._collect_run()


def test_path_mesh_still_fails_collect(qapp, popups) -> None:
    steps = _make_manager(_MESH)
    steps.set_step_context_item(MESH_POSTPROCESSING.index, "mesh", object())
    popup = EseMeshPopup(ESE_MESH, steps)
    popups.append(popup)

    with pytest.raises(ValueError) as excinfo:
        popup._collect_run()
    assert str(excinfo.value) == _NO_MESH
