"""Unit tests for :class:`virda_gui_new.widgets.popup.base_mesh_extraction`.

Runs under the offscreen Qt platform; skips without a Qt install. The
``ops.generate_scalp_surface`` facade is mocked, so no heavy segmentation
runs; only form values flow to the mock and a tiny real result back.
"""

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from virda.models.mesh import Mesh
from virda.models.mri_volume import MRIVolume
from virda.models.segmentation_mask import SegmentationMask
from virda_gui.api.options import SealingOptions
from virda_gui_new.api import ops
from virda_gui_new.api.ops import ScalpSurface
from virda_gui_new.dto.steps import BASE_MESH, STEPS
from virda_gui_new.services import artifact_saving
from virda_gui_new.services.steps_manager import StepsManager
from virda_gui_new.widgets.popup.base_mesh_extraction import BaseMeshExtractionPopup

_NO_VOLUME = "Base mesh needs the NIfTI volume from step 0 (run NIfTI import first)."
_VOLUME = MRIVolume(
    np.zeros((4, 4, 4), dtype="float32"), np.eye(4), (1.0, 1.0, 1.0), ("R", "A", "S")
)


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


@pytest.fixture
def manager(qapp) -> StepsManager:
    """Manager at step 1 with a ready NIfTI volume in the step-0 context."""
    steps = StepsManager(tuple(step.required_keys for step in STEPS))
    steps.set_step_context_item(0, "volume", _VOLUME)
    steps.next_step()
    return steps


@pytest.fixture
def popups(qapp):
    """Collect popups and stop their runner threads at teardown."""
    created: list[Any] = []
    yield created
    for popup in created:
        popup.shutdown()


def _fake_result() -> ScalpSurface:
    """Tiny real scalp surface: a flat mask and a one-face mesh."""
    mask = SegmentationMask(np.ones((3, 3, 3), dtype="uint8"))
    mesh = Mesh(np.zeros((4, 3)), np.zeros((1, 3), dtype=int))
    return ScalpSurface(mask=mask, mesh=mesh)


_RESULT = _fake_result()


def _patch_generate(monkeypatch, calls: list[Any]) -> None:
    def fake(volume, sealing, voxel_size_mm=None):
        calls.append((volume, sealing, voxel_size_mm))
        return _RESULT

    monkeypatch.setattr(ops, "generate_scalp_surface", fake)


def _settle(qapp, popup, timeout_s: float = 5.0) -> None:
    """Pump events until the run settles (busy flag cleared)."""
    deadline = time.monotonic() + timeout_s
    while popup._busy and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


def test_form_defaults(qapp, manager, popups) -> None:
    popup = BaseMeshExtractionPopup(BASE_MESH, manager)
    popups.append(popup)

    assert popup._run_button.isEnabled()
    assert popup._voxel_spin.value() == 0.0
    assert popup._voxel_spin.specialValueText() == "native"
    assert popup._seal_box.isChecked()
    assert popup._radius_spin.value() == 4
    assert popup._radius_spin.isEnabled()


def test_collect_run_defaults_stay_native_and_sealed(qapp, manager, popups, monkeypatch) -> None:
    calls: list[Any] = []
    _patch_generate(monkeypatch, calls)
    popup = BaseMeshExtractionPopup(BASE_MESH, manager)
    popups.append(popup)

    work = popup._collect_run()
    assert calls == []  # nothing runs until the work callable is invoked

    work()
    assert calls == [(_VOLUME, SealingOptions(seal_enabled=True, seal_radius=4), None)]


def test_collect_run_maps_voxel_and_seal(qapp, manager, popups, monkeypatch) -> None:
    calls: list[Any] = []
    _patch_generate(monkeypatch, calls)
    popup = BaseMeshExtractionPopup(BASE_MESH, manager)
    popups.append(popup)

    popup._voxel_spin.setValue(2.0)
    popup._seal_box.setChecked(False)
    assert not popup._radius_spin.isEnabled()  # radius follows the seal toggle

    popup._collect_run()()
    assert calls == [(_VOLUME, SealingOptions(seal_enabled=False, seal_radius=4), 2.0)]


def test_apply_result_writes_mesh_and_mask(qapp, manager, popups) -> None:
    popup = BaseMeshExtractionPopup(BASE_MESH, manager)
    popups.append(popup)

    popup._apply_result(_RESULT)

    assert manager.get_step_context_item(1, "scalp_surface") is _RESULT.mesh
    assert manager.get_step_context_item(1, "scalp_mask") is _RESULT.mask


def test_mocked_run_writes_context_and_advances(qapp, manager, popups, monkeypatch) -> None:
    calls: list[Any] = []
    _patch_generate(monkeypatch, calls)
    popup = BaseMeshExtractionPopup(BASE_MESH, manager)
    popups.append(popup)

    popup._run_button.click()
    _settle(qapp, popup)

    assert popup._hint_label.text() == "Done."
    assert manager.get_step_context_item(1, "scalp_surface") is _RESULT.mesh
    assert manager.get_step_context_item(1, "scalp_mask") is _RESULT.mask
    assert manager.get_current_step() == 2  # base advanced past step 1


def test_run_with_project_persists_mask_and_base_mesh(
    qapp, manager, popups, monkeypatch, tmp_path: Path
) -> None:
    _patch_generate(monkeypatch, [])
    popup = BaseMeshExtractionPopup(BASE_MESH, manager, project=tmp_path)
    popups.append(popup)

    popup._run_button.click()
    _settle(qapp, popup)

    assert popup._hint_label.text() == "Done."
    assert (tmp_path / "segmentation" / "head_mask.nii.gz").is_file()
    assert (tmp_path / "scalp" / "base_mesh.ply").is_file()


def test_save_failure_fails_the_run(qapp, manager, popups, monkeypatch, tmp_path: Path) -> None:

    _patch_generate(monkeypatch, [])
    failed: list[str] = []

    def fail_save(*args: Any) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(artifact_saving, "save_scalp_surface", fail_save)
    popup = BaseMeshExtractionPopup(BASE_MESH, manager, project=tmp_path)
    popups.append(popup)
    popup.failed.connect(failed.append)

    popup._run_button.click()
    _settle(qapp, popup)

    assert popup._hint_label.text() == "disk full"
    assert failed == ["disk full"]
    assert manager.get_current_step() == 1  # the step did not advance


def test_no_volume_disables_run_and_fails_collect(qapp, popups) -> None:
    steps = StepsManager(tuple(step.required_keys for step in STEPS))
    popup = BaseMeshExtractionPopup(BASE_MESH, steps)
    popups.append(popup)

    assert not popup._run_button.isEnabled()
    with pytest.raises(ValueError, match="Base mesh needs the NIfTI volume from step 0"):
        popup._collect_run()


def test_path_volume_loads_lazily_in_the_work_callable(
    qapp, popups, monkeypatch, tmp_path: Path
) -> None:
    """A seeded project stores a Path; the file is read by the work callable."""
    import nibabel as nib

    (tmp_path / "input").mkdir()
    nifti_path = tmp_path / "input" / "a.nii"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), "float32"), np.eye(4)), nifti_path)

    steps = StepsManager(tuple(step.required_keys for step in STEPS))
    steps.set_step_context_item(0, "volume", nifti_path)
    steps.next_step()
    popup = BaseMeshExtractionPopup(BASE_MESH, steps)
    popups.append(popup)
    assert popup._run_button.isEnabled()  # the path alone is enough to run

    calls: list[Any] = []
    _patch_generate(monkeypatch, calls)
    popup._collect_run()()
    loaded, _sealing, _voxel = calls[0]
    assert isinstance(loaded, MRIVolume)
    assert loaded.data.size == 64
