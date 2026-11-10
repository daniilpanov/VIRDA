"""Unit tests for :class:`virda_gui_new.widgets.popup.main_menu.StepsMenu`.

Runs under the offscreen Qt platform; dialogs and file pickers are stubbed,
so no real window pops up and a real tiny NIfTI file drives the import flow.
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog

from virda.models.mri_volume import MRIVolume
from virda_gui_new.dto.steps import STEPS
from virda_gui_new.services.steps_manager import StepsManager
from virda_gui_new.widgets.popup import main_menu


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
def manager() -> StepsManager:
    return StepsManager(tuple(step.required_keys for step in STEPS))


@pytest.fixture
def menu(qapp, manager):
    from virda_gui_new.widgets.popup.main_menu import StepsMenu

    widget = StepsMenu(manager)
    yield widget
    widget.deleteLater()


def _write_nifti(path: Path) -> None:
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), "float32"), np.eye(4)), path)


def _row_texts(menu) -> list[str]:
    widget = menu._list
    return [widget.item(i).text() for i in range(widget.count())]


def _row_enabled(menu, row: int) -> bool:
    from PySide6.QtCore import Qt

    return bool(menu._items[row].flags() & Qt.ItemFlag.ItemIsEnabled)


def _connect(menu) -> tuple[list[int], list[int]]:
    requested: list[int] = []
    invalidated: list[int] = []
    menu.stepRequested.connect(requested.append)
    menu.stepsInvalidated.connect(invalidated.append)
    return requested, invalidated


def test_fresh_rows_show_current_and_pending(qapp, manager, menu) -> None:
    assert _row_texts(menu) == [
        "0. NIfTI import [current]",
        "1. Base mesh",
        "2. Mesh postprocessing",
        "3. ESE mesh",
        "4. Localization",
    ]
    assert [_row_enabled(menu, n) for n in range(5)] == [True, False, False, False, False]


def test_rows_update_on_manager_notify(qapp, manager, menu) -> None:
    assert _row_texts(menu)[0] == "0. NIfTI import [current]"
    manager.set_step_context_item(0, "volume", "ready")
    manager.next_step()
    manager.set_step_context_item(1, "scalp_surface", "ready")
    manager.next_step()

    assert _row_texts(menu) == [
        "0. NIfTI import [done]",
        "1. Base mesh [done]",
        "2. Mesh postprocessing [current]",
        "3. ESE mesh",
        "4. Localization",
    ]
    assert [_row_enabled(menu, n) for n in range(5)] == [True, True, True, False, False]


def test_click_on_future_step_is_guarded(qapp, manager, menu, monkeypatch) -> None:
    requested, invalidated = _connect(menu)

    called = []
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: called.append(1))
    )
    menu._list.itemClicked.emit(menu._items[3])

    assert requested == [] and invalidated == [] and called == []
    assert manager.get_current_step() == 0


def test_fresh_step0_import_advances_and_emits(qapp, manager, menu, monkeypatch, tmp_path) -> None:
    nifti = tmp_path / "head.nii"
    _write_nifti(nifti)
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(nifti), ""))
    )
    requested, invalidated = _connect(menu)

    menu._on_step_clicked(0)

    volume = manager.get_step_context_item(0, "volume")
    assert isinstance(volume, MRIVolume)
    assert manager.get_current_step() == 1
    assert invalidated == []
    assert requested == [1]


def test_fresh_step0_import_cancelled_keeps_state(qapp, manager, menu, monkeypatch) -> None:
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("", ""))
    )
    requested, invalidated = _connect(menu)

    menu._on_step_clicked(0)

    assert manager.get_step_context_item(0, "volume") is None
    assert manager.get_current_step() == 0
    assert requested == [] and invalidated == []


def test_failed_import_warns_and_keeps_state(qapp, manager, menu, monkeypatch, tmp_path) -> None:
    bad = tmp_path / "head.nii"
    bad.write_text("not a nifti", encoding="utf-8")
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(bad), ""))
    )

    warnings: list[tuple[object, str, str]] = []

    def fake_warning(parent, title, text) -> None:
        warnings.append((parent, title, text))

    monkeypatch.setattr(main_menu, "show_warning", fake_warning)
    requested, invalidated = _connect(menu)

    menu._on_step_clicked(0)

    assert warnings == [(menu, "Import NIfTI...", warnings[0][2])]
    assert warnings[0][2]
    assert manager.get_step_context_item(0, "volume") is None
    assert manager.get_current_step() == 0
    assert requested == [] and invalidated == []


def test_redo_declined_on_done_step(qapp, manager, menu, monkeypatch, tmp_path) -> None:
    nifti = tmp_path / "head.nii"
    _write_nifti(nifti)
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(nifti), ""))
    )
    menu._on_step_clicked(0)
    old_volume = manager.get_step_context_item(0, "volume")
    monkeypatch.setattr(main_menu, "confirm_step_redo", lambda *a, **k: False)
    requested, invalidated = _connect(menu)

    menu._on_step_clicked(0)

    assert manager.get_current_step() == 1
    assert manager.get_step_context_item(0, "volume") is old_volume
    assert requested == [] and invalidated == []


def test_redo_agreed_on_done_step0_reimports_and_advances(
    qapp, manager, menu, monkeypatch, tmp_path
) -> None:
    first = tmp_path / "first.nii"
    second = tmp_path / "second.nii"
    _write_nifti(first)
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: (str(first), "")),
    )
    menu._on_step_clicked(0)
    old_volume = manager.get_step_context_item(0, "volume")
    _write_nifti(second)
    monkeypatch.setattr(main_menu, "confirm_step_redo", lambda *a, **k: True)
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: (str(second), "")),
    )
    requested, invalidated = _connect(menu)

    menu._on_step_clicked(0)

    new_volume = manager.get_step_context_item(0, "volume")
    assert isinstance(new_volume, MRIVolume)
    assert new_volume is not old_volume
    assert manager.get_current_step() == 1
    assert invalidated == [0]
    assert requested == [1]


def test_redo_agreed_on_done_step1_emits_requested(qapp, manager, menu, monkeypatch) -> None:
    manager.next_step()
    manager.set_step_context_item(1, "scalp_surface", "ready")
    manager.next_step()  # current = 2
    monkeypatch.setattr(main_menu, "confirm_step_redo", lambda *a, **k: True)
    requested, invalidated = _connect(menu)

    menu._on_step_clicked(1)

    assert manager.get_current_step() == 1
    assert manager.all_contexts() == [{}, {"scalp_surface": "ready"}]
    assert invalidated == [1]
    assert requested == [1]
