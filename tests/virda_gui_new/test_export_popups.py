"""Unit tests for :mod:`virda_gui_new.widgets.popup.export`.

Runs under the offscreen Qt platform. The result dialogs are stubbed
(like existing tests stub ``QFileDialog``), so the success/error flow is
exercised without real modal boxes; the export work itself is either the
real service writing into ``tmp_path`` or a recorded stub.
"""

import json
import os
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PySide6.QtWidgets import QComboBox, QFileDialog

from virda.models.electrode import Electrode, Electrodes
from virda.models.mesh import Mesh
from virda_gui_new.dto.steps import STEPS
from virda_gui_new.dto.viewer3d import LabeledPoints
from virda_gui_new.services.artifact_exporting import ExportKind, MeshExportKind
from virda_gui_new.services.coordinates import CoordinateFrame
from virda_gui_new.services.steps_manager import StepsManager
from virda_gui_new.widgets.popup.export import (
    ElectrodesExportPopup,
    FiducialsExportPopup,
    MeasurementsExportPopup,
    MeshExportPopup,
)
from virda_gui_new.widgets.popup.export import base as export_base
from virda_gui_new.widgets.popup.localization.frame_selector import FrameSelector

# World fiducials whose frame comes out axis-aligned (NAS at the origin).
_WORLD_FIDUCIALS = np.array([[0.0, 100.0, 0.0], [-70.0, 0.0, 0.0], [70.0, 0.0, 0.0]])
_FIDUCIAL_LABELS = ("NASION", "LPA", "RPA")
# Mesh whose centroid is (7.5, 7.5, 7.5): the head frame shifts by that.
_MESH = Mesh(
    np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]]),
    np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
)
_MEASUREMENTS = Electrodes([Electrode("Fz", {"NAS": 120.5})])


class _FakeMessageBox:
    """Stand-in for ``QMessageBox`` in the export popup base module."""

    class Icon:
        Critical = "critical"

    class StandardButton:
        Close = "standard-close"

    class ButtonRole:
        ActionRole = "action-role"

    log: list[dict[str, str]] = []
    #: Index of the added button that ``exec`` simulates clicking.
    clicked_index = 0

    def __init__(self, icon: object, title: str, text: str) -> None:
        self.icon = icon
        self.title = title
        self.text = text
        self.buttons: list[object] = []
        _FakeMessageBox.log.append({"kind": "error", "title": title, "text": text})

    def addButton(self, button: object, _role: object = None) -> object:  # noqa: N802
        self.buttons.append(button)
        return button

    def setDefaultButton(self, button: object) -> None:  # noqa: N802 - Qt API
        self.default = button

    def exec(self) -> None:
        self.clicked = self.buttons[_FakeMessageBox.clicked_index]

    def clickedButton(self) -> object:  # noqa: N802 - Qt API
        return self.clicked

    @staticmethod
    def information(
        _parent: object, title: str, text: str, _buttons: object = None, _default: object = None
    ) -> None:
        _FakeMessageBox.log.append({"kind": "info", "title": title, "text": text})


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
def stub_dialogs(monkeypatch: pytest.MonkeyPatch):
    """Replace the base module's ``QMessageBox`` with the fake recorder."""
    _FakeMessageBox.log = []
    _FakeMessageBox.clicked_index = 0
    monkeypatch.setattr(export_base, "QMessageBox", _FakeMessageBox)
    return _FakeMessageBox


@pytest.fixture
def manager() -> StepsManager:
    """Manager at step 4 with geometry, electrodes and measurements set."""
    steps = StepsManager(tuple(step.required_keys for step in STEPS))
    steps.next_step()
    steps.set_step_context_item(1, "scalp_surface", _MESH)
    steps.next_step()
    steps.set_step_context_item(2, "mesh", _MESH)
    steps.next_step()
    steps.set_step_context_item(3, "ese_mesh", _MESH)
    steps.next_step()
    steps.set_step_context_item(4, "electrodes", _MEASUREMENTS)
    steps.set_step_context_item(4, "measurements", _MEASUREMENTS)
    steps.set_global_item("fiducials", LabeledPoints(_WORLD_FIDUCIALS, _FIDUCIAL_LABELS))
    return steps


def _make_manager_without_geometry() -> StepsManager:
    """Manager at step 4 without meshes (no head frame) and no electrodes."""
    steps = StepsManager(tuple(step.required_keys for step in STEPS))
    for _ in range(4):
        steps.next_step()
    return steps


def _fiducials_item_enabled(selector: QComboBox) -> bool:
    """Whether the selector's fiducials item is enabled."""
    from PySide6.QtGui import QStandardItemModel

    frame_selector = selector
    index = frame_selector.findData(CoordinateFrame.FIDUCIALS)
    model = frame_selector.model()
    assert isinstance(model, QStandardItemModel)
    return bool(model.item(index).isEnabled())


def _head_item_enabled(selector: QComboBox) -> bool:
    from PySide6.QtGui import QStandardItemModel

    index = selector.findData(CoordinateFrame.HEAD)
    model = selector.model()
    assert isinstance(model, QStandardItemModel)
    return bool(model.item(index).isEnabled())


# --- structure and default paths ---


def test_measurements_popup_has_no_frame_selector(qapp, manager) -> None:
    popup = MeasurementsExportPopup(manager)
    popup.close()

    assert popup._frame_selector is None
    assert popup._frame() is CoordinateFrame.WORLD
    assert popup._path_edit.text() == ""  # no project -> empty prefill


def test_default_path_prefill_per_kind(qapp, manager, tmp_path: Path) -> None:
    measurements = MeasurementsExportPopup(manager, project=tmp_path)
    electrodes = ElectrodesExportPopup(manager, project=tmp_path)
    fiducials = FiducialsExportPopup(manager, project=tmp_path)
    mesh = MeshExportPopup(manager, project=tmp_path)
    try:
        measurements_path = str(tmp_path / "localization" / "measurements.json")
        electrodes_path = str(tmp_path / "localization" / "electrodes.csv")
        fiducials_path = str(tmp_path / "localization" / "fiducials.json")
        assert measurements._path_edit.text() == measurements_path
        assert electrodes._path_edit.text() == electrodes_path
        assert fiducials._path_edit.text() == fiducials_path
        assert mesh._path_edit.text() == str(tmp_path / "scalp" / "base_mesh.ply")
    finally:
        for popup in (measurements, electrodes, fiducials, mesh):
            popup.close()


def test_mesh_type_change_resets_default_path(qapp, manager, tmp_path: Path) -> None:
    popup = MeshExportPopup(manager, project=tmp_path)
    try:
        cast("QComboBox", popup._mesh_combo).setCurrentIndex(
            cast("QComboBox", popup._mesh_combo).findData(MeshExportKind.POSTPROCESSED)
        )
        assert popup._path_edit.text() == str(tmp_path / "scalp" / "final_mesh.ply")

        cast("QComboBox", popup._mesh_combo).setCurrentIndex(
            cast("QComboBox", popup._mesh_combo).findData(MeshExportKind.ESE)
        )
        assert popup._path_edit.text() == str(tmp_path / "ese" / "mesh.ply")

        cast("QComboBox", popup._mesh_combo).setCurrentIndex(
            cast("QComboBox", popup._mesh_combo).findData(MeshExportKind.BASE)
        )
        assert popup._path_edit.text() == str(tmp_path / "scalp" / "base_mesh.ply")
    finally:
        popup.close()


def test_browse_button_fills_the_path(qapp, manager, monkeypatch, tmp_path: Path) -> None:
    def fake_get_save_file_name(*_args: object, **_kwargs: object) -> tuple[str, str]:
        return (str(tmp_path / "custom.csv"), "")

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(fake_get_save_file_name))
    popup = ElectrodesExportPopup(manager)
    try:
        popup._browse()
        assert popup._path_edit.text() == str(tmp_path / "custom.csv")
    finally:
        popup.close()


# --- frame availability ---


def test_electrodes_popup_fiducials_frame_availability(qapp, manager) -> None:
    """The fiducials item is greyed without three fiducials, else usable."""
    bare = ElectrodesExportPopup(_make_manager_without_geometry())
    full = ElectrodesExportPopup(manager)
    try:
        assert not _fiducials_item_enabled(cast("FrameSelector", bare._frame_selector))
        assert _fiducials_item_enabled(cast("FrameSelector", full._frame_selector))
    finally:
        bare.close()
        full.close()


def test_head_frame_falls_back_to_world_without_mesh(qapp) -> None:
    popup = ElectrodesExportPopup(_make_manager_without_geometry())
    try:
        assert not _head_item_enabled(cast("FrameSelector", popup._frame_selector))
        assert popup._frame() is CoordinateFrame.WORLD
    finally:
        popup.close()


def test_fiducials_popup_never_offers_the_fiducials_frame(qapp, manager) -> None:
    popup = FiducialsExportPopup(manager)
    try:
        assert not _fiducials_item_enabled(cast("FrameSelector", popup._frame_selector))
        assert _head_item_enabled(cast("FrameSelector", popup._frame_selector))
    finally:
        popup.close()


def test_mesh_popup_mesh_kind_selection(qapp, manager) -> None:
    popup = MeshExportPopup(manager)
    try:
        cast("QComboBox", popup._mesh_combo).setCurrentIndex(
            cast("QComboBox", popup._mesh_combo).findData(MeshExportKind.ESE)
        )
        assert popup._mesh_kind() is MeshExportKind.ESE
    finally:
        popup.close()


# --- export click and result dialogs ---


def test_successful_export_closes_host_and_shows_success_box(
    qapp, manager, stub_dialogs, tmp_path: Path
) -> None:
    from virda_gui_new.windows.widget_windows import open_widget_window

    target = tmp_path / "out" / "measurements.json"
    popup = MeasurementsExportPopup(manager, project=tmp_path)
    popup._path_edit.setText(str(target))
    host = open_widget_window(popup, title="Export measurements")
    try:
        popup._export_button.click()

        assert target.is_file()
        assert json.loads(target.read_text(encoding="utf-8"))["electrodes"][0] == {
            "electrode_id": "Fz",
            "measured_distances": {"NAS": 120.5},
        }
        # Flow: the popup host closed first, then the modal success box.
        assert not host.isVisible()
        assert stub_dialogs.log == [
            {"kind": "info", "title": "Export successful", "text": f"Saved: {target}"}
        ]
    finally:
        host.close()
        qapp.processEvents()


def test_missing_source_shows_error_dialog_and_keeps_popup(
    qapp, stub_dialogs, tmp_path: Path
) -> None:
    _FakeMessageBox.clicked_index = 1  # the Retry button: the popup stays open
    popup = ElectrodesExportPopup(_make_manager_without_geometry(), project=tmp_path)
    default = str(tmp_path / "localization" / "electrodes.csv")
    popup.show()
    try:
        popup._export_button.click()

        assert stub_dialogs.log[0]["title"] == "Export error"
        assert "No localized electrodes to export" in stub_dialogs.log[0]["text"]
        # The popup stays open with its settings untouched.
        assert popup.isVisible()
        assert popup._path_edit.text() == default
    finally:
        popup.close()


def test_error_retry_keeps_popup_settings_without_auto_retry(
    qapp, manager, stub_dialogs, tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[object, ...]] = []

    def fail_export(*args: object, **kwargs: object) -> Path:
        calls.append(args)
        raise ValueError("boom")

    monkeypatch.setattr(export_base, "export_artifact", fail_export)
    _FakeMessageBox.clicked_index = 1  # the Retry button
    popup = FiducialsExportPopup(manager, project=tmp_path)
    custom = str(tmp_path / "adjusted.json")
    popup._path_edit.setText(custom)
    popup.show()
    try:
        popup._export_button.click()
        popup._export_button.click()  # second attempt after "Retry"

        assert len(calls) == 2  # no auto-retry: one export per explicit click
        assert stub_dialogs.log[0]["title"] == "Export error"
        assert len(stub_dialogs.log) == 2
        # The popup kept the user's previous settings untouched.
        assert popup.isVisible()
        assert popup._path_edit.text() == custom
        assert popup._frame() is CoordinateFrame.HEAD
    finally:
        popup.close()


def test_error_close_closes_the_popup(qapp, stub_dialogs, tmp_path: Path) -> None:
    popup = MeasurementsExportPopup(_make_manager_without_geometry(), project=tmp_path)
    popup._path_edit.setText(str(tmp_path / "any.json"))
    popup.show()
    try:
        popup._export_button.click()

        assert stub_dialogs.log[0]["title"] == "Export error"
        assert not popup.isVisible()  # standalone popup: Close closed it
    finally:
        qapp.processEvents()


def test_empty_path_is_an_error_without_calling_the_service(
    qapp, manager, stub_dialogs, monkeypatch
) -> None:
    calls: list[tuple[object, ...]] = []

    def fail_export(*args: object, **kwargs: object) -> Path:
        calls.append(args)
        raise AssertionError("the service must not run without a path")

    monkeypatch.setattr(export_base, "export_artifact", fail_export)
    popup = ElectrodesExportPopup(manager)  # no project -> empty prefill
    try:
        popup._export_button.click()

        assert calls == []
        assert stub_dialogs.log[0]["title"] == "Export error"
        assert "path" in stub_dialogs.log[0]["text"].lower()
    finally:
        popup.close()


def test_export_click_passes_frame_and_mesh_kind_to_the_service(
    qapp, manager, stub_dialogs, tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict[str, object]] = []

    def fake_export_artifact(mgr, kind, path, *, frame, mesh_kind) -> Path:
        calls.append({"kind": kind, "frame": frame, "mesh_kind": mesh_kind})
        written = Path(path)
        written.parent.mkdir(parents=True, exist_ok=True)
        written.write_text("", encoding="utf-8")
        return written

    monkeypatch.setattr(export_base, "export_artifact", fake_export_artifact)
    popup = MeshExportPopup(manager, project=tmp_path)
    cast("QComboBox", popup._mesh_combo).setCurrentIndex(
        cast("QComboBox", popup._mesh_combo).findData(MeshExportKind.ESE)
    )
    cast("FrameSelector", popup._frame_selector).setCurrentIndex(
        cast("FrameSelector", popup._frame_selector).findData(CoordinateFrame.FIDUCIALS)
    )
    try:
        popup._export_button.click()

        assert calls == [
            {
                "kind": ExportKind.MESH,
                "frame": CoordinateFrame.FIDUCIALS,
                "mesh_kind": MeshExportKind.ESE,
            }
        ]
        assert stub_dialogs.log == [
            {
                "kind": "info",
                "title": "Export successful",
                "text": f"Saved: {tmp_path / 'ese' / 'mesh.ply'}",
            }
        ]
    finally:
        popup.close()
