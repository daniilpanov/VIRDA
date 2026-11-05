"""Unit tests for the PySide6 application logic that does not need a display.

These tests exercise the pure helper logic of :mod:`virda_gui` by calling
the methods unbound against lightweight stubs, so no ``QApplication`` is
created (CI runners have no display).  The one exception is the offscreen
smoke tests, which construct the real :class:`IdeWindow` to catch signal
wiring regressions that the stubs cannot.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import QWidget

from virda_gui.constants import ADVANCED_FIELD_DEFAULTS, CONFIG_KEY_TO_ADVANCED
from virda_gui.dialogs.project_dialog import (
    ProjectStartDialog,
    ask_create_project_folder,
)
from virda_gui.main_window import IdeWindow
from virda_gui.preferences import Preferences


def _make_prefs(tmp_path: Path) -> Preferences:
    """Preferences backed by an isolated INI file so user config stays clean."""
    return Preferences(QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat))


def _write_mini_nifti(path: Path) -> Path:
    """Write a tiny NIfTI volume with a solid central block."""
    import nibabel as nib

    data = np.zeros((16, 16, 16), dtype=np.uint8)
    data[4:12, 4:12, 4:12] = 200
    nib.Nifti1Image(data, np.eye(4)).to_filename(path)
    return path


class _StubViewer(QWidget):
    """Stand-in for ``ViewerWidget`` used by the offscreen file-tab tests."""

    sceneLoaded = Signal(object)  # noqa: N815
    sceneFailed = Signal(str)  # noqa: N815

    def __init__(self, log: object | None = None) -> None:
        super().__init__()
        self.log = log
        self.load_calls: list[dict[str, str]] = []
        self.shut_down = False

    def load(self, **kwargs: str) -> None:
        self.load_calls.append(dict(kwargs))

    def set_project_dir(self, project: object) -> None:
        return None

    def shutdown(self) -> None:
        self.shut_down = True


def test_advanced_defaults_cover_stage3() -> None:
    assert ADVANCED_FIELD_DEFAULTS["residual_threshold_mm"] == "10.0"
    assert ADVANCED_FIELD_DEFAULTS["calibrate_ese_offset"] == "true"


def test_config_keys_map_stage3_fields() -> None:
    assert CONFIG_KEY_TO_ADVANCED["residual_threshold_mm"] == "residual_threshold_mm"
    assert CONFIG_KEY_TO_ADVANCED["calibrate_ese_offset"] == "calibrate_ese_offset"


def test_viewer_widget_importable_from_main_window() -> None:
    """The 3D viewer tab embeds ``ViewerWidget`` from ``virda_gui.viewer.viewer``."""
    import virda_gui.main_window as main_window_module
    from virda_gui.viewer.viewer import ViewerWidget

    assert vars(main_window_module)["ViewerWidget"] is ViewerWidget


def test_ide_window_base_generation_tab_offscreen(tmp_path: Path) -> None:
    """The Base Generation tab opens from the menu and clears on project close.

    Builds the real ``IdeWindow`` under the offscreen Qt platform to catch
    ``AttributeError`` wiring regressions the stub-based tests cannot see
    (e.g. a ``Signal.connect`` target that was dropped during a refactor).
    Skips when the headless Qt platform is unavailable.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if os.environ.get("PYVISTA_OFF_SCREEN") is None:
        os.environ["PYVISTA_OFF_SCREEN"] = "true"

    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on local Qt install
        pytest.skip(f"Qt platform unavailable: {exc}")

    app = QApplication.instance() or QApplication([])
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        assert window._tabs.count() == 0

        project = tmp_path / "sample-project"
        (project / "mesh").mkdir(parents=True)
        (project / "mesh" / "final_mesh.ply").write_bytes(b"ply\n")

        window.open_project(project)
        assert window._tabs.count() == 0
        assert window._sidebar._tree.topLevelItemCount() == 1

        window._show_base_tab()
        assert window._tabs.count() == 1
        assert window._tabs.tabText(0) == "Base Generation"

        window.close_project()
        assert window._project is None
        assert window._tabs.count() == 0
        assert window._base_tab.current_base_mesh() is None
    finally:
        window._on_close()
        app.quit()


def test_ide_window_constructs_and_manages_project_offscreen(
    tmp_path: Path,
) -> None:
    """IdeWindow builds, opens/closes a project and closes tabs."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if os.environ.get("PYVISTA_OFF_SCREEN") is None:
        os.environ["PYVISTA_OFF_SCREEN"] = "true"

    try:
        from PySide6.QtWidgets import QApplication, QWidget
    except Exception as exc:  # pragma: no cover - depends on local Qt install
        pytest.skip(f"Qt platform unavailable: {exc}")

    app = QApplication.instance() or QApplication([])
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        assert window.project() is None
        assert window._sidebar.project is None
        assert window._sidebar._tree.topLevelItemCount() == 0

        project = tmp_path / "sample-project"
        (project / "mesh").mkdir(parents=True)
        (project / "mesh" / "final_mesh.ply").write_bytes(b"ply\n")
        (project / "note.txt").write_text("x", encoding="utf-8")

        window.open_project(project)
        assert window.project() == project
        assert window._sidebar.project == project
        assert window._sidebar._tree.topLevelItemCount() == 1
        root = window._sidebar._tree.topLevelItem(0)
        assert root.text(0) == "sample-project"
        assert root.childCount() == 2  # mesh group + loose note.txt

        tab = QWidget()
        window._tabs.addTab(tab, "Untitled")
        assert window._tabs.count() == 1
        window._close_tab(0)
        assert window._tabs.count() == 0
        assert window.project() == project

        window.close_project()
        assert window.project() is None
        assert window._sidebar._tree.topLevelItemCount() == 0
    finally:
        window.close()
        app.quit()


def test_ide_window_does_not_auto_restore_offscreen(tmp_path: Path) -> None:
    """A fresh IdeWindow starts empty — auto-restore now lives in the dialog."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if os.environ.get("PYVISTA_OFF_SCREEN") is None:
        os.environ["PYVISTA_OFF_SCREEN"] = "true"

    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on local Qt install
        pytest.skip(f"Qt platform unavailable: {exc}")

    app = QApplication.instance() or QApplication([])
    project = tmp_path / "remembered"
    (project / "note.txt").parent.mkdir(parents=True)
    (project / "note.txt").write_text("x", encoding="utf-8")

    prefs = _make_prefs(tmp_path)
    first = IdeWindow(prefs=prefs)
    try:
        first.open_project(project)
    finally:
        first.close()
    assert project in prefs.recent_projects()

    second = IdeWindow(prefs=prefs)
    try:
        assert second.project() is None
    finally:
        second.close()
    app.quit()


def test_project_start_dialog_picks_path_offscreen(tmp_path: Path) -> None:
    """The startup dialog returns the project the user accepted."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if os.environ.get("PYVISTA_OFF_SCREEN") is None:
        os.environ["PYVISTA_OFF_SCREEN"] = "true"

    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on local Qt install
        pytest.skip(f"Qt platform unavailable: {exc}")

    app = QApplication.instance() or QApplication([])
    prefs = _make_prefs(tmp_path)
    chosen = tmp_path / "published"
    (chosen / "note.txt").parent.mkdir(parents=True)
    (chosen / "note.txt").write_text("x", encoding="utf-8")
    prefs.note_project_opened(chosen)

    dialog = ProjectStartDialog(prefs=prefs)
    try:
        dialog._accept(chosen)
        assert dialog.project() == chosen
    finally:
        dialog.close()
    app.quit()


def test_ask_create_project_folder_warns_if_non_empty(tmp_path: Path, monkeypatch) -> None:
    """Creating a project in a non-empty folder asks before opening it."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if os.environ.get("PYVISTA_OFF_SCREEN") is None:
        os.environ["PYVISTA_OFF_SCREEN"] = "true"

    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on local Qt install
        pytest.skip(f"Qt platform unavailable: {exc}")

    from PySide6.QtWidgets import QFileDialog, QMessageBox

    app = QApplication.instance() or QApplication([])
    target = tmp_path / "wanted"
    (target / "existing.txt").parent.mkdir(parents=True)
    (target / "existing.txt").write_text("y", encoding="utf-8")

    monkeypatch.setattr(QFileDialog, "exec", lambda *a, **k: True)
    monkeypatch.setattr(QFileDialog, "selectedFiles", lambda *a, **k: [str(target)])

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    assert ask_create_project_folder(parent=None) == target

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    assert ask_create_project_folder(parent=None) is None
    app.quit()


def test_ide_window_opens_project_files_in_tabs_offscreen(tmp_path: Path, monkeypatch) -> None:
    """Double-clicking a sidebar artifact opens a viewer/preview tab."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if os.environ.get("PYVISTA_OFF_SCREEN") is None:
        os.environ["PYVISTA_OFF_SCREEN"] = "true"

    import virda_gui.main_window as main_window_module

    monkeypatch.setattr(main_window_module, "ViewerWidget", _StubViewer)

    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on local Qt install
        pytest.skip(f"Qt platform unavailable: {exc}")

    app = QApplication.instance() or QApplication([])
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "sample-project"
        mesh = project / "mesh" / "final_mesh.ply"
        mesh.parent.mkdir(parents=True)
        mesh.write_bytes(b"ply\n")
        (project / "note.txt").write_text("hello", encoding="utf-8")
        window.open_project(project)

        window._sidebar.fileActivated.emit(mesh)
        mesh_tab = window._tabs.widget(window._tabs.count() - 1)
        assert isinstance(mesh_tab, _StubViewer)
        assert mesh_tab.load_calls == [{"mesh_path": str(mesh)}]

        window._sidebar.fileActivated.emit(mesh)
        assert window._tabs.count() == 1  # one mesh tab, reused

        window._sidebar.fileActivated.emit(project / "note.txt")
        from virda_gui.tabs.preview_tab import PreviewTab

        preview_index = next(
            i for i in range(window._tabs.count()) if isinstance(window._tabs.widget(i), PreviewTab)
        )
        assert window._tabs.tabText(preview_index) == "note.txt"

        window._close_tab(preview_index)
        assert not any(
            isinstance(window._tabs.widget(i), PreviewTab) for i in range(window._tabs.count())
        )

        mesh_index = next(
            i
            for i in range(window._tabs.count())
            if isinstance(window._tabs.widget(i), _StubViewer)
        )
        window._close_tab(mesh_index)
        assert mesh_tab.shut_down
    finally:
        window._on_close()
        app.quit()


def test_ide_window_prefills_base_tab_from_artifacts_offscreen(
    tmp_path: Path,
) -> None:
    """Opening a project points the base tab at the project's scan."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if os.environ.get("PYVISTA_OFF_SCREEN") is None:
        os.environ["PYVISTA_OFF_SCREEN"] = "true"

    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on local Qt install
        pytest.skip(f"Qt platform unavailable: {exc}")

    app = QApplication.instance() or QApplication([])
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "ready-project"
        inputs = project / "input"
        inputs.mkdir(parents=True)
        (inputs / "head.nii.gz").write_bytes(b"\x00")

        window.open_project(project)

        assert window._base_tab._source_path == inputs / "head.nii.gz"
    finally:
        window._on_close()
        app.quit()


def test_perform_import_copies_into_project_and_refreshes_sidebar(
    tmp_path: Path,
) -> None:
    """Importing a role copies the file and repopulates the sidebar tree."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if os.environ.get("PYVISTA_OFF_SCREEN") is None:
        os.environ["PYVISTA_OFF_SCREEN"] = "true"

    from virda_gui.importing import ROLE_REGISTRY

    role = next(role for role in ROLE_REGISTRY if role.key == "mesh")

    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on local Qt install
        pytest.skip(f"Qt platform unavailable: {exc}")

    app = QApplication.instance() or QApplication([])
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        assert window._perform_import(role, tmp_path / "mesh.ply") is None  # no project yet

        project = tmp_path / "sample-project"
        project.mkdir()
        window.open_project(project)
        source = tmp_path / "final_mesh.ply"
        source.write_bytes(b"ply\n")

        target = window._perform_import(role, source)

        assert target == project / "mesh" / "final_mesh.ply"
        assert target.read_bytes() == b"ply\n"
        root = window._sidebar._tree.topLevelItem(0)
        assert root is not None
        mesh_group = None
        for i in range(root.childCount()):
            child = root.child(i)
            if child is not None and child.text(0) == "mesh":
                mesh_group = child
                break
        assert mesh_group is not None
        assert mesh_group.childCount() == 1
    finally:
        window._on_close()
        app.quit()


def test_base_tab_step_labels_follow_voxel_spin_offscreen(tmp_path: Path) -> None:
    """Editing voxel size recomputes only the step/real-voxel labels."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if os.environ.get("PYVISTA_OFF_SCREEN") is None:
        os.environ["PYVISTA_OFF_SCREEN"] = "true"

    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on local Qt install
        pytest.skip(f"Qt platform unavailable: {exc}")

    app = QApplication.instance() or QApplication([])

    from virda_gui.state import AppState
    from virda_gui.tabs.base_generation_tab import BaseGenerationTab

    tab = BaseGenerationTab(AppState())
    try:
        assert "marching cube step = --" in tab._real_label.text()
        tab.set_source(_write_mini_nifti(tmp_path / "mini.nii.gz"))
        assert tab._voxel_spin.value() == 1.0
        assert "marching cube step = 1" in tab._real_label.text()

        tab._voxel_spin.setValue(2.5)
        assert "marching cube step = 2" in tab._real_label.text()
        assert "2 x 2 x 2" in tab._real_label.text()
        assert tab._seal_chk.isChecked()
        assert tab._seal_radius_spin.value() == 4
        assert tab._cleaner_min_vertices_spin.value() == 100
        assert tab._cleaner_merge_digits_spin.value() == 7
    finally:
        tab.shutdown()
        tab.close()
        app.quit()


def test_base_tab_store_result_writes_meshes_and_sidecar_offscreen(
    tmp_path: Path,
) -> None:
    """Storing a generated mesh writes base, final and the source sidecar."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if os.environ.get("PYVISTA_OFF_SCREEN") is None:
        os.environ["PYVISTA_OFF_SCREEN"] = "true"

    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on local Qt install
        pytest.skip(f"Qt platform unavailable: {exc}")

    app = QApplication.instance() or QApplication([])

    from virda_gui.options import CleanOptions, SealingOptions
    from virda_gui.scan_hash import read_source_hash, sha256_file
    from virda_gui.state import AppState
    from virda_gui.tabs.base_generation_tab import BaseGenerationTab, generate_mesh_from_nifti

    state = AppState()
    tab = None
    try:
        project = tmp_path / "gen-project"
        (project / "input").mkdir(parents=True)
        source = _write_mini_nifti(project / "input" / "head.nii.gz")
        state.last_project_dir = str(project)

        mesh = generate_mesh_from_nifti(
            source,
            SealingOptions(seal_enabled=True, seal_radius=1),
            CleanOptions(min_component_vertices=1, merge_digits=7),
            voxel_size_mm=2.0,
        )
        tab = BaseGenerationTab(state)
        fired: list[str] = []
        tab.baseMesh.connect(lambda _m: fired.append("base"))
        tab.saved.connect(lambda: fired.append("saved"))
        tab._store_result(mesh, source)

        assert (project / "mesh" / "base_mesh.ply").is_file()
        assert (project / "mesh" / "final_mesh.ply").is_file()
        assert read_source_hash(project / "mesh") == sha256_file(source)
        assert tab.current_base_mesh() is mesh
        assert fired == ["base", "saved"]
    finally:
        if tab is not None:
            tab.shutdown()
            tab.close()
        app.quit()
