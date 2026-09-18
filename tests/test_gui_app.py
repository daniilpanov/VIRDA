"""Unit tests for the PySide6 application logic that does not need a display.

These tests exercise the pure helper logic of :mod:`virda_gui` by calling
the methods unbound against lightweight stubs, so no ``QApplication`` is
created (CI runners have no display).  The one exception is the offscreen
smoke tests, which construct the real :class:`IdeWindow` to catch signal
wiring regressions that the stubs cannot.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import QTreeWidgetItem, QWidget

from virda_gui.constants import ADVANCED_FIELD_DEFAULTS, CONFIG_KEY_TO_ADVANCED
from virda_gui.main_window import IdeWindow
from virda_gui.preferences import Preferences
from virda_gui.tabs.config_tab import ConfigTab
from virda_gui.tabs.results_tab import ResultsTab


def _make_prefs(tmp_path: Path) -> Preferences:
    """Preferences backed by an isolated INI file so user config stays clean."""
    return Preferences(QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat))


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

    def shutdown(self) -> None:
        self.shut_down = True


class _FakeRow:
    """Duck-typed stand-in for ``ElectrodeGroupRow``."""

    def __init__(self, path: str, color: str) -> None:
        self._path = path
        self._color = color

    def get(self) -> str:
        return self._path

    def get_color(self) -> str:
        return self._color


def test_advanced_defaults_cover_stage3() -> None:
    assert ADVANCED_FIELD_DEFAULTS["residual_threshold_mm"] == "10.0"
    assert ADVANCED_FIELD_DEFAULTS["calibrate_ese_offset"] == "true"


def test_config_keys_map_stage3_fields() -> None:
    assert CONFIG_KEY_TO_ADVANCED["residual_threshold_mm"] == "residual_threshold_mm"
    assert CONFIG_KEY_TO_ADVANCED["calibrate_ese_offset"] == "calibrate_ese_offset"


def test_collect_electrode_specs_skips_empty_and_duplicates(tmp_path) -> None:
    rows = [
        _FakeRow(str(tmp_path / "a.tsv"), "yellow"),
        _FakeRow("", "lime"),  # empty path is skipped
        _FakeRow("   ", "cyan"),  # whitespace-only path is skipped
        _FakeRow(str(tmp_path / "a.tsv"), "red"),  # duplicate path is skipped
        _FakeRow(str(tmp_path / "b.json"), "magenta"),
    ]
    stub = SimpleNamespace(_state=SimpleNamespace(electrode_rows=rows))
    specs = ConfigTab.collect_electrode_specs(cast("ConfigTab", stub))

    assert specs == [
        (str(tmp_path / "a.tsv"), "yellow"),
        (str(tmp_path / "b.json"), "magenta"),
    ]


def test_ensure_stage3_group_without_project_dir(tmp_path) -> None:
    stub = SimpleNamespace(_state=SimpleNamespace(last_project_dir=None, electrode_rows=[]))
    assert ConfigTab.ensure_stage3_electrodes_group(cast("ConfigTab", stub)) is None


def test_ensure_stage3_group_without_output_file(tmp_path) -> None:
    state = SimpleNamespace(last_project_dir=str(tmp_path), electrode_rows=[])
    stub = SimpleNamespace(_state=state)
    assert ConfigTab.ensure_stage3_electrodes_group(cast("ConfigTab", stub)) is None


def test_ensure_stage3_group_already_present(tmp_path) -> None:
    electrodes = tmp_path / "stage3" / "electrodes.json"
    electrodes.parent.mkdir()
    electrodes.write_text("[]", encoding="utf-8")
    rows = [_FakeRow(str(electrodes), "yellow")]
    state = SimpleNamespace(last_project_dir=str(tmp_path), electrode_rows=rows)
    stub = SimpleNamespace(_state=state)
    assert ConfigTab.ensure_stage3_electrodes_group(cast("ConfigTab", stub)) is None


def test_viewer_widget_importable_from_main_window() -> None:
    """The 3D viewer tab embeds ``ViewerWidget`` from ``virda_gui.viewer.viewer``."""
    import virda_gui.main_window as main_window_module
    from virda_gui.viewer.viewer import ViewerWidget

    assert vars(main_window_module)["ViewerWidget"] is ViewerWidget


class _FakeViewer:
    def __init__(self) -> None:
        self.load_calls: list[dict[str, str]] = []

    def load(self, **kwargs: str) -> None:
        self.load_calls.append(dict(kwargs))


class _FakeStack:
    def setCurrentWidget(self, widget: object) -> None:  # noqa: N802 - Qt naming
        self.current = widget


class _FakeItem:
    def __init__(self, path: object) -> None:
        self._path = path

    def data(self, _role: object, _value: object) -> object:
        return self._path


def _double_click_stub(viewer: _FakeViewer) -> SimpleNamespace:
    return SimpleNamespace(
        _results_viewer_widget=viewer,
        _preview_stack=_FakeStack(),
        _preview_load_seq=0,
        _preview_worker=None,
        _ensure_results_viewer=lambda: viewer,
    )


def test_double_click_mesh_loads_interactive_preview(tmp_path) -> None:
    mesh = tmp_path / "final_mesh.ply"
    mesh.write_bytes(b"ply\n")
    viewer = _FakeViewer()
    stub = _double_click_stub(viewer)
    item = _FakeItem(mesh)

    ResultsTab._on_results_artifact_double_clicked(
        cast("ResultsTab", stub),
        cast("QTreeWidgetItem", item),
        0,
    )

    assert viewer.load_calls == [{"mesh_path": str(mesh)}]
    assert stub._preview_stack.current is viewer


def test_double_click_nifti_loads_interactive_preview(tmp_path) -> None:
    nifti = tmp_path / "head.nii.gz"
    nifti.write_bytes(b"\x00")
    viewer = _FakeViewer()
    stub = _double_click_stub(viewer)
    item = _FakeItem(nifti)

    ResultsTab._on_results_artifact_double_clicked(
        cast("ResultsTab", stub),
        cast("QTreeWidgetItem", item),
        0,
    )

    assert viewer.load_calls == [{"nifti_path": str(nifti)}]


def test_double_click_non_visual_file_does_not_load(tmp_path) -> None:
    csv_file = tmp_path / "electrode_coords.csv"
    csv_file.write_text("id\n1\n", encoding="utf-8")
    viewer = _FakeViewer()
    stub = _double_click_stub(viewer)
    item = _FakeItem(csv_file)

    ResultsTab._on_results_artifact_double_clicked(
        cast("ResultsTab", stub),
        cast("QTreeWidgetItem", item),
        0,
    )

    assert viewer.load_calls == []


def test_double_click_directory_does_not_load(tmp_path) -> None:
    directory = tmp_path / "mesh"
    directory.mkdir()
    viewer = _FakeViewer()
    stub = _double_click_stub(viewer)
    item = _FakeItem(directory)

    ResultsTab._on_results_artifact_double_clicked(
        cast("ResultsTab", stub),
        cast("QTreeWidgetItem", item),
        0,
    )

    assert viewer.load_calls == []


def test_ide_window_runs_pipeline_tab_offscreen(tmp_path: Path) -> None:
    """Opening a project shows the Run Pipeline tab; the sidebar can reopen it.

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
        assert window._tabs.count() == 1
        assert window._tabs.tabText(0) == "Run Pipeline"
        assert window._config_tab.project_dir() == str(project)
        assert window._sidebar._tree.topLevelItemCount() == 1

        window._close_tab(0)
        assert window._tabs.count() == 0
        window._sidebar.runPipelineRequested.emit()
        assert window._tabs.count() == 1
        assert window._tabs.tabText(0) == "Run Pipeline"

        window.close_project()
        assert window._project is None
        assert window._tabs.count() == 0
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
        assert window._tabs.count() == 2  # run pipeline tab + embedded test tab
        window._close_tab(1)
        assert window._tabs.count() == 1
        assert window.project() == project

        window.close_project()
        assert window.project() is None
        assert window._sidebar._tree.topLevelItemCount() == 0
    finally:
        window.close()
        app.quit()


def test_ide_window_restores_last_project_offscreen(tmp_path: Path) -> None:
    """A fresh IdeWindow reopens the last project remembered via QSettings."""
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

    second = IdeWindow(prefs=prefs)
    try:
        assert second.project() == project
        recent = second._prefs.recent_projects()
        assert project in recent
    finally:
        second.close()
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
        assert window._tabs.count() == 2  # run pipeline tab + one mesh tab

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


def test_ide_window_prefills_run_tab_from_artifacts_offscreen(
    tmp_path: Path,
) -> None:
    """Opening a project fills the Run Pipeline fields from its artifacts."""
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
        (inputs / "fiducials.json").write_text('{"fiducials": []}', encoding="utf-8")
        (inputs / "measurements.json").write_text("{}", encoding="utf-8")
        (inputs / "config.json").write_text(
            json.dumps(
                {
                    "nifti_path": str(inputs / "head.nii.gz"),
                    "project_dir": str(project),
                }
            ),
            encoding="utf-8",
        )

        window.open_project(project)

        assert window._config_tab.nifti_path() == str(inputs / "head.nii.gz")
        assert window._config_tab.project_dir() == str(project)
        assert window._config_tab.measurements.get() == str(inputs / "measurements.json")
        assert window._config_tab._fiducials.get() == str(inputs / "fiducials.json")
        assert window._config_tab._config_file.get() == str(inputs / "config.json")
    finally:
        window._on_close()
        app.quit()
