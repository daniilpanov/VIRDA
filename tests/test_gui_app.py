"""Unit tests for the PySide6 application logic that does not need a display.

These tests exercise the pure helper logic of :mod:`virda_gui.app` by calling
the methods unbound against lightweight stubs, so no ``QApplication`` is
created (CI runners have no display).  The one exception is the offscreen
smoke test below, which constructs the real :class:`VirdaApp` to catch signal
wiring regressions that the stubs cannot.
"""

import os
from types import SimpleNamespace
from typing import cast

import pytest
from PySide6.QtWidgets import QTreeWidgetItem

from virda_gui.config_tab import ConfigTab
from virda_gui.constants import ADVANCED_FIELD_DEFAULTS, CONFIG_KEY_TO_ADVANCED
from virda_gui.results_tab import ResultsTab


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


def test_viewer_widget_importable_from_app() -> None:
    """The 3D viewer tab embeds ``ViewerWidget`` from ``virda_gui.viewer``."""
    import virda_gui.app as app_module
    from virda_gui.viewer import ViewerWidget

    assert vars(app_module)["ViewerWidget"] is ViewerWidget


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


def test_virda_app_constructs_offscreen() -> None:
    """The full widget tree builds, so every connected slot exists.

    Builds the real ``VirdaApp`` under the offscreen Qt platform to catch
    ``AttributeError`` wiring regressions the stub-based tests cannot see
    (e.g. a ``Signal.connect`` target that was dropped during a refactor).
    Skips when the headless Qt platform is unavailable.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if os.environ.get("PYVISTA_OFF_SCREEN") is None:
        os.environ["PYVISTA_OFF_SCREEN"] = "true"

    try:
        from PySide6.QtWidgets import QApplication

        from virda_gui.app import VirdaApp
    except Exception as exc:  # pragma: no cover - depends on local Qt install
        pytest.skip(f"Qt platform unavailable: {exc}")

    app = QApplication.instance() or QApplication([])
    try:
        window = VirdaApp()
        assert window._notebook.count() == 3
        assert window._notebook.isTabEnabled(window._viewer_tab_index) is False
    finally:
        window._on_close()
        app.quit()
