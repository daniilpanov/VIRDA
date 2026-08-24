"""Unit tests for the tkinter application logic that does not need a display.

These tests exercise the pure helper logic of :mod:`virda_gui.app` by calling
the methods unbound against lightweight stubs, so no ``Tk`` root window is
created (CI runners have no display).
"""

from types import SimpleNamespace
from typing import cast

from virda_gui.app import (
    _ADVANCED_FIELD_DEFAULTS,
    _CONFIG_KEY_TO_ADVANCED,
    VirdaApp,
)


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
    assert _ADVANCED_FIELD_DEFAULTS["residual_threshold_mm"] == "10.0"
    assert _ADVANCED_FIELD_DEFAULTS["calibrate_ese_offset"] == "true"


def test_config_keys_map_stage3_fields() -> None:
    assert _CONFIG_KEY_TO_ADVANCED["residual_threshold_mm"] == "residual_threshold_mm"
    assert _CONFIG_KEY_TO_ADVANCED["calibrate_ese_offset"] == "calibrate_ese_offset"


def test_collect_electrode_specs_skips_empty_and_duplicates(tmp_path) -> None:
    rows = [
        _FakeRow(str(tmp_path / "a.tsv"), "yellow"),
        _FakeRow("", "lime"),  # empty path is skipped
        _FakeRow("   ", "cyan"),  # whitespace-only path is skipped
        _FakeRow(str(tmp_path / "a.tsv"), "red"),  # duplicate path is skipped
        _FakeRow(str(tmp_path / "b.json"), "magenta"),
    ]
    stub = SimpleNamespace(_electrode_rows=rows)
    specs = VirdaApp._collect_electrode_specs(cast("VirdaApp", stub))

    assert specs == [
        (str(tmp_path / "a.tsv"), "yellow"),
        (str(tmp_path / "b.json"), "magenta"),
    ]


def test_ensure_stage3_group_without_project_dir(tmp_path) -> None:
    stub = SimpleNamespace(_last_project_dir=None, _electrode_rows=[])
    assert VirdaApp._ensure_stage3_electrodes_group(cast("VirdaApp", stub)) is None


def test_ensure_stage3_group_without_output_file(tmp_path) -> None:
    stub = SimpleNamespace(_last_project_dir=str(tmp_path), _electrode_rows=[])
    assert VirdaApp._ensure_stage3_electrodes_group(cast("VirdaApp", stub)) is None


def test_ensure_stage3_group_already_present(tmp_path) -> None:
    electrodes = tmp_path / "stage3" / "electrodes.json"
    electrodes.parent.mkdir()
    electrodes.write_text("[]", encoding="utf-8")
    rows = [_FakeRow(str(electrodes), "yellow")]
    stub = SimpleNamespace(_last_project_dir=str(tmp_path), _electrode_rows=rows)
    assert VirdaApp._ensure_stage3_electrodes_group(cast("VirdaApp", stub)) is None


def test_viewer_thread_module_imports_show_viewer() -> None:
    """``_run_viewer_thread`` calls ``show_viewer`` — it must be importable."""
    import virda_gui.app as app_module
    from virda_gui.viewer import show_viewer

    assert vars(app_module)["show_viewer"] is show_viewer
