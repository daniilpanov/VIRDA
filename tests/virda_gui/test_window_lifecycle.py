"""Unit tests for :class:`virda_gui_new.windows.main_window` window lifecycle.

Runs under the offscreen Qt platform. A real ``MainWindow`` is built on a
prepared (empty, artifact-free) project so the tool-window parenting of
every popup and the close-everything exit path are exercised without any
heavy pipeline operation.
"""

import os
from pathlib import Path
from typing import Any, cast

import pytest
import shiboken6

from virda_gui_new.services import task_runner as task_runner_module
from virda_gui_new.windows.main_window import MainWindow


@pytest.fixture
def qapp():
    """Yield an offscreen ``QApplication``, skipping without a Qt install."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if os.environ.get("PYVISTA_OFF_SCREEN") is None:
        os.environ["PYVISTA_OFF_SCREEN"] = "true"
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on local Qt install
        pytest.skip(f"Qt platform unavailable: {exc}")

    app = QApplication.instance() or QApplication([])
    return app


def _build_main_window(tmp_path: Path):
    """Build the real MainWindow on an empty prepared project."""
    from virda_gui_new.services.project_import import prepare_project
    from virda_gui_new.windows.main_window import MainWindow

    project = tmp_path / "prepared-project"
    project.mkdir()
    prepared = prepare_project(project)
    return MainWindow(project, prepared=prepared)


def test_popup_windows_are_tool_windows_parented_to_main(qapp, tmp_path: Path) -> None:
    """Steps menu, step popups and file tree are Qt.Tool windows of MainWindow."""
    from PySide6.QtCore import Qt

    main = _build_main_window(tmp_path)
    try:
        main._open_step_popup(1)  # noqa: SLF001 - dispatcher under test

        steps_window = main._steps_window  # noqa: SLF001 - registry under test
        popup_window = main._popup_windows[1]  # noqa: SLF001 - registry under test
        filetree_window = main._filetree_window  # noqa: SLF001 - registry under test
        windows = [w for w in (steps_window, popup_window, filetree_window) if w is not None]
        assert len(windows) == 3
        for window in windows:
            assert window.windowType() == Qt.WindowType.Tool
            assert window.parent() is main
            assert window.windowModality() == Qt.WindowModality.NonModal
    finally:
        main.close()
        qapp.processEvents()


def test_menu_bar_has_file_export_with_four_actions(qapp, tmp_path: Path) -> None:
    """File -> Export holds one action per export kind."""
    main = _build_main_window(tmp_path)
    try:
        menubar = main.menuBar()
        assert [action.text() for action in menubar.actions()] == ["&File"]
        file_menu = menubar.actions()[0].menu()
        assert file_menu is not None
        assert [action.text() for action in file_menu.actions()] == ["&Export"]
        export_menu = file_menu.actions()[0].menu()
        assert export_menu is not None
        titles = [action.text() for action in export_menu.actions()]
        assert titles == ["&Measurements", "&Electrodes", "&Fiducials", "&Mesh"]
    finally:
        main.close()
        qapp.processEvents()


def test_export_actions_open_tool_windows_parented_to_main(qapp, tmp_path: Path) -> None:
    """Every File -> Export action opens its popup as a Qt.Tool child window."""
    from PySide6.QtCore import Qt

    from virda_gui_new.services.artifact_exporting import ExportKind

    main = _build_main_window(tmp_path)
    try:
        for kind in ExportKind:
            main._open_export_popup(kind)  # noqa: SLF001 - dispatcher under test
            window = main._export_windows[kind]  # noqa: SLF001 - registry under test
            assert window.windowType() == Qt.WindowType.Tool
            assert window.parent() is main
            assert window.windowModality() == Qt.WindowModality.NonModal
            assert window.windowTitle() == f"Export {kind.value}"
    finally:
        main.close()
        qapp.processEvents()


def test_reopening_an_export_popup_replaces_the_open_one(qapp, tmp_path: Path) -> None:
    from virda_gui_new.services.artifact_exporting import ExportKind

    main = _build_main_window(tmp_path)
    try:
        main._open_export_popup(ExportKind.MESH)  # noqa: SLF001 - dispatcher under test
        first = main._export_windows[ExportKind.MESH]  # noqa: SLF001

        main._open_export_popup(ExportKind.MESH)  # noqa: SLF001 - dispatcher under test
        qapp.processEvents()

        assert not shiboken6.isValid(first)
        assert main._export_windows[ExportKind.MESH] is not first  # noqa: SLF001
    finally:
        main.close()
        qapp.processEvents()


def test_closing_main_window_closes_export_windows(qapp, tmp_path: Path) -> None:
    """The close-everything exit path includes the export popups."""
    from virda_gui_new.services.artifact_exporting import ExportKind

    main = _build_main_window(tmp_path)
    main._open_export_popup(ExportKind.FIDUCIALS)  # noqa: SLF001 - dispatcher under test
    export_window = main._export_windows[ExportKind.FIDUCIALS]  # noqa: SLF001

    main.close()
    qapp.processEvents()

    assert not shiboken6.isValid(export_window)
    assert main._export_windows == {}  # noqa: SLF001 - registry under test
    assert not task_runner_module._ORPHANED  # noqa: SLF001 - registry under test


def test_closing_main_window_closes_all_popup_windows(qapp, tmp_path: Path) -> None:
    """Closing MainWindow closes every popup and drops the dead references."""
    main = _build_main_window(tmp_path)
    main._open_step_popup(1)  # noqa: SLF001 - dispatcher under test
    steps_window = main._steps_window
    popup_window = main._popup_windows[1]  # noqa: SLF001 - registry under test
    filetree_window = main._filetree_window

    main.close()
    qapp.processEvents()

    assert not main.isVisible()
    for window in (steps_window, popup_window, filetree_window):
        assert window is not None
        # WA_DeleteOnClose: the closed host windows are destroyed outright,
        # and the destroyed wiring dropped the main window's references.
        assert not shiboken6.isValid(window)
    assert main._popup_windows == {}  # noqa: SLF001 - registry under test
    assert main._popups == {}  # noqa: SLF001 - registry under test
    # The destroyed wiring dropped every dead reference.
    assert main._steps_window is None  # noqa: SLF001 - registry under test
    assert main._filetree is None  # noqa: SLF001 - registry under test
    # Stop-on-close ran: no worker thread was left behind in the registry.
    assert not task_runner_module._ORPHANED


def test_forgotten_steps_window_toggle_is_guarded(qapp, tmp_path: Path) -> None:
    """After MainWindow closed, toggling a destroyed steps menu is a no-op."""
    main = _build_main_window(tmp_path)
    main.close()
    qapp.processEvents()

    MainWindow._toggle_steps_menu(main)  # no window left: guarded no-op

    assert main._steps_window is None  # noqa: SLF001 - registry under test
    assert not task_runner_module._ORPHANED
    assert not task_runner_module._ORPHANED


def test_fiducials_store_mutation_rebuilds_the_scene(qapp, tmp_path: Path) -> None:
    """A global-store write triggers the pull-model scene rebuild."""
    import numpy as np

    from virda_gui_new.dto.viewer3d import LabeledPoints

    main = _build_main_window(tmp_path)
    try:
        scenes: list[object] = []
        main._viewer3d.sceneLoaded.connect(scenes.append)

        points = LabeledPoints(np.array([[0.0, 0.0, 1.0]]), ("NAS",))
        main._steps.set_global_item("fiducials", points)
        qapp.processEvents()

        assert scenes, "the store write did not reach the viewer"
        scene = scenes[-1]
        assert cast("Any", scene).fiducials is points
    finally:
        main.close()
        qapp.processEvents()
