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
from PySide6.QtWidgets import QTableWidgetItem

from virda_gui_new.services import task_runner as task_runner_module
from virda_gui_new.windows.main_window import MainWindow
from virda_gui_new.windows.widget_windows import WidgetWindow


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


class _RaiseSpy(WidgetWindow):
    """Tool window that counts how often it was raised."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.raises = 0

    def raise_(self) -> None:
        self.raises += 1
        super().raise_()


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
        main._files_toggle.click()  # noqa: SLF001 - toggle under test

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


def test_steps_pill_sits_below_the_menu_bar(qapp, tmp_path: Path) -> None:
    """The Steps pill is anchored below the menu bar of the main window."""
    main = _build_main_window(tmp_path)
    try:
        menu_bar = main.menuBar()
        pill = main._steps_toggle  # noqa: SLF001 - overlay under test
        assert menu_bar.isVisible()
        assert pill.geometry().top() > menu_bar.geometry().bottom()
    finally:
        main.close()
        qapp.processEvents()


def test_steps_pill_stays_below_the_menu_bar_after_a_resize(qapp, tmp_path: Path) -> None:
    """Resizing the main window keeps the pill clear of the menu bar."""
    main = _build_main_window(tmp_path)
    try:
        pill = main._steps_toggle  # noqa: SLF001 - overlay under test
        before = pill.pos()

        main.resize(900, 700)
        qapp.processEvents()

        menu_bar = main.menuBar()
        assert pill.geometry().top() > menu_bar.geometry().bottom()
        assert pill.pos() == before
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
    main._files_toggle.click()  # noqa: SLF001 - toggle under test
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


def test_file_tree_window_is_hidden_at_startup(qapp, tmp_path: Path) -> None:
    """The file tree is hidden until the Files pill is clicked."""
    main = _build_main_window(tmp_path)
    try:
        # The window is not even created before the first toggle.
        assert main._filetree_window is None  # noqa: SLF001 - registry under test
        # The file tree widget itself is ready and holds the scan result.
        assert main._filetree is not None  # noqa: SLF001 - registry under test
        assert main._filetree.isVisible() is False  # noqa: SLF001
    finally:
        main.close()
        qapp.processEvents()


def test_files_pill_toggles_the_file_tree_window(qapp, tmp_path: Path) -> None:
    """Clicking the Files pill shows the file tree and hides it again."""
    main = _build_main_window(tmp_path)
    try:
        main._files_toggle.click()  # noqa: SLF001 - toggle under test
        window = main._filetree_window  # noqa: SLF001 - registry under test
        assert window is not None
        assert window.isVisible()
        assert window.windowTitle() == "File tree"

        main._files_toggle.click()  # noqa: SLF001 - toggle under test
        assert not window.isVisible()
    finally:
        main.close()
        qapp.processEvents()


def test_files_pill_is_guarded_after_the_window_was_destroyed(qapp, tmp_path: Path) -> None:
    """A destroyed file tree window leaves the Files pill a no-op."""
    main = _build_main_window(tmp_path)
    try:
        main._files_toggle.click()  # noqa: SLF001 - toggle under test
        window = main._filetree_window  # noqa: SLF001 - registry under test
        assert window is not None

        window.close()
        qapp.processEvents()

        assert main._filetree_window is None  # noqa: SLF001 - registry under test
        assert main._filetree is None  # noqa: SLF001 - registry under test

        main._files_toggle.click()  # noqa: SLF001 - guarded no-op, not a crash

        assert main._filetree_window is None  # noqa: SLF001 - registry under test
    finally:
        main.close()
        qapp.processEvents()


def test_files_pill_sits_below_the_steps_pill(qapp, tmp_path: Path) -> None:
    """The Files pill is anchored below the steps pill and the menu bar."""
    main = _build_main_window(tmp_path)
    try:
        steps_pill = main._steps_toggle  # noqa: SLF001 - overlay under test
        files_pill = main._files_toggle  # noqa: SLF001 - overlay under test
        menu_bar = main.menuBar()

        assert files_pill.text() == "Files"
        assert files_pill.geometry().top() > steps_pill.geometry().bottom()
        assert files_pill.geometry().top() > menu_bar.geometry().bottom()
        assert files_pill.pos().x() == steps_pill.pos().x()
    finally:
        main.close()
        qapp.processEvents()


def test_files_pill_stays_below_the_steps_pill_after_a_resize(qapp, tmp_path: Path) -> None:
    """Resizing the main window keeps the two pills stacked."""
    main = _build_main_window(tmp_path)
    try:
        steps_pill = main._steps_toggle  # noqa: SLF001 - overlay under test
        files_pill = main._files_toggle  # noqa: SLF001 - overlay under test

        main.resize(900, 700)
        qapp.processEvents()

        assert files_pill.geometry().top() > steps_pill.geometry().bottom()
        assert files_pill.geometry().top() > main.menuBar().geometry().bottom()
    finally:
        main.close()
        qapp.processEvents()


def test_file_tree_refreshes_on_step_invalidation(qapp, tmp_path: Path) -> None:
    """The file tree still lists the artifacts of the project."""
    from virda_gui_new.services.project_import import prepare_project

    project = tmp_path / "prepared-project"
    project.mkdir()
    (project / "segmentation").mkdir()
    (project / "segmentation" / "head_mask.nii.gz").write_text("")
    main = MainWindow(project, prepared=prepare_project(project))
    try:
        main._files_toggle.click()  # noqa: SLF001 - toggle under test
        window = main._filetree_window  # noqa: SLF001 - registry under test
        assert window is not None

        main._on_steps_invalidated(1)  # noqa: SLF001 - dispatcher under test

        filetree = main._filetree  # noqa: SLF001 - registry under test
        assert filetree is not None
        rows = [
            cast("QTableWidgetItem", filetree._list.item(i)).text()
            for i in range(filetree._list.count())
        ]  # noqa: SLF001
        assert rows == ["segmentation/head_mask.nii.gz"]
    finally:
        main.close()
        qapp.processEvents()


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


def test_coming_to_the_front_raises_every_tool_window(qapp, tmp_path: Path, monkeypatch) -> None:
    """Raising the main window brings its tool windows up with it.

    The z-order itself is not observable offscreen, so the mechanism is
    checked: every tracked tool window is raised and stays visible.
    """
    from PySide6.QtCore import QEvent

    from virda_gui_new.services.artifact_exporting import ExportKind
    from virda_gui_new.windows import widget_windows

    monkeypatch.setattr(widget_windows, "WidgetWindow", _RaiseSpy)
    main = _build_main_window(tmp_path)
    try:
        main._open_step_popup(1)  # noqa: SLF001 - dispatcher under test
        main._open_export_popup(ExportKind.MESH)  # noqa: SLF001 - dispatcher under test
        qapp.processEvents()

        tracked = main._tool_windows()  # noqa: SLF001 - registry under test
        assert main._steps_window in tracked  # noqa: SLF001 - registry under test
        assert main._popup_windows[1] in tracked  # noqa: SLF001 - registry under test
        assert main._export_windows[ExportKind.MESH] in tracked  # noqa: SLF001
        before = [window.raises for window in tracked]

        qapp.sendEvent(main, QEvent(QEvent.Type.WindowActivate))

        assert all(window.raises > count for window, count in zip(tracked, before, strict=True))
        assert all(window.isVisible() for window in tracked)
    finally:
        main.close()
        qapp.processEvents()


def test_activation_change_raises_the_tool_windows(qapp, tmp_path: Path, monkeypatch) -> None:
    """An activation change of the main window raises its tool windows."""
    from PySide6.QtCore import QEvent

    from virda_gui_new.windows import widget_windows

    monkeypatch.setattr(widget_windows, "WidgetWindow", _RaiseSpy)
    main = _build_main_window(tmp_path)
    try:
        tracked = main._tool_windows()  # noqa: SLF001 - registry under test
        before = [window.raises for window in tracked]

        qapp.sendEvent(main, QEvent(QEvent.Type.ActivationChange))

        assert all(window.raises > count for window, count in zip(tracked, before, strict=True))
    finally:
        main.close()
        qapp.processEvents()


def test_destroyed_tool_window_is_skipped_without_raising(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    """A tool window destroyed behind the back of the main window is skipped."""
    from PySide6.QtWidgets import QWidget

    from virda_gui_new.windows import widget_windows

    monkeypatch.setattr(widget_windows, "WidgetWindow", _RaiseSpy)
    main = _build_main_window(tmp_path)
    try:
        window = widget_windows.open_widget_window(QWidget(), parent=main, title="Steps")
        main._popup_windows[99] = window  # noqa: SLF001 - registry under test
        window.close()
        qapp.processEvents()
        assert not shiboken6.isValid(window)
        raises_before = cast("Any", window).raises

        main._raise_tool_windows()  # noqa: SLF001 - mechanism under test

        assert cast("Any", window).raises == raises_before
    finally:
        main.close()
        qapp.processEvents()


def test_popup_dispatchers_raise_the_new_window(qapp, tmp_path: Path, monkeypatch) -> None:
    """A popup opened by a dispatcher is raised above the main window."""
    from virda_gui_new.services.artifact_exporting import ExportKind
    from virda_gui_new.windows import widget_windows

    monkeypatch.setattr(widget_windows, "WidgetWindow", _RaiseSpy)
    main = _build_main_window(tmp_path)
    try:
        main._open_step_popup(1)  # noqa: SLF001 - dispatcher under test
        step_window = main._popup_windows[1]  # noqa: SLF001 - registry under test
        assert step_window.raises >= 1

        main._open_export_popup(ExportKind.MESH)  # noqa: SLF001 - dispatcher under test
        export_window = main._export_windows[ExportKind.MESH]  # noqa: SLF001
        assert export_window.raises >= 1
    finally:
        main.close()
        qapp.processEvents()
