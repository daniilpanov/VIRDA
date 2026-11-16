"""Unit tests for the project actions of the File menu.

Runs under the offscreen Qt platform. The menu is built by a real
window; the dialogs and the opening of a project are stubbed, so only
the wiring of the actions is exercised.
"""

import os
from pathlib import Path
from typing import cast

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMenu

from virda_gui_new.services.preferences import Preferences
from virda_gui_new.services.project_import import prepare_project
from virda_gui_new.windows.main_window import MainWindow


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


def _build_main_window(tmp_path: Path, preferences: Preferences | None = None) -> MainWindow:
    """A shown main window over an empty project."""
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    window = MainWindow(project, prepared=prepare_project(project), preferences=preferences)
    window.show()
    return window


def _file_menu(main: MainWindow) -> QMenu:
    """The File menu of the window."""
    menu = main.menuBar().actions()[0].menu()
    assert menu is not None
    return cast("QMenu", menu)


def _recent_menu(main: MainWindow) -> QMenu:
    """The Recent submenu of the File menu."""
    for action in _file_menu(main).actions():
        if action.text() == "&Recent":
            menu = action.menu()
            assert menu is not None
            return cast("QMenu", menu)
    raise AssertionError("the File menu has no Recent submenu")


def test_the_file_menu_has_the_project_actions(qapp, tmp_path: Path) -> None:
    """The File menu offers the project actions."""
    main = _build_main_window(tmp_path)
    try:
        labels = [action.text() for action in _file_menu(main).actions()]
        assert "&Open project..." in labels
        assert "&New project..." in labels
        assert "&Recent" in labels
    finally:
        main.close()


def test_the_recent_menu_lists_the_existing_projects(qapp, tmp_path: Path) -> None:
    """The Recent submenu lists the projects that still exist."""
    settings = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    preferences = Preferences(settings)
    present = tmp_path / "present"
    present.mkdir()
    missing = tmp_path / "missing"
    preferences.note_project_opened(present)
    preferences.note_project_opened(missing)

    main = _build_main_window(tmp_path, preferences=preferences)
    try:
        labels = [action.text() for action in _recent_menu(main).actions()]
        assert str(present) in labels
        assert str(missing) not in labels
    finally:
        main.close()


def test_the_open_action_opens_the_picked_project(qapp, tmp_path: Path, monkeypatch) -> None:
    """The Open project action opens the folder the user picked."""
    picked = tmp_path / "picked"
    picked.mkdir()
    opened: list[Path] = []

    monkeypatch.setattr(
        "virda_gui_new.windows.main_window.ask_open_project_folder",
        lambda _parent: picked,
    )
    monkeypatch.setattr(
        "virda_gui_new.windows.main_window.open_project_window",
        lambda project, parent=None: opened.append(project),
    )
    main = _build_main_window(tmp_path)
    try:
        main._on_open_project()  # noqa: SLF001 - action under test
        assert opened == [picked]
    finally:
        main.close()


def test_the_new_action_opens_the_created_project(qapp, tmp_path: Path, monkeypatch) -> None:
    """The New project action opens the folder the user created."""
    created = tmp_path / "created"
    opened: list[Path] = []

    monkeypatch.setattr(
        "virda_gui_new.windows.main_window.ask_create_project_folder",
        lambda _parent: created,
    )
    monkeypatch.setattr(
        "virda_gui_new.windows.main_window.open_project_window",
        lambda project, parent=None: opened.append(project),
    )
    main = _build_main_window(tmp_path)
    try:
        main._on_new_project()  # noqa: SLF001 - action under test
        assert opened == [created]
    finally:
        main.close()


def test_opening_a_project_closes_the_current_window(qapp, tmp_path: Path, monkeypatch) -> None:
    """A project opened from the menu replaces the window."""
    monkeypatch.setattr(
        "virda_gui_new.windows.main_window.open_project_window",
        lambda project, parent=None: MainWindow,
    )
    main = _build_main_window(tmp_path)
    try:
        main._open_project(tmp_path)  # noqa: SLF001 - action under test
        assert main.isVisible() is False
    finally:
        main.close()
