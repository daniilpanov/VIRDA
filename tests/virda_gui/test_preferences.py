"""Unit tests for preference persistence backed by QSettings."""

from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings

from virda_gui.preferences import MAX_RECENT, Preferences


def make_settings(tmp_path: Path) -> QSettings:
    ini = tmp_path / "prefs.ini"
    return QSettings(str(ini), QSettings.Format.IniFormat)


def test_last_project_is_none_when_unset(tmp_path: Path) -> None:
    prefs = Preferences(make_settings(tmp_path))
    assert prefs.last_project() is None
    assert prefs.recent_projects() == []


def test_note_project_opened_updates_last_and_recent(tmp_path: Path) -> None:
    prefs = Preferences(make_settings(tmp_path))
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    prefs.note_project_opened(first)
    prefs.note_project_opened(second)

    assert prefs.last_project() == second
    assert prefs.recent_projects() == [second, first]


def test_recent_projects_are_deduplicated(tmp_path: Path) -> None:
    prefs = Preferences(make_settings(tmp_path))
    project = tmp_path / "project"
    project.mkdir()

    prefs.note_project_opened(project)
    prefs.note_project_opened(project)

    assert prefs.recent_projects() == [project]


def test_recent_projects_are_capped(tmp_path: Path) -> None:
    prefs = Preferences(make_settings(tmp_path))
    for index in range(MAX_RECENT + 3):
        project = tmp_path / f"project-{index}"
        project.mkdir()
        prefs.note_project_opened(project)

    recent = prefs.recent_projects()
    assert len(recent) == MAX_RECENT
    assert recent[0] == tmp_path / f"project-{MAX_RECENT + 2}"


def test_values_survive_a_fresh_preferences_instance(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    project = tmp_path / "kept"
    project.mkdir()

    Preferences(settings).note_project_opened(project)
    reloaded = Preferences(make_settings(tmp_path))

    assert reloaded.last_project() == project
    assert reloaded.recent_projects() == [project]


def test_single_recent_entry_roundtrip(tmp_path: Path) -> None:
    prefs = Preferences(make_settings(tmp_path))
    project = tmp_path / "only"
    project.mkdir()
    prefs.note_project_opened(project)

    raw: Any = prefs._settings.value("projects/recent_projects", [])
    assert raw == str(project) or raw == [str(project)]


def test_new_gui_preferences_share_the_store(tmp_path: Path) -> None:
    """The new GUI service reads projects written by the previous GUI."""
    from virda_gui_new.services.preferences import Preferences as NewPreferences

    project = tmp_path / "shared"
    project.mkdir()

    Preferences(make_settings(tmp_path)).note_project_opened(project)
    new_prefs = NewPreferences(make_settings(tmp_path))

    assert new_prefs.last_project() == project
    assert new_prefs.recent_projects() == [project]


def test_new_gui_preferences_roundtrip(tmp_path: Path) -> None:
    """The new GUI service records and reloads recents on its own."""
    from virda_gui_new.services.preferences import Preferences as NewPreferences

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    NewPreferences(make_settings(tmp_path)).note_project_opened(first)
    reloaded = NewPreferences(make_settings(tmp_path))
    reloaded.note_project_opened(second)

    assert reloaded.last_project() == second
    assert reloaded.recent_projects() == [second, first]
