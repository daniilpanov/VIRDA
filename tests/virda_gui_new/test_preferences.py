from pathlib import Path

from PySide6.QtCore import QSettings

from virda_gui_new.services.preferences import Preferences


def make_settings(tmp_path: Path) -> QSettings:
    ini = tmp_path / "prefs.ini"
    return QSettings(str(ini), QSettings.Format.IniFormat)


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
