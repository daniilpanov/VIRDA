"""Persisted GUI preferences: last project and recently opened projects."""

from pathlib import Path

from PySide6.QtCore import QSettings

_ORG = "VIRDA"
_APP = "virda-gui"
_KEY_LAST_PROJECT = "projects/last_project"
_KEY_RECENT_PROJECTS = "projects/recent_projects"
MAX_RECENT = 8


class Preferences:
    """Thin typed wrapper around the ``QSettings`` store for GUI state.

    A single ``QSettings`` object is shared for every read/write so the
    backing file is only opened once.  Tests may inject a different
    :class:`QSettings` (e.g. bound to a temporary INI file) to avoid
    touching the real user configuration.
    """

    def __init__(self, settings: QSettings | None = None) -> None:
        self._settings = settings or QSettings(_ORG, _APP)

    def last_project(self) -> Path | None:
        value = self._settings.value(_KEY_LAST_PROJECT, "")
        return Path(str(value)) if value else None

    def recent_projects(self) -> list[Path]:
        raw = self._settings.value(_KEY_RECENT_PROJECTS, [])
        entries = raw if isinstance(raw, list) else ([raw] if raw else [])
        return [Path(str(entry)) for entry in entries]

    def note_project_opened(self, project: Path) -> None:
        self._settings.setValue(_KEY_LAST_PROJECT, str(project))
        recent = [entry for entry in self.recent_projects() if entry != project]
        recent.insert(0, project)
        self._settings.setValue(_KEY_RECENT_PROJECTS, [str(entry) for entry in recent[:MAX_RECENT]])
        self._settings.sync()
