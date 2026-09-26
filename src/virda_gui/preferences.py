"""Persisted GUI preferences: last project and recently opened projects."""

from pathlib import Path

from PySide6.QtCore import QByteArray, QSettings

_ORG = "VIRDA"
_APP = "virda-gui"
_KEY_LAST_PROJECT = "projects/last_project"
_KEY_RECENT_PROJECTS = "projects/recent_projects"
_KEY_GEOMETRY = "ui/main_geometry"
_KEY_MAIN_SPLITTER = "ui/main_splitter"
_KEY_MESH_SPLITTER = "ui/mesh_splitter"
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

    def save_blob(self, key: str, data: QByteArray) -> None:
        """Persist a Qt geometry/splitter blob under *key*."""
        self._settings.setValue(key, data)
        self._settings.sync()

    def load_blob(self, key: str) -> QByteArray | None:
        """Return the blob stored under *key*, or None when missing."""
        value = self._settings.value(key)
        if isinstance(value, QByteArray) and not value.isEmpty():
            return value
        return None

    def save_window_geometry(self, data: QByteArray) -> None:
        self.save_blob(_KEY_GEOMETRY, data)

    def window_geometry(self) -> QByteArray | None:
        return self.load_blob(_KEY_GEOMETRY)

    def save_main_splitter(self, data: QByteArray) -> None:
        self.save_blob(_KEY_MAIN_SPLITTER, data)

    def main_splitter(self) -> QByteArray | None:
        return self.load_blob(_KEY_MAIN_SPLITTER)

    def save_mesh_splitter(self, data: QByteArray) -> None:
        self.save_blob(_KEY_MESH_SPLITTER, data)

    def mesh_splitter(self) -> QByteArray | None:
        return self.load_blob(_KEY_MESH_SPLITTER)
