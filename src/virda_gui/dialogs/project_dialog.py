"""Project startup dialog and shared project-folder pickers.

The :class:`ProjectStartDialog` shown before the main window asks whether to
open an existing project, create a new one, or re-open a recently used one.
The two ``ask_*`` helpers are thin wrappers around :class:`QFileDialog` so the
confirm-on-non-empty-folder question lives in exactly one place and can be
shared with the File menu actions of the main window.
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from virda_gui.preferences import Preferences


def ask_open_project_folder(parent: QWidget | None) -> Path | None:
    """Ask the user to pick an *existing* project folder."""
    directory = QFileDialog.getExistingDirectory(parent, "Open project")
    if not directory:
        return None
    project = Path(directory)
    if not project.is_dir():
        QMessageBox.warning(parent, "Project error", f"Not a directory:\n{project}")
        return None
    return project


def ask_create_project_folder(parent: QWidget | None) -> Path | None:
    """Ask the user to pick the folder of a *new* project.

    Creates the folder when it does not exist yet.  When the folder already
    holds files the user is warned that a project already exists there and is
    asked whether to open it instead; ``None`` means the user cancelled or
    declined.
    """
    directory = QFileDialog.getExistingDirectory(parent, "Create new project")
    if not directory:
        return None
    project = Path(directory)
    if not project.exists():
        project.mkdir(parents=True)
        return project
    if any(project.iterdir()):
        answer = QMessageBox.question(
            parent,
            "Project already exists",
            f"A project already exists in this folder:\n{project}\n\nOpen it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return None
    return project


class ProjectStartDialog(QDialog):
    """Modal startup dialog: open or create a project, or pick a recent one."""

    def __init__(self, prefs: Preferences, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._prefs = prefs
        self._project: Path | None = None

        self.setWindowTitle("VIRDA — Electrode Localization System")
        self.setModal(True)

        layout = QVBoxLayout(self)
        intro = QLabel("Open an existing project or create a new one.")
        layout.addWidget(intro)

        open_btn = QPushButton("Open project...")
        open_btn.clicked.connect(self._on_open)
        layout.addWidget(open_btn)

        create_btn = QPushButton("Create project...")
        create_btn.clicked.connect(self._on_create)
        layout.addWidget(create_btn)

        recent_label = QLabel("Recent projects")
        layout.addWidget(recent_label)
        self._recent_list = QListWidget(self)
        recent = [entry for entry in prefs.recent_projects() if entry.is_dir()]
        for entry in recent:
            self._recent_list.addItem(str(entry))
        self._recent_list.itemActivated.connect(self._on_recent_activated)
        layout.addWidget(self._recent_list)
        if recent:
            self._recent_list.setCurrentRow(0)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)

        self.setMinimumWidth(420)

    def project(self) -> Path | None:
        """Return the chosen project folder, or None when cancelled."""
        return self._project

    def _on_open(self) -> None:
        self._accept(ask_open_project_folder(self))

    def _on_create(self) -> None:
        self._accept(ask_create_project_folder(self))

    def _on_recent_activated(self, item: QListWidgetItem) -> None:
        self._accept(Path(item.text()))

    def _accept(self, project: Path | None) -> None:
        if project is None:
            return
        self._project = project
        self.accept()
