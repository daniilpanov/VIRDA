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


def ask_open_project_folder(parent: QWidget | None) -> Path | None:
    directory = QFileDialog.getExistingDirectory(parent, "Open project")
    if not directory:
        return None

    project = Path(directory)
    if not project.is_dir():
        QMessageBox.warning(parent, "Project error", f"Not a directory:\n{project}")
        return None

    return project


def ask_create_project_folder(parent: QWidget | None) -> Path | None:
    dialog = QFileDialog(parent, "Create new project")
    dialog.setFileMode(QFileDialog.FileMode.Directory)
    if not dialog.exec():
        return None

    selected = dialog.selectedFiles()
    if not selected:
        return None

    project = Path(selected[0])
    if not project.exists():
        try:
            project.mkdir(parents=True)
        except OSError as exc:
            QMessageBox.warning(parent, "Project error", f"Could not create folder:\n{exc}")
            return None

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

    def __init__(self, recent: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: Path | None = None

        self.setWindowTitle("VIRDA: Electrode Localization System")
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

        if recent:
            recent_label = QLabel("Recent projects")
            layout.addWidget(recent_label)
            self._recent_list = QListWidget(self)
            self._recent_list.addItems(recent)

            self._recent_list.itemActivated.connect(self._on_recent_activated)
            layout.addWidget(self._recent_list)

            self._recent_list.setCurrentRow(0)
            self._recent_list.setFocus()

            open_recent_btn = QPushButton("Open selected")
            open_recent_btn.setToolTip("Open the highlighted recent project.")
            open_recent_btn.clicked.connect(self._on_open_selected)
            layout.addWidget(open_recent_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)

        self.setMinimumWidth(420)

    def project(self) -> Path | None:
        return self._project

    def _on_open(self) -> None:
        self._accept(ask_open_project_folder(self))

    def _on_create(self) -> None:
        self._accept(ask_create_project_folder(self))

    def _on_recent_activated(self, item: QListWidgetItem) -> None:
        self._accept(Path(item.text()))

    def _on_open_selected(self) -> None:
        item = self._recent_list.currentItem()
        if item is None:
            return

        project = Path(item.text())
        if not project.is_dir():
            QMessageBox.warning(self, "Project missing", f"Folder no longer exists:\n{project}")
            self._recent_list.takeItem(self._recent_list.row(item))
            return

        self._accept(project)

    def _accept(self, project: Path | None) -> None:
        if project is None:
            return

        self._project = project
        self.accept()
