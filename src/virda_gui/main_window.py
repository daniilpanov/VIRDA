"""IDE-style main window: file sidebar, closable tabs and project management."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTabWidget,
)

from virda_gui.project import create_project
from virda_gui.sidebar import ProjectSidebar


class IdeWindow(QMainWindow):
    """IDE-style main window of the VIRDA GUI.

    The left panel is a :class:`~virda_gui.sidebar.ProjectSidebar` listing
    the project artifacts; the right panel is a closable tab bar where the
    run pipeline form, the 3D viewer and individual project files open in
    their own tabs.
    """

    def __init__(self) -> None:
        super().__init__()
        self._project: Path | None = None

        self.setWindowTitle("VIRDA — Electrode Localization System")
        self.resize(1100, 720)

        self._sidebar = ProjectSidebar(self)
        self._tabs = QTabWidget(self)
        self._tabs.setTabsClosable(True)
        self._tabs.setDocumentMode(True)
        self._tabs.tabCloseRequested.connect(self._close_tab)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._sidebar)
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 800])
        self.setCentralWidget(splitter)

        self._build_menu()
        self.statusBar().showMessage("")

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        new_action = QAction("&New project...", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._create_project)
        file_menu.addAction(new_action)

        open_action = QAction("&Open project...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_project_dialog)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        self._close_action = QAction("&Close project", self)
        self._close_action.setEnabled(False)
        self._close_action.triggered.connect(self.close_project)
        file_menu.addAction(self._close_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    # ------------------------------------------------------------------
    # Project management
    # ------------------------------------------------------------------

    def project(self) -> Path | None:
        """Return the open project directory, or None when none is open."""
        return self._project

    def open_project(self, project: Path) -> None:
        """Open *project* and populate the sidebar file tree."""
        self._project = project
        self._sidebar.set_project(project)
        self._close_action.setEnabled(True)
        self.setWindowTitle(f"VIRDA — {project.name}")
        self.statusBar().showMessage(f"Project opened: {project}", 5000)

    def close_project(self) -> None:
        """Close the project and reset the window to the empty state."""
        self._project = None
        self._sidebar.set_project(None)
        self._close_action.setEnabled(False)
        self.setWindowTitle("VIRDA — Electrode Localization System")

    def _create_project(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Create new project")
        if not directory:
            return
        project = create_project(Path(directory))
        self.open_project(project)

    def _open_project_dialog(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Open project")
        if not directory:
            return
        project = Path(directory)
        if not project.is_dir():
            QMessageBox.warning(self, "Project error", f"Not a directory:\n{project}")
            return
        self.open_project(project)

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------

    def _close_tab(self, index: int) -> None:
        widget = self._tabs.widget(index)
        self._tabs.removeTab(index)
        if widget is not None:
            widget.deleteLater()
