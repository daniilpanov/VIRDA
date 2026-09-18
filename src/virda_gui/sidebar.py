"""Project sidebar: file tree and quick actions for the IDE main window."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from virda_gui.project import format_file_size, format_mtime, scan_project


class ProjectSidebar(QWidget):
    """Left panel of the IDE main window.

    Shows the artifact tree of the open project, grouped by the well-known
    artifact subdirectories in pipeline order, with the *Open 3D viewer* and
    *Run pipeline* quick actions pinned to the bottom.  Double-clicking a file
    emits :attr:`fileActivated`.
    """

    openViewerRequested = Signal()  # noqa: N815
    runPipelineRequested = Signal()  # noqa: N815
    fileActivated = Signal(object)  # noqa: N815  # path: Path

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: Path | None = None

        self._hint = QLabel(
            "No project selected.\n\nCreate or open a project to populate the file list."
        )
        self._hint.setWordWrap(True)
        self._hint.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._tree = QTreeWidget(self)
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(["Artifact", "Size", "Modified"])
        self._tree.setColumnWidth(0, 190)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)

        self._open_viewer_btn = QPushButton("Open 3D viewer")
        self._open_viewer_btn.clicked.connect(self.openViewerRequested)
        self._run_pipeline_btn = QPushButton("Run pipeline")
        self._run_pipeline_btn.clicked.connect(self.runPipelineRequested)

        self._build_ui()
        self.set_project(None)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        self._stack = QStackedWidget(self)
        self._stack.addWidget(self._hint)
        self._stack.addWidget(self._tree)
        outer.addWidget(self._stack, 1)

        buttons = QFrame(self)
        buttons_layout = QHBoxLayout(buttons)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.addWidget(self._run_pipeline_btn)
        buttons_layout.addWidget(self._open_viewer_btn)
        outer.addWidget(buttons)

    @property
    def project(self) -> Path | None:
        """The project currently shown in the sidebar, if any."""
        return self._project

    def set_project(self, project: Path | None) -> None:
        """Populate the tree from *project* or clear it when None."""
        self._project = project
        self._tree.clear()
        enabled = project is not None
        self._open_viewer_btn.setEnabled(enabled)
        self._run_pipeline_btn.setEnabled(enabled)
        if project is None:
            self._stack.setCurrentWidget(self._hint)
            return

        scan = scan_project(project)
        root_item = QTreeWidgetItem([project.name, "<dir>", ""])
        root_item.setExpanded(True)
        root_item.setData(0, Qt.ItemDataRole.UserRole, project)
        self._tree.addTopLevelItem(root_item)

        for node in scan.groups + scan.extra_dirs:
            group_item = QTreeWidgetItem([node.name, "<dir>", ""])
            group_item.setData(0, Qt.ItemDataRole.UserRole, node)
            root_item.addChild(group_item)
            for file_path in sorted(node.rglob("*")):
                if not file_path.is_file():
                    continue
                rel = file_path.relative_to(node).as_posix()
                child = QTreeWidgetItem([rel, format_file_size(file_path), format_mtime(file_path)])
                child.setData(0, Qt.ItemDataRole.UserRole, file_path)
                group_item.addChild(child)

        for file_path in scan.loose_files:
            child = QTreeWidgetItem(
                [file_path.name, format_file_size(file_path), format_mtime(file_path)]
            )
            child.setData(0, Qt.ItemDataRole.UserRole, file_path)
            root_item.addChild(child)

        self._stack.setCurrentWidget(self._tree)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path is not None and path.is_file():
            self.fileActivated.emit(path)
