"""Read-only browser for the files of a project.

Rows list the files of the project directory, labelled with their path
relative to the project root. Double-clicking a file emits
``fileActivated`` with the :class:`~pathlib.Path`; consumers decide what
to do with it. This widget complements the steps navigator (which manages
steps) by browsing the files; it does not read or open them.
"""

from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QVBoxLayout, QWidget


class FileTree(QWidget):
    """List of the files of a project, one row per file.

    ``set_files`` rebuilds the rows from the project files; every row
    carries its path and shows it relative to the project root.
    Double-clicking a row whose file still exists emits ``fileActivated``
    with its path.
    """

    fileActivated = Signal(object)  # noqa: N815

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._list = QListWidget(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._list)
        self._list.itemDoubleClicked.connect(self._on_item_double_clicked)

    def set_files(self, paths: Sequence[Path], project: Path | None = None) -> None:
        """Rebuild the rows from the project files.

        The rows show the path relative to the project, so the folders
        of the project stay visible. The rows are sorted by path, so the
        order of the files does not depend on the scan.
        """
        self._list.clear()
        for path in sorted(paths):
            label = str(path.relative_to(project)) if project is not None else path.name
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self._list.addItem(item)

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path is not None and path.is_file():
            self.fileActivated.emit(path)
