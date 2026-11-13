"""Read-only browser for scanned project artifacts.

Rows list the artifact paths reported by
:func:`virda_gui_new.services.project_scan.find_project_artifacts`, ordered
by slot and labelled with their real project subdirectory. Double-clicking
an existing file emits ``fileActivated`` with the
:class:`~pathlib.Path`; consumers decide what to do with it. This widget
complements the steps navigator (which manages steps) by browsing
artifacts; it does not scan, load, or open dialogs.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from virda_gui_new.services.project_scan import ProjectArtifacts

#: ``ProjectArtifacts`` slots in display order, one row per present slot.
_SLOT_ORDER: tuple[str, ...] = (
    "nifti",
    "fiducials",
    "measurements",
    "coordsystem",
    "pipeline_config",
    "head_mask",
    "base_mesh",
    "scalp_mesh",
    "ese_mesh",
    "scalp_points",
    "ese_points",
)


class FileTree(QWidget):
    """List of ready project artifact files, one row per present slot.

    ``set_artifacts`` rebuilds the rows from a scan result; rows only exist
    for present artifacts and carry their real subdirectory in the label.
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

    def set_artifacts(self, artifacts: ProjectArtifacts) -> None:
        """Rebuild the rows from ``artifacts``; missing artifacts stay absent."""
        self._list.clear()
        for slot in _SLOT_ORDER:
            path = getattr(artifacts, slot)
            if path is None or not path.is_file():
                continue
            item = QListWidgetItem(f"{path.parent.name}/{path.name}")
            item.setData(Qt.ItemDataRole.UserRole, path)
            self._list.addItem(item)

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path is not None and path.is_file():
            self.fileActivated.emit(path)
