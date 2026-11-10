"""Editable fiducial input grid for the localization popup."""

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from virda_gui_new.services.localization import FiducialRows

_COLUMNS = ("Name", "X", "Y", "Z")


class FiducialsTable(QWidget):
    """Manual fiducial entry: one row per ``(name, xyz)`` point.

    Starts empty; "+ Add" appends one blank editable row and "- Remove
    selected" removes the selected rows. Any edit or row-set change
    emits :attr:`rowsChanged`. :meth:`collect_fiducial_rows` returns
    only complete rows: names stripped, XYZ parsed as floats, no
    further validation (duplicate names stay the service's concern).
    """

    rowsChanged = Signal()  # noqa: N815 - Qt signal API

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._table = QTableWidget(0, len(_COLUMNS), self)
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.itemChanged.connect(lambda _item: self.rowsChanged.emit())

        add = QPushButton("+ Add", self)
        add.clicked.connect(self.add_row)
        remove = QPushButton("- Remove selected", self)
        remove.clicked.connect(self.remove_selected_rows)
        buttons = QHBoxLayout()
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self._table)
        layout.addLayout(buttons)

    def add_row(self) -> None:
        """Append one blank editable row."""
        self._table.blockSignals(True)
        self._table.insertRow(self._table.rowCount())
        self._table.blockSignals(False)
        self.rowsChanged.emit()

    def remove_selected_rows(self) -> None:
        """Remove every selected row, ignoring nothing-to-remove calls."""
        rows = sorted({index.row() for index in self._table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        self._table.blockSignals(True)
        for row in rows:
            self._table.removeRow(row)
        self._table.blockSignals(False)
        self.rowsChanged.emit()

    def collect_fiducial_rows(self) -> FiducialRows:
        """Return the complete ``(name, (3,) xyz)`` rows; skip the rest."""
        rows: FiducialRows = []
        for row in range(self._table.rowCount()):
            name = self._text(row, 0)
            xyz = self._xyz(row)
            if name and xyz is not None:
                rows.append((name, xyz))
        return rows

    def _text(self, row: int, column: int) -> str:
        item = self._table.item(row, column)
        return "" if item is None else item.text().strip()

    def _xyz(self, row: int) -> np.ndarray | None:
        values = []
        for column in (1, 2, 3):
            try:
                values.append(float(self._text(row, column)))
            except ValueError:
                return None
        return np.array(values)
