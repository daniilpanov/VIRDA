"""Editable measurement input grid for the localization popup."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from virda_gui_new.services.localization import MeasurementTable

_COLUMNS = ("Electrode", "Fiducial", "Distance (mm)")


class MeasurementsTable(QWidget):
    """Manual distance entry: one row per ``(electrode, fiducial)`` pair.

    Starts empty; "+ Add" appends one blank editable row and "- Remove
    selected" removes the selected rows. Any edit or row-set change
    emits :attr:`rowsChanged`. :meth:`collect_measurement_table` keeps
    only complete rows (all three cells filled, distance parsed as a
    float); a repeated electrode id + fiducial name collides, and the
    later row wins.
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

    def collect_measurement_table(self) -> MeasurementTable:
        """Return electrode -> fiducial -> distance; later rows overwrite."""
        table: MeasurementTable = {}
        for row in range(self._table.rowCount()):
            electrode = self._text(row, 0)
            fiducial = self._text(row, 1)
            if not electrode or not fiducial:
                continue
            try:
                distance = float(self._text(row, 2))
            except ValueError:
                continue
            table.setdefault(electrode, {})[fiducial] = distance
        return table

    def _text(self, row: int, column: int) -> str:
        item = self._table.item(row, column)
        return "" if item is None else item.text().strip()
