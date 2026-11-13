"""Editable measurement input grid for the localization popup."""

from collections.abc import Mapping

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
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

    :meth:`set_rows` replaces the whole grid with an imported
    electrode -> fiducial -> distance table (one row per pair), and the
    "From file" button only emits :attr:`importRequested` -- the popup
    owns the file dialog and the import service.
    """

    rowsChanged = Signal()  # noqa: N815 - Qt signal API
    importRequested = Signal()  # noqa: N815 - Qt signal API

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._table = QTableWidget(0, len(_COLUMNS), self)
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.itemChanged.connect(lambda _item: self.rowsChanged.emit())

        from_file = QPushButton("From file", self)
        from_file.clicked.connect(self.importRequested)
        add = QPushButton("+ Add", self)
        add.clicked.connect(self.add_row)
        remove = QPushButton("- Remove selected", self)
        remove.clicked.connect(self.remove_selected_rows)
        buttons = QHBoxLayout()
        buttons.addWidget(from_file)
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

    def set_rows(self, table: Mapping[str, Mapping[str, float]]) -> None:
        """Replace the grid with one row per ``(electrode, fiducial)`` pair."""
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        for electrode, distances in table.items():
            for fiducial, distance in distances.items():
                row = self._table.rowCount()
                self._table.insertRow(row)
                cells = (electrode, fiducial, f"{float(distance):g}")
                for column, text in enumerate(cells):
                    self._table.setItem(row, column, QTableWidgetItem(text))
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
