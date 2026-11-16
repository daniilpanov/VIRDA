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

from virda.geometry import match_fiducial_name
from virda.models.fiducial import SLOT_IDS, FiducialType
from virda_gui_new.services.localization import MeasurementTable

#: The fiducial columns, in the fixed order of the canonical slots.

_COLUMNS = ("Electrode", *(fiducial.value for fiducial in SLOT_IDS))


class MeasurementsTable(QWidget):
    """Pivot grid of the measured distances: one row per electrode.

    The columns are the canonical fiducial slots (NAS, LPA, RPA) and the
    rows the electrodes; a cell holds the measured distance in
    millimetres. An empty cell is a measurement that is missing.

    The table is a free input: "+ Add" appends one blank electrode row
    and "- Remove selected" removes the selected rows. A changed cell
    emits :attr:`electrodeEdited` with the electrode it belongs to, and
    an added or removed row emits :attr:`electrodeSetChanged`; the
    whole table is replaced through :attr:`rowsChanged`.
    :meth:`collect_measurement_table`
    returns the electrode -> fiducial -> distance table, skipping the
    empty cells and the rows without an electrode id; a repeated
    electrode id merges, and the later row wins.

    :meth:`set_rows` replaces the whole grid with an imported
    electrode -> fiducial -> distance table (one row per electrode), and
    the "From file" button only emits :attr:`importRequested` -- the
    popup owns the file dialog and the import service.
    """

    rowsChanged = Signal()  # noqa: N815 - Qt signal API
    electrodeEdited = Signal(str)  # noqa: N815 - Qt signal API
    electrodeSetChanged = Signal()  # noqa: N815 - Qt signal API
    importRequested = Signal()  # noqa: N815 - Qt signal API

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._table = QTableWidget(0, len(_COLUMNS), self)
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.itemChanged.connect(self._on_item_changed)

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

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Report the edited electrode.

        A changed distance moves its own electrode only, so the row is
        reported on its own: the whole set is not recomputed.
        """
        self.electrodeEdited.emit(self._electrode_id_of_row(item.row()))

    def _electrode_id_of_row(self, row: int) -> str:
        """The electrode id of a grid row (empty when the row is blank)."""
        cell = self._table.item(row, 0)
        return cell.text().strip() if cell is not None else ""

    def add_row(self) -> None:
        """Append one blank electrode row."""
        self._table.blockSignals(True)
        self._table.insertRow(self._table.rowCount())
        self._table.blockSignals(False)
        # A new row changes the electrode set: the results follow the
        # table, but nothing is recomputed.
        self.electrodeSetChanged.emit()

    def set_rows(self, table: Mapping[str, Mapping[str, float]]) -> None:
        """Replace the grid with one row per electrode.

        The distances are matched onto the canonical columns through the
        shared alias table, so a file that names the nasion point
        ``NASION`` still fills the ``NAS`` column; distances to unknown
        fiducials are dropped.
        """
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        for electrode, distances in table.items():
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(str(electrode)))
            for column, fiducial in enumerate(SLOT_IDS, start=1):
                distance = _distance_for(distances, fiducial)
                if distance is not None:
                    self._table.setItem(row, column, QTableWidgetItem(f"{distance:g}"))
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
        # A removed row changes the electrode set: the results follow
        # the table, but nothing is recomputed.
        self.electrodeSetChanged.emit()

    def collect_measurement_table(self) -> MeasurementTable:
        """Return electrode -> fiducial -> distance; empty cells are skipped."""
        table: MeasurementTable = {}
        for row in range(self._table.rowCount()):
            electrode = self._text(row, 0)
            if not electrode:
                continue
            distances: dict[str, float] = {}
            for column, fiducial in enumerate(SLOT_IDS, start=1):
                text = self._text(row, column)
                if not text:
                    continue
                try:
                    distances[fiducial.value] = float(text)
                except ValueError:
                    continue
            if distances:
                table.setdefault(electrode, {}).update(distances)
        return table

    def _text(self, row: int, column: int) -> str:
        item = self._table.item(row, column)
        return "" if item is None else item.text().strip()


def _distance_for(distances: Mapping[str, float], fiducial: FiducialType) -> float | None:
    """The distance of one canonical slot in a measurement mapping."""
    for name, value in distances.items():
        if match_fiducial_name(name) is fiducial:
            return float(value)
    return None
