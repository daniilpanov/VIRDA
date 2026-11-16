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

_COLUMNS = ("Electrode", *(fiducial.value for fiducial in SLOT_IDS), "Predict")

#: The column of the "Predict" button.
_PREDICT_COLUMN = len(_COLUMNS) - 1

#: Why a prediction cannot run when the electrode has too few measurements.
_TOO_FEW = "two measured distances are needed"


class MeasurementsTable(QWidget):
    """Pivot grid of the measured distances: one row per electrode.

    The columns are the canonical fiducial slots (NAS, LPA, RPA) and the
    rows the electrodes; a cell holds the measured distance in
    millimetres and an empty cell is a measurement that is missing.

    The last column predicts the missing measurement of a row: clicking
    its button emits :attr:`predictRequested` with the electrode and the
    fiducial to predict, and the popup fills the cell with a predicted
    value (in italics, so a prediction stays recognizable; editing it
    makes it an ordinary measurement again). The button is active when
    the row holds exactly two of the three measurements.

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
    predictRequested = Signal(str, str)  # noqa: N815 - Qt signal API

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._table = QTableWidget(0, len(_COLUMNS), self)
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.itemChanged.connect(self._on_item_changed)
        self._predict_buttons: dict[int, QPushButton] = {}
        self._prediction_available = True
        self._prediction_reason = ""

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
        if item.column() > 0:
            # An edited cell is an ordinary measurement from now on.
            self._clear_prediction(item.row(), item.column())
        self.electrodeEdited.emit(self._electrode_id_of_row(item.row()))
        self._refresh_predict_buttons()

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
        self._rebuild_predict_buttons()

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
        self._rebuild_predict_buttons()

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
        self._rebuild_predict_buttons()

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

    def set_prediction_state(self, available: bool, reason: str = "") -> None:
        """Whether a prediction can run at all, and why it cannot."""
        self._prediction_available = available
        self._prediction_reason = reason
        self._refresh_predict_buttons()

    def set_predicted(
        self, electrode_id: str, fiducial_id: str, distance: float, note: str
    ) -> bool:
        """Fill a cell with a predicted distance, marked in italics.

        ``note`` is the tooltip of the cell (the residual, the ambiguity).
        Returns whether the cell was found.
        """
        row = self._row_of(electrode_id)
        column = self._column_of(fiducial_id)
        if row is None or column is None:
            return False
        self._table.blockSignals(True)
        item = QTableWidgetItem(f"{distance:g}")
        font = item.font()
        font.setItalic(True)
        item.setFont(font)
        item.setToolTip(note)
        self._table.setItem(row, column, item)
        self._table.blockSignals(False)
        self._refresh_predict_buttons()
        # The predicted distance moves its own electrode: the whole set
        # is not recomputed.
        self.electrodeEdited.emit(electrode_id)
        return True

    def clear_predicted(self, electrode_id: str, fiducial_id: str) -> bool:
        """Empty the cell of a predicted measurement.

        Returns whether the cell was found.
        """
        row = self._row_of(electrode_id)
        column = self._column_of(fiducial_id)
        if row is None or column is None:
            return False
        self._table.blockSignals(True)
        self._table.setItem(row, column, QTableWidgetItem())
        self._table.blockSignals(False)
        self._refresh_predict_buttons()
        # The cleared cell was a measurement of this electrode: only it
        # is recomputed.
        self.electrodeEdited.emit(electrode_id)
        return True

    def predicted_cells(self) -> list[tuple[str, str]]:
        """The cells that hold a predicted value."""
        cells: list[tuple[str, str]] = []
        for row in range(self._table.rowCount()):
            electrode = self._electrode_id_of_row(row)
            for column, fiducial in enumerate(SLOT_IDS, start=1):
                item = self._table.item(row, column)
                if item is not None and item.font().italic():
                    cells.append((electrode, fiducial.value))
        return cells

    def _row_of(self, electrode_id: str) -> int | None:
        """The row of an electrode, or ``None`` when it is not in the grid."""
        for row in range(self._table.rowCount()):
            if self._electrode_id_of_row(row) == electrode_id:
                return row
        return None

    def _column_of(self, fiducial_id: str) -> int | None:
        """The column of a fiducial, or ``None`` when it has none."""
        for column, fiducial in enumerate(SLOT_IDS, start=1):
            if fiducial.value == fiducial_id:
                return column
        return None

    def _clear_prediction(self, row: int, column: int) -> None:
        """Drop the prediction marking of an edited cell."""
        item = self._table.item(row, column)
        if item is None or not item.font().italic():
            return
        font = item.font()
        font.setItalic(False)
        item.setFont(font)
        item.setToolTip("")

    def _rebuild_predict_buttons(self) -> None:
        """Give every row a button that predicts its missing measurement."""
        self._predict_buttons.clear()
        for row in range(self._table.rowCount()):
            button = QPushButton("Predict", self._table)
            button.clicked.connect(lambda _checked=False, r=row: self._on_predict_clicked(r))
            self._predict_buttons[row] = button
            self._table.setCellWidget(row, _PREDICT_COLUMN, button)
        self._refresh_predict_buttons()

    def _refresh_predict_buttons(self) -> None:
        """Enable the buttons of the rows whose measurement can be predicted."""
        for row, button in self._predict_buttons.items():
            missing = self._missing_fiducial(row)
            if missing is None:
                button.setEnabled(False)
                complete = len(self._row_distances(row)) == len(SLOT_IDS)
                button.setToolTip("" if complete else _TOO_FEW)
            elif not self._prediction_available:
                button.setEnabled(False)
                button.setToolTip(self._prediction_reason)
            else:
                button.setEnabled(True)
                button.setToolTip("")

    def _missing_fiducial(self, row: int) -> FiducialType | None:
        """The fiducial whose measurement the row lacks, if there is one."""
        missing = [
            fiducial
            for column, fiducial in enumerate(SLOT_IDS, start=1)
            if not self._text(row, column)
        ]
        return missing[0] if len(missing) == 1 else None

    def _on_predict_clicked(self, row: int) -> None:
        """Ask the popup to predict the missing measurement of a row."""
        electrode = self._electrode_id_of_row(row)
        missing = self._missing_fiducial(row)
        if not electrode or missing is None:
            return
        self.predictRequested.emit(electrode, missing.value)

    def _row_distances(self, row: int) -> dict[str, float]:
        """The distances of one row, keyed by the fiducial slot."""
        distances: dict[str, float] = {}
        for column, fiducial in enumerate(SLOT_IDS, start=1):
            text = self._text(row, column)
            if not text:
                continue
            try:
                distances[fiducial.value] = float(text)
            except ValueError:
                continue
        return distances

    def _text(self, row: int, column: int) -> str:
        item = self._table.item(row, column)
        return "" if item is None else item.text().strip()


def _distance_for(distances: Mapping[str, float], fiducial: FiducialType) -> float | None:
    """The distance of one canonical slot in a measurement mapping."""
    for name, value in distances.items():
        if match_fiducial_name(name) is fiducial:
            return float(value)
    return None
