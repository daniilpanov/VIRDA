"""Live fiducials and Stage 3 measurements editors for the main window.

Each editor widget presents a JSON artifact (``input/fiducials.json`` and
``input/measurements.json``) as an editable table and can load / save it back
to disk.  The JSON <-> model conversion happens in pure, Qt-free helper
functions so round-trips are testable without a ``QApplication``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from virda.io.fiducial_helpers import load_fiducials, save_fiducials
from virda.models.fiducial import Fiducial, Fiducials
from virda_gui.constants import DEFAULT_FIDUCIALS_FILENAME

FIDUCIAL_HEADERS = ["ID", "Name", "X", "Y", "Z", "Method", "Weight"]
COL_ID, COL_NAME, COL_X, COL_Y, COL_Z, COL_METHOD, COL_WEIGHT = range(7)
COORDINATE_SYSTEMS = ["world", "voxel"]
DEFINITION_METHODS = ["manual", "auto", "imported"]


@dataclass(frozen=True)
class FiducialRow:
    """One row of the fiducials table (Qt-free)."""

    fiducial_id: str
    name: str
    coordinates: tuple[float, float, float]
    coordinate_system: str
    definition_method: str
    weight: float


def fiducials_to_rows(fiducials: Fiducials) -> list[FiducialRow]:
    """Flatten a :class:`Fiducials` model into editor rows."""
    return [
        FiducialRow(
            fiducial_id=fiducial.fiducial_id,
            name=fiducial.name,
            coordinates=(
                float(fiducial.coordinates[0]),
                float(fiducial.coordinates[1]),
                float(fiducial.coordinates[2]),
            ),
            coordinate_system=fiducial.coordinate_system,
            definition_method=fiducial.definition_method,
            weight=float(fiducial.weight),
        )
        for fiducial in fiducials.items
    ]


def rows_to_fiducials(rows: list[FiducialRow]) -> Fiducials:
    """Rebuild a :class:`Fiducials` model, validating ids, systems and weights."""
    return Fiducials(
        items=[
            Fiducial(
                fiducial_id=row.fiducial_id,
                name=row.name,
                coordinates=np.asarray(row.coordinates, dtype=np.float64),
                coordinate_system=row.coordinate_system,  # type: ignore[arg-type]
                definition_method=row.definition_method,  # type: ignore[arg-type]
                weight=row.weight,
            )
            for row in rows
        ]
    )


class FiducialsEditor(QWidget):
    """Editable table of fiducials stored in ``input/fiducials.json``."""

    rowsChanged = Signal()  # noqa: N815

    def __init__(
        self,
        parent: QWidget | None = None,
        default_dir: Callable[[], str | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._default_dir = default_dir
        self._path: Path | None = None
        self._coord_systems: list[str] = []
        self._loading = False

        self._table = QTableWidget(0, len(FIDUCIAL_HEADERS), self)
        self._table.setHorizontalHeaderLabels(FIDUCIAL_HEADERS)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_NAME, QHeaderView.ResizeMode.Stretch)
        self._table.cellChanged.connect(self._on_cell_changed)

        add_btn = QPushButton("Add row")
        add_btn.clicked.connect(self.add_row)
        remove_btn = QPushButton("Remove row")
        remove_btn.clicked.connect(self.remove_selected)
        load_btn = QPushButton("Load...")
        load_btn.clicked.connect(self._on_load)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._on_save)
        save_as_btn = QPushButton("Save As...")
        save_as_btn.clicked.connect(self._on_save_as)

        buttons = QHBoxLayout()
        buttons.addWidget(add_btn)
        buttons.addWidget(remove_btn)
        buttons.addStretch(1)
        buttons.addWidget(load_btn)
        buttons.addWidget(save_btn)
        buttons.addWidget(save_as_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._table, 1)
        layout.addLayout(buttons)

    # ---- table helpers ----

    def _set_item(self, row: int, col: int, text: str) -> None:
        self._table.setItem(row, col, QTableWidgetItem(text))

    def _text(self, row: int, col: int) -> str:
        item = self._table.item(row, col)
        return item.text() if item is not None else ""

    def _method(self, row: int) -> str:
        combo = self._table.cellWidget(row, COL_METHOD)
        if isinstance(combo, QComboBox):
            return combo.currentText()
        return "manual"

    def _make_method_combo(self, method: str) -> QComboBox:
        combo = QComboBox(self._table)
        combo.addItems(DEFINITION_METHODS)
        if method in DEFINITION_METHODS:
            combo.setCurrentText(method)
        return combo

    def _on_cell_changed(self, _row: int, _col: int) -> None:
        if not self._loading:
            self.rowsChanged.emit()

    # ---- table content ----

    def set_rows(self, rows: list[FiducialRow]) -> None:
        """Replace the table contents from the given rows."""
        self._loading = True
        try:
            self._coord_systems = [row.coordinate_system for row in rows]
            self._table.setRowCount(len(rows))
            for index, row in enumerate(rows):
                self._set_item(index, COL_ID, row.fiducial_id)
                self._set_item(index, COL_NAME, row.name)
                self._set_item(index, COL_X, f"{row.coordinates[0]}")
                self._set_item(index, COL_Y, f"{row.coordinates[1]}")
                self._set_item(index, COL_Z, f"{row.coordinates[2]}")
                self._table.setCellWidget(
                    index, COL_METHOD, self._make_method_combo(row.definition_method)
                )
                self._set_item(index, COL_WEIGHT, f"{row.weight}")
        finally:
            self._loading = False
        self.rowsChanged.emit()

    def fiducial_ids(self) -> list[str]:
        """Current non-empty ids, read directly from the table (no parsing)."""
        return [
            self._text(row, COL_ID).strip()
            for row in range(self._table.rowCount())
            if self._text(row, COL_ID).strip()
        ]

    def fiducial_rows(self) -> list[FiducialRow]:
        """Parse the table into rows, raising :class:`ValueError` on bad input."""
        rows: list[FiducialRow] = []
        for index in range(self._table.rowCount()):
            fiducial_id = self._text(index, COL_ID).strip()
            if not fiducial_id:
                continue
            rows.append(
                FiducialRow(
                    fiducial_id=fiducial_id,
                    name=self._text(index, COL_NAME).strip(),
                    coordinates=(
                        self._parse_float(index, COL_X, fiducial_id),
                        self._parse_float(index, COL_Y, fiducial_id),
                        self._parse_float(index, COL_Z, fiducial_id),
                    ),
                    coordinate_system=self._coord_systems[index]
                    if index < len(self._coord_systems)
                    else COORDINATE_SYSTEMS[0],
                    definition_method=self._method(index),
                    weight=self._parse_weight(index, fiducial_id),
                )
            )
        return rows

    def _parse_float(self, row: int, col: int, fiducial_id: str) -> float:
        text = self._text(row, col).strip()
        try:
            return float(text)
        except ValueError:
            raise ValueError(
                f"Fiducial {fiducial_id!r} row {row + 1}: {FIDUCIAL_HEADERS[col]} must be a number"
            ) from None

    def _parse_weight(self, row: int, fiducial_id: str) -> float:
        text = self._text(row, COL_WEIGHT).strip()
        if not text:
            return 1.0
        try:
            return float(text)
        except ValueError:
            raise ValueError(
                f"Fiducial {fiducial_id!r} row {row + 1}: Weight must be a number"
            ) from None

    def add_row(self) -> None:
        index = self._table.rowCount()
        self._table.insertRow(index)
        self._coord_systems.append(COORDINATE_SYSTEMS[0])
        self._table.setCellWidget(index, COL_METHOD, self._make_method_combo("manual"))
        self._set_item(index, COL_WEIGHT, "1.0")
        self._table.scrollToBottom()
        self.rowsChanged.emit()

    def remove_selected(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            return
        self._table.removeRow(row)
        if row < len(self._coord_systems):
            self._coord_systems.pop(row)
        self.rowsChanged.emit()

    # ---- load / save ----

    def load(self, path: Path, *, interactive: bool = True) -> bool:
        """Load *path* into the table; returns False when *path* is invalid."""
        try:
            fiducials = load_fiducials(path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            if interactive:
                QMessageBox.critical(self, "Load fiducials", f"Could not load fiducials:\n{exc}")
            return False
        self.set_rows(fiducials_to_rows(fiducials))
        self._path = path
        return True

    def save_to(self, path: Path) -> bool:
        """Save the current table to *path*; returns False on validation failure."""
        try:
            fiducials = rows_to_fiducials(self.fiducial_rows())
        except ValueError as exc:
            QMessageBox.critical(self, "Save fiducials", f"Invalid table:\n{exc}")
            return False
        try:
            save_fiducials(path, fiducials)
        except OSError as exc:
            QMessageBox.critical(self, "Save fiducials", f"Could not write file:\n{exc}")
            return False
        self._path = path
        return True

    def _on_load(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self, "Load fiducials", self._start_dir(), "JSON (*.json);;All files (*)"
        )
        if path:
            self.load(Path(path))

    def _on_save(self) -> None:
        if self._path is None:
            self._on_save_as()
            return
        self.save_to(self._path)

    def _on_save_as(self) -> None:
        path, _selected_filter = QFileDialog.getSaveFileName(
            self, "Save fiducials as", self._start_path(), "JSON (*.json);;All files (*)"
        )
        if path:
            self.save_to(Path(path))

    def _start_dir(self) -> str:
        """Default directory for open dialogs."""
        if self._path is not None:
            return str(self._path.parent)
        default_dir = self._default_dir() if self._default_dir else None
        return default_dir or ""

    def _start_path(self) -> str:
        """Default location for save dialogs."""
        if self._path is not None:
            return str(self._path)
        default_dir = self._start_dir()
        if default_dir:
            return str(Path(default_dir) / "input" / DEFAULT_FIDUCIALS_FILENAME)
        return DEFAULT_FIDUCIALS_FILENAME