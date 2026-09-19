"""Live fiducials and Stage 3 measurements editors for the main window.

Each editor widget presents a JSON artifact (``input/fiducials.json`` and
``input/measurements.json``) as an editable table and can load / save it back
to disk.  The JSON <-> model conversion happens in pure, Qt-free helper
functions so round-trips are testable without a ``QApplication``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from virda.io.fiducial_helpers import load_fiducials, save_fiducials
from virda.models.fiducial import Fiducial, Fiducials
from virda_gui.constants import DEFAULT_FIDUCIALS_FILENAME, DEFAULT_MEASUREMENTS_FILENAME

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


COL_ELECTRODE = 0


@dataclass(frozen=True)
class MeasurementRow:
    """One electrode row of the measurements table (Qt-free)."""

    electrode_id: str
    measured_distances: dict[str, float] = field(default_factory=dict)


def measurements_fiducial_ids(
    rows: list[MeasurementRow], weights: dict[str, float] | None = None
) -> list[str]:
    """Unique fiducial ids (in order) referenced by rows and weights."""
    seen: set[str] = set()
    ids: list[str] = []
    for row in rows:
        for fiducial_id in row.measured_distances:
            if fiducial_id not in seen:
                seen.add(fiducial_id)
                ids.append(fiducial_id)
    for fiducial_id in weights or {}:
        if fiducial_id not in seen:
            seen.add(fiducial_id)
            ids.append(fiducial_id)
    return ids


def measurements_schema_to_rows(data: dict[str, Any]) -> list[MeasurementRow]:
    """Convert a measurements JSON object into editor rows."""
    return [
        MeasurementRow(
            electrode_id=str(item.get("electrode_id") or ""),
            measured_distances={
                str(fiducial_id): float(distance)
                for fiducial_id, distance in item["measured_distances"].items()
            },
        )
        for item in data.get("electrodes", [])
    ]


def measurements_rows_to_schema(rows: list[MeasurementRow]) -> dict[str, Any]:
    """Serialize editor rows into the measurements JSON object."""
    return {
        "electrodes": [
            {
                "electrode_id": row.electrode_id,
                "measured_distances": row.measured_distances,
            }
            for row in rows
        ]
    }


class MeasurementsEditor(QWidget):
    """Editable table of Stage 3 measurements for ``input/measurements.json``."""

    def __init__(
        self,
        parent: QWidget | None = None,
        default_dir: Callable[[], str | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._default_dir = default_dir
        self._path: Path | None = None
        self._fiducial_ids: list[str] = []
        self._weights: dict[str, QLineEdit] = {}
        self._loading = False

        self._table = QTableWidget(0, 1, self)
        self._table.setHorizontalHeaderLabels(["Electrode"])
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_ELECTRODE, QHeaderView.ResizeMode.Stretch)

        self._weights_row = QWidget(self)
        self._weights_layout = QHBoxLayout(self._weights_row)
        self._weights_layout.setContentsMargins(0, 0, 0, 0)
        self._weights_layout.setSpacing(6)
        self._weights_row.setVisible(False)

        add_btn = QPushButton("Add electrode")
        add_btn.clicked.connect(self.add_row)
        remove_btn = QPushButton("Remove electrode")
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
        layout.addWidget(self._weights_row)
        layout.addWidget(self._table, 1)
        layout.addLayout(buttons)

    # ---- table helpers ----

    def _set_item(self, row: int, col: int, text: str) -> None:
        self._table.setItem(row, col, QTableWidgetItem(text))

    def _text(self, row: int, col: int) -> str:
        item = self._table.item(row, col)
        return item.text() if item is not None else ""

    def _capture_text_rows(self) -> list[tuple[str, dict[str, str]]]:
        """Snapshot the table as (electrode, {fiducial_id: text}) before a rebuild."""
        captured: list[tuple[str, dict[str, str]]] = []
        for row_index in range(self._table.rowCount()):
            distances = {
                self._fiducial_ids[col - 1]: self._text(row_index, col)
                for col in range(1, self._table.columnCount())
                if col - 1 < len(self._fiducial_ids)
            }
            captured.append((self._text(row_index, COL_ELECTRODE), distances))
        return captured

    # ---- table content ----

    def set_fiducial_ids(self, fiducial_ids: list[str]) -> None:
        """Set the fiducial columns, preserving already-typed distances."""
        if fiducial_ids == self._fiducial_ids:
            return
        captured = self._capture_text_rows()
        self._fiducial_ids = list(fiducial_ids)
        self._loading = True
        try:
            self._table.clear()
            self._table.setColumnCount(len(fiducial_ids) + 1)
            self._table.setHorizontalHeaderLabels(["Electrode", *fiducial_ids])
            self._table.setRowCount(len(captured))
            for row_index, (electrode_id, distances) in enumerate(captured):
                self._set_item(row_index, COL_ELECTRODE, electrode_id)
                for col, fiducial_id in enumerate(fiducial_ids, start=1):
                    self._set_item(row_index, col, distances.get(fiducial_id, ""))
        finally:
            self._loading = False
        self._rebuild_weights_row()

    def set_measurement_rows(
        self, rows: list[MeasurementRow], weights: dict[str, float] | None = None
    ) -> None:
        """Replace the table contents from the given rows."""
        self._loading = True
        try:
            self._table.setRowCount(len(rows))
            for index, row in enumerate(rows):
                self._set_item(index, COL_ELECTRODE, row.electrode_id)
                for col, fiducial_id in enumerate(self._fiducial_ids, start=1):
                    if fiducial_id in row.measured_distances:
                        self._set_item(index, col, f"{row.measured_distances[fiducial_id]}")
        finally:
            self._loading = False
        if weights is not None:
            self.set_weights(weights)

    def measurement_rows(self) -> list[MeasurementRow]:
        """Parse the table into rows, raising :class:`ValueError` on bad input."""
        rows: list[MeasurementRow] = []
        for row_index in range(self._table.rowCount()):
            electrode_id = self._text(row_index, COL_ELECTRODE).strip()
            distances: dict[str, float] = {}
            for col, fiducial_id in enumerate(self._fiducial_ids, start=1):
                text = self._text(row_index, col).strip()
                if not text:
                    continue
                try:
                    distances[fiducial_id] = float(text)
                except ValueError:
                    raise ValueError(
                        f"Electrode row {row_index + 1}: distance to "
                        f"{fiducial_id!r} must be a number"
                    ) from None
            if distances:
                rows.append(MeasurementRow(electrode_id=electrode_id, measured_distances=distances))
        return rows

    def add_row(self) -> None:
        index = self._table.rowCount()
        self._table.insertRow(index)
        self._table.scrollToBottom()
        self._table.setCurrentCell(index, COL_ELECTRODE)

    def remove_selected(self) -> None:
        row = self._table.currentRow()
        if row >= 0:
            self._table.removeRow(row)

    # ---- fiducial weights ----

    def _rebuild_weights_row(self) -> None:
        while self._weights_layout.count():
            item = self._weights_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._weights.clear()
        self._weights_row.setVisible(bool(self._fiducial_ids))
        if not self._fiducial_ids:
            return
        self._weights_layout.addWidget(QLabel("Fiducial weights:", self))
        for fiducial_id in self._fiducial_ids:
            edit = QLineEdit(self)
            edit.setFixedWidth(70)
            edit.setToolTip(f"Weight of fiducial {fiducial_id}")
            self._weights[fiducial_id] = edit
            self._weights_layout.addWidget(edit)
            self._weights_layout.addWidget(QLabel(fiducial_id, self))
        self._weights_layout.addStretch(1)

    def set_weights(self, weights: dict[str, float]) -> None:
        for fiducial_id, edit in self._weights.items():
            if fiducial_id in weights:
                edit.setText(f"{weights[fiducial_id]}")

    def weight_values(self) -> dict[str, str]:
        """Return the non-empty weight fields as strings."""
        return {
            fiducial_id: edit.text().strip()
            for fiducial_id, edit in self._weights.items()
            if edit.text().strip()
        }

    def _parsed_weights(self) -> dict[str, float]:
        weights: dict[str, float] = {}
        for fiducial_id, text in self.weight_values().items():
            try:
                weights[fiducial_id] = float(text)
            except ValueError:
                raise ValueError(f"Invalid weight for {fiducial_id!r}: {text!r}") from None
        return weights

    # ---- load / save ----

    def load(self, path: Path, *, interactive: bool = True) -> bool:
        """Load *path* into the table; returns False when *path* is invalid."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            if interactive:
                QMessageBox.critical(
                    self, "Load measurements", f"Could not load measurements:\n{exc}"
                )
            return False
        if not isinstance(data, dict):
            if interactive:
                QMessageBox.critical(self, "Load measurements", "Expected a JSON object.")
            return False
        rows = measurements_schema_to_rows(data)
        raw_weights = data.get("fiducial_weights")
        weights = (
            {str(fiducial_id): float(value) for fiducial_id, value in raw_weights.items()}
            if isinstance(raw_weights, dict)
            else {}
        )
        file_ids = measurements_fiducial_ids(rows, weights)
        ids = list(self._fiducial_ids)
        for fiducial_id in file_ids:
            if fiducial_id not in ids:
                ids.append(fiducial_id)
        self._path = path
        self.set_fiducial_ids(ids)
        self.set_measurement_rows(rows, weights)
        return True

    def save_to(self, path: Path) -> bool:
        """Save the current table to *path*; returns False on validation failure."""
        try:
            schema = self.collected_schema()
        except ValueError as exc:
            QMessageBox.critical(self, "Save measurements", f"Invalid table:\n{exc}")
            return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            QMessageBox.critical(self, "Save measurements", f"Could not write file:\n{exc}")
            return False
        self._path = path
        return True

    def collected_schema(self) -> dict[str, Any]:
        """Return the measurements JSON object for the current table."""
        schema = measurements_rows_to_schema(self.measurement_rows())
        weights = self._parsed_weights()
        if weights:
            schema["fiducial_weights"] = weights
        return schema

    def _on_load(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self, "Load measurements", self._start_dir(), "JSON (*.json);;All files (*)"
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
            self, "Save measurements as", self._start_path(), "JSON (*.json);;All files (*)"
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
            return str(Path(default_dir) / "input" / DEFAULT_MEASUREMENTS_FILENAME)
        return DEFAULT_MEASUREMENTS_FILENAME