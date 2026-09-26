"""Live fiducials and Stage 3 measurements editors for the main window.

Each editor widget presents a JSON artifact (``input/fiducials.json`` and
``input/measurements.json``) as an editable table and can load / save it back
to disk.  The JSON <-> model conversion happens in pure, Qt-free helper
functions so round-trips are testable without a ``QApplication``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QStandardItemModel
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from virda.io.exporters.fiducials import export_fiducials
from virda.io.exporters.localization_table import export_localization_table
from virda.io.importers.fiducials import import_fiducials
from virda.models.electrode import Electrodes
from virda.models.fiducial import Fiducial, Fiducials
from virda_gui.constants import DEFAULT_FIDUCIALS_FILENAME, DEFAULT_MEASUREMENTS_FILENAME
from virda_gui.state import AppState
from virda_gui.viewer.frames import (
    FRAME_HEAD,
    FRAME_SCANNER,
    FRAME_VOXEL,
    frame_available,
    frame_label,
    world_to_frame_matrix,
)
from virda_gui.viewer.scene import transform_points

FIDUCIAL_HEADERS = ["ID", "X", "Y", "Z", "Weight"]
COL_ID, COL_X, COL_Y, COL_Z, COL_W = range(5)
COL_ELECTRODE = 0
COORDINATE_SYSTEMS = ["world", "voxel"]

#: The coordinate systems the fiducial X/Y/Z columns are entered in.  ``head``
#: is scanner RAS relative to the NIfTI volume centre (same maths as cRAS);
#: ``voxel`` needs a loaded NIfTI scan and is rejected otherwise.
EDITOR_FRAME_IDS: tuple[str, str, str] = (FRAME_SCANNER, FRAME_HEAD, FRAME_VOXEL)

#: The three canonical fiducials the fixed measurements columns map onto.
CANONICAL_FIDUCIALS: tuple[str, str, str] = ("LPA", "RPA", "NAS")
MEASUREMENT_HEADERS = ["Electrode", *CANONICAL_FIDUCIALS]

#: Read-only preview table columns for the live localization output.  The X/Y/Z
#: columns show the localized coordinates in the frame picked with the
#: coordinate-system combo above the table.
LOCALIZATION_HEADERS = [
    "Name",
    "X",
    "Y",
    "Z",
    "LPA",
    "RPA",
    "NAS",
    "Residual (mm)",
    "Flagged",
]
LOC_COL_NAME = 0
LOC_COL_COORDS = 1
LOC_COL_DISTANCES = 4
LOC_COL_RESIDUAL = 7
LOC_COL_FLAGGED = 8


def canonical_fiducial_id(fiducial_id: str) -> str:
    """Return the canonical (uppercase) form of NAS/LPA/RPA fiducial ids.

    Legacy projects sometimes store the canonical fiducials with different
    casing (e.g. ``"nas"``); the fixed measurements columns and the live
    localization both key distances by the three canonical ids, so every
    reader normalises them here.
    """
    lowered = fiducial_id.lower()
    for canonical in CANONICAL_FIDUCIALS:
        if lowered == canonical.lower():
            return canonical
    return fiducial_id


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
            fiducial_id=canonical_fiducial_id(fiducial.fiducial_id),
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
                fiducial_id=canonical_fiducial_id(row.fiducial_id),
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
    fileSaved = Signal()  # noqa: N815
    inputFrameChanged = Signal(object, object)  # noqa: N815  # old_frame: str, new_frame: str

    def __init__(
        self,
        parent: QWidget | None = None,
        default_dir: Callable[[], str | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._default_dir = default_dir
        self._path: Path | None = None
        self._coord_systems: list[str] = []
        self._row_meta: list[tuple[str, str, str, float]] = []
        self._loading = False
        self._dirty = False
        self._input_frame: str = EDITOR_FRAME_IDS[0]

        self._table = QTableWidget(0, len(FIDUCIAL_HEADERS), self)
        self._table.setHorizontalHeaderLabels(FIDUCIAL_HEADERS)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_ID, QHeaderView.ResizeMode.Stretch)
        self._table.cellChanged.connect(self._on_cell_changed)

        self._frame_row = QWidget(self)
        self._frame_layout = QHBoxLayout(self._frame_row)
        self._frame_layout.setContentsMargins(0, 0, 0, 0)
        self._frame_layout.setSpacing(6)
        self._frame_layout.addWidget(QLabel("Coordinate system:", self))
        self._frame_combo = QComboBox(self._frame_row)
        for frame_id in EDITOR_FRAME_IDS:
            self._frame_combo.addItem(frame_label(frame_id), frame_id)
        self._frame_combo.blockSignals(True)
        self._frame_combo.setCurrentIndex(EDITOR_FRAME_IDS.index(EDITOR_FRAME_IDS[0]))
        self._frame_combo.blockSignals(False)
        self._frame_combo.currentIndexChanged.connect(self._on_frame_selected)
        self._frame_layout.addWidget(self._frame_combo)
        self._frame_layout.addWidget(
            QLabel("The X/Y/Z columns above are interpreted in this system.", self._frame_row)
        )
        self._frame_layout.addStretch(1)

        add_btn = QPushButton("Add row")
        add_btn.clicked.connect(self.add_row)
        remove_btn = QPushButton("Remove row")
        remove_btn.clicked.connect(self.remove_selected)
        clear_btn = QPushButton("Clear all")
        clear_btn.clicked.connect(self.clear_all)
        load_btn = QPushButton("Load...")
        load_btn.clicked.connect(self._on_load)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._on_save)
        save_as_btn = QPushButton("Save As...")
        save_as_btn.clicked.connect(self._on_save_as)

        buttons = QHBoxLayout()
        buttons.addWidget(add_btn)
        buttons.addWidget(remove_btn)
        buttons.addWidget(clear_btn)
        buttons.addStretch(1)
        buttons.addWidget(load_btn)
        buttons.addWidget(save_btn)
        buttons.addWidget(save_as_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._frame_row)
        layout.addWidget(self._table, 1)
        layout.addLayout(buttons)

    # ---- table helpers ----

    def _set_item(self, row: int, col: int, text: str) -> None:
        self._table.setItem(row, col, QTableWidgetItem(text))

    def _text(self, row: int, col: int) -> str:
        item = self._table.item(row, col)
        return item.text() if item is not None else ""

    def _on_cell_changed(self, _row: int, _col: int) -> None:
        if not self._loading:
            self._dirty = True
            self._highlight_duplicates()
            self.rowsChanged.emit()

    def _on_frame_selected(self) -> None:
        if self._loading:
            return
        old_frame = self._input_frame
        new_frame = self._frame_combo.currentData()
        if isinstance(new_frame, str) and not isinstance(new_frame, bytes):
            self._input_frame = new_frame
            self.inputFrameChanged.emit(old_frame, new_frame)
        self.rowsChanged.emit()

    def input_frame(self) -> str:
        """The coordinate system the X/Y/Z columns are interpreted in."""
        frame = self._frame_combo.currentData()
        return frame if isinstance(frame, str) else EDITOR_FRAME_IDS[0]

    def set_input_frame(self, frame: str) -> None:
        """Select the input coordinate system, ignoring unknown frames."""
        if frame in EDITOR_FRAME_IDS:
            self._input_frame = frame
            self._frame_combo.setCurrentIndex(EDITOR_FRAME_IDS.index(frame))

    # ---- table content ----

    def set_rows(self, rows: list[FiducialRow]) -> None:
        """Replace the table contents from the given rows."""
        self._loading = True
        try:
            self._coord_systems = [row.coordinate_system for row in rows]
            self._row_meta = [
                (row.name, row.coordinate_system, row.definition_method, row.weight) for row in rows
            ]
            self._table.setRowCount(len(rows))
            for index, row in enumerate(rows):
                self._set_item(index, COL_ID, row.fiducial_id)
                self._set_item(index, COL_X, f"{row.coordinates[0]}")
                self._set_item(index, COL_Y, f"{row.coordinates[1]}")
                self._set_item(index, COL_Z, f"{row.coordinates[2]}")
                self._set_item(index, COL_W, f"{row.weight}")
        finally:
            self._loading = False
        self._dirty = True
        self._highlight_duplicates()
        self.rowsChanged.emit()

    def is_dirty(self) -> bool:
        """Whether the table holds edits not yet written to disk."""
        return self._dirty

    def fiducial_ids(self) -> list[str]:
        """Current non-empty canonical ids, read directly from the table."""
        return [
            canonical_fiducial_id(self._text(row, COL_ID).strip())
            for row in range(self._table.rowCount())
            if self._text(row, COL_ID).strip()
        ]

    def fiducial_rows(self) -> list[FiducialRow]:
        """Parse the table into rows, raising :class:`ValueError` on bad input."""
        rows: list[FiducialRow] = []
        for index in range(self._table.rowCount()):
            fiducial_id = canonical_fiducial_id(self._text(index, COL_ID).strip())
            if not fiducial_id:
                continue
            if index < len(self._row_meta):
                name, coordinate_system, definition_method, meta_weight = self._row_meta[index]
            elif index < len(self._coord_systems):
                name, coordinate_system, definition_method, meta_weight = (
                    fiducial_id,
                    self._coord_systems[index],
                    "manual",
                    1.0,
                )
            else:
                name, coordinate_system, definition_method, meta_weight = (
                    fiducial_id,
                    COORDINATE_SYSTEMS[0],
                    "manual",
                    1.0,
                )
            rows.append(
                FiducialRow(
                    fiducial_id=fiducial_id,
                    name=name or fiducial_id,
                    coordinates=(
                        self._parse_float(index, COL_X, fiducial_id),
                        self._parse_float(index, COL_Y, fiducial_id),
                        self._parse_float(index, COL_Z, fiducial_id),
                    ),
                    coordinate_system=coordinate_system,
                    definition_method=definition_method,
                    weight=self._parse_weight(index, fiducial_id, meta_weight),
                )
            )
        return rows

    def _parse_weight(self, row: int, fiducial_id: str, fallback: float) -> float:
        text = self._text(row, COL_W).strip()
        if not text:
            return fallback
        try:
            weight = float(text)
        except ValueError:
            raise ValueError(
                f"Fiducial {fiducial_id!r} row {row + 1}: Weight must be a number"
            ) from None
        if weight <= 0:
            raise ValueError(f"Fiducial {fiducial_id!r} row {row + 1}: Weight must be positive")
        return weight

    def _parse_float(self, row: int, col: int, fiducial_id: str) -> float:
        text = self._text(row, col).strip()
        try:
            return float(text)
        except ValueError:
            raise ValueError(
                f"Fiducial {fiducial_id!r} row {row + 1}: {FIDUCIAL_HEADERS[col]} must be a number"
            ) from None

    def add_row(self) -> None:
        index = self._table.rowCount()
        self._table.insertRow(index)
        self._coord_systems.append(COORDINATE_SYSTEMS[0])
        self._row_meta.append(("", COORDINATE_SYSTEMS[0], "manual", 1.0))
        self._table.scrollToBottom()

    def append_point(self, fiducial_id: str, coordinates: tuple[float, float, float]) -> None:
        """Append a row with the given id and coordinates (e.g. surface-picked)."""
        index = self._table.rowCount()
        self._table.insertRow(index)
        self._coord_systems.append(COORDINATE_SYSTEMS[0])
        self._row_meta.append((fiducial_id, COORDINATE_SYSTEMS[0], "manual", 1.0))
        self._loading = True
        try:
            self._set_item(index, COL_ID, fiducial_id)
            self._set_item(index, COL_X, f"{coordinates[0]}")
            self._set_item(index, COL_Y, f"{coordinates[1]}")
            self._set_item(index, COL_Z, f"{coordinates[2]}")
            self._set_item(index, COL_W, "1.0")
        finally:
            self._loading = False
        self._dirty = True
        self._highlight_duplicates()
        self.rowsChanged.emit()
        self._table.setCurrentCell(index, COL_ID)
        self._table.scrollToBottom()

    def remove_selected(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            return
        self._table.removeRow(row)
        if row < len(self._coord_systems):
            self._coord_systems.pop(row)
        if row < len(self._row_meta):
            self._row_meta.pop(row)
        self._dirty = True
        self._highlight_duplicates()
        self.rowsChanged.emit()

    def clear_all(self) -> None:
        """Remove all rows after an explicit confirmation."""
        if self._table.rowCount() == 0:
            return
        answer = QMessageBox.question(
            self,
            "Clear fiducials",
            "Remove all fiducial rows?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.set_rows([])

    def _highlight_duplicates(self) -> None:
        """Mark ID cells sharing a canonical id with a red background."""
        counts: dict[str, int] = {}
        for row in range(self._table.rowCount()):
            fiducial_id = canonical_fiducial_id(self._text(row, COL_ID).strip())
            if fiducial_id:
                counts[fiducial_id] = counts.get(fiducial_id, 0) + 1
        for row in range(self._table.rowCount()):
            item = self._table.item(row, COL_ID)
            if item is None:
                continue
            fiducial_id = canonical_fiducial_id(self._text(row, COL_ID).strip())
            if fiducial_id and counts.get(fiducial_id, 0) > 1:
                item.setBackground(QBrush(QColor(150, 40, 40)))
                item.setToolTip("Duplicate fiducial id")
            else:
                item.setBackground(QBrush())
                item.setToolTip("")

    def clear(self) -> None:
        """Reset the table and forget the current file path."""
        self._path = None
        self.set_rows([])
        self._dirty = False

    # ---- load / save ----

    def load(self, path: Path, *, interactive: bool = True) -> bool:
        """Load *path* into the table; returns False when *path* is invalid."""
        try:
            fiducials = import_fiducials(path)
            rows = fiducials_to_rows(fiducials)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            if interactive:
                QMessageBox.critical(self, "Load fiducials", f"Could not load fiducials:\n{exc}")
            return False
        self.set_rows(rows)
        self._path = path
        self._dirty = False
        return True

    def save_to(self, path: Path) -> bool:
        """Save the current table to *path*; returns False on validation failure."""
        try:
            fiducials = rows_to_fiducials(self.fiducial_rows())
        except ValueError as exc:
            QMessageBox.critical(self, "Save fiducials", f"Invalid table:\n{exc}")
            return False
        if not _confirm_outside_project(self, path, "Save fiducials"):
            return False
        try:
            export_fiducials(path, fiducials)
        except OSError as exc:
            QMessageBox.critical(self, "Save fiducials", f"Could not write file:\n{exc}")
            return False
        self._path = path
        self._dirty = False
        self.fileSaved.emit()
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


def _is_outside_project(path: Path, default_dir: Callable[[], str | None] | None) -> bool:
    """Whether *path* lies outside the project folder behind *default_dir*."""
    if default_dir is None:
        return False
    root = default_dir()
    if not root:
        return False
    try:
        path.resolve().relative_to(Path(root).resolve())
    except ValueError:
        return True
    return False


def _confirm_outside_project(parent: QWidget, path: Path, title: str) -> bool:
    """Confirm saving to *path* outside the project; False aborts the save."""
    if not _is_outside_project(path, getattr(parent, "_default_dir", None)):
        return True
    answer = QMessageBox.question(
        parent,
        title,
        f"{path}\nis outside the project folder. Save there anyway?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Cancel,
    )
    return answer == QMessageBox.StandardButton.Yes


class MeasurementsEditor(QWidget):
    """Editable table of Stage 3 measurements for ``input/measurements.json``.

    The distance columns are fixed to the three canonical fiducials
    (LPA / RPA / NAS) and the whole editor is locked until those fiducials
    exist in the fiducials table (:meth:`set_fiducial_ready`).
    """

    rowsChanged = Signal()  # noqa: N815
    fileSaved = Signal()  # noqa: N815

    def __init__(
        self,
        parent: QWidget | None = None,
        default_dir: Callable[[], str | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._default_dir = default_dir
        self._path: Path | None = None
        self._loading = False
        self._ready = False
        self._dirty = False

        self._table = QTableWidget(0, len(MEASUREMENT_HEADERS), self)
        self._table.setHorizontalHeaderLabels(MEASUREMENT_HEADERS)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_ELECTRODE, QHeaderView.ResizeMode.Stretch)
        self._table.cellChanged.connect(self._on_cell_changed)

        self._hint = QLabel(
            "Fill in the NAS, LPA and RPA fiducials above to enable measurements.", self
        )
        self._hint.setWordWrap(True)

        add_btn = QPushButton("Add electrode")
        add_btn.clicked.connect(self.add_row)
        remove_btn = QPushButton("Remove electrode")
        remove_btn.clicked.connect(self.remove_selected)
        clear_btn = QPushButton("Clear all")
        clear_btn.clicked.connect(self.clear_all)
        load_btn = QPushButton("Load...")
        load_btn.clicked.connect(self._on_load)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._on_save)
        save_as_btn = QPushButton("Save As...")
        save_as_btn.clicked.connect(self._on_save_as)
        self._controls = [add_btn, remove_btn, clear_btn, load_btn, save_btn, save_as_btn]

        buttons = QHBoxLayout()
        buttons.addWidget(add_btn)
        buttons.addWidget(remove_btn)
        buttons.addWidget(clear_btn)
        buttons.addStretch(1)
        buttons.addWidget(load_btn)
        buttons.addWidget(save_btn)
        buttons.addWidget(save_as_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._hint)
        layout.addWidget(self._table, 1)
        layout.addLayout(buttons)

        self._apply_ready(False)

    # ---- table helpers ----

    def _set_item(self, row: int, col: int, text: str) -> None:
        self._table.setItem(row, col, QTableWidgetItem(text))

    def _text(self, row: int, col: int) -> str:
        item = self._table.item(row, col)
        return item.text() if item is not None else ""

    def _on_cell_changed(self, _row: int, _col: int) -> None:
        if not self._loading:
            self._dirty = True
            self._highlight_duplicates()
            self.rowsChanged.emit()

    # ---- readiness ----

    def set_fiducial_ready(self, ready: bool) -> None:
        """Lock or unlock the table depending on the canonical fiducials."""
        self._apply_ready(bool(ready))

    def _apply_ready(self, ready: bool) -> None:
        self._ready = ready
        self._table.setEnabled(ready)
        for control in self._controls:
            control.setEnabled(ready)
        self._hint.setVisible(not ready)

    @property
    def fiducial_ready(self) -> bool:
        """Whether measurements are enabled (canonical fiducials are present)."""
        return self._ready

    # ---- table content ----

    def _distance_text(self, distances: dict[str, float], fiducial_id: str) -> str:
        fiducial_lower = fiducial_id.lower()
        for key, value in distances.items():
            if key.lower() == fiducial_lower:
                return f"{value}"
        return ""

    def set_measurement_rows(self, rows: list[MeasurementRow]) -> None:
        """Replace the table contents from the given rows."""
        self._loading = True
        try:
            self._table.setRowCount(len(rows))
            for index, row in enumerate(rows):
                self._set_item(index, COL_ELECTRODE, row.electrode_id)
                for col, fiducial_id in enumerate(CANONICAL_FIDUCIALS, start=1):
                    self._set_item(
                        index, col, self._distance_text(row.measured_distances, fiducial_id)
                    )
        finally:
            self._loading = False
        self._dirty = True
        self._highlight_duplicates()
        self.rowsChanged.emit()

    def is_dirty(self) -> bool:
        """Whether the table holds edits not yet written to disk."""
        return self._dirty

    def measurement_rows(self) -> list[MeasurementRow]:
        """Parse the table into rows, raising :class:`ValueError` on bad input."""
        rows: list[MeasurementRow] = []
        for row_index in range(self._table.rowCount()):
            electrode_id = self._text(row_index, COL_ELECTRODE).strip()
            distances: dict[str, float] = {}
            for col, fiducial_id in enumerate(CANONICAL_FIDUCIALS, start=1):
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
            elif electrode_id:
                raise ValueError(
                    f"Electrode row {row_index + 1}: {electrode_id!r} has no measured distances"
                )
        return rows

    def add_row(self) -> None:
        index = self._table.rowCount()
        self._table.insertRow(index)
        self._table.scrollToBottom()
        self._table.setCurrentCell(index, COL_ELECTRODE)
        if not self._loading:
            self._dirty = True
            self.rowsChanged.emit()

    def remove_selected(self) -> None:
        row = self._table.currentRow()
        if row >= 0:
            self._table.removeRow(row)
            if not self._loading:
                self._dirty = True
                self._highlight_duplicates()
                self.rowsChanged.emit()

    def clear_all(self) -> None:
        """Remove all electrode rows after an explicit confirmation."""
        if self._table.rowCount() == 0:
            return
        answer = QMessageBox.question(
            self,
            "Clear measurements",
            "Remove all measurement rows?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.set_measurement_rows([])

    def _highlight_duplicates(self) -> None:
        """Mark electrode cells sharing a non-empty id with a red background."""
        counts: dict[str, int] = {}
        for row in range(self._table.rowCount()):
            electrode_id = self._text(row, COL_ELECTRODE).strip()
            if electrode_id:
                counts[electrode_id] = counts.get(electrode_id, 0) + 1
        for row in range(self._table.rowCount()):
            item = self._table.item(row, COL_ELECTRODE)
            if item is None:
                continue
            electrode_id = self._text(row, COL_ELECTRODE).strip()
            if electrode_id and counts.get(electrode_id, 0) > 1:
                item.setBackground(QBrush(QColor(150, 40, 40)))
                item.setToolTip("Duplicate electrode id")
            else:
                item.setBackground(QBrush())
                item.setToolTip("")

    def clear(self) -> None:
        """Reset the table and forget the current file path."""
        self._path = None
        self._loading = True
        try:
            self._table.clear()
            self._table.setHorizontalHeaderLabels(MEASUREMENT_HEADERS)
            self._table.setRowCount(0)
        finally:
            self._loading = False
        self._dirty = False

    # ---- load / save ----

    def load(self, path: Path, *, interactive: bool = True) -> bool:
        """Load *path* into the table; reports nothing when *interactive* is False."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object.")
            rows = measurements_schema_to_rows(data)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            if interactive:
                QMessageBox.critical(
                    self, "Load measurements", f"Could not load measurements:\n{exc}"
                )
            return False
        self._path = path
        self.set_measurement_rows(rows)
        self._dirty = False
        return True

    def save_to(self, path: Path) -> bool:
        """Save the current table to *path*; returns False on validation failure."""
        try:
            schema = self.collected_schema()
        except ValueError as exc:
            QMessageBox.critical(self, "Save measurements", f"Invalid table:\n{exc}")
            return False
        if not _confirm_outside_project(self, path, "Save measurements"):
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
        self._dirty = False
        self.fileSaved.emit()
        return True

    def collected_schema(self) -> dict[str, Any]:
        """Return the measurements JSON object for the current table."""
        return measurements_rows_to_schema(self.measurement_rows())

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


class LocalizationPreview(QWidget):
    """Read-only preview of the live localization results.

    Shows each electrode's localized coordinates in the selected frame side by
    side with the measured fiducial distances, plus the residual error and
    flag.  The backing CSV export is Qt-free.  Double-clicking a row emits
    :attr:`electrodeActivated` so the host can focus the 3D view on it.
    """

    electrodeActivated = Signal(str)  # noqa: N815 - electrode_id
    exported = Signal()  # noqa: N815 - fired after a successful CSV export

    def __init__(
        self,
        parent: QWidget | None = None,
        default_dir: Callable[[], str | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._default_dir = default_dir
        self._electrodes: Electrodes | None = None
        self._cras_offset: np.ndarray | None = None
        self._affine: np.ndarray | None = None
        self._coord_frame: str = FRAME_SCANNER

        self._table = QTableWidget(0, len(LOCALIZATION_HEADERS), self)
        self._table.setHorizontalHeaderLabels(LOCALIZATION_HEADERS)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.cellDoubleClicked.connect(self._on_row_activated)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

        self._frame_row = QWidget(self)
        self._frame_layout = QHBoxLayout(self._frame_row)
        self._frame_layout.setContentsMargins(0, 0, 0, 0)
        self._frame_layout.setSpacing(6)
        self._frame_layout.addWidget(QLabel("Coordinates:", self))
        self._frame_combo = QComboBox(self._frame_row)
        for frame_id in (FRAME_SCANNER, FRAME_HEAD, FRAME_VOXEL):
            self._frame_combo.addItem(frame_label(frame_id), frame_id)
        self._frame_combo.currentIndexChanged.connect(self._on_frame_selected)
        self._frame_layout.addWidget(self._frame_combo)
        self._flagged_only_chk = QCheckBox("Flagged only", self._frame_row)
        self._flagged_only_chk.setToolTip("Show only electrodes over the residual threshold.")
        self._flagged_only_chk.toggled.connect(self._render)
        self._frame_layout.addWidget(self._flagged_only_chk)
        self._frame_layout.addStretch(1)

        self._hint = QLabel(
            "Load or generate a scalp mesh and fill in the NAS/LPA/RPA fiducials and at "
            "least one measurement row to see localized electrodes here.",
            self,
        )
        self._hint.setWordWrap(True)
        self._hint.setVisible(True)
        self._default_hint = self._hint.text()

        export_btn = QPushButton("Export CSV...", self)
        export_btn.setToolTip("Save the table below as a CSV file.")
        export_btn.clicked.connect(self._on_export_csv)
        self._export_btn = export_btn

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(export_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._hint)
        layout.addWidget(self._frame_row)
        layout.addWidget(self._table, 1)
        layout.addLayout(buttons)

    # ---- table helpers ----

    def _set_item(self, row: int, col: int, text: str) -> None:
        self._table.setItem(row, col, QTableWidgetItem(text))

    def _set_coordinate_row(self, row: int, start_col: int, coordinates: np.ndarray | None) -> None:
        for axis in range(3):
            text = f"{float(coordinates[axis]):.3f}" if coordinates is not None else ""
            self._set_item(row, start_col + axis, text)

    # ---- result population ----

    def set_result(
        self,
        electrodes: Electrodes | None,
        cras_offset: np.ndarray | None,
        affine: np.ndarray | None = None,
    ) -> None:
        """Replace the table with the given localization result."""
        self._electrodes = electrodes
        self._cras_offset = (
            np.asarray(cras_offset, dtype=np.float64) if cras_offset is not None else None
        )
        self._affine = np.asarray(affine, dtype=np.float64) if affine is not None else None
        self._update_frame_availability()
        self._render()

    def _on_frame_selected(self) -> None:
        frame = self._frame_combo.currentData()
        if isinstance(frame, str):
            self._coord_frame = frame
        self._render()

    def _update_frame_availability(self) -> None:
        """Grey out coordinate systems the loaded scene cannot express."""
        model = self._frame_combo.model()
        if isinstance(model, QStandardItemModel):
            for index in range(self._frame_combo.count()):
                frame = self._frame_combo.itemData(index)
                item = model.item(index)
                if item is not None:
                    item.setEnabled(
                        isinstance(frame, str)
                        and frame_available(frame, self._affine, self._cras_offset)
                    )
        if not frame_available(self._coord_frame, self._affine, self._cras_offset):
            self._coord_frame = FRAME_SCANNER
            self._frame_combo.setCurrentIndex(0)

    def _frame_coords(self, world_coords: np.ndarray | None) -> np.ndarray | None:
        if world_coords is None:
            return None
        try:
            matrix = world_to_frame_matrix(self._coord_frame, self._affine, self._cras_offset)
        except ValueError:
            return None
        points = transform_points(np.asarray([world_coords], dtype=np.float64), matrix)
        return np.asarray(points[0], dtype=np.float64)

    def _render(self) -> None:
        """Repopulate the table from the cached result in the selected frame."""
        electrodes = self._electrodes
        self._table.setRowCount(0)
        if electrodes is None or not electrodes.items:
            self._hint.setText(self._default_hint)
            self._hint.setVisible(True)
            self._export_btn.setEnabled(False)
            return
        flagged_only = self._flagged_only_chk.isChecked()
        items = [e for e in electrodes.items if not flagged_only or e.flagged]
        if not items:
            self._hint.setText("No flagged electrodes.")
            self._hint.setVisible(True)
            self._export_btn.setEnabled(False)
            return
        self._hint.setVisible(False)
        self._export_btn.setEnabled(True)
        self._table.setRowCount(len(items))
        for index, electrode in enumerate(items):
            self._set_item(index, LOC_COL_NAME, electrode.electrode_id or "")
            self._set_coordinate_row(
                index, LOC_COL_COORDS, self._frame_coords(electrode.ese_coords)
            )
            for col, fiducial_id in enumerate(CANONICAL_FIDUCIALS, start=LOC_COL_DISTANCES):
                distance = self._distance(electrode.measured_distances, fiducial_id)
                self._set_item(index, col, f"{distance}" if distance is not None else "")
            self._set_item(
                index,
                LOC_COL_RESIDUAL,
                (
                    f"{float(electrode.residual_error):.3f}"
                    if electrode.is_localized and electrode.residual_error is not None
                    else ""
                ),
            )
            self._set_item(index, LOC_COL_FLAGGED, "yes" if electrode.flagged else "no")

    def _on_row_activated(self, row: int, _col: int) -> None:
        """Emit the double-clicked row's electrode id for camera focus."""
        item = self._table.item(row, LOC_COL_NAME)
        if item is not None and item.text():
            self.electrodeActivated.emit(item.text())

    def set_blocked(self, reason: str) -> None:
        """Show *reason* instead of the table until the blocker is resolved."""
        self._electrodes = None
        self._table.setRowCount(0)
        self._hint.setText(reason)
        self._hint.setVisible(True)
        self._export_btn.setEnabled(False)

    def _distance(self, distances: dict[str, float], fiducial_id: str) -> float | None:
        fiducial_lower = fiducial_id.lower()
        for key, value in distances.items():
            if key.lower() == fiducial_lower:
                return value
        return None

    # ---- export ----

    def _on_export_csv(self) -> None:
        path, _selected_filter = QFileDialog.getSaveFileName(
            self, "Export localization table", self._start_path(), "CSV (*.csv);;All files (*)"
        )
        if not path:
            return
        if self._electrodes is None:
            return
        try:
            export_localization_table(Path(path), self._electrodes, self._cras_offset)
        except OSError as exc:
            QMessageBox.critical(self, "Export localization", f"Could not write file:\n{exc}")
            return
        self.exported.emit()

    def _start_path(self) -> str:
        """Default location for the CSV export dialog."""
        default_dir = self._default_dir() if self._default_dir else None
        filename = DEFAULT_MEASUREMENTS_FILENAME.replace("measurements.json", "localization.csv")
        if default_dir:
            return str(Path(default_dir) / "input" / filename)
        return filename


class EditorsTab(QWidget):
    """Fiducials, measurements and localization preview stacked vertically.

    Measurements are unlocked once the canonical NAS/LPA/RPA fiducials exist;
    the read-only localization preview is fed live by the main window as
    localization runs.
    """

    advancedRequested = Signal()  # noqa: N815
    localizeRequested = Signal()  # noqa: N815
    filesSaved = Signal()  # noqa: N815 - any editor wrote a file to disk

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._fiducials = FiducialsEditor(self, default_dir=lambda: self._default_dir())
        self._measurements = MeasurementsEditor(self, default_dir=lambda: self._default_dir())
        self._localization = LocalizationPreview(self, default_dir=lambda: self._default_dir())
        self._fiducials.rowsChanged.connect(self._on_fiducials_changed)
        self._fiducials.set_input_frame(self._state.fiducial_frame)
        self._fiducials.inputFrameChanged.connect(self._on_input_frame_changed)
        self._fiducials.fileSaved.connect(self.filesSaved.emit)
        self._measurements.fileSaved.connect(self.filesSaved.emit)
        self._localization.exported.connect(self.filesSaved.emit)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.addWidget(self._fiducials)
        splitter.addWidget(self._measurements)
        splitter.addWidget(self._localization)

        advanced_btn = QPushButton("Advanced settings...", self)
        advanced_btn.setToolTip("Open the advanced mesh-generation and localization settings.")
        advanced_btn.clicked.connect(self._on_advanced_clicked)

        localize_btn = QPushButton("Localize now", self)
        localize_btn.setToolTip("Run localization once with the current tables.")
        localize_btn.clicked.connect(self._on_localize_clicked)

        buttons = QHBoxLayout()
        buttons.addWidget(advanced_btn)
        buttons.addWidget(localize_btn)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(splitter)
        layout.addLayout(buttons)

    @property
    def fiducials(self) -> FiducialsEditor:
        """The live fiducials editor widget."""
        return self._fiducials

    @property
    def measurements(self) -> MeasurementsEditor:
        """The live measurements editor widget."""
        return self._measurements

    @property
    def localization(self) -> LocalizationPreview:
        """The read-only live localization preview widget."""
        return self._localization

    def _default_dir(self) -> str | None:
        return self._state.last_project_dir

    def _canonical_fiducials_ready(self) -> bool:
        """Whether the NAS/LPA/RPA fiducials are all present (canonical ids)."""
        ids = {fiducial_id.lower() for fiducial_id in self._fiducials.fiducial_ids()}
        return {canonical.lower() for canonical in CANONICAL_FIDUCIALS}.issubset(ids)

    def _on_fiducials_changed(self) -> None:
        ready = self._canonical_fiducials_ready()
        self._measurements.set_fiducial_ready(ready)
        if not ready:
            self._localization.set_result(None, None)

    def _on_input_frame_changed(self, _old_frame: object, new_frame: object) -> None:
        """Persist the chosen fiducials input frame in the shared state."""
        if isinstance(new_frame, str):
            self._state.fiducial_frame = new_frame

    def _on_advanced_clicked(self) -> None:
        self.advancedRequested.emit()

    def _on_localize_clicked(self) -> None:
        self.localizeRequested.emit()

    def prefill_from_project(self, project: str | Path) -> None:
        """Load the project's canonical fiducials and measurements, if any."""
        self.clear()
        root = Path(project)
        fiducials = root / "input" / DEFAULT_FIDUCIALS_FILENAME
        if fiducials.is_file():
            self._fiducials.load(fiducials, interactive=False)
        measurements = root / "input" / DEFAULT_MEASUREMENTS_FILENAME
        if measurements.is_file():
            self._measurements.load(measurements, interactive=False)

    def clear(self) -> None:
        """Reset all three editors so no stale rows survive a project close."""
        self._measurements.clear()
        self._fiducials.clear()
        self._localization.set_result(None, None)

    def is_dirty(self) -> bool:
        """Whether either table holds edits not yet written to disk."""
        return self._fiducials.is_dirty() or self._measurements.is_dirty()
