"""Fixed three-row fiducial input grid for the localization popup.

A fiducial set is always exactly the three anatomical points NAS, LPA
and RPA (the canonical names the measurements files use), so the grid
is a fixed 3-row table instead of a free row list: the name cells are
read-only, only the XYZ cells are editable, and there are no add/remove
row buttons. An "Empty" button clears the whole grid (all three rows
become blank).

Row naming: each slot keeps a fixed canonical name (``NAS``, ``LPA``,
``RPA``) used for matching and for blank/manual rows. When rows are
prefilled from an artifact, the artifact's own label is displayed and
collected (e.g. ``NASION`` for the NAS slot), so the existing naming
convention survives a save/load round trip unchanged.

Internally the table keeps canonical world-coordinate rows (one
``(3,)`` array or ``None`` per slot); the cells are a view of that
state rendered in the current display frame (default ``world``; the
popup selects ``head`` when available). Switching the display frame
(:meth:`set_display_frame`) re-renders the set rows in the new frame
and clears rows without valid coordinates (their values cannot be
converted cell-wise); edits are read in the display frame and stored
back as world, so :meth:`collect_fiducial_rows` always returns world
coordinates in the fixed NAS/LPA/RPA order, skipping empty or
incomplete rows -- a partially filled table still prefills and saves
exactly like before. Editing one coordinate cell re-derives the whole
row from the displayed texts, so displayed numbers are the truth for
edited rows.

Buttons: an "Empty" button that clears the whole grid (emitting
:attr:`cleared` so the popup can sync the global store and the 3D
scene) and a "From file" button that only emits :attr:`importRequested`
-- the popup owns the file dialog and the import service.

Without a :class:`~virda_gui_new.services.coordinates.FrameConverter`
the table only supports the ``world`` display frame; any other frame
falls back to ``world`` rendering.
"""

from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from virda_gui_new.services.coordinates import CoordinateFrame, FrameConverter, match_fiducial_name
from virda_gui_new.services.localization import FiducialRows

_COLUMNS = ("Name", "X", "Y", "Z")
_ROW_IDS = ("NAS", "LPA", "RPA")


class FiducialsTable(QWidget):
    """Fixed NAS/LPA/RPA entry grid with world-coordinate rows.

    :meth:`set_rows` fills the slots whose names match the canonical
    row ids case-insensitively (via the shared alias table); unmatched
    names are ignored and unmatched slots are cleared. Any coordinate
    edit, refill or clear emits :attr:`rowsChanged`; :meth:`clear_grid`
    additionally emits :attr:`cleared` so the popup can sync the global
    store and the 3D scene. :meth:`collect_fiducial_rows` returns only
    complete rows (all three coordinates parsed as floats), in the
    fixed NAS/LPA/RPA order.
    """

    rowsChanged = Signal()  # noqa: N815 - Qt signal API
    cleared = Signal()  # noqa: N815 - Qt signal API
    importRequested = Signal()  # noqa: N815 - Qt signal API

    def __init__(
        self, converter: FrameConverter | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._converter = converter
        self._frame = CoordinateFrame.WORLD
        self._names = list(_ROW_IDS)
        self._world: list[np.ndarray | None] = [None] * len(_ROW_IDS)

        self._table = QTableWidget(len(_ROW_IDS), len(_COLUMNS), self)
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.itemChanged.connect(self._on_item_changed)

        from_file = QPushButton("From file", self)
        from_file.clicked.connect(self.importRequested)
        empty = QPushButton("Empty", self)
        empty.clicked.connect(self.clear_grid)
        buttons = QHBoxLayout()
        buttons.addWidget(from_file)
        buttons.addWidget(empty)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self._table)
        layout.addLayout(buttons)
        self._render()

    def set_rows(self, rows: Sequence[tuple[str, np.ndarray | Sequence[float]]]) -> None:
        """Replace the grid from ``(name, xyz)`` rows; one change signal.

        Rows are matched onto the fixed slots by name
        (case-insensitive alias match); a matched slot shows the given
        label and coordinates, unmatched slots are cleared.
        """
        self._names = list(_ROW_IDS)
        self._world = [None] * len(_ROW_IDS)
        for name, xyz in rows:
            slot = self._slot(name)
            if slot is not None:
                self._names[slot] = str(name).strip()
                self._world[slot] = np.asarray(xyz, dtype=np.float64)
        self._render()
        self.rowsChanged.emit()

    def clear_grid(self) -> None:
        """Blank the whole grid: canonical names, no coordinates."""
        self._names = list(_ROW_IDS)
        self._world = [None] * len(_ROW_IDS)
        self._render()
        self.cleared.emit()
        self.rowsChanged.emit()

    def set_display_frame(self, frame: CoordinateFrame) -> None:
        """Switch the displayed frame; the canonical rows stay world.

        Unavailable frames (no converter, or the converter cannot serve
        the frame) fall back to ``world``. A genuine frame change
        re-renders the grid, so incomplete rows are cleared; switching
        to the already-displayed frame is a no-op.
        """
        if frame is self._frame:
            return
        if frame is not CoordinateFrame.WORLD and (
            self._converter is None or not self._converter.is_available(frame)
        ):
            if CoordinateFrame.WORLD is self._frame:
                return
            frame = CoordinateFrame.WORLD
        self._frame = frame
        self._render()

    def collect_fiducial_rows(self) -> FiducialRows:
        """Return the complete ``(name, (3,) xyz)`` rows in fixed order."""
        rows: FiducialRows = []
        for name, xyz in zip(self._names, self._world, strict=True):
            if xyz is not None:
                rows.append((name, xyz))
        return rows

    def _slot(self, name: str) -> int | None:
        """Row index of the slot ``name`` matches, or ``None``."""
        slot = match_fiducial_name(name)
        return _ROW_IDS.index(slot) if slot is not None else None

    def _display(self, xyz: np.ndarray) -> np.ndarray:
        """Canonical world point -> current display frame point."""
        if self._frame is CoordinateFrame.WORLD or self._converter is None:
            return xyz
        return np.asarray(self._converter.from_world(xyz, self._frame), dtype=np.float64)

    def _canonicalize(self, displayed: np.ndarray) -> np.ndarray:
        """Displayed point -> canonical world coordinates."""
        if self._frame is CoordinateFrame.WORLD or self._converter is None:
            return displayed
        return np.asarray(self._converter.to_world(displayed, self._frame), dtype=np.float64)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Re-derive the edited row's canonical coordinates from its cells."""
        row = item.row()
        displayed = self._row_xyz(row)
        xyz = None if displayed is None else self._canonicalize(displayed)
        current = self._world[row]
        changed = (current is None) != (xyz is None) or (
            xyz is not None and current is not None and not np.array_equal(xyz, current)
        )
        if changed:
            self._world[row] = xyz
            self.rowsChanged.emit()

    def _row_xyz(self, row: int) -> np.ndarray | None:
        """The row's displayed coordinates as ``(3,)``, or ``None``."""
        values = []
        for column in (1, 2, 3):
            try:
                values.append(float(self._text(row, column)))
            except ValueError:
                return None
        return np.array(values)

    def _render(self) -> None:
        """Fill the cells from the canonical state (no change signals)."""
        table = self._table
        table.blockSignals(True)
        for row, (name, xyz) in enumerate(zip(self._names, self._world, strict=True)):
            cells = (name, *self._xyz_texts(None if xyz is None else self._display(xyz)))
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column == 0:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(row, column, item)
        table.blockSignals(False)

    @staticmethod
    def _xyz_texts(xyz: np.ndarray | None) -> tuple[str, str, str]:
        """Displayed coordinate texts of one row (blank when unset)."""
        if xyz is None:
            return ("", "", "")
        return (f"{xyz[0]:g}", f"{xyz[1]:g}", f"{xyz[2]:g}")

    def _text(self, row: int, column: int) -> str:
        item = self._table.item(row, column)
        return "" if item is None else item.text().strip()
