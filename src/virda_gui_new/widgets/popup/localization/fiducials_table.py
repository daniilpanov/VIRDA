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

Without a :class:`~virda.geometry.Geometry` service the table only
supports the ``world`` display frame; any other frame falls back to
``world`` rendering.
"""

from collections.abc import Mapping, Sequence

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

from virda.geometry import Geometry, match_fiducial_name
from virda.models.coordinates import CoordinateSystem
from virda.models.fiducial import SLOT_IDS, FiducialType
from virda_gui_new.services.localization import FiducialRows, SurfaceStatus

#: The status column (leftmost), the name, the coordinates and the picker.
_COLUMNS = ("", "Name", "X", "Y", "Z", "Pick")

#: The column of the "Pick" button.
_PICK_COLUMN = len(_COLUMNS) - 1

#: The dot that marks a fiducial off the scalp surface.
_OFF_SURFACE_DOT = "\u25cf"


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
    pickRequested = Signal(str)  # noqa: N815 - Qt signal API

    def __init__(self, converter: Geometry | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._converter = converter
        self._frame = CoordinateSystem.WORLD
        self._names = [str(slot) for slot in SLOT_IDS]
        self._world: list[np.ndarray | None] = [None] * len(SLOT_IDS)
        self._status: dict[str, SurfaceStatus] = {}
        self._pick_target: FiducialType | None = None
        self._pick_buttons: dict[int, QPushButton] = {}
        self._pick_available = True
        self._pick_blocked = False

        self._table = QTableWidget(len(SLOT_IDS), len(_COLUMNS), self)
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.itemChanged.connect(self._on_item_changed)
        for row, _slot in enumerate(SLOT_IDS):
            button = QPushButton("Pick", self._table)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, r=row: self._on_pick_clicked(r))
            self._pick_buttons[row] = button

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
        self._names = [str(slot) for slot in SLOT_IDS]
        self._world = [None] * len(SLOT_IDS)
        for name, xyz in rows:
            slot = self._slot(name)
            if slot is not None:
                self._names[slot] = str(name).strip()
                self._world[slot] = np.asarray(xyz, dtype=np.float64)
        self._render()
        self.rowsChanged.emit()

    def clear_grid(self) -> None:
        """Blank the whole grid: canonical names, no coordinates."""
        self._names = [str(slot) for slot in SLOT_IDS]
        self._world = [None] * len(SLOT_IDS)
        self._render()
        self.cleared.emit()
        self.rowsChanged.emit()

    def set_display_frame(self, frame: CoordinateSystem) -> None:
        """Switch the displayed frame; the canonical rows stay world.

        Unavailable frames (no converter, or the converter cannot serve
        the frame) fall back to ``world``. A genuine frame change
        re-renders the grid, so incomplete rows are cleared; switching
        to the already-displayed frame is a no-op.
        """
        if frame is self._frame:
            return
        if frame is not CoordinateSystem.WORLD and (
            self._converter is None or not self._converter.is_available(frame)
        ):
            if CoordinateSystem.WORLD is self._frame:
                return
            frame = CoordinateSystem.WORLD
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
        return SLOT_IDS.index(slot) if slot is not None else None

    def _display(self, xyz: np.ndarray) -> np.ndarray:
        """Canonical world point -> current display frame point."""
        if self._frame is CoordinateSystem.WORLD or self._converter is None:
            return xyz
        return np.asarray(self._converter.from_world(xyz, self._frame), dtype=np.float64)

    def _canonicalize(self, displayed: np.ndarray) -> np.ndarray:
        """Displayed point -> canonical world coordinates."""
        if self._frame is CoordinateSystem.WORLD or self._converter is None:
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
        for column in (2, 3, 4):
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
            for column, text in enumerate(cells, start=1):
                item = QTableWidgetItem(text)
                if column == 1:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(row, column, item)
            self._render_status(row, name)
            table.setCellWidget(row, _PICK_COLUMN, self._pick_buttons[row])
        table.blockSignals(False)

    def _on_pick_clicked(self, row: int) -> None:
        """Ask for a point on the surface to be picked for this row."""
        self.pickRequested.emit(SLOT_IDS[row].value)

    def set_pick_target(self, slot: FiducialType | None) -> None:
        """Mark the row whose point is being picked."""
        self._pick_target = slot
        for row, button in self._pick_buttons.items():
            button.setChecked(SLOT_IDS[row] is slot)

    def set_pick_available(self, available: bool) -> None:
        """Whether a point can be picked at all (the scalp mesh is needed)."""
        self._pick_available = available
        self._refresh_pick_buttons()

    def set_pick_blocked(self, blocked: bool) -> None:
        """Block picking while a localization run makes a pick stale.

        The block is independent of the availability: a button is active
        only when picking is available and not blocked, so unblocking
        restores exactly the buttons that were available before.
        """
        self._pick_blocked = blocked
        self._refresh_pick_buttons()

    def _refresh_pick_buttons(self) -> None:
        """Apply the availability and the busy block to every Pick button."""
        for button in self._pick_buttons.values():
            button.setEnabled(self._pick_available and not self._pick_blocked)

    def set_pick_point(self, name: str, xyz: np.ndarray) -> bool:
        """Write a picked point into the row of the given fiducial.

        The point is in world millimetres, like the canonical rows.
        Returns whether the slot was found.
        """
        slot = match_fiducial_name(name)
        if slot is None:
            return False
        row = SLOT_IDS.index(slot)
        self._world[row] = np.asarray(xyz, dtype=np.float64)
        self._render()
        self.rowsChanged.emit()
        return True

    def _render_status(self, row: int, name: str) -> None:
        """Fill the status cell of one row: the dot and its tooltip."""
        status = self._status.get(name)
        item = QTableWidgetItem()
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        if status is None:
            item.setToolTip("")
            self._table.setItem(row, 0, item)
            return
        if not status.on_surface:
            item.setText(_OFF_SURFACE_DOT)
            item.setForeground(Qt.GlobalColor.red)
        item.setToolTip(f"{name} is {status.distance_mm:.1f} mm from the scalp surface")
        self._table.setItem(row, 0, item)

    def set_surface_status(self, status: Mapping[str, SurfaceStatus]) -> None:
        """Show how far every fiducial is from the scalp surface.

        A fiducial that is not on the surface gets a red dot; the
        tooltip of the status cell always carries the distance. The
        status is the popup's own display: it changes no row, so the
        signals of the table stay silent.
        """
        self._status = dict(status)
        table = self._table
        table.blockSignals(True)
        for row, name in enumerate(self._names):
            self._render_status(row, name)
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
