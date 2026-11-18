"""Read-only electrode result grid for the localization popup."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from virda.geometry import Geometry
from virda.models.coordinates import CoordinateSystem
from virda.models.electrode import Electrodes

_COLUMNS = (
    "Electrode",
    "ESE X",
    "ESE Y",
    "ESE Z",
    "Scalp X",
    "Scalp Y",
    "Scalp Z",
    "Residual (mm)",
    "Flagged",
)


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def _coord_text(coords: np.ndarray | None, axis: int, system: CoordinateSystem) -> str:
    """One coordinate cell: integer voxel indices, millimetres otherwise."""
    if coords is None or coords.shape != (3,):
        return ""
    if system is CoordinateSystem.VOXEL:
        return f"{coords[axis]:.0f}"
    return f"{coords[axis]:.2f}"


class ElectrodesTable(QWidget):
    """Read-only grid of the localized electrodes.

    :meth:`show_electrodes` re-renders the grid from an
    :class:`Electrodes` model: coordinates and residuals at two decimals
    (missing values blank), the flag as ``yes``/``no``. The coordinates
    are rendered in the current display frame while the model keeps its
    world-canonical coordinates; the voxel frame shows integer indices,
    the other frames millimetres at two decimals. Residuals are distances
    and do not depend on the frame.
    """

    def __init__(self, converter: Geometry | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._converter = converter
        self._frame = CoordinateSystem.WORLD
        self._electrodes: Electrodes | None = None
        self._table = QTableWidget(0, len(_COLUMNS), self)
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout = QVBoxLayout(self)
        layout.addWidget(self._table)

    def show_electrodes(self, electrodes: Electrodes) -> None:
        """Render one row per electrode from the model."""
        self._electrodes = electrodes
        self._render()

    def set_display_frame(self, frame: CoordinateSystem) -> None:
        """Switch the displayed frame; the stored model stays world."""
        if frame is self._frame:
            return
        if frame is not CoordinateSystem.WORLD and (
            self._converter is None or not self._converter.is_available(frame)
        ):
            if CoordinateSystem.WORLD is self._frame:
                return
            frame = CoordinateSystem.WORLD
        self._frame = frame
        if self._electrodes is not None:
            self._render()

    def _display(self, coords: np.ndarray | None) -> np.ndarray | None:
        """Canonical world point -> current display frame point."""
        if coords is None or self._frame is CoordinateSystem.WORLD or self._converter is None:
            return coords
        return np.asarray(self._converter.from_world(coords, self._frame), dtype=np.float64)

    def _render(self) -> None:
        """Fill the cells from the stored model (no editing)."""
        electrodes = self._electrodes
        self._table.setRowCount(0 if electrodes is None else len(electrodes.items))
        if electrodes is None:
            return
        for row, electrode in enumerate(electrodes.items):
            ese = self._display(electrode.ese_coords)
            scalp = self._display(electrode.scalp_coords)
            texts = (
                electrode.electrode_id or "",
                _coord_text(ese, 0, self._frame),
                _coord_text(ese, 1, self._frame),
                _coord_text(ese, 2, self._frame),
                _coord_text(scalp, 0, self._frame),
                _coord_text(scalp, 1, self._frame),
                _coord_text(scalp, 2, self._frame),
                _fmt(electrode.residual_error),
                "yes" if electrode.flagged else "no",
            )
            for column, text in enumerate(texts):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(row, column, item)
