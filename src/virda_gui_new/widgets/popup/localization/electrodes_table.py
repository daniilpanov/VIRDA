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

from virda.models.electrode import Electrodes
from virda_gui_new.services.coordinates import CoordinateFrame, FrameConverter

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


def _coord_text(coords: np.ndarray | None, axis: int) -> str:
    return "" if coords is None or coords.shape != (3,) else f"{coords[axis]:.2f}"


class ElectrodesTable(QWidget):
    """Display of the localized electrodes; the popup keeps it read-only.

    :meth:`show_electrodes` re-renders the whole grid from an
    :class:`Electrodes` model: coordinates and residuals at two decimals
    (missing values blank), the flag as ``yes``/``no``. The table takes
    no editing input.

    The displayed coordinates are rendered in the current display frame
    (default ``world``; the popup selects ``head`` when available) while
    the stored model keeps its world-canonical coordinates --
    :meth:`set_display_frame` only re-renders. Residuals are distances,
    so they do not depend on the frame. Unavailable frames (no
    converter, or the converter cannot serve the frame) fall back to
    ``world`` rendering.
    """

    def __init__(
        self, converter: FrameConverter | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._converter = converter
        self._frame = CoordinateFrame.WORLD
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

    def set_display_frame(self, frame: CoordinateFrame) -> None:
        """Switch the displayed frame; the stored model stays world."""
        if frame is self._frame:
            return
        if frame is not CoordinateFrame.WORLD and (
            self._converter is None or not self._converter.is_available(frame)
        ):
            if CoordinateFrame.WORLD is self._frame:
                return
            frame = CoordinateFrame.WORLD
        self._frame = frame
        if self._electrodes is not None:
            self._render()

    def _display(self, coords: np.ndarray | None) -> np.ndarray | None:
        """Canonical world point -> current display frame point."""
        if coords is None or self._frame is CoordinateFrame.WORLD or self._converter is None:
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
                _coord_text(ese, 0),
                _coord_text(ese, 1),
                _coord_text(ese, 2),
                _coord_text(scalp, 0),
                _coord_text(scalp, 1),
                _coord_text(scalp, 2),
                _fmt(electrode.residual_error),
                "yes" if electrode.flagged else "no",
            )
            for column, text in enumerate(texts):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(row, column, item)
