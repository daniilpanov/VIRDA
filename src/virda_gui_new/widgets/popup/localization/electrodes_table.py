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
    :class:`Electrodes` model: coordinates and residuals at two
    decimals (missing values blank), the flag as ``yes``/``no``. The
    table takes no editing input.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._table = QTableWidget(0, len(_COLUMNS), self)
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout = QVBoxLayout(self)
        layout.addWidget(self._table)

    def show_electrodes(self, electrodes: Electrodes) -> None:
        """Render one row per electrode from the model."""
        self._table.setRowCount(len(electrodes.items))
        for row, electrode in enumerate(electrodes.items):
            texts = (
                electrode.electrode_id or "",
                _coord_text(electrode.ese_coords, 0),
                _coord_text(electrode.ese_coords, 1),
                _coord_text(electrode.ese_coords, 2),
                _coord_text(electrode.scalp_coords, 0),
                _coord_text(electrode.scalp_coords, 1),
                _coord_text(electrode.scalp_coords, 2),
                _fmt(electrode.residual_error),
                "yes" if electrode.flagged else "no",
            )
            for column, text in enumerate(texts):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(row, column, item)
