"""Coordinate-frame selector combo for the localization popup tables."""

from collections.abc import Sequence

from PySide6.QtCore import Signal
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import QComboBox, QWidget

from virda.models.coordinates import CoordinateSystem

_ITEMS = (
    ("Head", CoordinateSystem.HEAD),
    ("World", CoordinateSystem.WORLD),
    ("Fiducials", CoordinateSystem.FIDUCIALS),
    ("Voxel (i, j, k)", CoordinateSystem.VOXEL),
)


class FrameSelector(QComboBox):
    """Combo of the display frames; unavailable ones are greyed out.

    Head is the default selection (item order: head, world, fiducials,
    voxel). ``frames`` limits the frames the combo offers: the fiducials
    table omits the fiducials frame because that frame is defined by the
    very rows the table shows.
    :meth:`set_frame_enabled` enables/disables a frame's item; picking
    an enabled frame emits :attr:`frameChanged` with the
    :class:`CoordinateSystem`. Changing availability never changes the
    selection by itself.
    """

    frameChanged = Signal(CoordinateSystem)  # noqa: N815 - Qt signal API

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        frames: Sequence[CoordinateSystem] | None = None,
    ) -> None:
        super().__init__(parent)
        for label, frame in _ITEMS:
            if frames is None or frame in frames:
                self.addItem(label, frame)
        self.currentIndexChanged.connect(self._on_index_changed)

    def frame(self) -> CoordinateSystem:
        """The currently selected frame."""
        current = self.currentData()
        return current if isinstance(current, CoordinateSystem) else CoordinateSystem.WORLD

    def set_frame_enabled(self, frame: CoordinateSystem, enabled: bool) -> None:
        """Grey out (or re-enable) the item of ``frame``."""
        index = self.findData(frame)
        if index < 0:
            return
        model = self.model()
        item = model.item(index) if isinstance(model, QStandardItemModel) else None
        if item is not None:
            item.setEnabled(enabled)

    def _on_index_changed(self, _index: int) -> None:
        self.frameChanged.emit(self.frame())
