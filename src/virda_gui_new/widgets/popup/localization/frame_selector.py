"""Coordinate-frame selector combo for the localization popup tables."""

from PySide6.QtCore import Signal
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import QComboBox, QWidget

from virda_gui_new.services.coordinates import CoordinateFrame

_ITEMS = (
    ("Head", CoordinateFrame.HEAD),
    ("World", CoordinateFrame.WORLD),
    ("Fiducials", CoordinateFrame.FIDUCIALS),
)


class FrameSelector(QComboBox):
    """Combo of the three display frames; unavailable ones are greyed out.

    Head is the default selection (item order: head, world, fiducials).
    :meth:`set_frame_enabled` enables/disables a frame's item (e.g. the
    fiducials frame is unavailable while fewer than three fiducials have
    coordinates); picking an enabled frame emits :attr:`frameChanged`
    with the :class:`CoordinateFrame`, which the popup forwards to the
    table it controls. Changing availability never changes the selection
    by itself -- the popup decides on a fallback.
    """

    frameChanged = Signal(CoordinateFrame)  # noqa: N815 - Qt signal API

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        for label, frame in _ITEMS:
            self.addItem(label, frame)
        self.currentIndexChanged.connect(self._on_index_changed)

    def frame(self) -> CoordinateFrame:
        """The currently selected frame."""
        current = self.currentData()
        return current if isinstance(current, CoordinateFrame) else CoordinateFrame.WORLD

    def set_frame_enabled(self, frame: CoordinateFrame, enabled: bool) -> None:
        """Grey out (or re-enable) the item of ``frame``."""
        index = self.findData(frame)
        model = self.model()
        item = model.item(index) if isinstance(model, QStandardItemModel) else None
        if item is not None:
            item.setEnabled(enabled)

    def _on_index_changed(self, _index: int) -> None:
        self.frameChanged.emit(self.frame())
