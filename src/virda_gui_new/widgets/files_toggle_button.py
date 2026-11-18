"""Floating pill button that toggles the file tree window.

The button is an overlay child of its parent window, anchored below the
steps pill: it keeps itself pinned through a resize event filter and
only emits ``toggleRequested`` - the show/hide decision stays with the
embedder (MainWindow).
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget

from virda_gui_new.widgets.overlay_pill import OverlayPill


class FilesToggleButton(OverlayPill):
    """High-contrast pill overlay that requests a file-tree toggle.

    Clicking emits :attr:`toggleRequested`; the button never touches the
    file tree window itself, so it keeps working across the open/close
    cycles of that window.
    """

    toggleRequested = Signal()  # noqa: N815

    def __init__(self, parent: QWidget | None = None, *, below: OverlayPill | None = None) -> None:
        super().__init__("Files", parent, below=below)
        self.clicked.connect(self.toggleRequested.emit)
