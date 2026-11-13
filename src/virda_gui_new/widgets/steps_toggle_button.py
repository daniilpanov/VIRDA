"""Floating pill button that toggles the steps menu window.

The button is an overlay child of its parent window: it keeps itself
pinned below the menu bar through a resize event filter and only emits
``toggleRequested`` - the show/hide decision stays with the embedder
(MainWindow).
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget

from virda_gui_new.widgets.overlay_pill import OverlayPill


class StepsToggleButton(OverlayPill):
    """High-contrast pill overlay that requests a steps-menu toggle.

    Clicking emits :attr:`toggleRequested`; it never touches the menu
    window itself, so the button keeps working across step changes and
    popup open/close cycles.
    """

    toggleRequested = Signal()  # noqa: N815

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Steps", parent)
        self.clicked.connect(self.toggleRequested.emit)
