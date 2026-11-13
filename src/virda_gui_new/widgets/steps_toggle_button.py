"""Floating pill button that toggles the steps menu window.

The button is an overlay child of its parent window: it keeps itself pinned
to a corner through a resize event filter and only emits ``toggleRequested``
- the show/hide decision stays with the embedder (MainWindow).
"""

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import QPushButton, QWidget

_STYLE = """
QPushButton {
    background-color: #1e1e1e;
    color: #f5f5f5;
    border: 1px solid #666666;
    border-radius: 14px;
    padding: 4px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #3c3c3c;
}
"""


class StepsToggleButton(QPushButton):
    """High-contrast pill overlay that requests a steps-menu toggle.

    Clicking emits :attr:`toggleRequested`; it never touches the menu
    window itself, so the button keeps working across step changes and
    popup open/close cycles.
    """

    toggleRequested = Signal()  # noqa: N815

    _MARGIN = 10

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Steps", parent)
        self.setStyleSheet(_STYLE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(self.toggleRequested.emit)
        if parent is not None:
            parent.installEventFilter(self)
            self.reposition()

    def reposition(self) -> None:
        """Pin the pill to the top-left corner of the parent window."""
        self.adjustSize()
        self.move(self._MARGIN, self._MARGIN)
        self.raise_()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt override
        """Follow the parent window's resizes to keep the overlay pinned."""
        if obj is self.parent() and event.type() == QEvent.Type.Resize:
            self.reposition()
        return super().eventFilter(obj, event)
