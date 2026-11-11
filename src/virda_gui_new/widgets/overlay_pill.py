"""Base class for the high-contrast pill overlays of the main window.

A pill is a plain child of its parent window (it is not an entry of a
layout) that stays pinned to the top-left corner of the viewer's content
area. The vertical anchor is the bottom edge of the parent's menu bar, so
a pill never covers the menu bar of a ``QMainWindow`` parent; a pill may
also be anchored below another pill, which keeps a stack of pills in a
single column.

The pill only draws itself and reports its own geometry: what a click
means (which window to toggle) stays with the embedder (MainWindow).
"""

from contextlib import suppress

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import QMenuBar, QPushButton, QWidget

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

#: Parent events that can move the content area (and with it the pill).
_PARENT_EVENTS = frozenset(
    {
        QEvent.Type.Resize,
        QEvent.Type.Show,
        QEvent.Type.LayoutRequest,
    }
)

#: Menu bar events that change the anchor of the pill.
_MENU_BAR_EVENTS = frozenset(
    {
        QEvent.Type.Resize,
        QEvent.Type.Show,
        QEvent.Type.Hide,
        QEvent.Type.Move,
    }
)


class OverlayPill(QPushButton):
    """High-contrast pill overlay pinned below the parent's menu bar.

    The pill keeps itself in place through an event filter on its parent
    window, so it follows the window while it stays clear of the menu
    bar. Subclasses add their own toggle signal.
    """

    #: Emitted after the pill was moved to its anchor.
    repositioned = Signal()  # noqa: N815

    _MARGIN = 10
    _GAP = 6

    def __init__(
        self,
        text: str,
        parent: QWidget | None = None,
        *,
        below: OverlayPill | None = None,
    ) -> None:
        super().__init__(text, parent)
        self.setStyleSheet(_STYLE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._below = below
        self._menu_bar: QMenuBar | None = None
        if below is not None:
            # Follow the anchor pill: it knows the menu bar offset.
            below.repositioned.connect(self.reposition)
        if parent is not None:
            parent.installEventFilter(self)
            self.reposition()

    def reposition(self) -> None:
        """Pin the pill to the top-left of the parent's content area."""
        self._sync_menu_bar_filter()
        self.adjustSize()
        self.move(self._MARGIN, self._top())
        self.raise_()
        self.repositioned.emit()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt override
        """Follow the parent window's resizes and the menu bar's changes."""
        event_type = event.type()
        parent_moved = obj is self.parent() and event_type in _PARENT_EVENTS
        menu_bar_moved = obj is self._menu_bar and event_type in _MENU_BAR_EVENTS
        if parent_moved or menu_bar_moved:
            self.reposition()
        return super().eventFilter(obj, event)

    def _top(self) -> int:
        """Vertical offset of the pill inside its parent window."""
        below = self._below
        if below is not None and below.parent() is self.parent():
            return below.geometry().bottom() + 1 + self._GAP
        return self._menu_bar_bottom() + self._MARGIN

    def _menu_bar_bottom(self) -> int:
        """Bottom edge of the parent's visible menu bar (0 without one)."""
        menu_bar = self._menu_bar_widget()
        if menu_bar is None or not menu_bar.isVisible():
            return 0
        return menu_bar.geometry().bottom() + 1

    def _menu_bar_widget(self) -> QMenuBar | None:
        """The menu bar of the parent window, when it has one."""
        parent = self.parent()
        if parent is None:
            return None
        menu_bar = parent.findChild(QMenuBar)
        if menu_bar is not None and menu_bar.parent() is parent:
            return menu_bar
        return None

    def _sync_menu_bar_filter(self) -> None:
        """Watch the menu bar so the pill follows its visibility changes."""
        menu_bar = self._menu_bar_widget()
        if menu_bar is self._menu_bar:
            return
        if self._menu_bar is not None:
            # The old menu bar may be gone with its parent window.
            with suppress(RuntimeError):
                self._menu_bar.removeEventFilter(self)
        self._menu_bar = menu_bar
        if menu_bar is not None:
            menu_bar.installEventFilter(self)
