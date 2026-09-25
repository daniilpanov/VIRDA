"""Floating HUD overlays stacked over the 3D viewer.

The HUD puts the base widget (the 3D viewer) and several floating panels in the
same layout cell: the viewer fills the tab while translucent panels drawn on
top of it carry the live-editing tables.
"""

from PySide6.QtCore import QEvent, QPoint, QSize, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class HUDContainer(QWidget):
    """A single-cell grid that stacks floating overlays above a base widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(0)

    def set_base(self, widget: QWidget) -> None:
        """Make *widget* fill the tab underneath every overlay."""
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._grid.addWidget(widget, 0, 0)

    def add_overlay(
        self,
        widget: QWidget,
        alignment: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignCenter,
        *,
        fixed_width: int | None = None,
        fixed_height: int | None = None,
    ) -> QWidget:
        """Float *widget* above the base using *alignment* in the shared cell.

        Fixed sizes optional; the widget keeps its size hint otherwise and is
        lifted so later overlays paint on top of earlier ones and of the base.
        """
        if fixed_width is not None:
            widget.setFixedWidth(fixed_width)
        if fixed_height is not None:
            widget.setFixedHeight(fixed_height)
        self._grid.addWidget(widget, 0, 0, alignment)
        widget.setProperty("hud_side", _side_name(alignment))
        widget.raise_()
        widget.show()
        return widget

    def move_overlay(
        self, widget: QWidget, alignment: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignCenter
    ) -> None:
        """Re-anchor *widget* to *alignment* inside the shared cell.

        Size constraints set by :meth:`add_overlay` are kept; the widget is
        lifted above the base again.
        """
        self._grid.removeWidget(widget)
        self._grid.addWidget(widget, 0, 0, alignment)
        widget.setProperty("hud_side", _side_name(alignment))
        widget.raise_()
        widget.show()


def _side_name(alignment: Qt.AlignmentFlag) -> str:
    """Horizontal side ("left"/"right"/"center") of a HUD *alignment*."""
    if alignment & Qt.AlignmentFlag.AlignRight:
        return "right"
    if alignment & Qt.AlignmentFlag.AlignLeft:
        return "left"
    return "center"


class HudPanel(QFrame):
    """A translucent, collapsible panel that floats over the viewer.

    Dragging the header moves the panel to the left or right side of its
    :class:`HUDContainer`; a plain click still toggles collapse.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("HudPanel")
        self.setStyleSheet(
            "QFrame#HudPanel {"
            "  background-color: rgba(20, 26, 38, 235);"
            "  border: 1px solid rgba(120, 140, 170, 140);"
            "  border-radius: 8px;"
            "}"
        )
        self.setMinimumSize(QSize(320, 240))

        self._title = QLabel(title, self)
        self._collapse_btn = QToolButton(self)
        self._collapse_btn.setAutoRaise(True)
        self._collapse_btn.setCheckable(True)
        self._collapse_btn.setToolTip("Collapse / expand")
        self._collapse_btn.setArrowType(Qt.ArrowType.LeftArrow)
        self._collapse_btn.toggled.connect(self._on_collapse_toggled)

        header = QHBoxLayout()
        header.setContentsMargins(8, 4, 4, 0)
        header.addWidget(self._title, 1)
        header.addWidget(self._collapse_btn)

        self._body_layout = QVBoxLayout(self)
        self._body_layout.setContentsMargins(8, 4, 8, 8)
        self._body_layout.setSpacing(6)
        self._body_layout.addLayout(header)

        self._body: QWidget | None = None
        self._drag_press: QPoint | None = None
        self._dragging = False

    def set_body(self, body: QWidget) -> None:
        """Attach the panel's content; it expands to fill the panel."""
        self._body = body
        self._body_layout.addWidget(body, 1)

    def _on_collapse_toggled(self, collapsed: bool) -> None:
        if self._body is not None:
            self._body.setVisible(not collapsed)
        if collapsed:
            self._set_header_only()
        else:
            self.setMaximumHeight(16777215)
        self._collapse_btn.setArrowType(
            Qt.ArrowType.RightArrow if collapsed else Qt.ArrowType.LeftArrow
        )

    def _set_header_only(self) -> None:
        self.adjustSize()
        self.setFixedHeight(self._title.sizeHint().height() + 16)

    # ---- dragging ----

    def _in_header(self, pos: QPoint) -> bool:
        """Whether *pos* (panel-local) hits the always-visible header strip."""
        limit = max(self._title.geometry().bottom(), self._collapse_btn.geometry().bottom())
        return pos.y() <= limit + 4

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt naming
        if event.button() == Qt.MouseButton.LeftButton and self._in_header(
            event.position().toPoint()
        ):
            self._drag_press = event.globalPosition().toPoint()
            self._dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt naming
        if self._drag_press is not None and not self._dragging:
            delta = event.globalPosition().toPoint() - self._drag_press
            if delta.manhattanLength() >= QApplication.startDragDistance():
                self._dragging = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt naming
        if self._drag_press is not None and self._dragging:
            self._snap_to_side(event.globalPosition().toPoint())
        self._drag_press = None
        if self._dragging:
            self._dragging = False
            self.unsetCursor()
        super().mouseReleaseEvent(event)

    def _snap_to_side(self, global_pos: QPoint) -> None:
        """Re-anchor the panel left/right depending on the drop position."""
        container = self.parentWidget()
        move = getattr(container, "move_overlay", None)
        if container is None or not callable(move):
            return
        local = container.mapFromGlobal(global_pos)
        side = (
            Qt.AlignmentFlag.AlignRight
            if local.x() >= container.width() / 2
            else Qt.AlignmentFlag.AlignLeft
        )
        move(self, side)

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Leave and not self._dragging:
            self._drag_press = None
        return super().event(event)
