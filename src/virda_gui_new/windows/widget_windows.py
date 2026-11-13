"""Host one widget in a non-modal top-level window."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QWidget


class WidgetWindow(QMainWindow):
    """Frameless-of-opinions top-level host: only wraps a widget."""

    def __init__(
        self,
        widget: QWidget,
        parent: QWidget | None = None,
        *,
        title: str | None = None,
    ) -> None:
        super().__init__(parent)
        orphan = widget.parent() is None
        self.widget = widget
        self.setCentralWidget(widget)
        self.setWindowTitle(title or widget.windowTitle() or "")
        # Non-modal so several per-step windows may be open at once.
        self.setWindowModality(Qt.WindowModality.NonModal)
        # Safe only when the widget is solely ours to delete: deleting it
        # while it still sits in another layout would break that layout,
        # so keep the attribute off parented widgets.
        if orphan:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        """Stop the hosted widget's background work before it is destroyed."""
        shutdown = getattr(self.widget, "shutdown", None)
        if callable(shutdown):
            shutdown()
        super().closeEvent(event)


def open_widget_window(
    widget: QWidget,
    parent: QWidget | None = None,
    *,
    title: str | None = None,
) -> WidgetWindow:
    """Build, show and return a non-modal window hosting ``widget``."""
    window = WidgetWindow(widget, parent, title=title)
    window.show()
    return window
