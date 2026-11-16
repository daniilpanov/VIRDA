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
        delete_on_close: bool = True,
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
        if orphan and delete_on_close:
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
    show: bool = True,
    delete_on_close: bool = True,
) -> WidgetWindow:
    """Build and return a non-modal window hosting ``widget``.

    The window is shown unless ``show`` is false, and it deletes the
    hosted widget when it is closed unless ``delete_on_close`` is false.
    """
    window = WidgetWindow(widget, parent, title=title, delete_on_close=delete_on_close)
    if show:
        window.show()
    return window
