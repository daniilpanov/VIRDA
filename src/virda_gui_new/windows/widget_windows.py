"""Host one widget in a non-modal tool window."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QWidget


def is_alive(window: QWidget | None) -> bool:
    """True when ``window`` still has a live C++ counterpart.

    A tool window can be destroyed (by its own close button, by the
    window manager or by the embedder) while the embedder still holds a
    reference to it. Touching such a dead wrapper raises a
    ``RuntimeError``, so callers check it before using a window.
    """
    if window is None:
        return False
    try:
        window.objectName()
    except RuntimeError:
        return False
    return True


class WidgetWindow(QMainWindow):
    """Non-modal Qt.Tool host: only wraps a widget."""

    def __init__(
        self,
        widget: QWidget,
        parent: QWidget | None = None,
        *,
        title: str | None = None,
    ) -> None:
        # Only a real widget can parent a window; anything else (e.g. a
        # duck-typed test host standing in for the dispatcher) is ignored.
        qt_parent = parent if isinstance(parent, QWidget) else None
        super().__init__(qt_parent)
        # Tool windows keep the hosted widget attached to its parent window
        # (typically the MainWindow) instead of looking like a free-standing
        # application window.
        self.setWindowFlag(Qt.WindowType.Tool, True)
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
    """Build, show and raise a non-modal tool window hosting ``widget``.

    The window is raised right after it was shown, so it lands above
    its parent window instead of behind it.
    """
    window = WidgetWindow(widget, parent, title=title)
    window.show()
    window.raise_()
    window.activateWindow()
    return window
