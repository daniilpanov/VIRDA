"""Unit tests for :mod:`virda_gui_new.windows.widget_windows`.

Runs under the offscreen Qt platform; skips without a Qt install.
"""

import os
import time

import pytest

from virda_gui_new.services import task_runner as task_runner_module
from virda_gui_new.windows.widget_windows import WidgetWindow, open_widget_window


@pytest.fixture
def qapp():
    """Yield an offscreen ``QApplication``, skipping without a Qt install."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on local Qt install
        pytest.skip(f"Qt platform unavailable: {exc}")

    app = QApplication.instance() or QApplication([])
    return app


def _spin(app, seconds: float = 2.0) -> None:
    """Pump events in short sleeps for up to ``seconds`` wall time."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)


def _wait_work_started(flag: list[bool], seconds: float = 2.0) -> None:
    """Block until the worker thread has entered its callable."""
    deadline = time.monotonic() + seconds
    while not flag and time.monotonic() < deadline:
        time.sleep(0.01)
    assert flag, "worker never started"


def test_open_widget_window_shows_hosted_widget(qapp):
    """The returned window shows the widget non-modal as its central widget."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QWidget

    widget = QWidget()
    widget.setWindowTitle("Step details")

    window = open_widget_window(widget, title="My step")
    try:
        assert isinstance(window, WidgetWindow)
        assert window.widget is widget
        assert window.centralWidget() is widget
        assert window.windowTitle() == "My step"
        assert window.windowModality() == Qt.WindowModality.NonModal
        assert window.isVisible()
        assert widget.isVisible()
    finally:
        window.close()
    qapp.processEvents()


def test_open_widget_window_is_tool_window_of_parent(qapp):
    """With a parent the window is a Qt.Tool tool window of that parent."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QWidget

    parent = QWidget()
    widget = QWidget()

    window = open_widget_window(widget, parent=parent, title="Step 1")
    try:
        assert window.windowType() == Qt.WindowType.Tool
        assert window.parent() is parent
        assert window.windowModality() == Qt.WindowModality.NonModal
        assert window.isVisible()
        assert widget.isVisible()
    finally:
        window.close()
        parent.deleteLater()
    qapp.processEvents()


def test_open_widget_window_defaults_to_parentless_tool(qapp):
    """Without a parent the window is still a Qt.Tool tool window."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QWidget

    window = open_widget_window(QWidget())
    try:
        assert window.windowType() == Qt.WindowType.Tool
        assert window.parent() is None
    finally:
        window.close()
    qapp.processEvents()


def test_open_widget_window_raises_the_window(qapp, monkeypatch):
    """Opening a window raises it, so it lands above the parent window."""
    from PySide6.QtWidgets import QWidget

    from virda_gui_new.windows import widget_windows

    class _RaiseSpy(widget_windows.WidgetWindow):
        """Tool window that counts how often it was raised."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.raises = 0

        def raise_(self) -> None:
            self.raises += 1
            super().raise_()

    monkeypatch.setattr(widget_windows, "WidgetWindow", _RaiseSpy)
    window = widget_windows.open_widget_window(QWidget())
    try:
        assert window.raises >= 1
        assert window.isVisible()
    finally:
        window.close()
    qapp.processEvents()


def test_is_alive_reports_destroyed_windows(qapp):
    """A destroyed window is not alive anymore."""
    from PySide6.QtWidgets import QWidget

    from virda_gui_new.windows.widget_windows import is_alive

    assert not is_alive(None)

    window = open_widget_window(QWidget())
    assert is_alive(window)

    window.close()
    qapp.processEvents()
    assert not is_alive(window)


def test_widget_window_title_derived_from_widget(qapp):
    """Without an explicit title the widget's own windowTitle() is used."""
    from PySide6.QtWidgets import QWidget

    widget = QWidget()
    widget.setWindowTitle("Localization")

    window = WidgetWindow(widget)
    try:
        assert window.windowTitle() == "Localization"
    finally:
        window.close()
    qapp.processEvents()


def test_widget_window_title_falls_back_empty(qapp):
    """No title anywhere still yields an empty window title."""
    from PySide6.QtWidgets import QWidget

    window = WidgetWindow(QWidget())
    try:
        assert window.windowTitle() == ""
    finally:
        window.close()
    qapp.processEvents()


def test_widget_window_close_deletes_parentless_widget(qapp):
    """Closing deletes a parentless hosted widget (WA_DeleteOnClose)."""
    from PySide6.QtWidgets import QWidget

    widget = QWidget()
    window = open_widget_window(widget)
    assert widget.parent() is window

    deleted = []
    widget.destroyed.connect(lambda: deleted.append(True))

    window.close()
    qapp.processEvents()
    assert deleted == [True]
    # The Python wrapper still holds, but the C++ object is gone: the
    # WA_DeleteOnClose semantics replaced the visibility check.
    assert window.widget is widget


def test_widget_window_close_keeps_parented_widget(qapp):
    """Closing a host of a parented widget skips WA_DeleteOnClose, so it survives."""
    from PySide6.QtWidgets import QWidget

    owner = QWidget()
    widget = QWidget(owner)
    window = WidgetWindow(widget)

    deleted = []
    widget.destroyed.connect(lambda: deleted.append(True))

    window.close()
    qapp.processEvents()
    assert deleted == []
    assert window.centralWidget() is widget
    assert not window.widget.isVisible()


def test_widget_window_close_calls_widget_shutdown(qapp):
    """Closing the host stops the hosted widget's background work first."""
    from PySide6.QtWidgets import QWidget

    class _ShutdownWidget(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.shutdowns = 0

        def shutdown(self) -> None:
            self.shutdowns += 1

    widget = _ShutdownWidget()
    window = open_widget_window(widget)

    window.close()
    qapp.processEvents()
    assert widget.shutdowns == 1


def test_widget_window_close_shutdowns_parented_widget(qapp):
    """shutdown() is called even when the widget is not deleted on close."""
    from PySide6.QtWidgets import QWidget

    class _ShutdownWidget(QWidget):
        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.shutdowns = 0

        def shutdown(self) -> None:
            self.shutdowns += 1

    owner = QWidget()
    widget = _ShutdownWidget(owner)
    window = WidgetWindow(widget)

    window.close()
    qapp.processEvents()
    assert widget.shutdowns == 1
    assert widget.parent() is not None


def test_widget_window_close_stops_running_task(qapp):
    """Closing a host while a task runs stops it: no delivery, no crash."""
    from PySide6.QtWidgets import QWidget

    from virda_gui_new.services.task_runner import TaskRunner

    class _BusyWidget(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.runner = TaskRunner(self)
            self.results = []
            self.shutdowns = 0
            self.runner.done.connect(lambda seq, result: self.results.append(result))

        def shutdown(self) -> None:
            self.shutdowns += 1
            self.runner.stop()

    widget = _BusyWidget()
    window = open_widget_window(widget)
    started = []
    widget.runner.start(lambda: (started.append(True), time.sleep(0.3)))
    _wait_work_started(started)
    qapp.processEvents()

    deleted = []
    widget.destroyed.connect(lambda: deleted.append(True))
    window.close()
    _spin(qapp, 2.0)

    assert widget.shutdowns == 1
    assert deleted == [True]
    assert widget.results == []
    assert not task_runner_module._ORPHANS
