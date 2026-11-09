"""Background task execution for GUI code: a callable on a worker thread.

:class:`TaskRunner` runs an arbitrary Python callable on a dedicated
:class:`PySide6.QtCore.QThread` and delivers its result (or exception
message) through signals back to the GUI thread.
"""

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot


class _TaskWorker(QObject):
    """Runs one ``fn()`` callable; results come back to the GUI thread."""

    done = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, fn: Callable[[], Any], seq: int) -> None:
        super().__init__()
        self._fn = fn
        self._seq = seq

    @Slot()
    def run(self) -> None:
        try:
            result = self._fn()
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a signal
            self.failed.emit(self._seq, str(exc))
        else:
            self.done.emit(self._seq, result)


class TaskRunner(QObject):
    """Lifecycle for background tasks: start newer, retire older, stop cleanly.

    Every widget that offloads work (scene collection, loading, previews)
    embeds one runner, connects its :attr:`done` / :attr:`failed` signals to
    handler slots and calls :meth:`start` with the work callable; a newer
    :meth:`start` replaces a still-running older task.

    Starting a newer task retires the still-running previous one so nothing
    stale is ever delivered: the ``done``/``failed`` handlers of the cancelled
    worker are disconnected and only the most recent sequence number is accepted
    by the caller (``runner.seq``). ``stop`` tears everything down and is safe
    to call at application teardown for widgets whose ``closeEvent`` is never
    delivered.
    """

    done = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._seq = 0
        self._thread: QThread | None = None
        self._worker: _TaskWorker | None = None

    @property
    def seq(self) -> int:
        """Sequence number of the last requested task."""
        return self._seq

    def start(self, fn: Callable[[], Any]) -> None:
        """Run ``fn()`` on a fresh worker thread, replacing any running task."""
        self._seq += 1
        self.stop()
        thread = QThread(self)
        worker = _TaskWorker(fn, self._seq)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.done.connect(self.done)
        worker.failed.connect(self.failed)
        thread.finished.connect(worker.deleteLater)
        self._thread = thread
        self._worker = worker
        thread.start()

    def stop(self) -> None:
        """Retire a running task so nothing stale is delivered afterwards."""
        thread = self._thread
        if thread is None:
            return

        worker = self._worker
        if worker is not None:
            worker.done.disconnect(self.done)
            worker.failed.disconnect(self.failed)
            # deleteLater() must be posted while the worker's event loop is
            # still live, otherwise the DeferredDelete event is never processed.
            worker.deleteLater()

        thread.quit()
        thread.wait(3000)
        # Never delete a thread that is still running (wait timed out);
        # destroying a live QThread is undefined behaviour.
        # In that case the thread keeps running under its parent until it finishes.
        if thread.isFinished():
            thread.deleteLater()

        self._thread = None
        self._worker = None
