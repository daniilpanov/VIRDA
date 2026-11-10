"""Background task execution for GUI code."""

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot

# How long stop() waits for a running task before leaving the thread to
# finish on its own. Patchable in tests.
_STOP_WAIT_MS = 3000

# Live references to detached threads that are still running after a stop()
# timeout; each entry is removed and deleted once its thread finishes.
_ORPHANS: set[QThread] = set()


def _detach_orphan(thread: QThread) -> None:
    """Keep a still-running thread alive without a parent until it finishes."""
    thread.setParent(None)
    _ORPHANS.add(thread)
    thread.finished.connect(lambda: _release_orphan(thread))


def _release_orphan(thread: QThread) -> None:
    """Drop an orphan thread reference once it can be destroyed safely."""
    _ORPHANS.discard(thread)
    thread.deleteLater()


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
    """Runs callables on worker threads, retiring stale runs."""

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
        # The event loop would otherwise keep running after the work is
        # over, leaving a live thread behind until the runner is destroyed.
        worker.done.connect(thread.quit)
        worker.failed.connect(thread.quit)
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
            try:
                worker.done.disconnect(self.done)
                worker.failed.disconnect(self.failed)
                # deleteLater() must be posted while the worker's event loop
                # is still live, otherwise the DeferredDelete event is never
                # processed.
                worker.deleteLater()
            except RuntimeError:
                # A finished task's worker was already deleted together
                # with its thread; delivery is dead with it.
                pass

        # quit() called before the thread's event loop has started is a
        # no-op, so one retry covers a task that was still being scheduled.
        thread.quit()
        if not thread.wait(_STOP_WAIT_MS):
            thread.quit()
        if thread.wait(_STOP_WAIT_MS):
            thread.deleteLater()
        else:
            # Destroying a live QThread is undefined behaviour; detach it
            # from this (possibly soon-destroyed) parent and let it finish
            # under the orphan registry instead.
            _detach_orphan(thread)

        self._thread = None
        self._worker = None
