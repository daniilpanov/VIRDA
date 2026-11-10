"""Background task execution for GUI code."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot

# Live references to threads that keep running after their runner retired
# them: each entry holds the thread and its worker alive until the thread
# finishes, so a retired task never dies with its owner.
_ORPHANED: dict[QThread, QObject] = {}


def _release_orphan(thread: QThread) -> None:
    """Drop an orphan thread reference once it can be destroyed safely."""
    _ORPHANED.pop(thread, None)
    thread.deleteLater()


def _detach_orphan(thread: QThread, worker: QObject | None) -> None:
    """Keep a still-running thread alive without a parent until it finishes."""
    thread.setParent(None)
    if worker is not None:
        _ORPHANED[thread] = worker
    thread.finished.connect(lambda: _release_orphan(thread))
    # The thread may have finished between the wait() and this call, in
    # which case its finished signal was already emitted and will never
    # arrive: release the reference right away.
    if thread.isFinished():
        _release_orphan(thread)


class _Signal[T](Protocol):
    """Static-only view of a Qt signal carrying one typed payload.

    Qt signals carry no type information: ``SignalInstance.connect`` takes
    any slot and ``emit`` takes any arguments, so this protocol is what
    gives ``done.connect(...)`` and ``done.emit(...)`` their parameter
    types. It exists for the type checker only - at runtime the real
    ``Signal`` objects are used.
    """

    def connect(self, slot: Callable[[T], object]) -> object: ...

    def emit(self, value: T) -> None: ...


class _TaskWorker(QObject):
    """Runs one ``fn()`` callable; results come back to the GUI thread."""

    done = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, fn: Callable[[], Any], generation: int) -> None:
        super().__init__()
        self._fn = fn
        self._generation = generation

    @Slot()
    def run(self) -> None:
        try:
            result = self._fn()
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a signal
            self.failed.emit(self._generation, str(exc))
        else:
            self.done.emit(self._generation, result)


class TaskRunner[T = Any](QObject):
    """Runs callables on worker threads, retiring stale runs."""

    # Qt reads the signals from the class body, the type checker reads
    # the branch above: both must stay in sync.
    if TYPE_CHECKING:
        done: _Signal[T]
        failed: _Signal[str]
    else:
        done = Signal(object)
        failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._generation = 0
        self._thread: QThread | None = None
        self._worker: _TaskWorker | None = None

    def start(self, fn: Callable[[], T]) -> None:
        """Run ``fn()`` on a fresh worker thread, replacing any running task."""
        self.stop()
        generation = self._generation

        thread = QThread(self)
        worker = _TaskWorker(fn, generation)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.done.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        thread.finished.connect(worker.deleteLater)
        # The event loop would otherwise keep running after the work is
        # over, leaving a live thread behind until the runner is destroyed.
        # The quit must reach the thread from the worker itself: a queued
        # connection would only run it once the GUI thread processes its
        # event queue, keeping the finished thread alive until then.
        worker.done.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.failed.connect(thread.quit, Qt.ConnectionType.DirectConnection)

        self._thread = thread
        self._worker = worker
        thread.start()

    @Slot(int, object)
    def _on_done(self, generation: int, result: T) -> None:
        """Deliver a result, unless the run was retired meanwhile."""
        if generation == self._generation:
            self.done.emit(result)

    @Slot(int, str)
    def _on_failed(self, generation: int, message: str) -> None:
        """Deliver a failure, unless the run was retired meanwhile."""
        if generation == self._generation:
            self.failed.emit(message)

    def stop(self) -> None:
        """Retire the running task; it finishes in the background, unobserved.

        The retired result is never delivered: the generation counter moves
        on, so a result that was already queued is dropped when it arrives.
        """
        self._generation += 1
        thread = self._thread
        worker = self._worker
        if thread is None:
            return

        self._thread = None
        self._worker = None
        thread.quit()
        if thread.wait(0):
            thread.deleteLater()
        else:
            _detach_orphan(thread, worker)
