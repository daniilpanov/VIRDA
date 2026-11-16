"""Background task execution for GUI code."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from PySide6.QtCore import QCoreApplication, QObject, QThread, QTimer, Signal, Slot


class _OrphanThreads(dict[QThread, tuple[QObject, QObject]]):
    """Worker threads kept alive outside the runner that started them.

    An entry holds the thread and the objects that live in it. Qt must not
    delete them and Python must not drop them while the thread runs:
    releasing a wrapper from a thread that is shutting down corrupts the
    heap. The reaper drops an entry only once the thread has really
    finished; truthiness counts the threads that still run, so a retired
    run stops counting the moment its thread is done.
    """

    def __len__(self) -> int:
        return sum(1 for thread in dict.keys(self) if not thread.isFinished())


_ORPHANED = _OrphanThreads()
_REAP_INTERVAL_MS = 50
_FINISHED: set[int] = set()
_REAPER: QTimer | None = None
_EXIT_HOOK: QObject | None = None


def _reap() -> None:
    """Drop the references of the threads that have really finished."""
    for thread in list(_ORPHANED):
        if not thread.isFinished():
            _FINISHED.discard(id(thread))
            continue
        if id(thread) not in _FINISHED:
            # First tick past QThreadPrivate::finish(): wait one more, so
            # nothing is released while the thread is still unwinding.
            _FINISHED.add(id(thread))
            continue
        # Only the references go away: the wrappers are freed by Python in
        # the GUI thread, never from inside the thread that is exiting.
        _ORPHANED.pop(thread, None)
        _FINISHED.discard(id(thread))
    if not dict.__len__(_ORPHANED) and _REAPER is not None:
        _REAPER.stop()


def _watch_orphans() -> None:
    """Run the reaper on the GUI event loop."""
    global _REAPER
    if _REAPER is None:
        _REAPER = QTimer()
        _REAPER.setInterval(_REAP_INTERVAL_MS)
        _REAPER.timeout.connect(_reap)
    if not _REAPER.isActive():
        _REAPER.start()


def _quit_all() -> None:
    """Wind every worker thread down when the application quits."""
    for thread in list(_ORPHANED):
        thread.quit()
        thread.wait(5_000)


def _quit_on_exit() -> None:
    """Quit the worker threads when the application quits, once per process."""
    global _EXIT_HOOK
    app = QCoreApplication.instance()
    if app is None or _EXIT_HOOK is not None:
        return
    app.aboutToQuit.connect(_quit_all)
    _EXIT_HOOK = app


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


class _Generation:
    """The newest run number, shared with the worker thread."""

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = 0


class _Invoker(QObject):
    """Hands a task from the GUI thread to the worker thread."""

    submit = Signal(int, object)


class _TaskWorker(QObject):
    """Runs the callables posted to it; results come back to the GUI thread."""

    done = Signal(int, object)
    failed = Signal(int, str)
    released = Signal(int, object)

    def __init__(self, generation: _Generation) -> None:
        super().__init__()
        self._generation = generation

    @Slot(int, object)
    def run(self, generation: int, fn: Callable[[], Any]) -> None:
        try:
            # A queued run that was superseded meanwhile never starts at all.
            if generation != self._generation.value:
                return
            try:
                result = fn()
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a signal
                self.failed.emit(generation, str(exc))
            else:
                self.done.emit(generation, result)
        finally:
            # Hand the callable back: what it captured (often a widget)
            # may only be released by the GUI thread.
            self.released.emit(generation, fn)


class TaskRunner[T = Any](QObject):
    """Runs callables on one worker thread, retiring stale runs."""

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
        self._generation = _Generation()
        self._thread: QThread | None = None
        self._invoker: _Invoker | None = None
        self._quitting = False
        # The callable of the last finished run. Python must release it in
        # the GUI thread: the worker's own last drop happens inside the
        # thread, which would destroy captured widgets there.
        self._retained: object | None = None

    def start(self, fn: Callable[[], T]) -> None:
        """Run ``fn()`` on the worker thread, replacing any running task."""
        self._retained = None
        if self._quitting:
            # The thread of the previous burst is winding down: the next
            # task gets a fresh one, so a queued submit is never lost.
            self._thread = None
            self._invoker = None
        invoker = self._ensure_thread()
        self._generation.value += 1
        invoker.submit.emit(self._generation.value, fn)

    def stop(self) -> None:
        """Retire the running task and wind the worker thread down.

        The callable itself is not interrupted: its result is dropped and
        the thread exits once it returns. The next :meth:`start` gets a
        fresh thread.
        """
        self._generation.value += 1
        self._quit_thread()
        # Release the finished run here, while the GUI thread still owns it.
        self._retained = None

    def _quit_thread(self) -> None:
        """Ask the worker thread to exit once its queue is drained."""
        thread = self._thread
        if thread is None or self._quitting:
            return
        self._quitting = True
        thread.quit()

    def _ensure_thread(self) -> _Invoker:
        """Start the worker thread on first use and reuse it afterwards."""
        if self._invoker is not None:
            return self._invoker

        # Never parented: Qt must not destroy a thread that still runs.
        thread = QThread()
        worker = _TaskWorker(self._generation)
        worker.moveToThread(thread)
        invoker = _Invoker()
        invoker.submit.connect(worker.run)
        worker.done.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        worker.released.connect(self._on_task_released)

        _ORPHANED[thread] = (worker, invoker)
        _watch_orphans()
        _quit_on_exit()
        thread.start()

        self._thread = thread
        self._invoker = invoker
        self._quitting = False
        return invoker

    @Slot(int, object)
    def _on_done(self, generation: int, result: T) -> None:
        """Deliver a result, unless the run was retired meanwhile."""
        if generation != self._generation.value:
            return
        self.done.emit(result)
        # Nothing newer is waiting: let the thread go instead of keeping
        # an idle one per runner alive.
        if generation == self._generation.value:
            self._quit_thread()

    @Slot(int, str)
    def _on_failed(self, generation: int, message: str) -> None:
        """Deliver a failure, unless the run was retired meanwhile."""
        if generation != self._generation.value:
            return
        self.failed.emit(message)
        if generation == self._generation.value:
            self._quit_thread()

    @Slot(int, object)
    def _on_task_released(self, generation: int, fn: object) -> None:
        """Drop a finished callable in the GUI thread.

        The callable holds whatever it captured, which is frequently the
        popup the task belongs to. A widget may only be destroyed in the
        GUI thread: destroyed from a worker, its still-registered timers
        stay behind in the GUI dispatcher and crash it later. Taking the
        reference here guarantees the last drop happens in this thread.
        """
        del generation
        self._retained = None
        self._retained = fn
