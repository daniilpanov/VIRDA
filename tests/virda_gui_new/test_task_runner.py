"""Unit tests for :mod:`virda_gui_new.services.task_runner`.

Runs under the offscreen Qt platform; skips without a Qt install.
"""

import time

from virda_gui_new.services import task_runner as task_runner_module
from virda_gui_new.services.task_runner import TaskRunner


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


def test_done_delivers_result(qapp):
    """A successful run delivers its result once."""
    runner = TaskRunner()
    done: list[object] = []
    failed: list[str] = []
    runner.done.connect(done.append)
    runner.failed.connect(failed.append)

    runner.start(lambda: 7)
    _spin(qapp, 1.0)

    assert done == [7]
    assert failed == []


def test_typed_runner_delivers_result(qapp):
    """A parameterised runner keeps its signals working."""
    runner = TaskRunner[int]()
    done: list[int] = []
    runner.done.connect(done.append)

    runner.start(lambda: 7)
    _spin(qapp, 1.0)

    assert done == [7]


def test_failed_delivers_message(qapp):
    """A raising run delivers its message via failed."""
    runner = TaskRunner()
    done: list[object] = []
    failed: list[str] = []
    runner.done.connect(done.append)
    runner.failed.connect(failed.append)

    def boom() -> None:
        raise ValueError("no can do")

    runner.start(boom)
    _spin(qapp, 1.0)

    assert failed == ["no can do"]
    assert done == []


def test_stop_retires_running_task(qapp):
    """stop() drops the result of a running task."""
    runner = TaskRunner()
    done: list[object] = []
    runner.done.connect(done.append)

    started = []
    runner.start(lambda: (started.append(True), time.sleep(0.2)))  # type: ignore[func-returns-value]
    _wait_work_started(started)
    qapp.processEvents()
    runner.stop()
    _spin(qapp, 1.0)

    assert done == []
    assert not task_runner_module._ORPHANED


def test_stop_drops_queued_result(qapp):
    """A result queued before stop() is dropped when it arrives."""
    runner = TaskRunner()
    done: list[object] = []
    runner.done.connect(done.append)

    runner.start(lambda: 42)
    thread = runner._thread
    assert thread is not None
    # The worker is done, but the GUI thread never processed the queue:
    # its result is waiting for delivery.
    assert thread.wait(2000)

    runner.stop()
    qapp.processEvents()

    assert done == []


def test_stop_does_not_block(qapp):
    """stop() returns at once; the task keeps running in the background."""
    runner = TaskRunner()
    done: list[object] = []
    runner.done.connect(done.append)

    started = []
    runner.start(lambda: (started.append(True), time.sleep(0.6)))  # type: ignore[func-returns-value]
    _wait_work_started(started)
    qapp.processEvents()

    began = time.monotonic()
    runner.stop()
    elapsed = time.monotonic() - began

    assert elapsed < 0.2
    _spin(qapp, 2.0)
    assert done == []
    assert not task_runner_module._ORPHANED


def test_stop_detaches_running_thread(qapp):
    """A still-running task keeps its thread alive outside the runner."""
    runner = TaskRunner()
    done: list[object] = []
    runner.done.connect(done.append)

    started = []
    runner.start(lambda: (started.append(True), time.sleep(0.3)))  # type: ignore[func-returns-value]
    _wait_work_started(started)
    qapp.processEvents()

    assert runner._thread
    thread = runner._thread
    gone = []
    thread.destroyed.connect(lambda: gone.append(True))
    runner.stop()

    assert thread.parent() is None
    assert thread in task_runner_module._ORPHANED

    # The orphan finishes later and is then released from the registry.
    _spin(qapp, 2.0)
    assert done == []
    assert not task_runner_module._ORPHANED
    assert gone == [True]


def test_replacement_runs_the_newest_task(qapp):
    """A second start() retires the first run; only the newest is delivered."""
    runner = TaskRunner()
    done: list[object] = []
    runner.done.connect(done.append)

    started = []
    runner.start(lambda: (started.append(True), time.sleep(0.3), "old"))  # type: ignore[func-returns-value]
    _wait_work_started(started)
    runner.start(lambda: "new")
    _spin(qapp, 2.0)

    assert done == ["new"]
