"""Unit tests for :mod:`virda_gui_new.services.task_runner`.

Runs under the offscreen Qt platform; skips without a Qt install.
"""

import os
import time

import pytest

from virda_gui_new.services import task_runner as task_runner_module
from virda_gui_new.services.task_runner import TaskRunner


@pytest.fixture
def qapp():
    """Yield an offscreen ``QApplication``, skipping without a Qt install."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on local Qt install
        pytest.skip(f"Qt platform unavailable: {exc}")

    return QApplication.instance() or QApplication([])


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
    """A successful run delivers (seq, result) once."""
    runner = TaskRunner()
    done, failed = [], []
    runner.done.connect(lambda seq, result: done.append((seq, result)))
    runner.failed.connect(lambda seq, msg: failed.append((seq, msg)))

    runner.start(lambda: 7)
    _spin(qapp, 1.0)

    assert done == [(1, 7)]
    assert failed == []


def test_failed_delivers_message(qapp):
    """A raising run delivers (seq, message) via failed."""
    runner = TaskRunner()
    done, failed = [], []
    runner.done.connect(lambda seq, result: done.append((seq, result)))
    runner.failed.connect(lambda seq, msg: failed.append((seq, msg)))

    def boom() -> None:
        raise ValueError("no can do")

    runner.start(boom)
    _spin(qapp, 1.0)

    assert failed == [(1, "no can do")]
    assert done == []


def test_stop_retires_running_task(qapp):
    """stop() disconnects delivery and the finished thread is cleaned up."""
    runner = TaskRunner()
    done = []
    runner.done.connect(lambda seq, result: done.append((seq, result)))

    started = []
    runner.start(lambda: (started.append(True), time.sleep(0.2)))
    _wait_work_started(started)
    qapp.processEvents()
    runner.stop()
    _spin(qapp, 1.0)

    assert done == []
    assert not task_runner_module._ORPHANS


def test_stop_timeout_detaches_orphan_thread(qapp, monkeypatch):
    """On a wait timeout the running thread is detached, not destroyed."""
    monkeypatch.setattr(task_runner_module, "_STOP_WAIT_MS", 1)
    runner = TaskRunner()
    done = []
    runner.done.connect(lambda seq, result: done.append((seq, result)))

    started = []
    runner.start(lambda: (started.append(True), time.sleep(0.3)))
    _wait_work_started(started)
    qapp.processEvents()
    thread = runner._thread  # noqa: SLF001 - regression check on the detach
    gone = []
    thread.destroyed.connect(lambda: gone.append(True))
    runner.stop()

    assert thread is not None
    assert thread.parent() is None
    assert thread in task_runner_module._ORPHANS

    # The orphan finishes later and is then released from the registry.
    _spin(qapp, 2.0)
    assert done == []
    assert not task_runner_module._ORPHANS
    assert gone == [True]
