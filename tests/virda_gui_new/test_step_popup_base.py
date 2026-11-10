"""Unit tests for :class:`virda_gui_new.widgets.popup.abstract.StepPopupBase`.

Runs under the offscreen Qt platform; skips without a Qt install. The
:class:`TaskRunner` thread result is awaited in a bounded ``processEvents``
loop.
"""

import time
from collections.abc import Callable

from tests.virda_gui_new.fixtures.steps import advance_to
from virda_gui_new.dto.steps import BASE_MESH, STEPS
from virda_gui_new.widgets.popup.abstract import StepPopupBase
from virda_gui_new.windows.main_window import MainWindow


class _ProbePopup(StepPopupBase):
    """Minimal concrete popup with one counter option and fixed work."""

    def __init__(
        self,
        step,
        manager,
        *,
        result=7,
        work_error: str | None = None,
        collect_error: str | None = None,
    ) -> None:
        super().__init__(step, manager)
        self.probe_result = result
        self.work_error = work_error
        self.collect_error = collect_error
        self.applied: list[int] = []

    def build_form(self, layout) -> None:
        from PySide6.QtWidgets import QLabel

        self.counter = QLabel("0", self)
        layout.addWidget(self.counter)

    def _collect_run(self):
        if self.collect_error is not None:
            raise RuntimeError(self.collect_error)

        def work():
            if self.work_error is not None:
                raise RuntimeError(self.work_error)
            return self.probe_result

        return work

    def _apply_result(self, result):
        self.applied.append(result)
        self._manager.set_step_context_item(self._step.index, "result", result)
        # The pipeline only advances when the step wrote its artifacts.
        for key in self._step.required_keys:
            self._manager.set_step_context_item(self._step.index, key, result)


def _connect(popup) -> tuple[list[int], list[str]]:
    completed: list[int] = []
    failed: list[str] = []
    popup.completed.connect(completed.append)
    popup.failed.connect(failed.append)
    return completed, failed


def _settle(qapp, popup, timeout_s: float = 5.0) -> None:
    """Pump events until the run settles (busy flag cleared)."""
    deadline = time.monotonic() + timeout_s
    while popup._busy and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


def test_initial_state(qapp, manager, popups) -> None:
    popup = _ProbePopup(BASE_MESH, manager)
    popups.append(popup)
    assert popup._name_label.text() == "Base mesh"
    assert popup._hint_label.text() == ""
    assert popup._run_button.text() == "Run"
    assert popup._run_button.isEnabled()
    assert popup.counter.text() == "0"
    assert manager.get_current_step() == 0


def test_successful_run_applies_and_advances(qapp, manager, popups) -> None:
    popup = _ProbePopup(BASE_MESH, manager)
    popups.append(popup)
    advance_to(manager, BASE_MESH.index)
    completed, failed = _connect(popup)

    popup._run_button.click()
    assert not popup._run_button.isEnabled()
    assert popup._hint_label.text() == "Running..."

    _settle(qapp, popup)
    assert failed == []
    assert completed == [BASE_MESH.index]
    assert popup.applied == [7]
    assert manager.get_step_context_item(1, "result") == 7
    assert manager.get_current_step() == 2
    assert popup._hint_label.text() == "Done."
    assert popup._run_button.isEnabled()


def test_work_failure_fails_without_advance(qapp, manager, popups) -> None:
    popup = _ProbePopup(BASE_MESH, manager, work_error="nifti unreadable")
    popups.append(popup)
    advance_to(manager, BASE_MESH.index)
    completed, failed = _connect(popup)

    popup._run_button.click()
    _settle(qapp, popup)

    assert failed == ["nifti unreadable"]
    assert completed == []
    assert popup.applied == []
    assert manager.get_current_step() == 1
    assert manager.get_step_context_item(1, "result") is None
    assert popup._hint_label.text() == "nifti unreadable"
    assert popup._run_button.isEnabled()


def test_collect_hook_exception_fails_immediately(qapp, manager, popups) -> None:
    popup = _ProbePopup(BASE_MESH, manager, collect_error="bad options")
    popups.append(popup)
    completed, failed = _connect(popup)

    popup._run_button.click()

    assert failed == ["bad options"]
    assert completed == []
    assert manager.get_current_step() == 0
    assert popup._hint_label.text() == "bad options"
    assert popup._run_button.isEnabled()


class _Host:
    """No-op stand-in for the dispatcher state a :class:`MainWindow` holds."""

    def __init__(self) -> None:
        from PySide6.QtWidgets import QMainWindow

        self.logs: list[str] = []
        self._popup_factories: dict[int, Callable[[], StepPopupBase]] = {}
        self._popup_windows: dict[int, QMainWindow] = {}
        self._popups: dict[int, StepPopupBase] = {}

    def _log(self, message: str) -> None:
        self.logs.append(message)

    def _on_step_completed(self, step_n: int) -> None:
        MainWindow._on_step_completed(self, step_n)  # type: ignore[arg-type]

    def _open_step_popup(self, step_n: int) -> None:
        MainWindow._open_step_popup(self, step_n)  # type: ignore[arg-type]

    def _close_popup(self, step_n: int) -> None:
        MainWindow._close_popup(self, step_n)  # type: ignore[arg-type]

    def _forget_popup(self, step_n: int) -> None:
        MainWindow._forget_popup(self, step_n)  # type: ignore[arg-type]


def test_dispatcher_missing_factory_logs(qapp) -> None:
    host = _Host()

    MainWindow._open_step_popup(host, 1)  # type: ignore[arg-type]

    assert host.logs == ["Step 1: editor not implemented yet."]
    assert host._popups == {} and host._popup_windows == {}


def test_dispatcher_full_flow(qapp, manager, popups) -> None:
    host = _Host()
    host._popup_factories[BASE_MESH.index] = lambda: _ProbePopup(BASE_MESH, manager)
    next_step = BASE_MESH.index + 1
    host._popup_factories[next_step] = lambda: _ProbePopup(STEPS[next_step], manager)
    advance_to(manager, BASE_MESH.index)

    MainWindow._open_step_popup(host, BASE_MESH.index)  # type: ignore[arg-type]
    popup = host._popups[BASE_MESH.index]
    popups.append(popup)
    window = host._popup_windows[BASE_MESH.index]
    assert window.windowTitle() == "Base mesh"
    assert window.isVisible()
    gone = []
    window.destroyed.connect(lambda: gone.append(True))

    popup._run_button.click()
    _settle(qapp, popup)
    qapp.processEvents()

    # The finished popup is closed and destroyed; the next popup took over.
    assert gone == [True]
    assert list(host._popups) == [next_step]
    assert list(host._popup_windows) == [next_step]
    assert host._popup_windows[next_step].windowTitle() == STEPS[next_step].name
    assert manager.get_current_step() == next_step
    assert host.logs == [f"Step {BASE_MESH.index} done."]

    popup.shutdown()
    qapp.processEvents()


def test_dispatcher_user_closed_popup_window(qapp, manager, popups) -> None:
    """A manually closed popup window is dropped from dispatcher tracking."""
    host = _Host()
    host._popup_factories[BASE_MESH.index] = lambda: _ProbePopup(BASE_MESH, manager)
    advance_to(manager, BASE_MESH.index)

    MainWindow._open_step_popup(host, BASE_MESH.index)  # type: ignore[arg-type]
    window = host._popup_windows[BASE_MESH.index]
    window.close()
    qapp.processEvents()

    assert host._popup_windows == {}
    assert host._popups == {}

    # Reopening must not touch the destroyed window again.
    MainWindow._open_step_popup(host, BASE_MESH.index)  # type: ignore[arg-type]
    assert list(host._popups) == [BASE_MESH.index]
    assert list(host._popup_windows) == [BASE_MESH.index]

    for popup in host._popups.values():
        popup.shutdown()
    for window in host._popup_windows.values():
        window.close()
    qapp.processEvents()
