"""Base widget for the pipeline step popups: form, run, apply, advance."""

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from virda_gui_new.dto.steps import Step
from virda_gui_new.services.steps_manager import StepsManager
from virda_gui_new.services.task_runner import TaskRunner


class StepPopupBase(QWidget):
    """Shared lifecycle for a step popup: form, run, apply, advance.

    Subclasses add option widgets to ``self._form_layout`` in
    ``build_form``, return the heavy zero-arg work callable from
    ``_collect_run`` and write context entries from the ready result in
    ``_apply_result``. A Run click collects the work callable (a hook
    exception fails immediately) and runs it on this popup's own
    :class:`TaskRunner`; while in flight the Run button is disabled and the
    hint label shows ``"Running..."``. The runner retires superseded runs
    itself, so results are never delivered stale. On success
    ``_apply_result`` runs on the GUI thread, then the manager advances
    when this step is still current and holds its required artifacts; a
    step that finished without them keeps the pipeline where it is and
    says so in the hint. The hint shows ``"Done."`` and
    ``completed`` carries the step index, which is also the failure payload
    of ``failed``, whose message the hint shows. ``shutdown`` stops any
    in-flight run before the host closes the popup.
    """

    completed = Signal(int)  # noqa: N815
    failed = Signal(str)

    def __init__(self, step: Step, manager: StepsManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._step = step
        self._manager = manager
        self._runner = TaskRunner(self)
        self._runner.done.connect(self._on_done)
        self._runner.failed.connect(self._on_failed)
        self._busy = False

        self._name_label = QLabel(step.name, self)
        self._hint_label = QLabel("", self)
        self._hint_label.setStyleSheet("color: gray;")
        self._form_layout = QVBoxLayout()
        self._run_button = QPushButton("Run", self)
        self._run_button.clicked.connect(self._on_run)

        layout = QVBoxLayout(self)
        layout.addWidget(self._name_label)
        layout.addWidget(self._hint_label)
        layout.addLayout(self._form_layout)
        layout.addWidget(self._run_button)
        self.build_form(self._form_layout)

    def build_form(self, layout: QVBoxLayout) -> None:
        """Extension hook: add the option widgets to the form area."""

    def shutdown(self) -> None:
        """Stop any in-flight run; called by the host before closing."""
        self._runner.stop()

    def _on_run(self) -> None:
        if self._busy:
            return
        try:
            work = self._collect_run()
        except Exception as exc:  # noqa: BLE001 - any hook error is user-facing
            self._show_failure(str(exc))
            return
        self._busy = True
        self._run_button.setEnabled(False)
        self._hint_label.setText("Running...")
        self._runner.start(work)

    def _on_done(self, result: Any) -> None:
        try:
            self._apply_result(result)
        except Exception as exc:  # noqa: BLE001 - any apply error is user-facing
            self._show_failure(str(exc))
            return
        if self._manager.get_current_step() == self._step.index:
            if not self._manager.can_advance():
                # The step finished without its required artifacts, so the
                # pipeline stays here instead of skipping ahead.
                self._show_failure(
                    f"Step {self._step.index} finished without its required artifacts; "
                    "the pipeline stays on this step."
                )
                return
            self._manager.next_step()
        self._busy = False
        self._run_button.setEnabled(True)
        self._hint_label.setText("Done.")
        self.completed.emit(self._step.index)

    def _on_failed(self, message: str) -> None:
        self._show_failure(message)

    def _show_failure(self, message: str) -> None:
        self._busy = False
        self._run_button.setEnabled(True)
        self._hint_label.setText(message)
        self.failed.emit(message)

    def _collect_run(self) -> Callable[[], Any]:
        """Return the zero-arg work callable; must be overridden."""
        raise NotImplementedError(f"{type(self).__name__} must override _collect_run")

    def _apply_result(self, result: Any) -> None:
        """Write the step context from the ready result; must be overridden."""
        raise NotImplementedError(f"{type(self).__name__} must override _apply_result")
