"""Clickable pipeline stepper bar for the guided workflow.

Sits above the tab bar of the main window; each chip reflects one
:class:`~virda_gui.pipeline_status.PipelineStep` and navigates to the tab
where that step is performed.  Plain-language titles, ASCII marks only.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from virda_gui.pipeline_status import PipelineStep

_MARKS = {
    "done": "[ok]",
    "unsaved": "[!]",
    "active": "[>]",
    "waiting": "[ ]",
}


class PipelineBar(QWidget):
    """Four-step pipeline navigator driven by :func:`set_steps`."""

    stepActivated = Signal(str)  # noqa: N815 - step key

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 4, 8, 4)
        self._layout.setSpacing(6)
        self._buttons: list[QPushButton] = []
        self._key_buttons: dict[str, QPushButton] = {}
        self._steps: list[PipelineStep] = []
        self._current: str | None = None

    def is_enabled(self, key: str) -> bool:
        """Whether the step with *key* is currently unlocked."""
        return next((step.enabled for step in self._steps if step.key == key), False)

    def current_key(self) -> str | None:
        """The highlighted step key, or None when no pipeline tab is active."""
        return self._current

    def set_current(self, key: str | None) -> None:
        """Highlight the chip of the active tab; None clears the highlight."""
        self._current = key
        for step_key, button in self._key_buttons.items():
            font = button.font()
            font.setBold(step_key == key)
            button.setFont(font)

    def set_steps(self, steps: list[PipelineStep]) -> None:
        """Rebuild the chips from the given pipeline steps."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._buttons = []
        self._key_buttons = {}
        self._steps = list(steps)
        for step in steps:
            button = QPushButton(f"{step.title}\n{_MARKS[step.state]}", self)
            button.setToolTip(step.detail)
            button.setEnabled(step.enabled)
            button.clicked.connect(
                lambda _checked=False, key=step.key: self.stepActivated.emit(key)
            )
            if step.state == "active":
                button.setDefault(True)
            self._layout.addWidget(button, 1)
            self._buttons.append(button)
            self._key_buttons[step.key] = button
        self.set_current(self._current)
