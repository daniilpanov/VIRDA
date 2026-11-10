"""Standardized modal warning and confirmation dialogs."""

from collections.abc import Sequence

from PySide6.QtWidgets import QMessageBox, QWidget

from virda_gui_new.dto.steps import STEPS


def confirm_partial_import(
    parent: QWidget | None,
    failures: Sequence[str],
    recoverable_step: int,
) -> bool:
    """Ask whether to open a project whose artifact files partially failed.

    ``failures`` holds one ``"<path>: <reason>"`` string per dropped file;
    ``recoverable_step`` is the last pipeline step still covered by the
    loaded artifacts (``-1`` when nothing could be restored).
    """
    listed = "\n".join(f"- {item}" for item in failures)
    if recoverable_step >= 0:
        step_name = STEPS[recoverable_step].name
        recovery = f"Recovery is available up to step {recoverable_step} ({step_name})."
    else:
        recovery = "No pipeline artifacts could be restored."

    answer = QMessageBox.question(
        parent,
        "Partial import",
        f"Some project files could not be imported:\n{listed}\n\n{recovery}\n\nContinue?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return answer == QMessageBox.StandardButton.Yes


def confirm_step_redo(
    parent: QWidget | None,
    step_name: str,
    discarded_step_names: Sequence[str],
) -> bool:
    """Ask whether to re-run a completed step and discard later step results."""
    names = list(discarded_step_names)
    if names:
        listed = "\n".join(f"- {name}" for name in names)
        text = f"Re-running discards the results of later steps:\n{listed}"
    else:
        text = "No later steps to discard."

    answer = QMessageBox.question(
        parent,
        f"Re-run {step_name}?",
        text,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return answer == QMessageBox.StandardButton.Yes


def show_warning(parent: QWidget | None, title: str, text: str) -> None:
    """Show a standardized modal warning dialog."""
    QMessageBox.warning(parent, title, text)
