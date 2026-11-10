"""Standardized modal warning and confirmation dialogs."""

from collections.abc import Sequence

from PySide6.QtWidgets import QMessageBox, QWidget


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
