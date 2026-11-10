"""Standardized modal warning and confirmation dialogs."""

from collections.abc import Sequence

from PySide6.QtWidgets import QMessageBox, QWidget

from virda_gui_new.dto.steps import STEPS


def confirm_partial_import(
    parent: QWidget | None,
    failures: Sequence[str],
    stop_step: int,
) -> bool:
    """Ask whether to open a project whose artifact files partially failed.

    ``failures`` holds one ``"<path>: <reason>"`` string per dropped file;
    ``stop_step`` is the step the pipeline continues from (``-1`` when
    nothing was restored). ``True`` means "open it anyway".
    """
    parts: list[str] = []
    if failures:
        listed = "\n".join(f"- {item}" for item in failures)
        parts.append(f"Some project files could not be imported:\n{listed}")
    if 0 <= stop_step < len(STEPS):
        parts.append(f"The pipeline continues from step {stop_step} ({STEPS[stop_step].name}).")
    if not parts:
        return True

    answer = QMessageBox.question(
        parent,
        "Partial import",
        "\n\n".join(parts) + "\n\nOpen the project anyway?",
        QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Ok,
    )
    return answer == QMessageBox.StandardButton.Ok


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
