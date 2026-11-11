"""Unit tests for :mod:`virda_gui_new.dialogs.warnings`.

The static ``QMessageBox`` methods are stubbed, so no real dialog pops up;
the tests only assert the title/text/buttons built by the module and the
returned choice.
"""

import os
from typing import Any

import pytest

from virda_gui_new.dialogs.warnings import confirm_partial_import, confirm_step_redo, show_warning


@pytest.fixture
def qapp():
    """Yield an offscreen ``QApplication``, skipping without a Qt install."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on local Qt install
        pytest.skip(f"Qt platform unavailable: {exc}")

    app = QApplication.instance() or QApplication([])
    return app


def test_confirm_step_redo_true_lists_discards(qapp, monkeypatch) -> None:
    """Yes returns True and the dialog lists each later step by name."""
    from PySide6.QtWidgets import QMessageBox

    captured: dict[str, Any] = {}

    def fake_question(parent, title, text, buttons=None, default=None):
        captured.update(parent=parent, title=title, text=text, buttons=buttons, default=default)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))

    discard = ["Mesh postprocessing", "ESE mesh", "Localization"]
    assert confirm_step_redo(None, "Base mesh", discard) is True
    assert captured["parent"] is None
    assert captured["title"] == "Re-run Base mesh?"
    assert captured["text"] == (
        "Re-running discards the results of later steps:\n"
        "- Mesh postprocessing\n- ESE mesh\n- Localization"
    )
    assert captured["buttons"] == QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    assert captured["default"] == QMessageBox.StandardButton.No


def test_confirm_step_redo_false_passthrough(qapp, monkeypatch) -> None:
    """No (the default) returns False."""

    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.No),
    )
    assert confirm_step_redo(None, "NIfTI import", ["Base mesh"]) is False


def test_confirm_step_redo_empty_discard_list(qapp, monkeypatch) -> None:
    """An empty discard list still builds a graceful dialog."""

    from PySide6.QtWidgets import QMessageBox

    captured: dict[str, Any] = {}

    def fake_question(parent, title, text, buttons=None, default=None):
        captured.update(title=title, text=text, default=default)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))

    assert confirm_step_redo(None, "Localization", []) is True
    assert captured["title"] == "Re-run Localization?"
    assert captured["text"] == "No later steps to discard."
    assert captured["default"] == QMessageBox.StandardButton.No


def test_show_warning_delegates(qapp, monkeypatch) -> None:
    """show_warning forwards title and text to a modal QMessageBox.warning."""
    from PySide6.QtWidgets import QMessageBox

    captured: dict[str, Any] = {}

    def fake_warning(parent, title, text, *a, **k):
        captured.update(parent=parent, title=title, text=text)
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(fake_warning))

    assert show_warning(None, "VIRDA warning", "Something failed.") is None  # type: ignore[func-returns-value]
    assert captured["parent"] is None
    assert captured["title"] == "VIRDA warning"
    assert captured["text"] == "Something failed."


def test_confirm_partial_import_yes_lists_failures_and_step(qapp, monkeypatch) -> None:
    """Yes returns True; the dialog lists each dropped file and the recovery step."""
    from PySide6.QtWidgets import QMessageBox

    captured: dict[str, Any] = {}

    def fake_question(parent, title, text, buttons=None, default=None):
        captured.update(parent=parent, title=title, text=text, default=default)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))

    failures = ["localization/electrodes_scalp.csv: no electrode points found"]
    assert confirm_partial_import(None, failures, 2) is True
    assert captured["parent"] is None
    assert captured["title"] == "Partial import"
    assert captured["text"] == (
        "Some project files could not be imported:\n"
        "- localization/electrodes_scalp.csv: no electrode points found\n\n"
        "Recovery is available up to step 2 (Mesh postprocessing).\n\nContinue?"
    )
    assert captured["default"] == QMessageBox.StandardButton.No


def test_confirm_partial_import_nothing_restored(qapp, monkeypatch) -> None:
    """With no recoverable step the dialog says so and No returns False."""
    from PySide6.QtWidgets import QMessageBox

    captured: dict[str, Any] = {}

    def fake_question(parent, title, text, buttons=None, default=None):
        captured.update(text=text)
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))

    assert confirm_partial_import(None, ["input/a.nii: unreadable"], -1) is False
    assert "No pipeline artifacts could be restored." in captured["text"]
