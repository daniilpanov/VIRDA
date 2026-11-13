"""Unit tests for the file-tree toggle button wiring.

Runs under the offscreen Qt platform. The button is exercised on its own
against a real parent window, so no full MainWindow (and no pipeline) is
built.
"""

import os

import pytest

from virda_gui_new.widgets.files_toggle_button import FilesToggleButton
from virda_gui_new.widgets.steps_toggle_button import StepsToggleButton


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


def test_files_button_requests_a_toggle(qapp) -> None:
    """Clicking the pill only asks for a toggle, it touches no window."""
    button = FilesToggleButton()
    toggled: list[bool] = []
    button.toggleRequested.connect(lambda: toggled.append(True))

    button.click()

    assert toggled == [True]

    button.deleteLater()


def test_files_button_sits_below_the_steps_button(qapp) -> None:
    """The Files pill is anchored under the steps pill, below the menu bar."""
    from PySide6.QtWidgets import QMainWindow

    parent = QMainWindow()
    parent.menuBar().addMenu("&File")
    steps = StepsToggleButton(parent)
    files = FilesToggleButton(parent, below=steps)
    parent.show()
    qapp.processEvents()

    menu_bar = parent.menuBar()
    assert files.text() == "Files"
    assert files.geometry().top() > steps.geometry().bottom()
    assert files.geometry().top() > menu_bar.geometry().bottom()
    assert files.pos().x() == steps.pos().x()

    parent.deleteLater()


def test_files_button_follows_the_steps_button(qapp) -> None:
    """The Files pill follows the anchor pill when the anchor moves."""
    from PySide6.QtWidgets import QMainWindow

    parent = QMainWindow()
    parent.menuBar().addMenu("&File")
    steps = StepsToggleButton(parent)
    files = FilesToggleButton(parent, below=steps)
    parent.show()
    qapp.processEvents()
    before = files.pos()

    # Without a menu bar the steps pill moves up, the Files pill follows.
    parent.menuBar().hide()
    qapp.processEvents()

    assert steps.pos().y() == StepsToggleButton._MARGIN
    assert files.geometry().top() > steps.geometry().bottom()
    assert files.pos().y() < before.y()

    parent.deleteLater()
