"""Unit tests for the steps-menu toggle button wiring.

Runs under the offscreen Qt platform. The MainWindow toggle/forget methods
are exercised on a duck-typed host (as in :mod:`tests.virda_gui.test_main_window`)
against a real ``StepsMenu`` hosted in a real top-level window, so no full
MainWindow (and no pipeline) is built.
"""

import os
from typing import cast

import pytest

from virda_gui_new.dto.steps import STEPS
from virda_gui_new.services.steps_manager import StepsManager
from virda_gui_new.widgets.popup.main_menu import StepsMenu
from virda_gui_new.widgets.steps_toggle_button import StepsToggleButton
from virda_gui_new.windows.main_window import MainWindow
from virda_gui_new.windows.widget_windows import WidgetWindow, open_widget_window


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


class _Host:
    """Duck-typed stand-in carrying only the steps-menu glue state."""

    def __init__(self) -> None:
        self._steps_window: WidgetWindow | None = None

    _toggle_steps_menu = MainWindow._toggle_steps_menu
    _forget_steps_menu = MainWindow._forget_steps_menu


@pytest.fixture
def host(qapp):
    """A host with a live steps menu window tracked like MainWindow does."""
    menu = StepsMenu(StepsManager(tuple(step.required_keys for step in STEPS)))
    host = _Host()
    steps_window = open_widget_window(menu, title="Steps")
    host._steps_window = steps_window
    steps_window.destroyed.connect(cast("MainWindow", host)._forget_steps_menu)
    yield host
    if host._steps_window is not None:
        # Deleting the window deletes its hosted menu with it; a destroyed
        # window (closed by the test) has already taken the menu down.
        host._steps_window.deleteLater()


def test_toggle_button_shows_and_hides_steps_menu(qapp, host) -> None:
    button = StepsToggleButton()
    button.toggleRequested.connect(host._toggle_steps_menu)

    assert host._steps_window.isVisible()
    button.click()
    assert not host._steps_window.isVisible()
    button.click()
    assert host._steps_window.isVisible()

    button.deleteLater()


def test_toggle_survives_hide_show_cycle_and_stays_functional(qapp, host) -> None:
    """Several toggles in a row keep the same window usable."""
    button = StepsToggleButton()
    button.toggleRequested.connect(host._toggle_steps_menu)

    for _ in range(2):
        window = host._steps_window
        assert window is not None
        button.click()
        assert not window.isVisible()
        button.click()
        assert window.isVisible()

    button.deleteLater()


def test_destroyed_steps_window_is_forgotten_and_toggle_is_guarded(qapp, host) -> None:
    button = StepsToggleButton()
    button.toggleRequested.connect(host._toggle_steps_menu)

    window = host._steps_window
    assert window is not None
    window.close()
    qapp.processEvents()

    assert host._steps_window is None
    button.click()  # no window left: a guarded no-op, not a crash

    button.deleteLater()


def test_button_pins_to_parent_corner_and_follows_resize(qapp) -> None:
    from PySide6.QtWidgets import QWidget

    parent = QWidget()
    button = StepsToggleButton(parent)
    assert button.parent() is parent

    parent.resize(500, 400)
    parent.resize(800, 600)

    assert button.pos().x() == StepsToggleButton._MARGIN
    assert button.pos().y() == StepsToggleButton._MARGIN

    parent.deleteLater()
    button.deleteLater()
