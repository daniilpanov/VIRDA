import pytest

from virda_gui_new.dto.steps import STEPS
from virda_gui_new.services.steps_manager import StepsManager


@pytest.fixture
def manager() -> StepsManager:
    return StepsManager(tuple(step.required_keys for step in STEPS))


@pytest.fixture
def menu(qapp, manager):
    from virda_gui_new.widgets.popup.main_menu import StepsMenu

    widget = StepsMenu(manager)
    yield widget
    widget.deleteLater()
