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


def advance(manager: StepsManager) -> None:
    """Finish the current step and move to the next one.

    The manager refuses to leave an unfinished step, so the missing
    required keys of the current step are filled with a placeholder
    first -- like a finished pipeline step would hold them.
    """
    step_n = manager.get_current_step()
    for key in STEPS[step_n].required_keys:
        if manager.get_step_context_item(step_n, key) is None:
            manager.set_step_context_item(step_n, key, "ready")
    manager.next_step()


def advance_to(manager: StepsManager, step_n: int) -> None:
    """Move the manager to ``step_n``, finishing the steps on the way."""
    while manager.get_current_step() < step_n:
        advance(manager)
