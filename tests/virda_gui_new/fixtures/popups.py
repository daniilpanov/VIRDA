import pytest

from virda_gui_new.widgets.popup.abstract import StepPopupBase


@pytest.fixture
def popups(qapp):
    """Collect popups and stop their runner threads at teardown."""
    created: list[StepPopupBase] = []
    yield created
    for popup in created:
        popup.shutdown()
