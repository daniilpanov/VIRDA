"""Entry point of the new GUI: pick a project, seed its steps, open the window."""

from PySide6.QtWidgets import QApplication

from virda_gui_new.dialogs.project_dialog import ProjectStartDialog
from virda_gui_new.dialogs.warnings import confirm_partial_import
from virda_gui_new.dto.steps import STEPS
from virda_gui_new.services.preferences import Preferences
from virda_gui_new.services.project_import import prepare_project
from virda_gui_new.services.step_seeding import StepChainIncompleteError, seed_steps_from_artifacts
from virda_gui_new.services.steps_manager import StepsManager
from virda_gui_new.windows.main_window import MainWindow


def main() -> None:
    app = QApplication.instance() or QApplication([])
    prefs = Preferences()

    while True:
        recent = [str(entry) for entry in prefs.recent_projects() if entry.is_dir()]
        dialog = ProjectStartDialog(recent)
        if dialog.exec() != ProjectStartDialog.DialogCode.Accepted:
            return

        project = dialog.project()
        if project is None:
            return

        prepared = prepare_project(project)
        # The steps are seeded here, before the window exists: a project
        # with a gap in its chain stops at that step, and the user decides
        # in the dialog whether to continue from there.
        steps = StepsManager(tuple(step.required_keys for step in STEPS))
        incomplete: StepChainIncompleteError | None = None
        try:
            seed_steps_from_artifacts(steps, prepared.loaded)
        except StepChainIncompleteError as error:
            incomplete = error
        if (prepared.failures or incomplete) and not confirm_partial_import(
            None, prepared.failures, steps.get_current_step(), incomplete
        ):
            continue

        prefs.note_project_opened(project)

        window = MainWindow(project, prepared=prepared, steps=steps)
        window.show()
        app.exec()
        return
