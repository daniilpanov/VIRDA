from typing import cast

from PySide6.QtWidgets import QApplication

from virda_gui_new.dialogs.project_dialog import ProjectStartDialog
from virda_gui_new.dialogs.warnings import confirm_partial_import
from virda_gui_new.services.preferences import Preferences
from virda_gui_new.services.project_import import prepare_project
from virda_gui_new.windows.main_window import MainWindow


def main() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    # Quitting with the last (primary) window closed: closing the MainWindow
    # also closes its tool popups, so the event loop ends with it.
    app.setQuitOnLastWindowClosed(True)
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
        if prepared.failures and not confirm_partial_import(
            None, prepared.failures, prepared.recoverable_step
        ):
            continue

        prefs.note_project_opened(project)

        window = MainWindow(project, prepared=prepared)
        window.show()
        app.exec()
        return
