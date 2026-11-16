"""Entry point of the new GUI: pick a project, seed its steps, open the window."""

from typing import cast

from PySide6.QtWidgets import QApplication

from virda_gui_new.dialogs.project_dialog import ProjectStartDialog
from virda_gui_new.services.preferences import Preferences
from virda_gui_new.windows.main_window import open_project_window


def main() -> None:
    app = cast(QApplication, QApplication.instance() or QApplication([]))
    # Quitting with the last (primary) window closed: closing the MainWindow
    # also closes its tool popups, so the event loop ends with it.
    app.setQuitOnLastWindowClosed(True)
    prefs = Preferences()

    while True:
        recent = [str(entry) for entry in prefs.recent_project_paths()]
        dialog = ProjectStartDialog(recent)
        if dialog.exec() != ProjectStartDialog.DialogCode.Accepted:
            return

        project = dialog.project()
        if project is None:
            return

        # The project is opened the same way the File menu opens it:
        # the import is checked with the user and the window follows.
        if open_project_window(project) is None:
            continue

        app.exec()
        return
