from PySide6.QtWidgets import QApplication

from virda_gui_new.dialogs.project_dialog import ProjectStartDialog
from virda_gui_new.services.preferences import Preferences
from virda_gui_new.windows.main_window import MainWindow


def main() -> None:
    app = QApplication.instance() or QApplication([])
    prefs = Preferences()

    recent = [str(entry) for entry in prefs.recent_projects() if entry.is_dir()]
    dialog = ProjectStartDialog(recent)
    if dialog.exec() != ProjectStartDialog.DialogCode.Accepted:
        return

    project = dialog.project()
    if project is None:
        return

    prefs.note_project_opened(project)

    window = MainWindow(project)
    window.show()
    app.exec()
    return
