from PySide6.QtWidgets import QApplication

from virda_gui_new.dialogs.project_dialog import ProjectStartDialog
from virda_gui_new.windows.main_window import MainWindow


def main() -> None:
    app = QApplication.instance() or QApplication([])

    dialog = ProjectStartDialog(["1", "2", "3"])
    if dialog.exec() == ProjectStartDialog.DialogCode.Accepted:
        project = dialog.project()
        if project is not None:
            window = MainWindow(project)
            window.show()
            app.exec()

    del app
