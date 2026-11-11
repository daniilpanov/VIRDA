from PySide6.QtWidgets import QApplication

from virda_gui_new.dialogs.project_dialog import ProjectStartDialog
from virda_gui_new.dialogs.warnings import confirm_partial_import
from virda_gui_new.services.project_import import prepare_project
from virda_gui_new.windows.main_window import MainWindow


def main() -> None:
    app = QApplication.instance() or QApplication([])

    while True:
        dialog = ProjectStartDialog(["1", "2", "3"])
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

        window = MainWindow(project, artifacts=prepared.artifacts)
        window.show()
        app.exec()
        return
