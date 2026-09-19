"""VIRDA GUI application — IDE-style main window with PySide6.

Launches a Qt GUI for configuring and running the VIRDA electrode
localisation pipeline (Stage 1 segmentation/mesh, Stage 2 ESE and Stage 3
localization).  After a successful run the 3D viewer opens in its own tab
(with electrode overlays) and an HTML viewer can be exported.

The startup window is a project picker — open, create or pick a recent
project — and only afterwards the IDE-style main window is shown.
"""

from PySide6.QtWidgets import QApplication

from virda_gui.dialogs.project_dialog import ProjectStartDialog
from virda_gui.main_window import IdeWindow
from virda_gui.preferences import Preferences


def main() -> None:
    """Entry point for ``virda-gui``."""
    app = QApplication.instance() or QApplication([])
    prefs = Preferences()

    dialog = ProjectStartDialog(prefs)
    window = IdeWindow(prefs=prefs)
    if dialog.exec() == ProjectStartDialog.DialogCode.Accepted:
        project = dialog.project()
        if project is not None:
            window.open_project(project)
    window.show()
    app.exec()
    del app
