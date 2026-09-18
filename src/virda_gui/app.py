"""VIRDA GUI application — IDE-style main window with PySide6.

Launches a Qt GUI for configuring and running the VIRDA electrode
localisation pipeline (Stage 1 segmentation/mesh, Stage 2 ESE and Stage 3
localization).  After a successful run the 3D viewer opens in its own tab
(with electrode overlays) and an HTML viewer can be exported.

The left panel is a project sidebar listing all saved artifacts (mesh,
fiducials, ESE, localization, QC reports, logs) for the opened project;
files open in their own closable tabs to the right.
"""

from PySide6.QtWidgets import QApplication

from virda_gui.main_window import IdeWindow


def main() -> None:
    """Entry point for ``virda-gui``."""
    app = QApplication.instance() or QApplication([])
    window = IdeWindow()
    window.show()
    app.exec()
    del app
