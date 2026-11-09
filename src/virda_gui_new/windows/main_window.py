from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QMainWindow

from virda_gui_new.widgets.viewer3d import AdvancedViewer3D


class MainWindow(QMainWindow):
    def __init__(self, project_path: Path) -> None:
        super().__init__()

        self.setWindowTitle("VIRDA: Electrode Localization System")
        self._project_path = project_path

        self._viewer3d = AdvancedViewer3D(self)
        self.setCentralWidget(self._viewer3d)

        self.showMaximized()
        self.maximumSize()

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        self._viewer3d.shutdown()
        super().closeEvent(event)
