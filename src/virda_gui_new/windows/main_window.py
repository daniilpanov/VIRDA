from pathlib import Path

from PySide6.QtWidgets import QMainWindow


class MainWindow(QMainWindow):
    def __init__(self, project_path: Path) -> None:
        super().__init__()

        self.setWindowTitle("VIRDA: Electrode Localization System")
        self._project_path = project_path

        self.showMaximized()
        self.maximumSize()
