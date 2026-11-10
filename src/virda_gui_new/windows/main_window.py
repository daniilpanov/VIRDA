"""Main window assembling the 3D viewer over the pipeline step contexts."""

from pathlib import Path

from PySide6.QtWidgets import QMainWindow

from virda_gui_new.dto.steps import STEPS
from virda_gui_new.services.project_scan import find_project_artifacts
from virda_gui_new.services.scene_builder import collect_scene_kwargs
from virda_gui_new.services.step_seeding import seed_steps_from_artifacts
from virda_gui_new.services.steps_manager import StepsManager
from virda_gui_new.widgets.popup.main_menu import StepsMenu
from virda_gui_new.widgets.viewer3d import SceneViewer3D
from virda_gui_new.windows.widget_windows import open_widget_window


class MainWindow(QMainWindow):
    def __init__(self, project_path: Path) -> None:
        super().__init__()

        self.setWindowTitle("VIRDA: Electrode Localization System")
        self._project_path = project_path

        self._viewer3d = SceneViewer3D(self, log=self._log)
        self._viewer3d.sceneLoaded.connect(self._on_scene_loaded)
        self._viewer3d.sceneFailed.connect(self._on_scene_failed)
        self.setCentralWidget(self._viewer3d)

        self._steps = StepsManager(tuple(step.required_keys for step in STEPS))
        seed_steps_from_artifacts(self._steps, find_project_artifacts(project_path))
        self._steps.subscribe(self._rebuild_scene)
        self._rebuild_scene()

        self._steps_menu = StepsMenu(self._steps)
        self._steps_window = open_widget_window(self._steps_menu, title="Steps")
        self._steps_menu.stepRequested.connect(lambda n: self._log(f"Step {n} requested."))
        self._steps_menu.stepsInvalidated.connect(
            lambda n: self._log(f"Steps after {n} invalidated.")
        )

        self.showMaximized()

    def _rebuild_scene(self) -> None:
        kwargs = collect_scene_kwargs(self._steps)
        if kwargs:
            self._viewer3d.set_scene(**kwargs)
        else:
            self._viewer3d.clear_scene()

    def _log(self, message: str) -> None:
        self.statusBar().showMessage(message)

    def _on_scene_loaded(self, scene: object) -> None:
        self._log("3D scene loaded.")

    def _on_scene_failed(self, message: str) -> None:
        self._log(f"3D viewer failed: {message}")
