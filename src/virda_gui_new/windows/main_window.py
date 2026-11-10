"""Main window assembling the 3D viewer over the pipeline step contexts."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QMainWindow

from virda_gui_new.dto.steps import BASE_MESH, ESE_MESH, MESH_POSTPROCESSING, STEPS
from virda_gui_new.services.project_scan import find_project_artifacts
from virda_gui_new.services.scene_builder import collect_scene_kwargs
from virda_gui_new.services.step_seeding import seed_steps_from_artifacts
from virda_gui_new.services.steps_manager import StepsManager
from virda_gui_new.widgets.popup.abstract import StepPopupBase
from virda_gui_new.widgets.popup.base_mesh_extraction import BaseMeshExtractionPopup
from virda_gui_new.widgets.popup.ese_mesh import EseMeshPopup
from virda_gui_new.widgets.popup.main_menu import StepsMenu
from virda_gui_new.widgets.popup.mesh_postprocessing import MeshPostprocessingPopup
from virda_gui_new.widgets.viewer3d import SceneViewer3D
from virda_gui_new.windows.widget_windows import WidgetWindow, open_widget_window


class MainWindow(QMainWindow):
    # The file tree widget arrives with the project file tree step; declaring
    # the attribute here keeps the steps-invalidated handler typed before it.
    _filetree: Any

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
        self._steps.subscribe(self._on_steps_changed)
        self._rebuild_scene()

        self._steps_menu = StepsMenu(self._steps)
        self._steps_window = open_widget_window(self._steps_menu, title="Steps")
        self._popup_factories: dict[int, Callable[[], StepPopupBase]] = {}
        self._popup_factories[1] = lambda: BaseMeshExtractionPopup(BASE_MESH, self._steps)
        self._popup_factories[2] = lambda: MeshPostprocessingPopup(MESH_POSTPROCESSING, self._steps)
        self._popup_factories[3] = lambda: EseMeshPopup(ESE_MESH, self._steps)
        self._popup_windows: dict[int, WidgetWindow] = {}
        self._popups: dict[int, StepPopupBase] = {}
        self._steps_menu.stepRequested.connect(self._open_step_popup)
        self._steps_menu.stepsInvalidated.connect(self._on_steps_invalidated)

        self.showMaximized()

    def _rebuild_scene(self) -> None:
        kwargs = collect_scene_kwargs(self._steps)
        if kwargs:
            self._viewer3d.set_scene(**kwargs)
        else:
            self._viewer3d.clear_scene()

    def _open_step_popup(self, step_n: int) -> None:
        factory = self._popup_factories.get(step_n)
        if factory is None:
            self._log(f"Step {step_n}: editor not implemented yet.")
            return
        old_window = self._popup_windows.get(step_n)
        if old_window is not None:
            self._popups[step_n].shutdown()
            old_window.close()
            del self._popups[step_n]
            del self._popup_windows[step_n]
        popup = factory()
        popup.completed.connect(self._on_step_completed)
        popup.failed.connect(lambda msg, s=step_n: self._log(f"Step {s} failed: {msg}"))
        window = open_widget_window(popup, title=STEPS[step_n].name)
        self._popups[step_n] = popup
        self._popup_windows[step_n] = window

    def _on_step_completed(self, step_n: int) -> None:
        self._log(f"Step {step_n} done.")
        if step_n + 1 < len(STEPS):
            self._open_step_popup(step_n + 1)
        elif step_n == len(STEPS) - 1:
            self._log("Pipeline complete.")

    def _on_steps_changed(self) -> None:
        self._close_stale_popups(self._steps.get_current_step())

    def _on_steps_invalidated(self, n: int) -> None:
        self._log(f"Steps after {n} invalidated.")
        self._filetree.set_artifacts(find_project_artifacts(self._project_path))

    def _close_stale_popups(self, step_n: int) -> None:
        """Close open popup windows of steps beyond the current step."""
        for step in [s for s in self._popup_windows if s > step_n]:
            self._popups.pop(step).shutdown()
            self._popup_windows.pop(step).close()

    def _log(self, message: str) -> None:
        self.statusBar().showMessage(message)

    def _on_scene_loaded(self, scene: object) -> None:
        self._log("3D scene loaded.")

    def _on_scene_failed(self, message: str) -> None:
        self._log(f"3D viewer failed: {message}")
