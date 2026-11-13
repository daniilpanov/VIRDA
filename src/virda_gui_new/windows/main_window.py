"""Main window assembling the 3D viewer over the pipeline step contexts."""

from collections.abc import Callable
from pathlib import Path

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow

from virda_gui_new.dto.steps import BASE_MESH, ESE_MESH, LOCALIZATION, MESH_POSTPROCESSING, STEPS
from virda_gui_new.services.project_import import PreparedProject, prepare_project
from virda_gui_new.services.scene_builder import collect_scene_kwargs
from virda_gui_new.services.step_seeding import seed_steps_from_artifacts
from virda_gui_new.services.steps_manager import StepsManager
from virda_gui_new.widgets.popup.abstract import StepPopupBase
from virda_gui_new.widgets.popup.base_mesh_extraction import BaseMeshExtractionPopup
from virda_gui_new.widgets.popup.ese_mesh import EseMeshPopup
from virda_gui_new.widgets.popup.localization import LocalizationPopup
from virda_gui_new.widgets.popup.main_menu import StepsMenu
from virda_gui_new.widgets.popup.mesh_postprocessing import MeshPostprocessingPopup
from virda_gui_new.widgets.steps_toggle_button import StepsToggleButton
from virda_gui_new.widgets.viewer3d import SceneViewer3D
from virda_gui_new.windows.widget_windows import WidgetWindow, open_widget_window


class MainWindow(QMainWindow):
    def __init__(
        self,
        project_path: Path,
        prepared: PreparedProject | None = None,
        steps: StepsManager | None = None,
    ) -> None:
        super().__init__()

        self.setWindowTitle("VIRDA: Electrode Localization System")
        self._project_path = project_path

        self._viewer3d = SceneViewer3D(self, log=self._log)
        self._viewer3d.sceneLoaded.connect(self._on_scene_loaded)
        self._viewer3d.sceneFailed.connect(self._on_scene_failed)
        self.setCentralWidget(self._viewer3d)

        if prepared is None:
            prepared = prepare_project(project_path)
        if steps is None:
            # No manager was handed over: seed one here. The seeding stops
            # at a gap in the chain and raises StepChainIncompleteError.
            steps = StepsManager(tuple(step.required_keys for step in STEPS))
            seed_steps_from_artifacts(steps, prepared.loaded)
        self._steps = steps
        self._steps.subscribe(self._rebuild_scene)
        self._steps.subscribe(self._on_steps_changed)
        self._rebuild_scene()

        self._steps_menu = StepsMenu(self._steps, project=self._project_path)
        self._steps_window: WidgetWindow | None = open_widget_window(
            self._steps_menu, parent=self, title="Steps"
        )
        self._steps_window.destroyed.connect(self._forget_steps_menu)
        self._steps_toggle = StepsToggleButton(self)
        self._steps_toggle.toggleRequested.connect(self._toggle_steps_menu)
        self._popup_factories: dict[int, Callable[[], StepPopupBase]] = {}
        self._popup_factories[1] = lambda: BaseMeshExtractionPopup(BASE_MESH, self._steps)
        self._popup_factories[2] = lambda: MeshPostprocessingPopup(MESH_POSTPROCESSING, self._steps)
        self._popup_factories[3] = lambda: EseMeshPopup(ESE_MESH, self._steps)
        self._popup_factories[4] = lambda: LocalizationPopup(LOCALIZATION, self._steps)
        self._popup_windows: dict[int, WidgetWindow] = {}
        self._popups: dict[int, StepPopupBase] = {}
        self._steps_menu.stepRequested.connect(self._open_step_popup)
        self._steps_menu.stepsInvalidated.connect(self._on_steps_invalidated)

        self.showMaximized()

    def _rebuild_scene(self) -> None:
        failures: list[str] = []
        kwargs = collect_scene_kwargs(self._steps, failures)
        for failure in failures:
            self._log(failure)
        if kwargs:
            self._viewer3d.set_scene(**kwargs)
        else:
            self._viewer3d.clear_scene()

    def _open_step_popup(self, step_n: int) -> None:
        factory = self._popup_factories.get(step_n)
        if factory is None:
            self._log(f"Step {step_n}: editor not implemented yet.")
            return
        self._close_popup(step_n)
        popup = factory()
        popup.completed.connect(self._on_step_completed)
        popup.failed.connect(lambda msg, s=step_n: self._log(f"Step {s} failed: {msg}"))
        window = open_widget_window(popup, parent=self, title=STEPS[step_n].name)

        def on_window_destroyed(*_args: object) -> None:
            self._forget_popup(step_n)

        window.destroyed.connect(on_window_destroyed)
        self._popups[step_n] = popup
        self._popup_windows[step_n] = window

    def _on_step_completed(self, step_n: int) -> None:
        self._log(f"Step {step_n} done.")
        self._close_popup(step_n)
        if step_n + 1 < len(STEPS):
            self._open_step_popup(step_n + 1)
        elif step_n == len(STEPS) - 1:
            self._log("Pipeline complete.")

    def _on_steps_changed(self) -> None:
        self._close_stale_popups(self._steps.get_current_step())

    def _on_steps_invalidated(self, n: int) -> None:
        self._log(f"Steps after {n} invalidated.")

    def _toggle_steps_menu(self) -> None:
        """Show (or re-raise) / hide the steps menu from the toggle button."""
        window = self._steps_window
        if window is None:
            return
        if window.isVisible():
            window.hide()
        else:
            window.show()
            window.raise_()
            window.activateWindow()

    def _forget_steps_menu(self) -> None:
        """Drop the steps menu reference once its window was destroyed."""
        self._steps_window = None

    def _close_stale_popups(self, step_n: int) -> None:
        """Close open popup windows of steps beyond the current step."""
        for step in [s for s in self._popup_windows if s > step_n]:
            self._close_popup(step)

    def _close_popup(self, step_n: int) -> None:
        """Shut down and close this step's popup window, if open."""
        popup = self._popups.pop(step_n, None)
        window = self._popup_windows.pop(step_n, None)
        if popup is not None:
            popup.shutdown()
        if window is not None:
            window.close()

    def _forget_popup(self, step_n: int) -> None:
        """Drop tracking of a popup window destroyed outside the dispatcher."""
        self._popup_windows.pop(step_n, None)
        self._popups.pop(step_n, None)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        """Close every tracked popup so nothing outlives the main window.

        Closing each host stops its hosted widget's background work
        (including any running :class:`~virda_gui_new.services.task_runner.
        TaskRunner` thread) before the application exits.
        """
        for step_n in list(self._popup_windows):
            self._close_popup(step_n)
        if self._steps_window is not None:
            self._steps_window.close()
        super().closeEvent(event)

    def _log(self, message: str) -> None:
        self.statusBar().showMessage(message)

    def _on_scene_loaded(self, scene: object) -> None:
        self._log("3D scene loaded.")

    def _on_scene_failed(self, message: str) -> None:
        self._log(f"3D viewer failed: {message}")
