"""IDE-style main window: file sidebar, closable tabs and project management."""

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QWidget,
)

from virda.logging_setup import add_log_handler, remove_log_handler

from .constants import ADVANCED_FIELD_DEFAULTS
from .project import create_project
from .services.pipeline_runner import PipelineRunner
from .sidebar import ProjectSidebar
from .state import AppState
from .tabs.config_tab import ConfigTab
from .viewer.viewer import ViewerWidget

_TAB_RUN = "run-pipeline"
_TAB_VIEWER = "3d-viewer"


class IdeWindow(QMainWindow):
    """IDE-style main window of the VIRDA GUI.

    The left panel is a :class:`~virda_gui.sidebar.ProjectSidebar` listing
    the project artifacts; the right panel is a closable tab bar where the
    run pipeline form, the 3D viewer and individual project files open in
    their own tabs.  The window owns the :class:`PipelineRunner`, the shared
    :class:`AppState` and the log stream forwarded to the active run tab.
    """

    def __init__(self) -> None:
        super().__init__()
        self._project: Path | None = None
        self._viewer_widget: ViewerWidget | None = None

        self.setWindowTitle("VIRDA — Electrode Localization System")
        self.resize(1100, 720)

        self._state = AppState(advanced=dict(ADVANCED_FIELD_DEFAULTS))

        self._pipe_runner = PipelineRunner(self._state)
        add_log_handler(self._pipe_runner.log_handler)
        self._pipe_runner.finished.connect(self._on_pipeline_done)
        self._pipe_runner.failed.connect(self._on_pipeline_error)
        self._pipe_runner.exportDone.connect(
            lambda: self._config_tab.log_viewer.append("HTML export completed.")
        )
        self._pipe_runner.exportFailed.connect(
            lambda: self._config_tab.log_viewer.append("HTML export failed — see log above.")
        )

        self._config_tab = ConfigTab(self._state)
        self._config_tab.runRequested.connect(self._on_run)
        self._config_tab.openViewer.connect(self._on_open_viewer)
        self._config_tab.exportHtml.connect(self._on_export_html)

        self._sidebar = ProjectSidebar(self)
        self._sidebar.openViewerRequested.connect(self._on_open_viewer)
        self._sidebar.runPipelineRequested.connect(self._show_run_tab)

        self._tabs = QTabWidget(self)
        self._tabs.setTabsClosable(True)
        self._tabs.setDocumentMode(True)
        self._tabs.tabCloseRequested.connect(self._close_tab)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._sidebar)
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 800])
        self.setCentralWidget(splitter)

        self._build_menu()
        self.statusBar().showMessage("")

        self._poll_timer = QTimer()
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._poll_log_queue)
        self._poll_timer.start()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        new_action = QAction("&New project...", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._create_project)
        file_menu.addAction(new_action)

        open_action = QAction("&Open project...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_project_dialog)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        self._close_action = QAction("&Close project", self)
        self._close_action.setEnabled(False)
        self._close_action.triggered.connect(self.close_project)
        file_menu.addAction(self._close_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    # ------------------------------------------------------------------
    # Project management
    # ------------------------------------------------------------------

    def project(self) -> Path | None:
        """Return the open project directory, or None when none is open."""
        return self._project

    def open_project(self, project: Path) -> None:
        """Open *project*: populate the sidebar and show the run pipeline tab."""
        self._project = project
        self._sidebar.set_project(project)
        self._close_action.setEnabled(True)
        self._state.last_project_dir = str(project)
        self._config_tab.set_project_dir(project)
        self.setWindowTitle(f"VIRDA — {project.name}")
        self._show_run_tab()
        self.statusBar().showMessage(f"Project opened: {project}", 5000)

    def close_project(self) -> None:
        """Close the project and reset the window to the empty state."""
        self._project = None
        self._sidebar.set_project(None)
        self._close_action.setEnabled(False)
        self._state.last_project_dir = None
        self._tabs.removeTab(self._tabs.indexOf(self._config_tab))
        self.setWindowTitle("VIRDA — Electrode Localization System")

    def _create_project(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Create new project")
        if not directory:
            return
        project = create_project(Path(directory))
        self.open_project(project)

    def _open_project_dialog(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Open project")
        if not directory:
            return
        project = Path(directory)
        if not project.is_dir():
            QMessageBox.warning(self, "Project error", f"Not a directory:\n{project}")
            return
        self.open_project(project)

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------

    def _add_tab(self, widget: QWidget, title: str) -> None:
        """Add or activate the tab holding *widget* (single tab per widget)."""
        index = self._tabs.indexOf(widget)
        if index < 0:
            index = self._tabs.addTab(widget, title)
        else:
            self._tabs.setTabText(index, title)
        self._tabs.setCurrentIndex(index)

    def _close_tab(self, index: int) -> None:
        # The widget is intentionally kept alive so its state (e.g. the run
        # form contents) survives closing and reopening the tab.
        self._tabs.removeTab(index)

    def _show_run_tab(self) -> None:
        self._add_tab(self._config_tab, "Run Pipeline")

    # ------------------------------------------------------------------
    # Pipeline execution (background thread)
    # ------------------------------------------------------------------

    def _on_run(self) -> None:
        try:
            config = self._config_tab.collect_config()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user as a dialog
            QMessageBox.critical(self, "Validation error", str(exc))
            return

        self._config_tab.run_btn.setEnabled(False)
        self._config_tab.viewer_btn.setEnabled(False)
        self._config_tab.export_btn.setEnabled(False)
        self._config_tab.log_viewer.clear()
        self._state.log_queue.put("Starting pipeline...")
        self._state.last_project_dir = config.project_dir
        self._state.stage3_summary = None

        measurements_path = self._config_tab.measurements.get().strip() or None
        if measurements_path and not Path(measurements_path).is_file():
            QMessageBox.critical(
                self,
                "Measurements error",
                f"Measurements file not found:\n{measurements_path}",
            )
            self._config_tab.run_btn.setEnabled(True)
            self._config_tab.viewer_btn.setEnabled(True)
            self._config_tab.export_btn.setEnabled(True)
            return

        self._pipe_runner.submit(config, measurements_path)

    # ------------------------------------------------------------------
    # Log queue polling (main thread)
    # ------------------------------------------------------------------

    def _poll_log_queue(self) -> None:
        self._pipe_runner.poll(self._config_tab.log_viewer.append)

    def _on_pipeline_done(self) -> None:
        added = self._config_tab.ensure_stage3_electrodes_group()
        if added:
            self._config_tab.log_viewer.append(f"Electrode group added: {added}")
        self._config_tab.run_btn.setEnabled(True)
        self._config_tab.viewer_btn.setEnabled(True)
        self._config_tab.export_btn.setEnabled(True)
        if self._state.last_project_dir:
            self._sidebar.set_project(Path(self._state.last_project_dir))
        self._update_results_info(success=True)

    def _on_pipeline_error(self) -> None:
        self._config_tab.run_btn.setEnabled(True)
        self._config_tab.viewer_btn.setEnabled(True)
        self._config_tab.export_btn.setEnabled(True)
        self._update_results_info(success=False)

    def _update_results_info(self, *, success: bool) -> None:
        project = self._state.last_project_dir
        if success:
            if project:
                self._open_viewer(Path(project))
            summary = self._state.stage3_summary
            if summary:
                shift = summary["offset_shift_mm"]
                shift_text = f", offset shift {shift:.2f} mm" if shift is not None else ""
                self._config_tab.log_viewer.append(
                    "Pipeline completed. "
                    f"Stage 3: {summary['localized']}/{summary['total']} electrodes "
                    f"localized ({summary['flagged']} flagged{shift_text})."
                )
            elif project:
                self._config_tab.log_viewer.append(
                    f"Pipeline completed. Project directory: {project}"
                )
        else:
            self._config_tab.log_viewer.append("Pipeline failed. Check the log for details.")

    # ------------------------------------------------------------------
    # 3D viewer
    # ------------------------------------------------------------------

    def _build_viewer_widget(self) -> None:
        if self._viewer_widget is not None:
            return
        self._viewer_widget = ViewerWidget(log=self._state.log_queue.put)
        self._viewer_widget.sceneLoaded.connect(self._on_viewer_scene_loaded)
        self._viewer_widget.sceneFailed.connect(self._on_viewer_scene_failed)

    def _on_open_viewer(self) -> None:
        resolved = self._state.last_project_dir or self._project
        if not resolved:
            QMessageBox.warning(
                self,
                "No project directory",
                "No project directory selected. Run the pipeline or pick a Project dir.",
            )
            return
        self._open_viewer(Path(resolved))

    def _open_viewer(self, project: Path) -> None:
        kwargs = self._collect_viewer_kwargs(project)
        if not kwargs:
            QMessageBox.warning(
                self, "Nothing to view", "No mesh or NIfTI file found in the project."
            )
            return
        if self._state.viewer_loading:
            QMessageBox.warning(
                self,
                "Viewer still loading",
                "The 3D viewer is still loading a scene. Wait for it to finish.",
            )
            return

        self._config_tab.log_viewer.append("Opening 3D viewer...")
        self._config_tab.viewer_btn.setEnabled(False)
        self._state.viewer_loading = True
        self._build_viewer_widget()
        self._add_tab(self._viewer_widget, "3D Viewer")
        self._tabs.setCurrentWidget(self._viewer_widget)
        self._viewer_widget.load(**kwargs)

    def _collect_viewer_kwargs(self, project: Path) -> dict[str, Any]:
        mesh_path = project / "mesh" / "final_mesh.ply"
        fiducials_path = project / "input" / "fiducials.json"
        normals_path = project / "ese" / "normals.npy"

        nifti = self._config_tab.nifti_path()
        kwargs: dict[str, Any] = {}
        if nifti:
            kwargs["nifti_path"] = nifti
        else:
            for pattern in ("input/*.nii.gz", "input/*.nii"):
                found = sorted(project.glob(pattern))
                if found:
                    kwargs["nifti_path"] = str(found[0])
                    break
        if mesh_path.exists():
            kwargs["mesh_path"] = str(mesh_path)
        if fiducials_path.exists():
            kwargs["fiducials_path"] = str(fiducials_path)
        if normals_path.exists():
            kwargs["normals_path"] = str(normals_path)

        self._config_tab.ensure_stage3_electrodes_group(project)
        electrode_specs = self._config_tab.collect_electrode_specs()
        if electrode_specs:
            kwargs["electrode_specs"] = electrode_specs
            kwargs["electrodes_cras"] = self._config_tab.electrodes_cras_check.isChecked()
        return kwargs

    def _on_viewer_scene_loaded(self, _scene: Any) -> None:
        self._state.viewer_loading = False
        self._config_tab.log_viewer.append("3D viewer scene loaded.")
        self._config_tab.viewer_btn.setEnabled(True)

    def _on_viewer_scene_failed(self, message: str) -> None:
        self._state.viewer_loading = False
        self._config_tab.viewer_btn.setEnabled(True)
        self._config_tab.log_viewer.append(f"3D viewer failed: {message}")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_export_html(self) -> None:
        resolved = self._state.last_project_dir or self._project
        if not resolved:
            QMessageBox.warning(
                self,
                "No project directory",
                "No project directory selected. Run the pipeline or pick a Project dir.",
            )
            return
        self._export_html(Path(resolved))

    def _export_html(self, project: Path) -> None:
        output = project / "viewer.html"
        self._config_tab.log_viewer.append(f"Exporting HTML viewer to {output}...")
        self._pipe_runner.start_html_export(project, output)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        try:
            self._on_close()
        finally:
            super().closeEvent(event)

    def _on_close(self) -> None:
        self._state.closed = True
        self._poll_timer.stop()
        if self._viewer_widget is not None:
            self._viewer_widget.shutdown()
        self._pipe_runner.shutdown()
        remove_log_handler(self._pipe_runner.log_handler)
