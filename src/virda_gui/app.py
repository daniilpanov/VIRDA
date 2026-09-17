"""VIRDA GUI application — main window with PySide6.

Launches a Qt GUI for configuring and running the VIRDA electrode
localisation pipeline (Stage 1 segmentation/mesh, Stage 2 ESE and Stage 3
localization).  After a successful run the 3D viewer opens automatically
(with electrode overlays) and an HTML viewer can be exported.

The *Saved Results* tab browses the artifacts of any project directory:
it lists everything saved by previous pipeline runs (mesh, fiducials, ESE,
localization, QC reports, logs) with previews and one-click access to the 3D
viewer / HTML export for that project.

Electrode overlay groups (Stage 3 ``electrodes.json`` or tabular TSV/CSV
tables) can be managed in the *Electrode Groups* section; after a run with
measurements the localized ``localization/electrodes.json`` is added
automatically. Fiducials from an MNE ``coordsystem.json`` loaded as the
config file are passed to the pipeline automatically.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from virda.logging_setup import add_log_handler, remove_log_handler

from .config_tab import ConfigTab
from .constants import ADVANCED_FIELD_DEFAULTS
from .pipeline_runner import PipelineRunner
from .results_tab import ResultsTab
from .state import AppState
from .viewer import ViewerWidget


class _VirdaMainWindow(QMainWindow):
    """Main window whose :meth:`closeEvent` triggers the app teardown hook."""

    def __init__(self, on_close: Callable[[], None]) -> None:
        super().__init__()
        self._on_close_callback = on_close

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        try:
            self._on_close_callback()
        finally:
            super().closeEvent(event)


class VirdaApp(QObject):
    """Main application window.

    Subclasses :class:`QObject` so slots connected to background-worker signals
    are queued to the GUI thread (a plain class would invoke them in the
    emitting thread and touch Qt widgets off-thread).
    """

    def __init__(self) -> None:
        super().__init__()
        self._root = _VirdaMainWindow(self._on_close)
        self._root.setWindowTitle("VIRDA — Electrode Localization System")
        self._root.resize(860, 640)

        self._state = AppState(
            advanced=dict(ADVANCED_FIELD_DEFAULTS),
        )

        # Capture pipeline/library logs into the log pane (console handlers
        # set up by the pipeline itself keep working).
        self._pipe_runner = PipelineRunner(self._state)
        add_log_handler(self._pipe_runner.log_handler)
        self._pipe_runner.finished.connect(self._on_pipeline_done)
        self._pipe_runner.failed.connect(self._on_pipeline_error)
        self._pipe_runner.exportDone.connect(
            lambda: self._log_viewer.append("HTML export completed.")
        )
        self._pipe_runner.exportFailed.connect(
            lambda: self._log_viewer.append("HTML export failed — see log above.")
        )

        self._build_ui()
        self._poll_timer = QTimer()
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._poll_log_queue)
        self._poll_timer.start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._notebook = QTabWidget()
        self._root.setCentralWidget(self._notebook)

        self._config_tab = ConfigTab(self._state)
        self._notebook.addTab(self._config_tab, "  Configuration  ")
        self._config_tab.runRequested.connect(self._on_run)
        self._config_tab.openViewer.connect(self._on_open_viewer)
        self._config_tab.exportHtml.connect(self._on_export_html)
        self._log_viewer = self._config_tab.log_viewer

        self._results_tab = ResultsTab(self._state)
        self._notebook.addTab(self._results_tab, "  Saved Results  ")
        self._results_tab.openViewer.connect(self._open_viewer)
        self._results_tab.exportHtml.connect(self._export_html)

        self._viewer_frame = QWidget()
        self._viewer_tab_index = self._notebook.addTab(self._viewer_frame, "  3D Viewer  ")
        self._notebook.setTabEnabled(self._viewer_tab_index, False)
        self._build_viewer_tab()

    # ---- 3D Viewer tab ----

    def _build_viewer_tab(self) -> None:
        parent = self._viewer_frame
        outer = QVBoxLayout(parent)
        outer.setContentsMargins(0, 0, 0, 0)

        self._viewer_widget = ViewerWidget(parent, log=self._state.log_queue.put)
        self._viewer_widget.sceneLoaded.connect(self._on_viewer_scene_loaded)
        self._viewer_widget.sceneFailed.connect(self._on_viewer_scene_failed)
        outer.addWidget(self._viewer_widget)

    # ------------------------------------------------------------------
    # Pipeline execution (background thread)
    # ------------------------------------------------------------------

    def _on_run(self) -> None:
        try:
            config = self._config_tab.collect_config()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user as a dialog
            QMessageBox.critical(self._root, "Validation error", str(exc))
            return

        self._config_tab.run_btn.setEnabled(False)
        self._config_tab.viewer_btn.setEnabled(False)
        self._config_tab.export_btn.setEnabled(False)
        self._log_viewer.clear()
        self._state.log_queue.put("Starting pipeline...")
        self._state.last_project_dir = config.project_dir
        self._state.stage3_summary = None

        measurements_path = self._config_tab.measurements.get().strip() or None
        if measurements_path and not Path(measurements_path).is_file():
            QMessageBox.critical(
                self._root,
                "Measurements error",
                f"Measurements file not found:\n{measurements_path}",
            )
            self._on_pipeline_error()
            return

        self._pipe_runner.submit(config, measurements_path)

    # ------------------------------------------------------------------
    # Log queue polling (main thread)
    # ------------------------------------------------------------------

    def _poll_log_queue(self) -> None:
        self._pipe_runner.poll(self._log_viewer.append)

    def _on_pipeline_done(self) -> None:
        added = self._config_tab.ensure_stage3_electrodes_group()
        if added:
            self._log_viewer.append(f"Electrode group added: {added}")
        self._config_tab.run_btn.setEnabled(True)
        self._config_tab.viewer_btn.setEnabled(True)
        self._config_tab.export_btn.setEnabled(True)
        if self._state.last_project_dir:
            self._results_tab.set_project_dir(self._state.last_project_dir)
            self._results_tab.refresh()
        self._update_results_info(success=True)

    def _on_pipeline_error(self) -> None:
        self._config_tab.run_btn.setEnabled(True)
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
                self._log_viewer.append(
                    "Pipeline completed. "
                    f"Stage 3: {summary['localized']}/{summary['total']} electrodes "
                    f"localized ({summary['flagged']} flagged{shift_text})."
                )
            elif project:
                self._log_viewer.append(f"Pipeline completed. Project directory: {project}")
        else:
            self._log_viewer.append("Pipeline failed. Check the log for details.")

    # ------------------------------------------------------------------
    # 3D viewer
    # ------------------------------------------------------------------

    def _open_viewer(self, project: Path) -> None:
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

        if not kwargs:
            QMessageBox.warning(
                self._root, "Nothing to view", "No mesh or NIfTI file found in the project."
            )
            return

        if self._state.viewer_loading:
            QMessageBox.warning(
                self._root,
                "Viewer still loading",
                "The 3D viewer is still loading a scene. Wait for it to finish.",
            )
            return

        self._log_viewer.append("Opening 3D viewer...")
        self._config_tab.viewer_btn.setEnabled(False)
        self._state.viewer_loading = True
        self._notebook.setTabEnabled(self._viewer_tab_index, False)
        self._notebook.setCurrentIndex(self._viewer_tab_index)
        self._viewer_widget.load(**kwargs)

    def _on_viewer_scene_loaded(self, _scene: Any) -> None:
        self._state.viewer_loading = False
        self._log_viewer.append("3D viewer scene loaded.")
        self._config_tab.viewer_btn.setEnabled(True)
        self._notebook.setTabEnabled(self._viewer_tab_index, True)
        self._notebook.setCurrentIndex(self._viewer_tab_index)

    def _on_viewer_scene_failed(self, _message: str) -> None:
        self._state.viewer_loading = False
        self._config_tab.viewer_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_open_viewer(self) -> None:
        resolved = self._state.last_project_dir or self._config_tab.project_dir()
        if not resolved:
            QMessageBox.warning(
                self._root,
                "No project directory",
                "No project directory selected. Run the pipeline or pick a Project dir.",
            )
            return
        self._open_viewer(Path(resolved))

    def _on_export_html(self) -> None:
        resolved = self._state.last_project_dir or self._config_tab.project_dir()
        if not resolved:
            QMessageBox.warning(
                self._root,
                "No project directory",
                "No project directory selected. Run the pipeline or pick a Project dir.",
            )
            return
        self._export_html(Path(resolved))

    def _export_html(self, project: Path) -> None:
        output = project / "viewer.html"
        self._log_viewer.append(f"Exporting HTML viewer to {output}...")
        self._pipe_runner.start_html_export(project, output)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the Qt event loop."""
        app = QApplication.instance()
        self._root.show()
        app.exec()

    def _on_close(self) -> None:
        self._state.closed = True
        self._results_tab.shutdown()
        self._viewer_widget.shutdown()
        remove_log_handler(self._pipe_runner.log_handler)


def main() -> None:
    """Entry point for ``virda-gui``."""
    app = QApplication.instance() or QApplication([])
    window = VirdaApp()
    window.run()
    del app
