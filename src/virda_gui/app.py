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

import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QThread, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from virda.logging_setup import add_log_handler, remove_log_handler

from .config_tab import ConfigTab
from .constants import (
    ADVANCED_FIELD_DEFAULTS,
    PROJECT_ARTIFACT_DIRS,
)
from .file_manager import open_in_file_manager
from .pipeline_runner import PipelineRunner
from .preview_worker import _PreviewBundle, _PreviewWorker
from .state import AppState
from .viewer import ViewerWidget
from .widgets import (
    DirectorySelector,
)


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
        self._preview_load_seq = 0
        self._preview_thread: QThread | None = None
        self._preview_worker: _PreviewWorker | None = None

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

        self._saved_results_frame = QWidget()
        self._notebook.addTab(self._saved_results_frame, "  Saved Results  ")

        self._build_saved_results_tab()

        self._viewer_frame = QWidget()
        self._viewer_tab_index = self._notebook.addTab(self._viewer_frame, "  3D Viewer  ")
        self._notebook.setTabEnabled(self._viewer_tab_index, False)
        self._build_viewer_tab()

    # ---- Saved Results tab ----

    def _build_saved_results_tab(self) -> None:
        parent = self._saved_results_frame
        outer = QVBoxLayout(parent)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(4)

        row = QFrame(parent)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        self._results_project_dir = DirectorySelector(row, label="Project dir")
        row_layout.addWidget(self._results_project_dir, 1)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_saved_results)
        row_layout.addWidget(refresh_btn)

        outer.addWidget(row)

        splitter = QSplitter(Qt.Orientation.Vertical, parent)

        tree_widget = QWidget(splitter)
        tree_layout = QVBoxLayout(tree_widget)
        tree_layout.setContentsMargins(0, 0, 0, 0)

        self._results_tree = QTreeWidget(tree_widget)
        self._results_tree.setColumnCount(3)
        self._results_tree.setHeaderLabels(["Artifact", "Size", "Modified"])
        self._results_tree.setColumnWidth(0, 260)
        self._results_tree.itemSelectionChanged.connect(self._on_results_artifact_selected)
        self._results_tree.itemDoubleClicked.connect(self._on_results_artifact_double_clicked)
        tree_layout.addWidget(self._results_tree)

        preview_box = QGroupBox("Preview", splitter)
        preview_layout = QVBoxLayout(preview_box)
        self._preview_stack = QStackedWidget(preview_box)
        preview_layout.addWidget(self._preview_stack)

        self._results_preview = QPlainTextEdit(self._preview_stack)
        self._results_preview.setReadOnly(True)
        self._preview_stack.addWidget(self._results_preview)

        self._results_table = QTableWidget(self._preview_stack)
        self._results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._results_table.setAlternatingRowColors(True)
        self._preview_stack.addWidget(self._results_table)

        self._results_viewer_widget: ViewerWidget | None = None

        self._preview_stack.setCurrentWidget(self._results_preview)

        splitter.addWidget(tree_widget)
        splitter.addWidget(preview_box)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, 1)

        actions = QFrame(parent)
        actions_layout = QHBoxLayout(actions)
        actions_layout.setContentsMargins(0, 4, 0, 0)

        viewer_btn = QPushButton("Open 3D Viewer")
        viewer_btn.clicked.connect(self._on_open_viewer_from_results)
        actions_layout.addWidget(viewer_btn)
        actions_layout.addSpacing(8)

        export_btn = QPushButton("Export HTML Viewer")
        export_btn.clicked.connect(self._on_export_html_from_results)
        actions_layout.addWidget(export_btn)
        actions_layout.addSpacing(8)

        self._reveal_btn = QPushButton("Show in Explorer")
        self._reveal_btn.clicked.connect(self._on_show_in_explorer)
        actions_layout.addWidget(self._reveal_btn)
        if self._file_manager_opener() is None:
            self._reveal_btn.hide()

        actions_layout.addStretch(1)

        self._results_summary_label = QLabel("")
        actions_layout.addWidget(self._results_summary_label)

        outer.addWidget(actions)

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
            self._results_project_dir.set(self._state.last_project_dir)
            self._refresh_saved_results()
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

    def _on_open_viewer_from_results(self) -> None:
        project = self._selected_results_project()
        if project is None:
            QMessageBox.warning(
                self._root,
                "No project directory",
                "Select a valid project directory on the Saved Results tab first.",
            )
            return
        self._open_viewer(project)

    def _open_viewer(self, project: Path) -> None:
        mesh_path = project / "mesh" / "final_mesh.ply"
        fiducials_path = project / "input" / "fiducials.json"
        normals_path = project / "ese" / "normals.npy"

        nifti = self._config_tab.nifti_path()

        kwargs: dict[str, Any] = {}
        if nifti:
            kwargs["nifti_path"] = nifti
        else:
            # Fall back to the MRI copy stored in the project directory.
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
        self._set_viewer_buttons_enabled(False)
        self._state.viewer_loading = True
        self._notebook.setTabEnabled(self._viewer_tab_index, False)
        self._notebook.setCurrentIndex(self._viewer_tab_index)
        self._viewer_widget.load(**kwargs)

    # ---- 3D viewer callbacks ----

    def _set_viewer_buttons_enabled(self, enabled: bool) -> None:
        self._config_tab.viewer_btn.setEnabled(enabled)

    def _on_viewer_scene_loaded(self, _scene: Any) -> None:
        self._state.viewer_loading = False
        self._log_viewer.append("3D viewer scene loaded.")
        self._set_viewer_buttons_enabled(True)
        self._notebook.setTabEnabled(self._viewer_tab_index, True)
        self._notebook.setCurrentIndex(self._viewer_tab_index)  # switch to 3D Viewer tab

    def _on_viewer_scene_failed(self, _message: str) -> None:
        self._state.viewer_loading = False
        self._set_viewer_buttons_enabled(True)

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

    def _on_export_html_from_results(self) -> None:
        project = self._selected_results_project()
        if project is None:
            QMessageBox.warning(
                self._root,
                "No project directory",
                "Select a valid project directory on the Saved Results tab first.",
            )
            return
        self._export_html(project)

    def _export_html(self, project: Path) -> None:
        output = project / "viewer.html"
        self._log_viewer.append(f"Exporting HTML viewer to {output}...")
        self._pipe_runner.start_html_export(project, output)

    # ------------------------------------------------------------------
    # Saved Results tab
    # ------------------------------------------------------------------

    def _selected_results_project(self) -> Path | None:
        raw = self._results_project_dir.get().strip()
        if not raw:
            return None
        project = Path(raw)
        return project if project.is_dir() else None

    def _refresh_saved_results(self) -> None:
        tree = self._results_tree
        tree.clear()
        self._clear_preview()

        project = self._selected_results_project()
        if project is None:
            self._results_summary_label.setText("Select a valid project directory.")
            return

        root_item = QTreeWidgetItem([project.name, "<dir>", ""])
        root_item.setExpanded(True)
        root_item.setData(0, Qt.ItemDataRole.UserRole, project)
        tree.addTopLevelItem(root_item)

        known = [name for name in PROJECT_ARTIFACT_DIRS if (project / name).is_dir()]
        extra_dirs = sorted(
            entry.name
            for entry in project.iterdir()
            if entry.is_dir() and entry.name not in PROJECT_ARTIFACT_DIRS
        )
        loose_files = sorted(entry for entry in project.iterdir() if entry.is_file())

        n_files = 0
        for subdir in known + extra_dirs:
            node = project / subdir
            group_item = QTreeWidgetItem([subdir, "<dir>", ""])
            group_item.setData(0, Qt.ItemDataRole.UserRole, node)
            root_item.addChild(group_item)
            for file_path in sorted(node.rglob("*")):
                if not file_path.is_file():
                    continue
                rel = file_path.relative_to(node).as_posix()
                child = QTreeWidgetItem(
                    [rel, self._format_file_size(file_path), self._format_mtime(file_path)]
                )
                child.setData(0, Qt.ItemDataRole.UserRole, file_path)
                group_item.addChild(child)
                n_files += 1

        for file_path in loose_files:
            child = QTreeWidgetItem(
                [file_path.name, self._format_file_size(file_path), self._format_mtime(file_path)]
            )
            child.setData(0, Qt.ItemDataRole.UserRole, file_path)
            root_item.addChild(child)
            n_files += 1

        self._results_summary_label.setText(
            f"{n_files} file(s)" if n_files else "No saved artifacts found yet."
        )

    def _on_results_artifact_selected(self) -> None:
        items = self._results_tree.selectedItems()
        if not items:
            return
        item = items[0]
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path is None:
            return
        self._start_preview_load(path)

    def _on_results_artifact_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path is None or path.is_dir():
            return
        name = path.name.lower()
        if not (name.endswith(".ply") or name.endswith((".nii.gz", ".nii"))):
            return
        viewer = self._ensure_results_viewer()
        if name.endswith(".ply"):
            viewer.load(mesh_path=str(path))
        else:
            viewer.load(nifti_path=str(path))
        self._preview_stack.setCurrentWidget(viewer)
        self._preview_load_seq += 1
        if self._preview_worker is not None:
            self._preview_worker.schedule(self._preview_load_seq, None)

    def _ensure_results_viewer(self) -> ViewerWidget:
        if self._results_viewer_widget is None:
            self._results_viewer_widget = ViewerWidget(
                self._preview_stack, log=self._state.log_queue.put
            )
            self._results_viewer_widget.sceneFailed.connect(self._on_results_viewer_failed)
            self._preview_stack.addWidget(self._results_viewer_widget)
        return self._results_viewer_widget

    def _on_results_viewer_failed(self, message: str) -> None:
        self._set_results_text(f"3D preview failed: {message}")

    def _start_preview_load(self, path: Path) -> None:
        self._preview_load_seq += 1
        seq = self._preview_load_seq
        self._clear_preview()
        self._set_results_text(f"Loading {path.name}...")
        self._ensure_preview_worker().schedule(seq, path)

    def _ensure_preview_worker(self) -> _PreviewWorker:
        if self._preview_thread is None:
            thread = QThread(self)
            worker = _PreviewWorker()
            worker.moveToThread(thread)
            worker.ready.connect(self._on_preview_ready)
            worker.failed.connect(self._on_preview_failed)
            thread.start()
            self._preview_thread = thread
            self._preview_worker = worker
        assert self._preview_worker is not None
        return self._preview_worker

    def _on_preview_ready(self, seq: int, bundle: _PreviewBundle) -> None:
        if self._state.closed or seq != self._preview_load_seq:
            return
        if bundle.mode == "table":
            self._set_results_table(bundle.headers, bundle.rows)
        else:
            self._set_results_text(bundle.text)

    def _on_preview_failed(self, seq: int, message: str) -> None:
        if self._state.closed or seq != self._preview_load_seq:
            return
        self._set_results_text(f"Failed to read preview:\n{message}")

    def _set_results_text(self, text: str) -> None:
        self._preview_stack.setCurrentWidget(self._results_preview)
        self._results_preview.setPlainText(text)

    def _set_results_table(self, headers: list[str], rows: list[list[str]]) -> None:
        table = self._results_table
        n_columns = max(len(headers), 1)
        table.clear()
        table.setColumnCount(n_columns)
        if headers:
            table.setHorizontalHeaderLabels(headers)
        else:
            table.setHorizontalHeaderLabels([f"Col {i}" for i in range(n_columns)])
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j in range(n_columns):
                table.setItem(i, j, QTableWidgetItem(row[j] if j < len(row) else ""))
        table.resizeColumnsToContents()
        self._preview_stack.setCurrentWidget(table)

    def _clear_preview(self) -> None:
        self._results_preview.setPlainText("")
        self._results_table.clear()
        self._results_table.setRowCount(0)
        self._preview_stack.setCurrentWidget(self._results_preview)

    @staticmethod
    def _format_file_size(path: Path) -> str:
        size = float(path.stat().st_size)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} GB"

    @staticmethod
    def _format_mtime(path: Path) -> str:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

    def _on_show_in_explorer(self) -> None:
        project = self._selected_results_project()
        if project is None:
            QMessageBox.warning(
                self._root,
                "No project directory",
                "Select a valid project directory on the Saved Results tab first.",
            )
            return
        try:
            open_in_file_manager(project)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            QMessageBox.critical(
                self._root, "File manager error", f"Could not open the folder:\n{exc}"
            )

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
        if self._preview_worker is not None:
            self._preview_worker.stop()
        if self._preview_thread is not None:
            self._preview_thread.quit()
        if self._results_viewer_widget is not None:
            self._results_viewer_widget.shutdown()
        remove_log_handler(self._pipe_runner.log_handler)


def main() -> None:
    """Entry point for ``virda-gui``."""
    app = QApplication.instance() or QApplication([])
    window = VirdaApp()
    window.run()
    del app
