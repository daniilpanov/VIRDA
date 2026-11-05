"""IDE-style main window: file sidebar, closable tabs and project management.

The window owns the shared :class:`AppState`, the project-aware sidebar and
the closable tab bar hosting the run pipeline form, the 3D viewer and
per-file preview tabs.  Every operation is reported through the status bar
(kept in the *Recent messages* history) or dialogs.
"""

import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .constants import ADVANCED_FIELD_DEFAULTS
from .dialogs.project_dialog import ask_create_project_folder, ask_open_project_folder
from .importing import (
    IMPORT_FALLBACK_ROLES,
    ImportRole,
    detect_role,
    import_file,
    import_target,
    validate_import_source,
)
from .preferences import Preferences
from .project import classify_artifact
from .scan_hash import read_source_hash, sha256_file
from .services.pipeline_runner import PipelineRunner
from .sidebar import ProjectSidebar
from .state import AppState
from .tabs.config_tab import ConfigTab
from .tabs.preview_tab import PreviewTab
from .viewer.viewer import ViewerWidget


class IdeWindow(QMainWindow):
    """IDE-style main window of the VIRDA GUI.

    The left panel is a :class:`~virda_gui.sidebar.ProjectSidebar` listing
    the project artifacts; the right panel is a closable tab bar where the
    run pipeline form, the 3D viewer and individual project files open in
    their own tabs.  The window owns the :class:`PipelineRunner` and the
    shared :class:`AppState`.
    """

    def __init__(self, prefs: Preferences | None = None) -> None:
        super().__init__()
        self._project: Path | None = None
        self._viewer_widget: ViewerWidget | None = None
        self._file_tabs: dict[str, QWidget] = {}
        self._status_history: list[str] = []
        self._prefs = prefs or Preferences()

        self.setWindowTitle("VIRDA — Electrode Localization System")
        self.resize(1100, 720)

        self._state = AppState(advanced=dict(ADVANCED_FIELD_DEFAULTS))

        self._pipe_runner = PipelineRunner(self._state)
        self._pipe_runner.finished.connect(self._on_pipeline_done)
        self._pipe_runner.failed.connect(self._on_pipeline_error)

        self._config_tab = ConfigTab(self._state)
        self._config_tab.runRequested.connect(self._on_run)
        self._config_tab.openViewer.connect(self._on_open_viewer)

        self._sidebar = ProjectSidebar(self)
        self._sidebar.openViewerRequested.connect(self._on_open_viewer)
        self._sidebar.importFilesRequested.connect(self._on_import_files)
        self._sidebar.fileActivated.connect(self._open_file_tab)

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
        self._splitter = splitter
        self.setCentralWidget(splitter)
        self._restore_layout()

        self._build_menu()
        self._status("")

    # ------------------------------------------------------------------
    # Status bar (with history)
    # ------------------------------------------------------------------

    def _status(self, message: str, timeout: int = 5000) -> None:
        """Show *message* in the status bar and remember it for the history."""
        if message:
            stamp = time.strftime("%H:%M:%S", time.localtime())
            self._status_history.append(f"[{stamp}] {message}")
            del self._status_history[:-100]
        self.statusBar().showMessage(message, timeout)

    def _show_status_history(self) -> None:
        """Show the remembered status-bar messages."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Recent messages")
        dialog.resize(560, 320)
        view = QPlainTextEdit(dialog)
        view.setReadOnly(True)
        view.setPlainText("\n".join(self._status_history) or "No messages yet.")
        close_btn = QPushButton("Close", dialog)
        close_btn.clicked.connect(dialog.accept)
        layout = QVBoxLayout(dialog)
        layout.addWidget(view)
        layout.addWidget(close_btn)
        dialog.exec()

    def _restore_layout(self) -> None:
        """Restore window geometry and splitter layout from the last session.

        The stored blob carries a screen number and position, so on a changed
        monitor setup (unplugged display, RDP, smaller resolution) the window
        could land outside every screen.  In that case the size is clamped to
        the primary screen and the window is centered on it; splitter blobs
        simply fall back to the defaults when they cannot be applied.
        """
        geometry = self._prefs.window_geometry()
        if geometry is not None and self.restoreGeometry(geometry):
            self._ensure_visible_on_screen()
        splitter = self._prefs.main_splitter()
        if splitter is not None:
            self._splitter.restoreState(splitter)

    def _ensure_visible_on_screen(self) -> None:
        """Center the window on the primary screen when it is fully off-screen."""
        screens = QApplication.screens()
        if not screens:
            return
        frame = self.frameGeometry()
        if any(screen.availableGeometry().intersects(frame) for screen in screens):
            return
        primary = QApplication.primaryScreen()
        area = (
            primary.availableGeometry() if primary is not None else screens[0].availableGeometry()
        )
        width = min(frame.width(), area.width())
        height = min(frame.height(), area.height())
        self.resize(width, height)
        self.move(
            area.center().x() - width // 2,
            area.center().y() - height // 2,
        )

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

        self._recent_menu = QMenu("&Recent Projects", self)
        file_menu.addMenu(self._recent_menu)

        file_menu.addSeparator()

        run_action = QAction("&Run pipeline...", self)
        run_action.triggered.connect(self._show_run_tab)
        file_menu.addAction(run_action)

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

        self._refresh_recent_menu()

        view_menu = self.menuBar().addMenu("&View")
        history_action = QAction("&Recent messages...", self)
        history_action.triggered.connect(self._show_status_history)
        view_menu.addAction(history_action)

    def _refresh_recent_menu(self) -> None:
        self._recent_menu.clear()
        recent = [p for p in self._prefs.recent_projects() if p.is_dir()]
        if not recent:
            placeholder = self._recent_menu.addAction("No recent projects")
            placeholder.setEnabled(False)
            return
        for project in recent:
            action = self._recent_menu.addAction(str(project))
            action.triggered.connect(lambda _checked=False, path=project: self.open_project(path))

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
        self._config_tab.prefill_from_project(project)
        self._prefs.note_project_opened(project)
        self._refresh_recent_menu()
        self.setWindowTitle(f"VIRDA — {project.name}")
        self._show_run_tab()
        self._status(f"Project opened: {project}", 5000)

    def close_project(self) -> None:
        """Close the project and reset the window to the empty state."""
        if self._viewer_widget is not None:
            index = self._tabs.indexOf(self._viewer_widget)
            if index >= 0:
                self._close_tab(index)
        for key in list(self._file_tabs):
            self._close_file_tab(key)
        self._project = None
        self._sidebar.set_project(None)
        self._close_action.setEnabled(False)
        self._state.last_project_dir = None
        self._tabs.removeTab(self._tabs.indexOf(self._config_tab))
        self.setWindowTitle("VIRDA — Electrode Localization System")

    def _create_project(self) -> None:
        project = ask_create_project_folder(self)
        if project is not None:
            self.open_project(project)

    def _open_project_dialog(self) -> None:
        project = ask_open_project_folder(self)
        if project is not None:
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
        widget = self._tabs.widget(index)
        self._tabs.removeTab(index)
        self._discard_tab_widget(widget)

    def _discard_tab_widget(self, widget: QWidget) -> None:
        """Release per-file tabs and tear down interactive widgets on close."""
        for key, memo in list(self._file_tabs.items()):
            if memo is widget:
                del self._file_tabs[key]
                break
        if widget is self._viewer_widget:
            self._viewer_widget.shutdown()
            self._viewer_widget = None
        elif isinstance(widget, (ViewerWidget, PreviewTab)):
            widget.shutdown()

    def _show_run_tab(self) -> None:
        self._add_tab(self._config_tab, "Run Pipeline")

    # ------------------------------------------------------------------
    # Project file tabs
    # ------------------------------------------------------------------

    def _open_file_tab(self, path: Path) -> None:
        if not path.is_file():
            return
        kind = classify_artifact(path)
        if kind == "mesh":
            self._open_visual_file_tab(path, {"mesh_path": str(path)})
        elif kind == "nifti":
            self._open_visual_file_tab(path, {"nifti_path": str(path)})
        else:
            self._open_preview_tab(path)

    def _open_visual_file_tab(self, path: Path, kwargs: dict[str, Any]) -> None:
        key = str(path)
        widget = self._file_tabs.get(key)
        if widget is None:
            widget = ViewerWidget(log=lambda message: self._status(message, 4000))
            widget.sceneLoaded.connect(lambda _scene, tab=widget: self._on_scene_tab_loaded(tab))
            widget.sceneFailed.connect(
                lambda message, tab=widget: self._on_scene_tab_failed(tab, message)
            )
            self._file_tabs[key] = widget
        if self._tabs.indexOf(widget) < 0:
            self._add_tab(widget, path.name)
            self._tabs.setTabToolTip(self._tabs.indexOf(widget), str(path))
        self._tabs.setCurrentWidget(widget)
        widget.load(**kwargs)

    def _open_preview_tab(self, path: Path) -> None:
        key = str(path)
        widget = self._file_tabs.get(key)
        if widget is None:
            widget = PreviewTab()
            self._file_tabs[key] = widget
        if self._tabs.indexOf(widget) < 0:
            self._add_tab(widget, path.name)
            self._tabs.setTabToolTip(self._tabs.indexOf(widget), str(path))
        self._tabs.setCurrentWidget(widget)
        widget.open(path)

    def _on_scene_tab_loaded(self, _tab: ViewerWidget) -> None:
        self._status("3D viewer scene loaded.", 4000)

    def _on_scene_tab_failed(self, _tab: ViewerWidget, message: str) -> None:
        self._status(f"3D viewer failed: {message}", 6000)

    def _close_file_tab(self, key: str) -> None:
        """Close the open file tab for *key* so it cannot show stale bytes."""
        widget = self._file_tabs.get(key)
        if widget is None:
            return
        index = self._tabs.indexOf(widget)
        if index >= 0:
            self._close_tab(index)

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def _on_import_files(self) -> None:
        """Pick files, detect each role and import them into the project."""
        if self._project is None:
            QMessageBox.warning(self, "No project", "Open a project first to import artifacts.")
            return
        sources, _selected_filter = QFileDialog.getOpenFileNames(self, "Import files...")
        for source in sources:
            role = detect_role(source)
            if role is None:
                labels = [candidate.label for candidate in IMPORT_FALLBACK_ROLES]
                label, ok = QInputDialog.getItem(
                    self,
                    "Import file...",
                    f"What does this file contain?\n{source}",
                    labels,
                    editable=False,
                )
                if not ok:
                    continue
                role = next(
                    candidate for candidate in IMPORT_FALLBACK_ROLES if candidate.label == label
                )
            try:
                validate_import_source(role, source)
            except ValueError as exc:
                QMessageBox.critical(self, "Import error", str(exc))
                continue
            if role.key == "nifti":
                self._import_nifti_scan(role, Path(source))
                continue
            self._perform_import(role, Path(source))

    def _import_nifti_scan(self, role: ImportRole, source: Path) -> None:
        """Import a NIfTI scan, dropping stale meshes when it is new.

        A scan identical to the recorded source is skipped with a warning;
        anything else replaces the project's scan and drops the derived
        base, final and ESE meshes.
        """
        assert self._project is not None
        try:
            incoming = sha256_file(source)
        except OSError as exc:
            QMessageBox.critical(self, "Import error", f"Could not read scan:\n{exc}")
            return
        stored = read_source_hash(self._project / "mesh")
        known = [stored] if stored is not None else []
        for candidate in sorted(self._project.glob("input/*.nii*")):
            try:
                known.append(sha256_file(candidate))
            except OSError:
                continue
        if incoming in known:
            self._warn_identical_scan()
            return

        if not self._confirm_nifti_replace():
            return
        target = self._perform_import(role, source)
        if target is None:
            return
        self._invalidate_derived_meshes(delete_base=True)
        self._status("Scan imported.", 5000)

    def _warn_identical_scan(self) -> None:
        """Tell the user the scan is already in the project (modal + status)."""
        self._status("Existing NIfTI is identical to the imported one.", 5000)
        QMessageBox.information(
            self,
            "Import NIfTI scan",
            "Existing NIfTI is identical to the imported one.",
        )

    def _invalidate_derived_meshes(self, *, delete_base: bool) -> None:
        """Drop derived meshes from disk.

        With *delete_base* the base and final mesh files go too (a new scan
        arrived); otherwise only the ESE artifacts are dropped (a fresh base
        was just generated).
        """
        project = self._project
        if project is None:
            return
        stale: list[Path] = []
        ese_dir = project / "ese"
        if ese_dir.is_dir():
            stale.extend(path for path in ese_dir.glob("*") if path.is_file())
        if delete_base:
            stale.append(project / "mesh" / "base_mesh.ply")
            stale.append(project / "mesh" / "final_mesh.ply")
        for path in stale:
            key = str(path)
            self._close_file_tab(key)
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                self._status(f"Could not remove stale file {path.name}:\n{exc}", 5000)
        self._sidebar.set_project(project)

    def _confirm_nifti_replace(self) -> bool:
        """Warn that a new scan supersedes the project's inputs; False aborts.

        The base/final/ESE meshes derived from the previous scan go stale, so
        the dialog offers a fresh project as an alternative.
        """
        assert self._project is not None
        if not any(self._project.glob("input/*.nii*")):
            return True
        box = QMessageBox(self)
        box.setWindowTitle("Import NIfTI scan")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(
            "The project already holds a brain scan. Importing another one makes "
            "the base, final and ESE meshes stale. To keep previous work, "
            "create a new project."
        )
        import_btn = box.addButton("Import anyway", QMessageBox.ButtonRole.AcceptRole)
        new_btn = box.addButton("New project...", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked == new_btn:
            project = ask_create_project_folder(self)
            if project is None:
                return False
            self.open_project(project)
            return True
        return clicked == import_btn

    def _perform_import(
        self, role: ImportRole, source: Path, *, overwrite_confirmed: bool = False
    ) -> Path | None:
        """Copy *source* into the project as *role*; return the target or None.

        With *overwrite_confirmed* an existing target is replaced without
        asking again (the caller already confirmed the replacement).
        """
        if self._project is None:
            return None
        target = import_target(role, self._project, source)
        exists = target.exists()
        if exists and not overwrite_confirmed:
            answer = QMessageBox.question(
                self,
                "Overwrite?",
                f"File already exists:\n{target}\n\nOverwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return None
        self._status(f"Importing {role.label}...", 2000)
        import_file(self._project, source, role, overwrite=exists)
        self._close_file_tab(str(target))
        self._sidebar.set_project(self._project)
        self._status(f"Imported {role.label} -> {target}", 5000)
        return target

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
            return

        self._pipe_runner.submit(config, measurements_path)

    # ------------------------------------------------------------------
    # Pipeline completion handlers (main thread)
    # ------------------------------------------------------------------

    def _on_pipeline_done(self) -> None:
        self._config_tab.ensure_stage3_electrodes_group()
        self._config_tab.run_btn.setEnabled(True)
        self._config_tab.viewer_btn.setEnabled(True)
        if self._state.last_project_dir:
            self._sidebar.set_project(Path(self._state.last_project_dir))
        self._update_results_info(success=True)

    def _on_pipeline_error(self) -> None:
        self._config_tab.run_btn.setEnabled(True)
        self._config_tab.viewer_btn.setEnabled(True)
        self._update_results_info(success=False)

    def _update_results_info(self, *, success: bool) -> None:
        project = self._state.last_project_dir
        if success:
            if project:
                self._open_viewer(Path(project))
            summary = self._state.stage3_summary
            if summary:
                self._status(
                    "Pipeline completed. "
                    f"Stage 3: {summary['localized']}/{summary['total']} electrodes "
                    f"localized ({summary['flagged']} flagged).",
                    8000,
                )
            elif project:
                self._status(f"Pipeline completed. Project directory: {project}", 8000)
        else:
            self._status("Pipeline failed.", 8000)

    # ------------------------------------------------------------------
    # 3D viewer
    # ------------------------------------------------------------------

    def _build_viewer_widget(self) -> None:
        if self._viewer_widget is not None:
            return
        self._viewer_widget = ViewerWidget(log=lambda message: self._status(message, 4000))
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
        self._status("3D viewer scene loaded.", 4000)
        self._config_tab.viewer_btn.setEnabled(True)

    def _on_viewer_scene_failed(self, message: str) -> None:
        self._state.viewer_loading = False
        self._config_tab.viewer_btn.setEnabled(True)
        self._status(f"3D viewer failed: {message}", 6000)

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
        self._prefs.save_window_geometry(self.saveGeometry())
        self._prefs.save_main_splitter(self._splitter.saveState())
        for widget in self._file_tabs.values():
            if isinstance(widget, (PreviewTab, ViewerWidget)):
                widget.shutdown()
        if self._viewer_widget is not None:
            self._viewer_widget.shutdown()
        self._pipe_runner.shutdown()
