"""IDE-style main window: file sidebar, closable tabs and project management.

The window owns the shared :class:`AppState`, the project-aware sidebar and
the closable tab bar hosting the mesh processing tab, the 3D viewer (with the
live-editing tables floating over it as a HUD) and per-file preview tabs.  It
has no pipeline runner, no config file and no log viewer: every operation is
driven from the interface and reported through the status bar or dialogs.
"""

import queue
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
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

from virda.models.electrode import Electrode, Electrodes
from virda.models.ese_mesh import ESEMesh
from virda.models.fiducial import Fiducial, Fiducials
from virda.models.scalp_mesh import ScalpMesh
from virda.ops.atoms import localize
from virda.ops.options import LocalizeOptions

from .constants import ADVANCED_FIELD_DEFAULTS, ELECTRODE_PALETTE
from .dialogs.advanced_settings import AdvancedSettingsDialog
from .dialogs.project_dialog import ask_create_project_folder, ask_open_project_folder
from .importing import (
    IMPORT_FALLBACK_ROLES,
    ImportRole,
    detect_role,
    import_file,
    import_target,
    validate_import_source,
)
from .pipeline_bar import PipelineBar
from .pipeline_status import describe_hole, pipeline_steps
from .preferences import Preferences
from .project import classify_artifact
from .scan_hash import read_source_hash, sha256_file
from .sidebar import ProjectSidebar
from .state import AppState
from .tabs.base_generation_tab import BaseGenerationTab
from .tabs.editors_tab import EditorsTab, FiducialRow
from .tabs.ese_tab import EseTab
from .tabs.mesh_processing_tab import MeshProcessingTab
from .tabs.preview_tab import PreviewTab
from .viewer.frames import (
    FRAME_SCANNER,
    frame_available,
    frame_label,
    frame_to_frame_matrix,
    frame_to_scene_matrix,
    scene_to_world_matrix,
)
from .viewer.hud import HUDContainer, HudPanel
from .viewer.scene import scene_placement, transform_points
from .viewer.viewer import ViewerWidget
from .widgets import ElectrodeGroupRow

_ESE_BLOCKED_REASON = (
    "Generate the ESE mesh first (open the ESE Surface tab): localization runs on the ESE surface."
)
_FIDUCIALS_BLOCKED_REASON = "Add at least three fiducial rows before localizing."
_MEASUREMENTS_BLOCKED_REASON = "Add at least one measurement row before localizing."
_FILL_COORDS_BLOCKED_REASON = "Fill in NAS/LPA/RPA coordinates before localizing."


class IdeWindow(QMainWindow):
    """IDE-style main window of the VIRDA GUI.

    The left panel is a :class:`~virda_gui.sidebar.ProjectSidebar` listing
    the project artifacts; the right panel is a closable tab bar where the
    live editing form, the mesh processing tab, the 3D viewer and individual
    project files open in their own tabs.
    """

    def __init__(self, prefs: Preferences | None = None) -> None:
        super().__init__()
        self._project: Path | None = None
        self._viewer_widget: ViewerWidget | None = None
        self._viewer_tab_widget: QWidget | None = None
        self._viewer_hud_panel: QWidget | None = None
        self._viewer_loading_label: QLabel | None = None
        self._electrode_group_widgets: list[ElectrodeGroupRow] = []
        self._file_tabs: dict[str, QWidget] = {}
        self._invalidated_file_tabs: list[str] = []
        self._status_history: list[str] = []
        self._prefs = prefs or Preferences()

        self.setWindowTitle("VIRDA — Electrode Localization System")
        self.resize(1100, 720)

        self._state = AppState(advanced=dict(ADVANCED_FIELD_DEFAULTS))

        self._editors_tab = EditorsTab(self._state)
        self._editors_tab.advancedRequested.connect(self._on_show_advanced_settings)
        self._editors_tab.localizeRequested.connect(self._on_localize_manual)
        self._editors_tab.localization.electrodeActivated.connect(self._on_electrode_activated)
        self._editors_tab.measurements.rowsChanged.connect(self._schedule_localization)
        self._editors_tab.fiducials.inputFrameChanged.connect(self._on_fiducial_frame_changed)

        self._mesh_processing_tab = MeshProcessingTab(self._state)
        self._mesh_processing_tab.previewMesh.connect(self._on_mesh_preview)
        self._mesh_processing_tab.saved.connect(self._on_mesh_saved)
        self._mesh_processing_tab.eseRequested.connect(self._show_ese_tab)
        self._mesh_processing_tab.baseRequested.connect(self._show_base_tab)
        self._mesh_processing_tab.status.connect(self._status)
        self._mesh_processing_tab.previewMesh.connect(lambda _m: self._refresh_pipeline())
        self._mesh_processing_tab.baseMeshChanged.connect(lambda _p: self._refresh_pipeline())
        self._editors_tab.fiducials.rowsChanged.connect(lambda: self._refresh_pipeline())
        self._editors_tab.measurements.rowsChanged.connect(lambda: self._refresh_pipeline())
        self._editors_tab.localization.resultChanged.connect(self._refresh_pipeline)

        self._ese_tab = EseTab(
            self._state,
            base_provider=lambda: self._mesh_processing_tab.current_scalp_mesh(),
        )
        self._mesh_processing_tab.previewMesh.connect(self._ese_tab.show_preview)
        self._ese_tab.eseMesh.connect(self._on_ese_mesh)
        self._ese_tab.eseMesh.connect(lambda _m: self._refresh_pipeline())
        self._ese_tab.saved.connect(self._on_ese_saved)
        self._ese_tab.continueRequested.connect(self._show_editors_tab)
        self._ese_tab.status.connect(self._status)
        self._ese_tab.busyChanged.connect(self._mesh_processing_tab.set_locked)

        self._base_tab = BaseGenerationTab(self._state)
        self._base_tab.baseMesh.connect(self._on_base_generated)
        self._base_tab.saved.connect(self._refresh_pipeline)
        self._base_tab.meshRequested.connect(self._show_mesh_processing_tab)
        self._base_tab.status.connect(self._status)
        self._base_tab.busyChanged.connect(self._mesh_processing_tab.set_locked)

        self._fiducial_overlay_timer = QTimer(self)
        self._fiducial_overlay_timer.setSingleShot(True)
        self._fiducial_overlay_timer.setInterval(150)
        self._fiducial_overlay_timer.timeout.connect(self._refresh_live_fiducials)
        self._editors_tab.fiducials.rowsChanged.connect(self._on_fiducials_edited)

        self._localize_timer = QTimer(self)
        self._localize_timer.setSingleShot(True)
        self._localize_timer.setInterval(200)
        self._localize_timer.timeout.connect(self._run_localize_auto)
        self._localized_electrodes: Electrodes | None = None
        self._last_preview_mesh: ScalpMesh | None = None
        self._localize_queue = queue.Queue()
        self._localize_thread: threading.Thread | None = None
        self._localize_generation = 0
        self._localize_rerun_pending = False
        self._localize_last_auto_skip = ""
        self._localize_poll = QTimer(self)
        self._localize_poll.setInterval(80)
        self._localize_poll.timeout.connect(self._drain_localize_queue)
        self._localize_poll.start()

        self._sidebar = ProjectSidebar(self)
        self._editors_tab.filesSaved.connect(self._sidebar.refresh)
        self._editors_tab.filesSaved.connect(self._refresh_pipeline)
        self._sidebar.openViewerRequested.connect(self._on_open_viewer)
        self._sidebar.importFilesRequested.connect(self._on_import_files)
        self._sidebar.fileActivated.connect(self._open_file_tab)

        self._tabs = QTabWidget(self)
        self._tabs.setTabsClosable(True)
        self._tabs.setDocumentMode(True)
        self._tabs.tabCloseRequested.connect(self._close_tab)
        self._tabs.currentChanged.connect(self._on_tab_changed)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._sidebar)
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 800])
        self._splitter = splitter

        self._pipeline_bar = PipelineBar(self)
        self._pipeline_bar.stepActivated.connect(self._on_pipeline_step)
        central = QWidget(self)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self._pipeline_bar)
        central_layout.addWidget(splitter, 1)
        self.setCentralWidget(central)
        self._restore_layout()
        self._refresh_pipeline()

        self._build_menu()
        self._status("")

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
        mesh_splitter = self._prefs.mesh_splitter()
        if mesh_splitter is not None:
            self._mesh_processing_tab.restore_splitter_state(mesh_splitter)

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

        live_action = QAction("&Live editing", self)
        live_action.triggered.connect(self._show_editors_tab)
        file_menu.addAction(live_action)

        mesh_action = QAction("&Mesh processing", self)
        mesh_action.triggered.connect(self._show_mesh_processing_tab)
        file_menu.addAction(mesh_action)

        base_action = QAction("&Base generation", self)
        base_action.triggered.connect(self._show_base_tab)
        file_menu.addAction(base_action)

        ese_action = QAction("&ESE surface", self)
        ese_action.triggered.connect(self._show_ese_tab)
        file_menu.addAction(ese_action)

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
        """Open *project*: populate the sidebar and prefill the editors."""
        if not self._confirm_discard():
            return
        self._project = project
        self._sidebar.set_project(project)
        self._close_action.setEnabled(True)
        self._state.last_project_dir = str(project)
        self._editors_tab.prefill_from_project(project)
        self._base_tab.prefill_from_project(project)
        self._mesh_processing_tab.prefill_from_project(project)
        self._ese_tab.prefill_from_project(project)
        self._prefs.note_project_opened(project)
        self._refresh_recent_menu()
        self.setWindowTitle(f"VIRDA — {project.name}")
        self._status(f"Project opened: {project}", 5000)
        self._refresh_pipeline()
        self._sync_ese_backdrop()
        self._maybe_warn_data_loss()

    def close_project(self) -> None:
        """Close the project and reset the window to the empty state."""
        if not self._confirm_discard():
            return
        if self._viewer_tab_widget is not None:
            index = self._tabs.indexOf(self._viewer_tab_widget)
            if index >= 0:
                self._close_tab(index)
        for key in list(self._file_tabs):
            self._close_file_tab(key)
        self._project = None
        self._sidebar.set_project(None)
        self._close_action.setEnabled(False)
        self._state.last_project_dir = None
        if self._viewer_widget is not None:
            self._viewer_widget.set_project_dir(None)
        self._tabs.removeTab(self._tabs.indexOf(self._editors_tab))
        self._tabs.removeTab(self._tabs.indexOf(self._base_tab))
        self._tabs.removeTab(self._tabs.indexOf(self._mesh_processing_tab))
        self._tabs.removeTab(self._tabs.indexOf(self._ese_tab))
        self._editors_tab.clear()
        self._base_tab.clear()
        self._mesh_processing_tab.clear()
        self._ese_tab.clear()
        self._localized_electrodes = None
        self._last_preview_mesh = None
        self.setWindowTitle("VIRDA — Electrode Localization System")
        self._refresh_pipeline()

    def _confirm_discard(self) -> bool:
        """Ask before dropping unsaved table edits; False aborts."""
        if not self._editors_tab.is_dirty():
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "Unsaved table edits will be lost. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Yes

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
        if widget is self._viewer_tab_widget:
            assert self._viewer_widget is not None
            self._sync_electrode_groups()
            self._viewer_widget.shutdown()
            self._viewer_widget = None
            self._viewer_tab_widget = None
            self._viewer_hud_panel = None
            self._viewer_loading_label = None
            self._electrode_group_widgets = []
        elif isinstance(widget, (ViewerWidget, PreviewTab)):
            widget.shutdown()

    def _show_editors_tab(self) -> None:
        """Bring live editing to the foreground.

        Opens the 3D viewer with the live-editing HUD when the project holds
        renderable data; otherwise falls back to the standalone editors tab so
        fiducials and measurements stay editable without a mesh or NIfTI.
        """
        if self._viewer_widget is not None and self._viewer_tab_widget is not None:
            self._tabs.setCurrentWidget(self._viewer_tab_widget)
            return
        project = self._project
        if project is None and self._state.last_project_dir:
            project = Path(self._state.last_project_dir)
        if project is not None and self._has_renderable_viewer_data(project):
            self._open_viewer(project)
            return
        self._add_tab(self._editors_tab, "Live Editing")

    def _has_renderable_viewer_data(self, project: Path) -> bool:
        """Return whether *project* holds something the 3D viewer can render."""
        return (project / "mesh" / "final_mesh.ply").is_file() or any(project.glob("input/*.nii*"))

    def _show_mesh_processing_tab(self) -> None:
        self._add_tab(self._mesh_processing_tab, "Mesh Processing")

    def _show_base_tab(self) -> None:
        self._add_tab(self._base_tab, "Base Generation")

    def _on_base_generated(self, _mesh: object) -> None:
        """Reload the mesh tab after the base tab saved; the ESE mesh went stale."""
        self._invalidate_derived_meshes(delete_base=False)
        project = self._project
        if project is not None:
            self._mesh_processing_tab.prefill_from_project(project)
        self._reopen_invalidated_file_tabs()
        self._reload_viewer_if_open()
        self._refresh_pipeline()

    def _reload_viewer_if_open(self) -> None:
        """Reload the open 3D viewer scene so regenerated data reappears."""
        if self._viewer_widget is None or self._viewer_tab_widget is None:
            return
        if self._state.viewer_loading:
            return
        project = self._project
        if project is None and self._state.last_project_dir:
            project = Path(self._state.last_project_dir)
        if project is None or not self._collect_viewer_kwargs(project):
            return
        self._open_viewer(project)

    def _reopen_invalidated_file_tabs(self) -> None:
        """Re-open file tabs closed by invalidation; missing files are skipped."""
        pending, self._invalidated_file_tabs = self._invalidated_file_tabs, []
        for key in pending:
            self._open_file_tab(Path(key))

    def _show_ese_tab(self) -> None:
        self._sync_ese_backdrop()
        self._add_tab(self._ese_tab, "ESE Surface")

    def _sync_ese_backdrop(self) -> None:
        """Push the working scalp mesh into the ESE tab backdrop, if any."""
        working = self._mesh_processing_tab.current_scalp_mesh()
        if working is not None:
            self._ese_tab.show_preview(working)

    def _pipeline_progress(self) -> dict[str, object]:
        """Gather the saved and in-memory inputs of the pipeline step model."""
        project = self._project
        progress: dict[str, object] = {
            "nifti_saved": False,
            "base_saved": False,
            "mesh_saved": False,
            "ese_saved": False,
        }
        if project is not None:
            progress["nifti_saved"] = any(project.glob("input/*.nii*"))
            progress["base_saved"] = (project / "mesh" / "base_mesh.ply").is_file()
            progress["mesh_saved"] = (project / "mesh" / "final_mesh.ply").is_file()
            progress["ese_saved"] = (project / "ese" / "mesh.ply").is_file()
        try:
            progress["fiducials_filled"] = len(self._editors_tab.fiducials.filled_rows())
        except ValueError:
            progress["fiducials_filled"] = 0
        try:
            progress["measurement_rows"] = len(self._editors_tab.measurements.measurement_rows())
        except ValueError:
            progress["measurement_rows"] = 0
        progress["base_in_memory"] = (
            self._mesh_processing_tab.base_mesh() is not None
            or self._base_tab.current_base_mesh() is not None
        )
        progress["mesh_in_memory"] = self._mesh_processing_tab.current_scalp_mesh() is not None
        progress["ese_in_memory"] = self._ese_tab.current_ese_mesh() is not None
        return progress

    def _refresh_pipeline(self) -> None:
        """Recompute the stepper bar from project files and tab memory."""
        self._pipeline_bar.set_steps(pipeline_steps(**self._pipeline_progress()))  # type: ignore[arg-type]

    def _maybe_warn_data_loss(self) -> None:
        """Report orphaned meshes on project open and offer a rebuild.

        Fiducials and measurements are never part of the loss: they survive
        in their tables and files, only the derived meshes go stale.
        """
        report = describe_hole(pipeline_steps(**self._pipeline_progress()))  # type: ignore[arg-type]
        if report is None:
            return
        answer = QMessageBox.question(
            self,
            "Derived data lost",
            "Derived data was lost: "
            + ", ".join(report.orphaned_titles)
            + f".\n\nThe last valid step is {report.last_valid_title}.\n"
            f"Go to {report.redo_title} to regenerate?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if report.redo_key == "base":
            self._show_base_tab()
        elif report.redo_key == "mesh":
            self._show_mesh_processing_tab()
        elif report.redo_key == "ese":
            self._show_ese_tab()

    def _on_tab_changed(self, _index: int) -> None:
        """Highlight the stepper chip of the active pipeline tab, if any."""
        widget = self._tabs.currentWidget()
        if widget is self._base_tab:
            key: str | None = "base"
        elif widget is self._mesh_processing_tab:
            key = "mesh"
        elif widget is self._ese_tab:
            key = "ese"
        elif widget is self._editors_tab:
            key = "points"
        else:
            key = None
        self._pipeline_bar.set_current(key)

    def _on_pipeline_step(self, key: str) -> None:
        """Navigate to the tab where pipeline step *key* is performed."""
        if not self._pipeline_bar.is_enabled(key):
            return
        if key == "scan":
            self._on_import_files()
        elif key == "base":
            self._show_base_tab()
        elif key == "mesh":
            self._show_mesh_processing_tab()
        elif key == "ese":
            self._show_ese_tab()
        elif key == "points":
            self._show_editors_tab()

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
        widget.set_project_dir(self._project)
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
        """Import a NIfTI scan, invalidating stale meshes when it is new.

        A scan identical to the recorded source is skipped with a warning;
        anything else replaces the project's scan and drops the derived
        base, final and ESE meshes (fiducials and measurements are kept).
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
        self._show_base_tab()
        self._base_tab.set_source(target)
        self._status("Scan imported. Press Generate base mesh when ready.", 5000)

    def _warn_identical_scan(self) -> None:
        """Tell the user the scan is already in the project (modal + status)."""
        self._status("Existing NIfTI is identical to the imported one.", 5000)
        QMessageBox.information(
            self,
            "Import NIfTI scan",
            "Existing NIfTI is identical to the imported one.",
        )

    def _invalidate_derived_meshes(self, *, delete_base: bool) -> None:
        """Drop derived meshes from disk and memory; tables are always kept.

        With *delete_base* the base and final mesh files go too (a new scan
        arrived); otherwise only the ESE artifacts are dropped (a fresh base
        was just generated).  Fiducials and measurements survive either way.
        """
        project = self._project
        if project is not None:
            stale: list[Path] = []
            ese_dir = project / "ese"
            if ese_dir.is_dir():
                stale.extend(path for path in ese_dir.glob("*") if path.is_file())
            if delete_base:
                stale.append(project / "mesh" / "base_mesh.ply")
                stale.append(project / "mesh" / "final_mesh.ply")
            for path in stale:
                key = str(path)
                if key in self._file_tabs and key not in self._invalidated_file_tabs:
                    self._invalidated_file_tabs.append(key)
                self._close_file_tab(key)
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    self._status(f"Could not remove stale file {path.name}:\n{exc}", 5000)
        if delete_base:
            self._mesh_processing_tab.clear()
            self._base_tab.clear()
        self._ese_tab.clear()
        self._last_preview_mesh = None
        self._localized_electrodes = None
        self._editors_tab.localization.set_blocked(_ESE_BLOCKED_REASON)
        if project is not None:
            self._sidebar.set_project(project)
        self._refresh_pipeline()

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

    def _close_file_tab(self, key: str) -> None:
        """Close the open file tab for *key* so it cannot show stale bytes."""
        widget = self._file_tabs.get(key)
        if widget is None:
            return
        index = self._tabs.indexOf(widget)
        if index >= 0:
            self._close_tab(index)

    # ------------------------------------------------------------------
    # Live localization overlay
    # ------------------------------------------------------------------

    def _schedule_localization(self) -> None:
        """Re-run live localization after inputs change (debounced by the timer).

        Runs once the prerequisites are in place (ESE mesh, the three
        canonical fiducials and at least one measurement row) so the read-only
        preview self-fills without a manual "Localize" trigger.
        """
        if self._ese_tab.current_ese_mesh() is None:
            self._localized_electrodes = None
            self._editors_tab.localization.set_blocked(_ESE_BLOCKED_REASON)
            return
        try:
            enough_fiducials = len(self._editors_tab.fiducials.fiducial_rows()) >= 3
            has_measurements = bool(self._editors_tab.measurements.measurement_rows())
        except ValueError:
            if self._localized_electrodes is None:
                self._editors_tab.localization.set_blocked(_FILL_COORDS_BLOCKED_REASON)
            return
        if not enough_fiducials or not has_measurements:
            if self._localized_electrodes is None:
                reason = (
                    _FIDUCIALS_BLOCKED_REASON
                    if not enough_fiducials
                    else _MEASUREMENTS_BLOCKED_REASON
                )
                self._editors_tab.localization.set_blocked(reason)
            return
        self._localize_timer.start()

    def _run_localize_auto(self) -> None:
        self._run_localize(interactive=False)

    def _on_localize_manual(self) -> None:
        """Run localization once from the explicit button with dialogs."""
        self._run_localize(interactive=True)

    def _on_electrode_activated(self, electrode_id: str) -> None:
        """Focus the 3D camera on the double-clicked localized electrode."""
        viewer = self._viewer_widget
        electrodes = self._localized_electrodes
        if viewer is None or electrodes is None:
            return
        match = next((e for e in electrodes.items if (e.electrode_id or "") == electrode_id), None)
        if match is None or match.ese_coords is None:
            return
        affine, cras_offset, mm_scene = viewer.scene_frame_params
        scene_points = transform_points(
            np.asarray([match.ese_coords], dtype=np.float64),
            frame_to_scene_matrix(FRAME_SCANNER, affine, cras_offset, mm_scene),
        )
        viewer.focus_scene_point(np.asarray(scene_points[0], dtype=np.float64))

    def _localize_warning(self, message: str, interactive: bool) -> None:
        if not interactive:
            if message == self._localize_last_auto_skip:
                return
            self._localize_last_auto_skip = message
            self._status(f"Localization skipped: {message}", 5000)
            return
        self._localize_last_auto_skip = ""
        self._status(f"Localization skipped: {message}", 5000)
        QMessageBox.warning(self, "Localize", message)

    def _localize_options(self) -> LocalizeOptions:
        """The localization options, from the advanced GUI settings."""
        advanced = self._state.advanced
        calibrate = str(advanced.get("calibrate_ese_offset", "true")).lower() == "true"
        try:
            threshold = float(advanced.get("residual_threshold_mm", 10.0))
        except TypeError, ValueError:
            threshold = 10.0
        return LocalizeOptions(calibrate_ese_offset=calibrate, residual_threshold_mm=threshold)

    def _run_localize(self, *, interactive: bool) -> None:
        """Snapshot the table inputs and localize on a background thread.

        The brute-force search is heavy, so it runs on a daemon thread; the
        result is applied back on the main thread by :meth:`_drain_localize_queue`.
        Only the latest snapshot is rendered, so an in-flight run can never
        overwrite a newer one (a follow-up run is queued instead of overlapping).
        """
        if self._localize_thread is not None and self._localize_thread.is_alive():
            self._localize_rerun_pending = True
            return

        mesh = self._ese_tab.current_ese_mesh()
        if mesh is None:
            self._localize_warning(_ESE_BLOCKED_REASON, interactive)
            self._editors_tab.localization.set_blocked(_ESE_BLOCKED_REASON)
            return
        surface: ESEMesh | ScalpMesh = mesh

        try:
            fiducial_rows = self._editors_tab.fiducials.fiducial_rows()
            measurement_rows = self._editors_tab.measurements.measurement_rows()
        except (ValueError, np.linalg.LinAlgError) as exc:
            self._localize_warning(f"Invalid table:\n{exc}", interactive)
            return

        if len(fiducial_rows) < 3:
            self._localize_warning(
                "Add at least three fiducial rows before localizing.", interactive
            )
            return
        if not measurement_rows:
            self._localize_warning(
                "Add at least one measurement row before localizing.", interactive
            )
            return

        affine, cras_offset, mm_scene = (
            self._viewer_widget.scene_frame_params
            if self._viewer_widget is not None
            else (None, None, True)
        )
        frame = self._editors_tab.fiducials.input_frame()
        if not frame_available(frame, affine, cras_offset):
            self._localize_warning(
                f"The {frame_label(frame)} frame is not available for this scene.", interactive
            )
            return

        to_world = scene_to_world_matrix(affine, mm_scene) @ frame_to_scene_matrix(
            frame, affine, cras_offset, mm_scene
        )
        world_points = transform_points(
            np.asarray([row.coordinates for row in fiducial_rows], dtype=np.float64), to_world
        )

        try:
            fiducials = Fiducials(
                items=[
                    Fiducial(
                        fiducial_id=row.fiducial_id,
                        name=row.name,
                        coordinates=point,
                        coordinate_system="world",
                        definition_method=row.definition_method,
                        weight=row.weight,
                    )
                    for row, point in zip(fiducial_rows, world_points, strict=True)
                ]
            )
            electrodes = Electrodes(
                items=[
                    Electrode(
                        electrode_id=row.electrode_id or None,
                        measured_distances=dict(row.measured_distances),
                    )
                    for row in measurement_rows
                ]
            )
        except ValueError as exc:
            self._localize_warning(f"Invalid table:\n{exc}", interactive)
            return

        options = self._localize_options()
        self._localize_generation += 1
        generation = self._localize_generation
        thread = threading.Thread(
            target=self._localize_worker_thread,
            args=(surface, fiducials, electrodes, options, generation),
            daemon=True,
        )
        self._localize_thread = thread
        thread.start()

    def _localize_worker_thread(
        self,
        surface: ESEMesh | ScalpMesh,
        fiducials: Fiducials,
        electrodes: Electrodes,
        options: LocalizeOptions,
        generation: int,
    ) -> None:
        """Background thread: run the localizer and post the outcome via queue."""
        try:
            result = localize(
                surface=surface, fiducials=fiducials, electrodes=electrodes, options=options
            )
            self._localize_queue.put((generation, result))
        except Exception as exc:  # noqa: BLE001 - surfaced on the main thread
            self._localize_queue.put((generation, exc))

    def _drain_localize_queue(self) -> None:
        """Apply finished localization results on the main thread."""
        if self._localize_thread is not None and not self._localize_thread.is_alive():
            self._localize_thread = None
        while True:
            try:
                generation, payload = self._localize_queue.get_nowait()
            except queue.Empty:
                break
            if generation != self._localize_generation:
                continue
            if isinstance(payload, Exception):
                self._localize_warning(f"Localization failed:\n{payload}", interactive=False)
                continue
            self._localized_electrodes = payload
            self._show_localized_electrodes(payload)
            self._refresh_localization_preview()
            self._refresh_pipeline()
            localized_count = sum(1 for electrode in payload.items if electrode.is_localized)
            self._status(
                f"Localized {localized_count}/{len(payload.items)} electrodes "
                f"(offset shift {payload.calibrated_offset_shift_mm or 0.0:g} mm).",
                5000,
            )
        if self._localize_rerun_pending:
            self._localize_rerun_pending = False
            self._localize_timer.start()

    def _show_localized_electrodes(self, localized: Electrodes) -> None:
        viewer = self._viewer_widget
        if viewer is None:
            return
        ids: list[str] = []
        points: list[np.ndarray] = []
        flags: list[bool] = []
        for electrode in localized.items:
            if electrode.ese_coords is not None:
                ids.append(electrode.electrode_id or "")
                points.append(electrode.ese_coords)
                flags.append(electrode.flagged)
        if not ids:
            return
        viewer.set_live_electrodes(
            ids,
            np.asarray(points, dtype=np.float64),
            np.asarray(flags, dtype=bool),
            frame=FRAME_SCANNER,
        )

    def _refresh_localization_preview(self) -> None:
        """Repopulate the read-only localization table from the cached result."""
        affine, cras_offset = (
            (self._viewer_widget.scene_frame_params[0], self._viewer_widget.scene_frame_params[1])
            if self._viewer_widget is not None
            else (None, None)
        )
        self._editors_tab.localization.set_result(self._localized_electrodes, cras_offset, affine)

    # ------------------------------------------------------------------
    # Fiducial frame conversion
    # ------------------------------------------------------------------

    def _on_fiducial_frame_changed(self, old_frame: str, new_frame: str) -> None:
        """Recalculate the fiducial X/Y/Z cells when the coordinate system changes."""
        editor = self._editors_tab.fiducials
        affine, cras_offset, _mm_scene = (
            self._viewer_widget.scene_frame_params
            if self._viewer_widget is not None
            else (None, None, True)
        )
        if not frame_available(new_frame, affine, cras_offset):
            QMessageBox.warning(
                self,
                "Coordinate system",
                f"The {frame_label(new_frame)} frame requires a loaded NIfTI scan.",
            )
            editor.set_input_frame(old_frame)
            return
        try:
            matrix = frame_to_frame_matrix(old_frame, new_frame, affine, cras_offset)
            rows = editor.filled_rows()
        except (ValueError, np.linalg.LinAlgError) as exc:
            QMessageBox.warning(self, "Coordinate system", f"Cannot convert coordinates:\n{exc}")
            editor.set_input_frame(old_frame)
            return
        if not rows:
            return
        points = np.asarray([row.coordinates for row in rows], dtype=np.float64)
        converted = transform_points(points, matrix)
        new_rows = [
            FiducialRow(
                fiducial_id=row.fiducial_id,
                name=row.name,
                coordinates=tuple(float(value) for value in point),
                coordinate_system="world",
                definition_method=row.definition_method,
                weight=row.weight,
            )
            for row, point in zip(rows, converted, strict=True)
        ]
        editor.set_rows(new_rows)

    # ------------------------------------------------------------------
    # 3D viewer + electrode overlays
    # ------------------------------------------------------------------

    def _sync_electrode_groups(self) -> None:
        """Persist the current overlay rows into the state record."""
        if self._electrode_group_widgets:
            self._state.electrode_rows = [
                (widget.get().strip(), widget.get_color())
                for widget in self._electrode_group_widgets
                if widget.get().strip()
            ]

    def _on_show_advanced_settings(self) -> None:
        dialog = AdvancedSettingsDialog(self, self._state.advanced)
        if dialog.exec():
            self._state.advanced = dict(dialog.result_values)
            self._status("Advanced settings applied.", 5000)

    def _build_viewer_widget(self) -> None:
        if self._viewer_widget is not None:
            return
        viewer = ViewerWidget(log=lambda message: self._status(message, 4000))
        self._viewer_widget = viewer
        overlay_bar = self._build_electrode_overlay_panel()

        hud = HUDContainer()
        hud.set_base(viewer)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body_layout.addWidget(overlay_bar)
        body_layout.addWidget(self._editors_tab, 1)

        panel = HudPanel("Live editing", hud)
        panel.set_body(body)
        hud.add_overlay(panel, Qt.AlignmentFlag.AlignLeft, fixed_width=430)
        self._viewer_hud_panel = panel

        viewer.sceneLoaded.connect(self._on_viewer_scene_loaded)
        viewer.sceneFailed.connect(self._on_viewer_scene_failed)
        self._viewer_tab_widget = hud

    def _build_electrode_overlay_panel(self) -> QWidget:
        """Rebuild the electrode-overlay controls from the state's row record."""
        outer = QWidget()
        layout = QVBoxLayout(outer)
        layout.setContentsMargins(6, 4, 6, 0)
        layout.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(QLabel("Electrode overlays:"))
        add_group_btn = QPushButton("Add group")
        add_group_btn.clicked.connect(self._on_add_electrode_group)
        header.addWidget(add_group_btn)
        self._electrodes_cras_check = QCheckBox("Force cRAS conversion")
        self._electrodes_cras_check.setChecked(self._state.electrodes_cras)
        self._electrodes_cras_check.setToolTip(
            "Applies only to tabular electrode files, not JSON or live results. "
            "Toggling reloads the viewer."
        )
        self._electrodes_cras_check.toggled.connect(self._on_electrodes_cras_toggled)
        header.addWidget(self._electrodes_cras_check)
        header.addStretch(1)
        layout.addLayout(header)

        legend = QLabel(
            "File groups use their swatch colors; live results are lime (ok) / red (flagged).",
            outer,
        )
        legend.setWordWrap(True)
        layout.addWidget(legend)

        self._electrode_groups_layout = QVBoxLayout()
        self._electrode_groups_layout.setContentsMargins(0, 0, 0, 0)
        self._electrode_groups_layout.setSpacing(2)
        layout.addLayout(self._electrode_groups_layout)

        self._electrode_group_widgets = []
        for path, color in self._state.electrode_rows:
            self._add_electrode_group_row(path=path, color=color)
        return outer

    def _on_add_electrode_group(self) -> None:
        self._add_electrode_group_row()
        self._sync_electrode_groups()
        self._reload_viewer_if_open()

    def _add_electrode_group_row(self, path: str = "", color: str | None = None) -> None:
        if color is None:
            color = ELECTRODE_PALETTE[self._state.palette_index % len(ELECTRODE_PALETTE)]
            self._state.palette_index += 1
        row = ElectrodeGroupRow(
            on_remove=lambda: self._on_remove_electrode_group(row),
            color=color,
        )
        if path:
            row.set(path)
        self._electrode_groups_layout.addWidget(row)
        self._electrode_group_widgets.append(row)

    def _on_remove_electrode_group(self, row: ElectrodeGroupRow) -> None:
        if row in self._electrode_group_widgets:
            self._electrode_group_widgets.remove(row)
            self._electrode_groups_layout.removeWidget(row)
        row.deleteLater()
        self._sync_electrode_groups()
        self._reload_viewer_if_open()

    def _on_electrodes_cras_toggled(self, checked: bool) -> None:
        self._state.electrodes_cras = checked
        if self._viewer_widget is None:
            return
        project = self._project
        if project is None and self._state.last_project_dir:
            project = Path(self._state.last_project_dir)
        if project is not None:
            self._open_viewer(project)

    def _on_open_viewer(self) -> None:
        resolved = self._state.last_project_dir or self._project
        if not resolved:
            QMessageBox.warning(
                self,
                "No project directory",
                "No project directory selected. Open or create a project first.",
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

        self._status("Opening 3D viewer...", 2000)
        self._state.viewer_loading = True
        self._build_viewer_widget()
        assert self._viewer_tab_widget is not None
        assert self._viewer_widget is not None
        self._viewer_widget.set_project_dir(project)
        self._add_tab(self._viewer_tab_widget, f"3D Viewer - {project.name}")
        self._tabs.setTabToolTip(self._tabs.indexOf(self._viewer_tab_widget), str(project))
        self._tabs.setCurrentWidget(self._viewer_tab_widget)
        self._show_viewer_loading(True)
        self._viewer_widget.load(**kwargs)

    def _show_viewer_loading(self, loading: bool) -> None:
        """Show or hide the loading overlay above the 3D viewer."""
        container = self._viewer_tab_widget
        if container is None or not isinstance(container, HUDContainer):
            return
        if self._viewer_loading_label is None:
            label = QLabel("Loading 3D scene...", container)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(
                "QLabel { background-color: rgba(20, 26, 38, 235);"
                " color: white; padding: 12px 24px; border-radius: 8px; }"
            )
            container.add_overlay(label, Qt.AlignmentFlag.AlignCenter)
            self._viewer_loading_label = label
        self._viewer_loading_label.setVisible(loading)
        if loading:
            self._viewer_loading_label.raise_()

    def _collect_viewer_kwargs(self, project: Path) -> dict[str, Any]:
        mesh_path = project / "mesh" / "final_mesh.ply"
        fiducials_path = project / "input" / "fiducials.json"
        normals_path = project / "ese" / "normals.npy"

        kwargs: dict[str, Any] = {}
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

        self._sync_electrode_groups()
        if self._state.electrode_rows:
            kwargs["electrode_specs"] = list(self._state.electrode_rows)
            kwargs["electrodes_cras"] = self._state.electrodes_cras
        return kwargs

    def _on_viewer_scene_loaded(self, _scene: Any) -> None:
        self._state.viewer_loading = False
        self._show_viewer_loading(False)
        self._status("3D viewer scene loaded.", 4000)
        self._refresh_live_fiducials()
        self._restore_mesh_overlays()
        if self._localized_electrodes is not None:
            self._show_localized_electrodes(self._localized_electrodes)
        self._refresh_localization_preview()

    def _restore_mesh_overlays(self) -> None:
        """Re-apply the tab meshes over a freshly loaded viewer scene."""
        viewer = self._viewer_widget
        if viewer is None:
            return
        if self._last_preview_mesh is not None:
            mesh = self._last_preview_mesh
            viewer.set_extra_mesh(self._mesh_to_scene_poly(mesh.vertices, mesh.faces), "scalp")
        ese = self._ese_tab.current_ese_mesh()
        if ese is not None:
            viewer.set_extra_mesh(self._mesh_to_scene_poly(ese.vertices, ese.faces), "ese")

    def _on_viewer_scene_failed(self, message: str) -> None:
        self._state.viewer_loading = False
        self._show_viewer_loading(False)
        self._status(f"3D viewer failed: {message}", 6000)

    def _on_fiducials_edited(self) -> None:
        """Debounce fast table edits before pushing rows to the viewer."""
        self._fiducial_overlay_timer.start()
        self._schedule_localization()

    def _refresh_live_fiducials(self) -> None:
        """Push the current fiducials table onto the viewer as a live overlay.

        Rows are interpreted in the editor's input coordinate system; the
        viewer converts them into the scene frame before rendering, so a
        coordinate-system switch immediately re-places the points on the mesh.
        """
        viewer = self._viewer_widget
        if viewer is None:
            return
        try:
            rows = self._editors_tab.fiducials.filled_rows()
            ids = [row.fiducial_id for row in rows]
            points = (
                np.asarray([row.coordinates for row in rows], dtype=np.float64)
                if rows
                else np.empty((0, 3))
            )
            viewer.set_live_fiducials(ids, points, frame=self._editors_tab.fiducials.input_frame())
        except (ValueError, np.linalg.LinAlgError) as exc:
            self._status(f"Live fiducials skipped: {exc}", 5000)

    # ------------------------------------------------------------------
    # Mesh processing overlay + save
    # ------------------------------------------------------------------

    def _mesh_to_scene_poly(self, vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
        """Build a pyvista mesh whose points live in the viewer's scene frame."""
        faces_ravel = np.column_stack(
            [np.full(len(faces), 3, dtype=np.int64), np.asarray(faces, dtype=np.int64)]
        ).ravel()
        poly = pv.PolyData(np.asarray(vertices, dtype=np.float64), faces_ravel)
        _, _, transform, mm_scene = scene_placement(self._viewer_widget.scene_frame_params[0])
        if not mm_scene:
            poly.transform(transform, inplace=True)
        return poly

    def _on_mesh_preview(self, mesh: ScalpMesh) -> None:
        self._last_preview_mesh = mesh
        self._schedule_localization()
        if self._viewer_widget is None:
            return
        self._viewer_widget.set_extra_mesh(
            self._mesh_to_scene_poly(mesh.vertices, mesh.faces), "scalp"
        )

    def _on_ese_mesh(self, ese: ESEMesh) -> None:
        self._schedule_localization()
        if self._viewer_widget is None:
            return
        self._viewer_widget.set_extra_mesh(self._mesh_to_scene_poly(ese.vertices, ese.faces), "ese")

    def _on_mesh_saved(self) -> None:
        if self._state.last_project_dir:
            self._sidebar.set_project(Path(self._state.last_project_dir))
        self._status("Mesh saved. Re-open the 3D viewer to inspect the persisted surface.", 5000)
        self._refresh_pipeline()

    def _on_ese_saved(self) -> None:
        if self._state.last_project_dir:
            self._sidebar.set_project(Path(self._state.last_project_dir))
        self._status("ESE mesh saved.", 5000)
        self._reopen_invalidated_file_tabs()
        self._reload_viewer_if_open()
        self._refresh_pipeline()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        if not self._confirm_discard():
            event.ignore()
            return
        try:
            self._on_close()
        finally:
            super().closeEvent(event)

    def _on_close(self) -> None:
        self._state.closed = True
        self._prefs.save_window_geometry(self.saveGeometry())
        self._prefs.save_main_splitter(self._splitter.saveState())
        self._prefs.save_mesh_splitter(self._mesh_processing_tab.splitter_state())
        self._localize_poll.stop()
        for widget in self._file_tabs.values():
            if isinstance(widget, (PreviewTab, ViewerWidget)):
                widget.shutdown()
        if self._viewer_widget is not None:
            self._viewer_widget.shutdown()
        self._mesh_processing_tab.shutdown()
        self._ese_tab.shutdown()
        self._base_tab.shutdown()
