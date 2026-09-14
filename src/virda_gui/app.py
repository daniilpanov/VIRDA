"""VIRDA GUI application — main window with PySide6.

Launches a Qt GUI for configuring and running the VIRDA electrode
localisation pipeline (Stage 1 segmentation/mesh, Stage 2 ESE and Stage 3
localization).  After a successful run the *Results* tab shows a summary and
the user can open the interactive 3D viewer (with electrode overlays) or
export an HTML viewer.

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

import json
import logging
import os
import queue
import shutil
import subprocess
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from virda.config import load_config_file
from virda.io.fiducial_helpers import load_fiducials
from virda.logging_setup import add_log_handler, remove_log_handler
from virda.main import run
from virda.models.config import Config
from virda.models.coordsystem import Coordsystem

from .viewer import ViewerWidget
from .widgets import (
    DirectorySelector,
    ElectrodeGroupRow,
    FileSelector,
    LabeledField,
    LogViewer,
)

_DONE_SENTINEL = "__DONE__"
_ERROR_SENTINEL = "__ERROR__"
_EXPORT_DONE_SENTINEL = "__EXPORT_DONE__"
_EXPORT_ERROR_SENTINEL = "__EXPORT_ERROR__"


class _QueueLogHandler(logging.Handler):
    """Forward ``virda.*`` log records into the GUI log queue.

    ``emit`` runs on whatever thread logged the record (pipeline and viewer
    run in background threads); :class:`queue.Queue` makes the hand-off to
    the main thread safe.
    """

    def __init__(self, log_queue: queue.Queue[str | None]) -> None:
        super().__init__()
        self._log_queue = log_queue
        self.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._log_queue.put(self.format(record))
        except Exception:  # pragma: no cover - logging must never raise
            self.handleError(record)


_ELECTRODE_PALETTE = ["yellow", "lime", "magenta", "cyan", "orange", "white"]

_PROJECT_ARTIFACT_DIRS = [
    "input",
    "segmentation",
    "mesh",
    "fiducials",
    "ese",
    "localization",
    "quality_control",
    "logs",
]

_TEXT_PREVIEW_SUFFIXES = {".txt", ".log", ".csv", ".tsv", ".md"}
_PREVIEW_MAX_CHARS = 8000

_ADVANCED_FIELD_DEFAULTS: dict[str, str] = {
    "otsu_scope": "all",
    "otsu_threshold_scale": "0.6",
    "closing_radius": "5",
    "seal_enabled": "true",
    "seal_radius": "4",
    "cleaner_min_vertices": "100",
    "cleaner_merge_digits": "7",
    "smoother_type": "laplacian",
    "smoother_iterations": "5",
    "smoother_lamb": "0.5",
    "smoother_nu": "-0.53",
    "ese_offset_mm": "",
    "neighborhood_radius_mm": "10.0",
    "k_neighbors": "",
    "pca_sigma_mm": "5.0",
    "min_neighbors": "5",
    "use_weighted_pca": "false",
    "residual_threshold_mm": "10.0",
    "calibrate_ese_offset": "true",
}

_CONFIG_KEY_TO_ADVANCED: dict[str, str] = {
    "otsu_scope": "otsu_scope",
    "otsu_threshold_scale": "otsu_threshold_scale",
    "closing_radius": "closing_radius",
    "seal_enabled": "seal_enabled",
    "seal_radius": "seal_radius",
    "cleaner_min_vertices": "cleaner_min_vertices",
    "cleaner_merge_digits": "cleaner_merge_digits",
    "smoother_type": "smoother_type",
    "smoother_iterations": "smoother_iterations",
    "smoother_lamb": "smoother_lamb",
    "smoother_nu": "smoother_nu",
    "ese_offset_mm": "ese_offset_mm",
    "neighborhood_radius_mm": "neighborhood_radius_mm",
    "k_neighbors": "k_neighbors",
    "pca_sigma_mm": "pca_sigma_mm",
    "min_neighbors": "min_neighbors",
    "use_weighted_pca": "use_weighted_pca",
    "residual_threshold_mm": "residual_threshold_mm",
    "calibrate_ese_offset": "calibrate_ese_offset",
}

_CONFIG_KEY_TO_INPUT: dict[str, str] = {
    "nifti_path": "nifti_path",
    "project_dir": "project_dir",
    "fiducials_path": "fiducials_path",
    "auto_detect_fiducials": "auto_detect_fiducials",
}

_ADVANCED_COMBO_FIELDS: dict[str, list[str]] = {
    "otsu_scope": ["all", "foreground"],
    "smoother_type": ["laplacian", "taubin"],
}


class AdvancedSettingsDialog(QDialog):
    """Modal dialog for advanced pipeline parameters."""

    def __init__(
        self,
        parent: QWidget | None,
        values: dict[str, str],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced Settings")
        self.setModal(True)

        self.result_values: dict[str, str] = dict(values)
        self.confirmed: bool = False

        self._fields: dict[str, LabeledField] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self._build_segmentation_section(container, layout)
        self._build_mesh_section(container, layout)
        self._build_ese_section(container, layout)
        self._build_neighborhood_section(container, layout)
        self._build_stage3_section(container, layout)

        btn_frame = QFrame(container)
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setContentsMargins(0, 8, 0, 0)

        ok_btn = QPushButton("OK")
        ok_btn.setFixedWidth(100)
        ok_btn.clicked.connect(self._on_ok)
        btn_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addWidget(btn_frame)
        scroll.setWidget(container)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def _build_segmentation_section(self, parent: QWidget, outer: QVBoxLayout) -> None:
        box = QGroupBox("Stage 1: Segmentation", parent)
        layout = QVBoxLayout(box)
        self._add_field(box, layout, "otsu_scope", "Otsu scope", "combo")
        self._add_field(box, layout, "otsu_threshold_scale", "Threshold scale", "entry")
        self._add_field(box, layout, "closing_radius", "Closing radius", "entry")
        outer.addWidget(box)

    def _build_mesh_section(self, parent: QWidget, outer: QVBoxLayout) -> None:
        box = QGroupBox("Stage 1: Mesh Processing", parent)
        layout = QVBoxLayout(box)
        self._add_field(box, layout, "seal_enabled", "Seal mask gaps", "check")
        self._add_field(box, layout, "seal_radius", "Seal radius", "entry")
        self._add_field(box, layout, "cleaner_min_vertices", "Min component vertices", "entry")
        self._add_field(box, layout, "cleaner_merge_digits", "Merge digits", "entry")
        self._add_field(box, layout, "smoother_type", "Smoother type", "combo")
        self._add_field(box, layout, "smoother_iterations", "Iterations", "entry")
        self._add_field(box, layout, "smoother_lamb", "Lambda", "entry")
        self._add_field(box, layout, "smoother_nu", "Nu (Taubin)", "entry")
        outer.addWidget(box)

    def _build_ese_section(self, parent: QWidget, outer: QVBoxLayout) -> None:
        box = QGroupBox("Stage 2: ESE Parameters", parent)
        layout = QVBoxLayout(box)
        self._add_field(box, layout, "ese_offset_mm", "Offset (mm)", "entry")
        outer.addWidget(box)

    def _build_neighborhood_section(self, parent: QWidget, outer: QVBoxLayout) -> None:
        box = QGroupBox("Stage 2: Neighborhood", parent)
        layout = QVBoxLayout(box)
        self._add_field(box, layout, "neighborhood_radius_mm", "Radius (mm)", "entry")
        self._add_field(box, layout, "k_neighbors", "K neighbors", "entry")
        self._add_field(box, layout, "pca_sigma_mm", "PCA sigma (mm)", "entry")
        self._add_field(box, layout, "min_neighbors", "Min neighbors", "entry")
        self._add_field(box, layout, "use_weighted_pca", "Weighted PCA", "check")
        outer.addWidget(box)

    def _build_stage3_section(self, parent: QWidget, outer: QVBoxLayout) -> None:
        box = QGroupBox("Stage 3: Localization", parent)
        layout = QVBoxLayout(box)
        self._add_field(box, layout, "residual_threshold_mm", "Residual threshold (mm)", "entry")
        self._add_field(box, layout, "calibrate_ese_offset", "Calibrate ESE offset", "check")
        outer.addWidget(box)

    def _add_field(
        self, parent: QWidget, layout: QVBoxLayout, key: str, label: str, widget_type: str
    ) -> None:
        values = _ADVANCED_COMBO_FIELDS.get(key)
        field = LabeledField(
            parent,
            label=label,
            widget_type=widget_type,  # type: ignore[arg-type]
            values=values,
            default=self.result_values.get(key, _ADVANCED_FIELD_DEFAULTS.get(key, "")),
        )
        layout.addWidget(field)
        self._fields[key] = field

    def _on_ok(self) -> None:
        for key, field in self._fields.items():
            self.result_values[key] = field.get()
        self.confirmed = True
        self.accept()


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


class VirdaApp:
    """Main application window."""

    def __init__(self) -> None:
        self._root = _VirdaMainWindow(self._on_close)
        self._root.setWindowTitle("VIRDA — Electrode Localization System")
        self._root.resize(860, 640)

        self._log_queue: queue.Queue[str | None] = queue.Queue()
        self._pipeline_thread: threading.Thread | None = None
        self._viewer_loading = False
        self._run_btn: QPushButton | None = None
        self._viewer_btn: QPushButton | None = None
        self._export_btn: QPushButton | None = None
        self._last_project_dir: str | None = None
        self._advanced_values: dict[str, str] = dict(_ADVANCED_FIELD_DEFAULTS)
        self._electrode_rows: list[ElectrodeGroupRow] = []
        self._palette_index = 0
        self._stage3_summary: dict[str, Any] | None = None
        self._coordsystem: Coordsystem | None = None

        # Capture pipeline/library logs into the log pane (console handlers
        # set up by the pipeline itself keep working).
        self._log_handler = _QueueLogHandler(self._log_queue)
        add_log_handler(self._log_handler)

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

        self._config_frame = QWidget()
        self._notebook.addTab(self._config_frame, "  Configuration  ")

        self._results_frame = QWidget()
        self._notebook.addTab(self._results_frame, "  Results  ")

        self._saved_results_frame = QWidget()
        self._notebook.addTab(self._saved_results_frame, "  Saved Results  ")

        self._build_config_tab()
        self._build_results_tab()
        self._build_saved_results_tab()

        self._viewer_frame = QWidget()
        self._viewer_tab_index = self._notebook.addTab(self._viewer_frame, "  3D Viewer  ")
        self._build_viewer_tab()

    # ---- Configuration tab ----

    def _build_config_tab(self) -> None:
        parent = self._config_frame
        outer = QVBoxLayout(parent)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(4)

        input_box = QGroupBox("Input Files", parent)
        input_layout = QVBoxLayout(input_box)

        self._config_file = FileSelector(
            input_box,
            label="Config file",
            filetypes=[("JSON", "*.json"), ("All files", "*")],
        )
        input_layout.addWidget(self._config_file)
        self._config_file.textChanged.connect(self._on_config_file_changed)

        self._nifti = FileSelector(
            input_box,
            label="NIfTI scan",
            filetypes=[("NIfTI", "*.nii.gz *.nii"), ("All files", "*")],
        )
        input_layout.addWidget(self._nifti)

        self._project_dir = DirectorySelector(input_box, label="Project dir")
        input_layout.addWidget(self._project_dir)

        self._fiducials = FileSelector(
            input_box,
            label="Fiducials",
            filetypes=[("JSON", "*.json"), ("All files", "*")],
        )
        input_layout.addWidget(self._fiducials)
        self._fiducials.textChanged.connect(self._on_fiducials_path_changed)

        self._measurements = FileSelector(
            input_box,
            label="Measurements",
            filetypes=[("JSON", "*.json"), ("All files", "*")],
        )
        input_layout.addWidget(self._measurements)

        self._auto_detect_fid = LabeledField(
            input_box,
            label="Auto detect fiducials",
            widget_type="check",
            default="false",
        )
        input_layout.addWidget(self._auto_detect_fid)

        outer.addWidget(input_box)

        groups_box = QGroupBox("Electrode Groups (viewer overlays)", parent)
        groups_layout = QVBoxLayout(groups_box)
        self._groups_inner = QWidget(groups_box)
        self._groups_layout = QVBoxLayout(self._groups_inner)
        self._groups_layout.setContentsMargins(0, 0, 0, 0)
        self._groups_layout.setSpacing(4)
        groups_layout.addWidget(self._groups_inner)

        groups_btns = QFrame(groups_box)
        groups_btns_layout = QHBoxLayout(groups_btns)
        groups_btns_layout.setContentsMargins(0, 0, 0, 0)

        add_group_btn = QPushButton("Add group")
        add_group_btn.clicked.connect(self._on_add_electrode_group)
        groups_btns_layout.addWidget(add_group_btn)

        self._electrodes_cras_check = QCheckBox("Force cRAS conversion")
        groups_btns_layout.addWidget(self._electrodes_cras_check)

        groups_btns_layout.addStretch(1)
        groups_layout.addWidget(groups_btns)

        outer.addWidget(groups_box)

        btn_frame = QFrame(parent)
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setContentsMargins(0, 4, 0, 4)

        self._advanced_btn = QPushButton("Advanced Settings")
        self._advanced_btn.clicked.connect(self._on_show_advanced)
        btn_layout.addWidget(self._advanced_btn)
        btn_layout.addSpacing(8)

        self._run_btn = QPushButton("Run Pipeline")
        self._run_btn.clicked.connect(self._on_run)
        btn_layout.addWidget(self._run_btn)
        btn_layout.addSpacing(8)

        self._viewer_btn = QPushButton("Open 3D Viewer")
        self._viewer_btn.clicked.connect(self._on_open_viewer)
        self._viewer_btn.setEnabled(False)
        btn_layout.addWidget(self._viewer_btn)
        btn_layout.addSpacing(8)

        self._export_btn = QPushButton("Export HTML")
        self._export_btn.clicked.connect(self._on_export_html)
        self._export_btn.setEnabled(False)
        btn_layout.addWidget(self._export_btn)

        btn_layout.addStretch(1)
        outer.addWidget(btn_frame)

        self._log_viewer = LogViewer(parent)
        outer.addWidget(self._log_viewer, 1)

    # ---- Results tab ----

    def _build_results_tab(self) -> None:
        parent = self._results_frame
        outer = QVBoxLayout(parent)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(4)

        info_box = QGroupBox("Pipeline Results", parent)
        info_layout = QVBoxLayout(info_box)

        self._result_label = QLabel("No results yet. Run the pipeline first.")
        self._result_label.setWordWrap(True)
        info_layout.addWidget(self._result_label)

        outer.addWidget(info_box)

        actions = QFrame(parent)
        actions_layout = QHBoxLayout(actions)
        actions_layout.setContentsMargins(0, 4, 0, 0)

        viewer_btn = QPushButton("Open 3D Viewer")
        viewer_btn.clicked.connect(self._on_open_viewer)
        viewer_btn.setEnabled(False)
        actions_layout.addWidget(viewer_btn)
        self._results_viewer_btn = viewer_btn
        actions_layout.addSpacing(8)

        export_btn = QPushButton("Export HTML Viewer")
        export_btn.clicked.connect(self._on_export_html)
        export_btn.setEnabled(False)
        actions_layout.addWidget(export_btn)
        self._results_export_btn = export_btn

        actions_layout.addStretch(1)
        outer.addWidget(actions)
        outer.addStretch(1)

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
        tree_layout.addWidget(self._results_tree)

        preview_box = QGroupBox("Preview", splitter)
        preview_layout = QVBoxLayout(preview_box)
        self._results_preview = QPlainTextEdit(preview_box)
        self._results_preview.setReadOnly(True)
        preview_layout.addWidget(self._results_preview)

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

        self._viewer_widget = ViewerWidget(parent, log=self._log_queue.put)
        self._viewer_widget.sceneLoaded.connect(self._on_viewer_scene_loaded)
        self._viewer_widget.sceneFailed.connect(self._on_viewer_scene_failed)
        outer.addWidget(self._viewer_widget)

    # ------------------------------------------------------------------
    # Config file handling
    # ------------------------------------------------------------------

    def _on_config_file_changed(self, _text: str) -> None:
        self._coordsystem = None
        path = self._config_file.get()
        if not path:
            return
        try:
            data = load_config_file(path)
        except Exception as exc:
            QMessageBox.critical(self._root, "Config error", f"Invalid config file:\n{exc}")
            self._config_file.set("")
            return
        self._populate_from_config(data)

    def _populate_from_config(self, data: dict[str, Any]) -> None:
        for config_key, attr_name in _CONFIG_KEY_TO_INPUT.items():
            if config_key in data:
                widget = getattr(self, f"_{attr_name}", None)
                if widget is not None and not widget.get():
                    widget.set(str(data[config_key]))

        if data.get("measurements_path") and not self._measurements.get():
            self._measurements.set(str(data["measurements_path"]))

        for config_key, adv_key in _CONFIG_KEY_TO_ADVANCED.items():
            if config_key in data and not self._advanced_values.get(adv_key):
                self._advanced_values[adv_key] = str(data[config_key])

        # Keep the parsed MNE coordsystem (its fiducials feed Stage 1).
        coordsystem = data.get("coordsystem")
        if isinstance(coordsystem, Coordsystem):
            self._coordsystem = coordsystem
        else:
            if coordsystem is not None:
                self._log_viewer.append(
                    "WARNING: 'coordsystem' entry in the config file was not parsed "
                    "from a coordsystem.json file — its fiducials are ignored."
                )
            self._coordsystem = None

    def _on_fiducials_path_changed(self, _text: str) -> None:
        path = self._fiducials.get()
        if not path:
            return
        try:
            load_fiducials(Path(path))
        except Exception as exc:
            QMessageBox.critical(self._root, "Fiducials error", f"Invalid fiducials file:\n{exc}")
            self._fiducials.set("")

    # ------------------------------------------------------------------
    # Advanced settings
    # ------------------------------------------------------------------

    def _on_show_advanced(self) -> None:
        dialog = AdvancedSettingsDialog(self._root, self._advanced_values)
        if dialog.exec():
            self._advanced_values = dialog.result_values

    # ------------------------------------------------------------------
    # Electrode groups
    # ------------------------------------------------------------------

    def _on_add_electrode_group(self, path: str = "", color: str | None = None) -> None:
        if color is None:
            color = _ELECTRODE_PALETTE[self._palette_index % len(_ELECTRODE_PALETTE)]
            self._palette_index += 1

        row = ElectrodeGroupRow(
            on_remove=lambda: self._on_remove_electrode_group(row),
            color=color,
        )
        if path:
            row.set(path)
        self._groups_layout.addWidget(row)
        self._electrode_rows.append(row)

    def _on_remove_electrode_group(self, row: ElectrodeGroupRow) -> None:
        if row in self._electrode_rows:
            self._electrode_rows.remove(row)
            self._groups_layout.removeWidget(row)
        row.deleteLater()

    def _collect_electrode_specs(self) -> list[tuple[str, str]]:
        """Return (path, color) pairs of all non-empty electrode group rows."""
        specs: list[tuple[str, str]] = []
        seen: set[str] = set()
        for row in self._electrode_rows:
            path = row.get().strip()
            if not path or path in seen:
                continue
            seen.add(path)
            specs.append((path, row.get_color()))
        return specs

    def _ensure_stage3_electrodes_group(self, project_dir: str | Path | None = None) -> str | None:
        """Add the Stage 3 output as a group unless already present.

        Returns the added path or None when there is nothing to add.
        """
        resolved = project_dir or self._last_project_dir
        if not resolved:
            return None
        electrodes_path = Path(resolved) / "localization" / "electrodes.json"
        if not electrodes_path.is_file():
            return None
        path_str = str(electrodes_path)
        if any(row.get().strip() == path_str for row in self._electrode_rows):
            return None
        self._on_add_electrode_group(path=path_str)
        return path_str

    # ------------------------------------------------------------------
    # Config collection
    # ------------------------------------------------------------------

    def _collect_config(self) -> Config:
        nifti = self._nifti.get() or None
        project = self._project_dir.get() or None
        fiducials = self._fiducials.get() or None

        if not nifti:
            raise ValueError("NIfTI scan path is required.")
        if not project:
            raise ValueError("Project directory is required.")

        adv = self._advanced_values

        def _int(val: str, default: int | None = None, *, key: str = "value") -> int | None:
            val = val.strip()
            if not val:
                return default
            try:
                return int(val)
            except ValueError:
                raise ValueError(f"{key}: expected an integer, got {val!r}") from None

        def _float(val: str, default: float | None = None, *, key: str = "value") -> float | None:
            val = val.strip()
            if not val:
                return default
            try:
                return float(val)
            except ValueError:
                raise ValueError(f"{key}: expected a number, got {val!r}") from None

        return Config(
            nifti_path=nifti,
            project_dir=project,
            fiducials_path=fiducials or None,
            auto_detect_fiducials=self._auto_detect_fid.get() == "true",
            coordsystem=self._coordsystem,
            closing_radius=_int(adv["closing_radius"], 5, key="closing_radius"),
            otsu_scope=adv["otsu_scope"] or "all",  # type: ignore[arg-type]
            otsu_threshold_scale=_float(
                adv["otsu_threshold_scale"], 0.6, key="otsu_threshold_scale"
            ),
            seal_enabled=adv["seal_enabled"] == "true",
            seal_radius=_int(adv["seal_radius"], 4, key="seal_radius"),
            cleaner_min_vertices=_int(adv["cleaner_min_vertices"], 100, key="cleaner_min_vertices"),
            cleaner_merge_digits=_int(adv["cleaner_merge_digits"], 7, key="cleaner_merge_digits"),
            smoother_type=adv["smoother_type"] or "laplacian",
            smoother_iterations=_int(adv["smoother_iterations"], 5, key="smoother_iterations"),
            smoother_lamb=_float(adv["smoother_lamb"], 0.5, key="smoother_lamb"),
            smoother_nu=_float(adv["smoother_nu"], -0.53, key="smoother_nu"),
            ese_offset_mm=_float(adv["ese_offset_mm"], key="ese_offset_mm"),
            neighborhood_radius_mm=_float(
                adv["neighborhood_radius_mm"], 10.0, key="neighborhood_radius_mm"
            ),
            k_neighbors=_int(adv["k_neighbors"], key="k_neighbors"),
            use_weighted_pca=adv["use_weighted_pca"] == "true",
            pca_sigma_mm=_float(adv["pca_sigma_mm"], 5.0, key="pca_sigma_mm"),
            min_neighbors=_int(adv["min_neighbors"], 5, key="min_neighbors"),
            residual_threshold_mm=_float(
                adv["residual_threshold_mm"],  # type: ignore[arg-type]
                10.0,
                key="residual_threshold_mm",
            ),
            calibrate_ese_offset=adv["calibrate_ese_offset"] == "true",
        )

    # ------------------------------------------------------------------
    # Pipeline execution (background thread)
    # ------------------------------------------------------------------

    def _on_run(self) -> None:
        try:
            config = self._collect_config()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user as a dialog
            QMessageBox.critical(self._root, "Validation error", str(exc))
            return

        self._run_btn.setEnabled(False)
        self._viewer_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._results_viewer_btn.setEnabled(False)
        self._results_export_btn.setEnabled(False)
        self._log_viewer.clear()
        self._log_queue.put("Starting pipeline...")
        self._last_project_dir = config.project_dir
        self._stage3_summary = None

        measurements_path = self._measurements.get().strip() or None
        if measurements_path and not Path(measurements_path).is_file():
            QMessageBox.critical(
                self._root,
                "Measurements error",
                f"Measurements file not found:\n{measurements_path}",
            )
            self._on_pipeline_error()
            return

        self._pipeline_thread = threading.Thread(
            target=self._run_pipeline, args=(config, measurements_path), daemon=True
        )
        self._pipeline_thread.start()

    def _run_pipeline(self, config: Config, measurements_path: str | None) -> None:
        """Background thread: run the pipeline and post results to the queue."""
        try:
            self._log_queue.put("Building configuration...")
            stage1_result, ese_mesh, electrodes = run(config, measurements_path)

            msg = f"Stage 1: mesh with {len(stage1_result.mesh.vertices)} vertices"
            self._log_queue.put(msg)

            if ese_mesh is not None:
                msg = f"Stage 2: ESE mesh with {len(ese_mesh.vertices)} vertices"
                self._log_queue.put(msg)

            if electrodes is not None:
                items = electrodes.items
                localized = sum(1 for e in items if e.is_localized)
                flagged = sum(1 for e in items if e.flagged)
                shift = electrodes.calibrated_offset_shift_mm
                msg = f"Stage 3: {localized}/{len(items)} electrodes localized ({flagged} flagged)"
                if shift is not None:
                    msg += f", ESE offset shift {shift:.2f} mm"
                self._log_queue.put(msg)
                self._stage3_summary = {
                    "total": len(items),
                    "localized": localized,
                    "flagged": flagged,
                    "offset_shift_mm": shift,
                }
            elif measurements_path:
                self._log_queue.put("Stage 3 skipped: ESE mesh or measurements are not available.")

            self._log_queue.put("Pipeline completed successfully.")
            self._log_queue.put(_DONE_SENTINEL)

        except Exception as exc:
            self._log_queue.put(f"ERROR: {exc}")
            self._log_queue.put(_ERROR_SENTINEL)

    # ------------------------------------------------------------------
    # Log queue polling (main thread)
    # ------------------------------------------------------------------

    def _poll_log_queue(self) -> None:
        try:
            while True:
                msg = self._log_queue.get_nowait()
                if msg == _DONE_SENTINEL:
                    self._on_pipeline_done()
                    break
                if msg == _ERROR_SENTINEL:
                    self._on_pipeline_error()
                    break
                if msg == _EXPORT_DONE_SENTINEL:
                    self._log_viewer.append("HTML export completed.")
                    break
                if msg == _EXPORT_ERROR_SENTINEL:
                    self._log_viewer.append("HTML export failed — see log above.")
                    break
                self._log_viewer.append(msg)
        except queue.Empty:
            pass

    def _on_pipeline_done(self) -> None:
        added = self._ensure_stage3_electrodes_group()
        if added:
            self._log_viewer.append(f"Electrode group added: {added}")
        self._run_btn.setEnabled(True)
        self._viewer_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
        self._results_viewer_btn.setEnabled(True)
        self._results_export_btn.setEnabled(True)
        if self._last_project_dir:
            self._results_project_dir.set(self._last_project_dir)
            self._refresh_saved_results()
        self._update_results_info(success=True)

    def _on_pipeline_error(self) -> None:
        self._run_btn.setEnabled(True)
        self._update_results_info(success=False)

    def _update_results_info(self, *, success: bool) -> None:
        if success:
            self._notebook.setCurrentIndex(1)  # switch to Results tab
            project = self._last_project_dir or "—"
            text = f"Pipeline completed.\nProject directory: {project}"
            summary = self._stage3_summary
            if summary:
                shift = summary["offset_shift_mm"]
                shift_text = f", offset shift {shift:.2f} mm" if shift is not None else ""
                text += (
                    f"\nStage 3: {summary['localized']}/{summary['total']} electrodes "
                    f"localized ({summary['flagged']} flagged{shift_text})"
                )
            else:
                text += "\nStage 3: not run (no measurements provided)."
            self._result_label.setText(text)
            self._result_label.setStyleSheet("color: green;")
        else:
            self._result_label.setText("Pipeline failed. Check the log for details.")
            self._result_label.setStyleSheet("color: red;")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_open_viewer(self) -> None:
        resolved = self._last_project_dir or self._project_dir.get().strip()
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
        fiducials_path = project / "fiducials" / "fiducials.json"
        normals_path = project / "ese" / "normals.npy"

        nifti = self._nifti.get()

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

        self._ensure_stage3_electrodes_group(project)
        electrode_specs = self._collect_electrode_specs()
        if electrode_specs:
            kwargs["electrode_specs"] = electrode_specs
            kwargs["electrodes_cras"] = self._electrodes_cras_check.isChecked()

        if not kwargs:
            QMessageBox.warning(
                self._root, "Nothing to view", "No mesh or NIfTI file found in the project."
            )
            return

        if self._viewer_loading:
            QMessageBox.warning(
                self._root,
                "Viewer still loading",
                "The 3D viewer is still loading a scene. Wait for it to finish.",
            )
            return

        self._log_viewer.append("Opening 3D viewer...")
        self._set_viewer_buttons_enabled(False)
        self._viewer_loading = True
        self._notebook.setCurrentIndex(self._viewer_tab_index)
        self._viewer_widget.load(**kwargs)

    # ---- 3D viewer callbacks ----

    def _set_viewer_buttons_enabled(self, enabled: bool) -> None:
        self._viewer_btn.setEnabled(enabled)
        self._results_viewer_btn.setEnabled(enabled)

    def _on_viewer_scene_loaded(self, _scene: Any) -> None:
        self._viewer_loading = False
        self._log_viewer.append("3D viewer scene loaded.")
        self._set_viewer_buttons_enabled(True)
        self._notebook.setCurrentIndex(self._viewer_tab_index)  # switch to 3D Viewer tab

    def _on_viewer_scene_failed(self, _message: str) -> None:
        self._viewer_loading = False
        self._set_viewer_buttons_enabled(True)

    def _on_export_html(self) -> None:
        resolved = self._last_project_dir or self._project_dir.get().strip()
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

        def _export() -> None:
            try:
                from .html_export import export_project

                export_project(str(project), output)
                self._log_queue.put(f"HTML exported: {output}")
                self._log_queue.put(_EXPORT_DONE_SENTINEL)
            except Exception as exc:
                self._log_queue.put(f"HTML export failed: {exc}")
                self._log_queue.put(_EXPORT_ERROR_SENTINEL)

        threading.Thread(target=_export, daemon=True).start()

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
        self._set_results_preview("")

        project = self._selected_results_project()
        if project is None:
            self._results_summary_label.setText("Select a valid project directory.")
            return

        root_item = QTreeWidgetItem([project.name, "<dir>", ""])
        root_item.setExpanded(True)
        root_item.setData(0, Qt.ItemDataRole.UserRole, project)
        tree.addTopLevelItem(root_item)

        known = [name for name in _PROJECT_ARTIFACT_DIRS if (project / name).is_dir()]
        extra_dirs = sorted(
            entry.name
            for entry in project.iterdir()
            if entry.is_dir() and entry.name not in _PROJECT_ARTIFACT_DIRS
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
        try:
            if path.is_dir():
                n = sum(1 for p in path.rglob("*") if p.is_file())
                preview = f"{path}\n\n{n} file(s) in this folder."
            else:
                preview = self._preview_artifact(path)
        except Exception as exc:
            preview = f"Failed to read {path}:\n{exc}"
        self._set_results_preview(preview)

    def _set_results_preview(self, text: str) -> None:
        self._results_preview.setPlainText(text)

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

    @staticmethod
    def _preview_artifact(path: Path) -> str:
        """Human-readable preview of a saved artifact (JSON/NIfTI/npy/PLY/text)."""
        name = path.name.lower()
        suffix = path.suffix.lower()
        header_text = f"{path}\n{'-' * 60}\n"

        def _truncate(body: str) -> str:
            if len(body) > _PREVIEW_MAX_CHARS:
                body = (
                    body[:_PREVIEW_MAX_CHARS]
                    + f"\n\n... (truncated to first {_PREVIEW_MAX_CHARS} characters)"
                )
            return header_text + body

        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return _truncate(json.dumps(data, indent=2, ensure_ascii=False))

        if name.endswith((".nii.gz", ".nii")):
            import nibabel as nib

            img: Any = nib.load(str(path))
            nii_header: Any = img.header
            shape = tuple(int(v) for v in nii_header.get_data_shape())
            zooms = tuple(round(float(z), 3) for z in nii_header.get_zooms()[:3])
            return (
                header_text + f"NIfTI volume\n  shape         : {shape}\n  spacing (mm)  : {zooms}"
            )

        if suffix == ".npy":
            array = np.load(path, allow_pickle=False)
            body = f"NumPy array\n  shape : {array.shape}\n  dtype : {array.dtype}"
            return _truncate(body)

        if suffix == ".ply":
            lines: list[str] = []
            with open(path, "rb") as fh:
                for line in fh:
                    decoded = line.decode("ascii", errors="replace").rstrip("\r\n")
                    lines.append(decoded)
                    if len(lines) >= 100 or decoded.strip() == "end_header":
                        break
            return _truncate("PLY header:\n" + "\n".join(lines))

        if suffix in _TEXT_PREVIEW_SUFFIXES:
            return _truncate(path.read_text(encoding="utf-8", errors="replace"))

        size_kb = path.stat().st_size / 1024
        return header_text + f"(binary file, no preview — {size_kb:.1f} KB)"

    def _on_show_in_explorer(self) -> None:
        project = self._selected_results_project()
        if project is None:
            QMessageBox.warning(
                self._root,
                "No project directory",
                "Select a valid project directory on the Saved Results tab first.",
            )
            return
        opener = self._file_manager_opener()
        if opener is None:
            return
        try:
            opener(project)
        except (OSError, subprocess.SubprocessError) as exc:
            QMessageBox.critical(
                self._root, "File manager error", f"Could not open the folder:\n{exc}"
            )

    @staticmethod
    def _file_manager_opener() -> Callable[[Path], None] | None:
        """Return a callable that reveals a folder, or None if unsupported."""
        if hasattr(os, "startfile"):  # Windows
            return lambda path: os.startfile(str(path))  # noqa: S606
        for command in ("xdg-open", "open"):  # Linux, macOS
            if shutil.which(command) is not None:
                return lambda path, command=command: subprocess.run(
                    [command, str(path)],
                    check=True,
                    timeout=10,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the Qt event loop."""
        app = QApplication.instance()
        self._root.show()
        app.exec()

    def _on_close(self) -> None:
        remove_log_handler(self._log_handler)


def main() -> None:
    """Entry point for ``virda-gui``."""
    app = QApplication.instance() or QApplication([])
    window = VirdaApp()
    window.run()
    del app
