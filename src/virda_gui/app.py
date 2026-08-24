"""VIRDA GUI application — main window with ttkbootstrap.

Launches a tkinter GUI for configuring and running the VIRDA electrode
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
import tkinter as tk
import tkinter.scrolledtext as scrolledtext
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from typing import Any

import numpy as np
import ttkbootstrap as ttkb
from ttkbootstrap.constants import BOTH, LEFT, RIGHT, YES, E, W, X, Y
from ttkbootstrap.dialogs import Messagebox

from virda.config import load_config_file
from virda.io.fiducial_helpers import load_fiducials
from virda.logging_setup import add_log_handler, remove_log_handler
from virda.main import run
from virda.models.config import Config
from virda.models.coordsystem import Coordsystem

from .viewer import show_viewer
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
_VIEWER_DONE_SENTINEL = "__VIEWER_DONE__"


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
    "config",
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
    "n_electrodes": "",
    "ese_offset_mm": "",
    "ese_reference": "electrode_body_center",
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
    "n_electrodes": "n_electrodes",
    "ese_offset_mm": "ese_offset_mm",
    "ese_reference": "ese_reference",
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
    "ese_reference": ["electrode_body_center", "electrode_capsule_center"],
}


class AdvancedSettingsDialog(tk.Toplevel):
    """Modal dialog for advanced pipeline parameters."""

    def __init__(
        self,
        parent: tk.Misc,
        values: dict[str, str],
    ) -> None:
        super().__init__(parent)
        self.title("Advanced Settings")
        self.resizable(False, False)
        self.grab_set()

        self.result_values: dict[str, str] = dict(values)
        self.confirmed: bool = False

        self._fields: dict[str, LabeledField] = {}

        self._build_ui()
        self._center_on_parent(parent)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _center_on_parent(self, parent: tk.Misc) -> None:
        self.update_idletasks()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        w = self.winfo_width()
        h = self.winfo_height()
        x = px + (pw - w) // 2
        y = py + (ph - h) // 2
        self.geometry(f"+{x}+{y}")

    def _build_ui(self) -> None:
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self._scroll_frame = ttk.Frame(canvas)

        self._scroll_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self._scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._build_segmentation_section(self._scroll_frame)
        self._build_mesh_section(self._scroll_frame)
        self._build_ese_section(self._scroll_frame)
        self._build_neighborhood_section(self._scroll_frame)
        self._build_stage3_section(self._scroll_frame)

        btn_frame = ttk.Frame(self._scroll_frame)
        btn_frame.pack(fill=X, padx=8, pady=8)

        ttkb.Button(btn_frame, text="OK", bootstyle="success", command=self._on_ok, width=10).pack(
            side=LEFT, padx=(0, 8)
        )
        ttkb.Button(
            btn_frame, text="Cancel", bootstyle="secondary", command=self._on_cancel, width=10
        ).pack(side=LEFT)

    def _build_segmentation_section(self, parent: tk.Misc) -> None:
        frame = ttk.LabelFrame(parent, text="  Stage 1: Segmentation  ")
        frame.pack(fill=X, padx=8, pady=4)

        self._add_field(frame, "otsu_scope", "Otsu scope", "combo")
        self._add_field(frame, "otsu_threshold_scale", "Threshold scale", "entry")
        self._add_field(frame, "closing_radius", "Closing radius", "entry")

    def _build_mesh_section(self, parent: tk.Misc) -> None:
        frame = ttk.LabelFrame(parent, text="  Stage 1: Mesh Processing  ")
        frame.pack(fill=X, padx=8, pady=4)

        self._add_field(frame, "seal_enabled", "Seal mask gaps", "check")
        self._add_field(frame, "seal_radius", "Seal radius", "entry")
        self._add_field(frame, "cleaner_min_vertices", "Min component vertices", "entry")
        self._add_field(frame, "cleaner_merge_digits", "Merge digits", "entry")
        self._add_field(frame, "smoother_type", "Smoother type", "combo")
        self._add_field(frame, "smoother_iterations", "Iterations", "entry")
        self._add_field(frame, "smoother_lamb", "Lambda", "entry")
        self._add_field(frame, "smoother_nu", "Nu (Taubin)", "entry")

    def _build_ese_section(self, parent: tk.Misc) -> None:
        frame = ttk.LabelFrame(parent, text="  Stage 2: ESE Parameters  ")
        frame.pack(fill=X, padx=8, pady=4)

        self._add_field(frame, "n_electrodes", "Number of electrodes", "entry")
        self._add_field(frame, "ese_offset_mm", "Offset (mm)", "entry")
        self._add_field(frame, "ese_reference", "Reference", "combo")

    def _build_neighborhood_section(self, parent: tk.Misc) -> None:
        frame = ttk.LabelFrame(parent, text="  Stage 2: Neighborhood  ")
        frame.pack(fill=X, padx=8, pady=4)

        self._add_field(frame, "neighborhood_radius_mm", "Radius (mm)", "entry")
        self._add_field(frame, "k_neighbors", "K neighbors", "entry")
        self._add_field(frame, "pca_sigma_mm", "PCA sigma (mm)", "entry")
        self._add_field(frame, "min_neighbors", "Min neighbors", "entry")
        self._add_field(frame, "use_weighted_pca", "Weighted PCA", "check")

    def _build_stage3_section(self, parent: tk.Misc) -> None:
        frame = ttk.LabelFrame(parent, text="  Stage 3: Localization  ")
        frame.pack(fill=X, padx=8, pady=4)

        self._add_field(frame, "residual_threshold_mm", "Residual threshold (mm)", "entry")
        self._add_field(frame, "calibrate_ese_offset", "Calibrate ESE offset", "check")

    def _add_field(self, parent: tk.Misc, key: str, label: str, widget_type: str) -> None:
        values = _ADVANCED_COMBO_FIELDS.get(key) if widget_type == "combo" else None
        field = LabeledField(
            parent,
            label=label,
            widget_type=widget_type,  # type: ignore[arg-type]
            values=values,
            default=self.result_values.get(key, _ADVANCED_FIELD_DEFAULTS.get(key, "")),
        )
        field.pack(fill=X, padx=4, pady=2)
        self._fields[key] = field

    def _on_ok(self) -> None:
        for key, field in self._fields.items():
            self.result_values[key] = field.get()
        self.confirmed = True
        self.destroy()

    def _on_cancel(self) -> None:
        self.confirmed = False
        self.destroy()


class VirdaApp:
    """Main application window."""

    def __init__(self) -> None:
        self._root = ttkb.Window(
            title="VIRDA — Electrode Localization System",
            themename="cosmo",
            size=(820, 600),
            resizable=(True, True),
        )

        self._log_queue: queue.Queue[str | None] = queue.Queue()
        self._pipeline_thread: threading.Thread | None = None
        self._viewer_thread: threading.Thread | None = None
        self._run_btn: ttkb.Button | None = None
        self._viewer_btn: ttkb.Button | None = None
        self._export_btn: ttkb.Button | None = None
        self._last_project_dir: str | None = None
        self._advanced_values: dict[str, str] = dict(_ADVANCED_FIELD_DEFAULTS)
        self._electrode_rows: list[ElectrodeGroupRow] = []
        self._palette_index = 0
        self._stage3_summary: dict[str, Any] | None = None
        self._coordsystem: Coordsystem | None = None
        self._results_tree_paths: dict[str, Path] = {}

        # Capture pipeline/library logs into the log pane (console handlers
        # set up by the pipeline itself keep working).
        self._log_handler = _QueueLogHandler(self._log_queue)
        add_log_handler(self._log_handler)

        self._build_ui()
        self._poll_log_queue()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._notebook = ttkb.Notebook(self._root, bootstyle="default")
        self._notebook.pack(fill=BOTH, expand=YES, padx=8, pady=8)

        self._config_frame = ttk.Frame(self._notebook)
        self._notebook.add(self._config_frame, text="  Configuration  ")

        self._results_frame = ttk.Frame(self._notebook)
        self._notebook.add(self._results_frame, text="  Results  ")

        self._saved_results_frame = ttk.Frame(self._notebook)
        self._notebook.add(self._saved_results_frame, text="  Saved Results  ")

        self._build_config_tab()
        self._build_results_tab()
        self._build_saved_results_tab()

    # ---- Configuration tab ----

    def _build_config_tab(self) -> None:
        parent = self._config_frame

        input_frame = ttk.LabelFrame(parent, text="  Input Files  ")
        input_frame.pack(fill=X, padx=8, pady=(8, 4))

        self._config_file = FileSelector(
            input_frame,
            label="Config file",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        self._config_file.pack(fill=X, padx=6, pady=4)
        self._config_file._var.trace_add("write", self._on_config_file_changed)

        self._nifti = FileSelector(
            input_frame,
            label="NIfTI scan",
            filetypes=[("NIfTI", "*.nii.gz *.nii"), ("All files", "*.*")],
        )
        self._nifti.pack(fill=X, padx=6, pady=4)

        self._project_dir = DirectorySelector(input_frame, label="Project dir")
        self._project_dir.pack(fill=X, padx=6, pady=4)

        self._fiducials = FileSelector(
            input_frame,
            label="Fiducials",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        self._fiducials.pack(fill=X, padx=6, pady=4)
        self._fiducials._var.trace_add("write", self._on_fiducials_path_changed)

        self._measurements = FileSelector(
            input_frame,
            label="Measurements",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        self._measurements.pack(fill=X, padx=6, pady=4)

        self._auto_detect_fid = LabeledField(
            input_frame,
            label="Auto detect fiducials",
            widget_type="check",
            default="false",
        )
        self._auto_detect_fid.pack(fill=X, padx=6, pady=(4, 6))

        groups_frame = ttk.LabelFrame(parent, text="  Electrode Groups (viewer overlays)  ")
        groups_frame.pack(fill=X, padx=8, pady=(4, 4))

        self._groups_inner = ttk.Frame(groups_frame)
        self._groups_inner.pack(fill=X, padx=6, pady=(4, 2))

        groups_btns = ttk.Frame(groups_frame)
        groups_btns.pack(fill=X, padx=6, pady=(0, 6))

        ttkb.Button(
            groups_btns,
            text="Add group",
            bootstyle="secondary-outline",
            command=self._on_add_electrode_group,
            width=12,
        ).pack(side=LEFT, padx=(0, 8))

        self._electrodes_cras_var = tk.StringVar(value="false")
        ttk.Checkbutton(
            groups_btns,
            text="Force cRAS conversion",
            variable=self._electrodes_cras_var,
            onvalue="true",
            offvalue="false",
        ).pack(side=LEFT)

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=X, padx=8, pady=(4, 2))

        self._advanced_btn = ttkb.Button(
            btn_frame,
            text="Advanced Settings",
            bootstyle="secondary-outline",
            command=self._on_show_advanced,
            width=18,
        )
        self._advanced_btn.pack(side=LEFT, padx=(0, 8))

        self._run_btn = ttkb.Button(
            btn_frame,
            text="Run Pipeline",
            bootstyle="success",
            command=self._on_run,
            width=16,
        )
        self._run_btn.pack(side=LEFT, padx=(0, 8))

        self._viewer_btn = ttkb.Button(
            btn_frame,
            text="Open 3D Viewer",
            bootstyle="info-outline",
            command=self._on_open_viewer,
            state=tk.DISABLED,
            width=16,
        )
        self._viewer_btn.pack(side=LEFT, padx=(0, 8))

        self._export_btn = ttkb.Button(
            btn_frame,
            text="Export HTML",
            bootstyle="secondary-outline",
            command=self._on_export_html,
            state=tk.DISABLED,
            width=16,
        )
        self._export_btn.pack(side=LEFT)

        self._log_viewer = LogViewer(parent)
        self._log_viewer.pack(fill=BOTH, expand=YES, padx=8, pady=(2, 8))

    # ---- Results tab ----

    def _build_results_tab(self) -> None:
        parent = self._results_frame

        info_frame = ttk.LabelFrame(parent, text="  Pipeline Results  ")
        info_frame.pack(fill=X, padx=8, pady=(8, 4))

        self._result_label = ttk.Label(info_frame, text="No results yet. Run the pipeline first.")
        self._result_label.pack(padx=8, pady=8, anchor=W)

        actions_frame = ttk.Frame(parent)
        actions_frame.pack(fill=X, padx=8, pady=4)

        viewer_btn = ttkb.Button(
            actions_frame,
            text="Open 3D Viewer",
            bootstyle="info",
            command=self._on_open_viewer,
            state=tk.DISABLED,
        )
        viewer_btn.pack(side=LEFT, padx=(0, 8))
        self._results_viewer_btn = viewer_btn

        export_btn = ttkb.Button(
            actions_frame,
            text="Export HTML Viewer",
            bootstyle="secondary",
            command=self._on_export_html,
            state=tk.DISABLED,
        )
        export_btn.pack(side=LEFT)
        self._results_export_btn = export_btn

    # ---- Saved Results tab ----

    def _build_saved_results_tab(self) -> None:
        parent = self._saved_results_frame

        row = ttk.Frame(parent)
        row.pack(fill=X, padx=8, pady=(8, 4))

        self._results_project_dir = DirectorySelector(row, label="Project dir")
        self._results_project_dir.pack(side=LEFT, fill=X, expand=True)

        ttkb.Button(
            row,
            text="Refresh",
            bootstyle="secondary-outline",
            command=self._refresh_saved_results,
            width=10,
        ).pack(side=LEFT, padx=(6, 0))

        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=BOTH, expand=YES, padx=8, pady=4)

        columns = ("size", "modified")
        self._results_tree = ttk.Treeview(
            tree_frame, columns=columns, selectmode="browse", show="tree headings", height=9
        )
        self._results_tree.heading("#0", text="Artifact", anchor=W)
        self._results_tree.heading("size", text="Size", anchor=E)
        self._results_tree.heading("modified", text="Modified", anchor=W)
        self._results_tree.column("#0", stretch=True)
        self._results_tree.column("size", width=90, minwidth=70, anchor=E, stretch=False)
        self._results_tree.column("modified", width=150, minwidth=120, anchor=W, stretch=False)

        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self._results_tree.yview)
        self._results_tree.configure(yscrollcommand=tree_scroll.set)
        self._results_tree.pack(side=LEFT, fill=BOTH, expand=YES)
        tree_scroll.pack(side=LEFT, fill=Y)
        self._results_tree.bind("<<TreeviewSelect>>", self._on_results_artifact_selected)

        preview_frame = ttk.LabelFrame(parent, text="  Preview  ")
        preview_frame.pack(fill=BOTH, expand=YES, padx=8, pady=(2, 4))

        self._results_preview = scrolledtext.ScrolledText(
            preview_frame, height=10, state=tk.DISABLED, wrap=tk.WORD, font=("TkDefaultFont", 9)
        )
        self._results_preview.pack(fill=BOTH, expand=True, padx=6, pady=6)

        actions = ttk.Frame(parent)
        actions.pack(fill=X, padx=8, pady=(0, 8))

        ttkb.Button(
            actions,
            text="Open 3D Viewer",
            bootstyle="info",
            command=self._on_open_viewer_from_results,
        ).pack(side=LEFT, padx=(0, 8))

        ttkb.Button(
            actions,
            text="Export HTML Viewer",
            bootstyle="secondary",
            command=self._on_export_html_from_results,
        ).pack(side=LEFT, padx=(0, 8))

        self._reveal_btn = ttkb.Button(
            actions,
            text="Show in Explorer",
            bootstyle="secondary-outline",
            command=self._on_show_in_explorer,
        )
        self._reveal_btn.pack(side=LEFT)
        if not self._file_manager_opener():
            # No known way to reveal a folder on this platform: hide the button.
            self._reveal_btn.pack_forget()
        self._results_summary_label = ttk.Label(actions, text="")
        self._results_summary_label.pack(side=RIGHT)

    # ------------------------------------------------------------------
    # Config file handling
    # ------------------------------------------------------------------

    def _on_config_file_changed(self, *_args: Any) -> None:
        self._coordsystem = None
        path = self._config_file.get()
        if not path:
            return
        try:
            data = load_config_file(path)
        except Exception as exc:
            Messagebox.show_error(
                f"Invalid config file:\n{exc}", title="Config error", parent=self._root
            )
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

    def _on_fiducials_path_changed(self, *_args: Any) -> None:
        path = self._fiducials.get()
        if not path:
            return
        try:
            load_fiducials(Path(path))
        except Exception as exc:
            Messagebox.show_error(
                f"Invalid fiducials file:\n{exc}",
                title="Fiducials error",
                parent=self._root,
            )
            self._fiducials.set("")

    # ------------------------------------------------------------------
    # Advanced settings
    # ------------------------------------------------------------------

    def _on_show_advanced(self) -> None:
        dialog = AdvancedSettingsDialog(self._root, self._advanced_values)
        self._root.wait_window(dialog)
        if dialog.confirmed:
            self._advanced_values = dialog.result_values

    # ------------------------------------------------------------------
    # Electrode groups
    # ------------------------------------------------------------------

    def _on_add_electrode_group(self, path: str = "", color: str | None = None) -> None:
        if color is None:
            color = _ELECTRODE_PALETTE[self._palette_index % len(_ELECTRODE_PALETTE)]
            self._palette_index += 1

        row = ElectrodeGroupRow(
            self._groups_inner,
            on_remove=lambda: self._on_remove_electrode_group(row),
            color=color,
        )
        if path:
            row.set(path)
        row.pack(fill=X, pady=2)
        self._electrode_rows.append(row)

    def _on_remove_electrode_group(self, row: ElectrodeGroupRow) -> None:
        if row in self._electrode_rows:
            self._electrode_rows.remove(row)
        row.destroy()

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
            n_electrodes=_int(adv["n_electrodes"], key="n_electrodes"),
            ese_offset_mm=_float(adv["ese_offset_mm"], key="ese_offset_mm"),
            ese_reference=adv["ese_reference"] or None,
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
            Messagebox.show_error(str(exc), title="Validation error", parent=self._root)
            return

        self._run_btn.configure(state=tk.DISABLED)
        self._viewer_btn.configure(state=tk.DISABLED)
        self._export_btn.configure(state=tk.DISABLED)
        self._results_viewer_btn.configure(state=tk.DISABLED)
        self._results_export_btn.configure(state=tk.DISABLED)
        self._log_viewer.clear()
        self._log_queue.put("Starting pipeline...")
        self._last_project_dir = config.project_dir
        self._stage3_summary = None

        measurements_path = self._measurements.get().strip() or None
        if measurements_path and not Path(measurements_path).is_file():
            Messagebox.show_error(
                f"Measurements file not found:\n{measurements_path}",
                title="Measurements error",
                parent=self._root,
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
        # Schedule the next tick first so a handler exception below can never
        # silently kill the polling loop.
        self._root.after(100, self._poll_log_queue)
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
                if msg == _VIEWER_DONE_SENTINEL:
                    self._log_viewer.append("3D viewer closed.")
                    self._viewer_btn.configure(state=tk.NORMAL)
                    self._results_viewer_btn.configure(state=tk.NORMAL)
                    break
                self._log_viewer.append(msg)
        except queue.Empty:
            pass

    def _on_pipeline_done(self) -> None:
        added = self._ensure_stage3_electrodes_group()
        if added:
            self._log_viewer.append(f"Electrode group added: {added}")
        self._run_btn.configure(state=tk.NORMAL)
        self._viewer_btn.configure(state=tk.NORMAL)
        self._export_btn.configure(state=tk.NORMAL)
        self._results_viewer_btn.configure(state=tk.NORMAL)
        self._results_export_btn.configure(state=tk.NORMAL)
        if self._last_project_dir:
            self._results_project_dir.set(self._last_project_dir)
            self._refresh_saved_results()
        self._update_results_info(success=True)

    def _on_pipeline_error(self) -> None:
        self._run_btn.configure(state=tk.NORMAL)
        self._update_results_info(success=False)

    def _update_results_info(self, *, success: bool) -> None:
        if success:
            self._notebook.select(1)  # switch to Results tab
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
            self._result_label.configure(text=text, foreground="green")
        else:
            self._result_label.configure(
                text="Pipeline failed. Check the log for details.",
                foreground="red",
            )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_open_viewer(self) -> None:
        resolved = self._last_project_dir or self._project_dir.get().strip()
        if not resolved:
            Messagebox.show_warning(
                "No project directory selected. Run the pipeline or pick a Project dir.",
                parent=self._root,
            )
            return
        self._open_viewer(Path(resolved))

    def _on_open_viewer_from_results(self) -> None:
        project = self._selected_results_project()
        if project is None:
            Messagebox.show_warning(
                "Select a valid project directory on the Saved Results tab first.",
                parent=self._root,
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
            kwargs["electrodes_cras"] = self._electrodes_cras_var.get() == "true"

        if not kwargs:
            Messagebox.show_warning(
                "No mesh or NIfTI file found in the project.", parent=self._root
            )
            return

        if self._viewer_thread is not None and self._viewer_thread.is_alive():
            Messagebox.show_warning(
                "The 3D viewer is already open. Close it before opening another one.",
                parent=self._root,
            )
            return

        self._log_viewer.append("Opening 3D viewer...")
        self._viewer_btn.configure(state=tk.DISABLED)
        self._results_viewer_btn.configure(state=tk.DISABLED)
        self._viewer_thread = threading.Thread(
            target=self._run_viewer_thread, kwargs=kwargs, daemon=True
        )
        self._viewer_thread.start()

    def _run_viewer_thread(self, **kwargs: Any) -> None:
        """Background thread body: run show_viewer, route failures to the log.

        The VTK window is created off the main thread, which some platforms
        tolerate only partially; any failure must reach the log pane instead
        of silently killing this daemon thread.
        """
        try:
            show_viewer(log=self._log_queue.put, **kwargs)
        except Exception as exc:
            self._log_queue.put(f"ERROR: 3D viewer failed: {exc}")
        self._log_queue.put(_VIEWER_DONE_SENTINEL)

    def _on_export_html(self) -> None:
        resolved = self._last_project_dir or self._project_dir.get().strip()
        if not resolved:
            Messagebox.show_warning(
                "No project directory selected. Run the pipeline or pick a Project dir.",
                parent=self._root,
            )
            return
        self._export_html(Path(resolved))

    def _on_export_html_from_results(self) -> None:
        project = self._selected_results_project()
        if project is None:
            Messagebox.show_warning(
                "Select a valid project directory on the Saved Results tab first.",
                parent=self._root,
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
        for iid in tree.get_children():
            tree.delete(iid)
        self._results_tree_paths.clear()
        self._set_results_preview("")

        project = self._selected_results_project()
        if project is None:
            self._results_summary_label.configure(text="Select a valid project directory.")
            return

        root_iid = tree.insert("", "end", text=project.name, open=True, values=("<dir>", ""))
        self._results_tree_paths[root_iid] = project

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
            group_iid = tree.insert(root_iid, "end", text=subdir, open=False, values=("<dir>", ""))
            self._results_tree_paths[group_iid] = node
            for file_path in sorted(node.rglob("*")):
                if not file_path.is_file():
                    continue
                rel = file_path.relative_to(node).as_posix()
                child = tree.insert(
                    group_iid,
                    "end",
                    text=rel,
                    values=(self._format_file_size(file_path), self._format_mtime(file_path)),
                )
                self._results_tree_paths[child] = file_path
                n_files += 1

        for file_path in loose_files:
            child = tree.insert(
                root_iid,
                "end",
                text=file_path.name,
                values=(self._format_file_size(file_path), self._format_mtime(file_path)),
            )
            self._results_tree_paths[child] = file_path
            n_files += 1

        self._results_summary_label.configure(
            text=f"{n_files} file(s)" if n_files else "No saved artifacts found yet."
        )

    def _on_results_artifact_selected(self, *_args: Any) -> None:
        selection = self._results_tree.selection()
        if not selection:
            return
        path = self._results_tree_paths.get(selection[0])
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
        self._results_preview.configure(state=tk.NORMAL)
        self._results_preview.delete("1.0", tk.END)
        self._results_preview.insert(tk.END, text)
        self._results_preview.configure(state=tk.DISABLED)

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
            Messagebox.show_warning(
                "Select a valid project directory on the Saved Results tab first.",
                parent=self._root,
            )
            return
        opener = self._file_manager_opener()
        if opener is None:
            return
        try:
            opener(project)
        except (OSError, subprocess.SubprocessError) as exc:
            Messagebox.show_error(
                f"Could not open the folder:\n{exc}",
                title="File manager error",
                parent=self._root,
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
        """Start the tkinter main loop."""
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.mainloop()

    def _on_close(self) -> None:
        remove_log_handler(self._log_handler)
        self._root.destroy()


def main() -> None:
    """Entry point for ``virda-gui``."""
    app = VirdaApp()
    app.run()
