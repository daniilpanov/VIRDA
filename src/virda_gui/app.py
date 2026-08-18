"""VIRDA GUI application — main window with ttkbootstrap.

Launches a tkinter GUI for configuring and running the VIRDA electrode
localisation pipeline.  After a successful run the *Results* tab becomes
available, allowing the user to open the interactive 3D viewer or export
an HTML viewer.

Stage 3 (real electrode localisation) is预留 placeholders — the *Stage 3*
section and *Results* tab will be fleshed out when that stage ships.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any

import ttkbootstrap as ttkb
from ttkbootstrap.constants import BOTH, LEFT, YES, W, X
from ttkbootstrap.dialogs import Messagebox

from virda.main import run
from virda.models.config import Config

from .widgets import (
    CollapsibleSection,
    DirectorySelector,
    FileSelector,
    LabeledField,
    LogViewer,
)

_DONE_SENTINEL = "__DONE__"
_ERROR_SENTINEL = "__ERROR__"


class VirdaApp:
    """Main application window."""

    def __init__(self) -> None:
        self._root = ttkb.Window(
            title="VIRDA — Electrode Localization System",
            themename="cosmo",
            size=(820, 760),
            resizable=(True, True),
        )

        self._log_queue: queue.Queue[str | None] = queue.Queue()
        self._pipeline_thread: threading.Thread | None = None
        self._run_btn: ttkb.Button | None = None
        self._viewer_btn: ttkb.Button | None = None
        self._export_btn: ttkb.Button | None = None
        self._last_project_dir: str | None = None

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

        self._build_config_tab()
        self._build_results_tab()

    # ---- Configuration tab ----

    def _build_config_tab(self) -> None:
        parent = self._config_frame

        input_frame = ttk.LabelFrame(parent, text="  Input Files  ")
        input_frame.pack(fill=X, padx=8, pady=(8, 4))

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
        self._fiducials.pack(fill=X, padx=6, pady=(4, 6))

        self._seg = self._build_segmentation_section(parent)
        self._mesh = self._build_mesh_section(parent)
        self._ese = self._build_ese_section(parent)
        self._nhood = self._build_neighborhood_section(parent)
        self._stage3_placeholder = self._build_stage3_placeholder(parent)

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=X, padx=8, pady=(4, 2))

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

    def _build_segmentation_section(self, parent: tk.Misc) -> CollapsibleSection:
        sec = CollapsibleSection(parent, title="Stage 1: Segmentation")
        sec.pack(fill=X, padx=8, pady=2)
        sec.collapse()

        self._otsu_scope = LabeledField(
            sec.body,
            label="Otsu scope",
            widget_type="combo",
            values=["all", "foreground"],
            default="all",
        )
        self._otsu_scope.pack(fill=X, padx=4, pady=2)

        self._otsu_threshold = LabeledField(
            sec.body, label="Threshold scale", widget_type="entry", default="0.6"
        )
        self._otsu_threshold.pack(fill=X, padx=4, pady=2)

        self._closing_radius = LabeledField(
            sec.body, label="Closing radius", widget_type="entry", default="5"
        )
        self._closing_radius.pack(fill=X, padx=4, pady=2)

        return sec

    def _build_mesh_section(self, parent: tk.Misc) -> CollapsibleSection:
        sec = CollapsibleSection(parent, title="Stage 1: Mesh Processing")
        sec.pack(fill=X, padx=8, pady=2)
        sec.collapse()

        self._seal_enabled = LabeledField(
            sec.body, label="Seal mask gaps", widget_type="check", default="true"
        )
        self._seal_enabled.pack(fill=X, padx=4, pady=2)

        self._seal_radius = LabeledField(sec.body, label="Seal radius", default="4")
        self._seal_radius.pack(fill=X, padx=4, pady=2)

        self._cleaner_min = LabeledField(sec.body, label="Min component vertices", default="100")
        self._cleaner_min.pack(fill=X, padx=4, pady=2)

        self._cleaner_digits = LabeledField(sec.body, label="Merge digits", default="7")
        self._cleaner_digits.pack(fill=X, padx=4, pady=2)

        self._smoother_type = LabeledField(
            sec.body,
            label="Smoother type",
            widget_type="combo",
            values=["laplacian", "taubin"],
            default="laplacian",
        )
        self._smoother_type.pack(fill=X, padx=4, pady=2)

        self._smoother_iters = LabeledField(sec.body, label="Iterations", default="5")
        self._smoother_iters.pack(fill=X, padx=4, pady=2)

        self._smoother_lamb = LabeledField(sec.body, label="Lambda", default="0.5")
        self._smoother_lamb.pack(fill=X, padx=4, pady=2)

        self._smoother_nu = LabeledField(sec.body, label="Nu (Taubin)", default="-0.53")
        self._smoother_nu.pack(fill=X, padx=4, pady=2)

        return sec

    def _build_ese_section(self, parent: tk.Misc) -> CollapsibleSection:
        sec = CollapsibleSection(parent, title="Stage 2: ESE Parameters")
        sec.pack(fill=X, padx=8, pady=2)
        sec.collapse()

        self._n_electrodes = LabeledField(sec.body, label="Number of electrodes", default="")
        self._n_electrodes.pack(fill=X, padx=4, pady=2)

        self._ese_offset = LabeledField(sec.body, label="Offset (mm)", default="")
        self._ese_offset.pack(fill=X, padx=4, pady=2)

        self._ese_reference = LabeledField(
            sec.body,
            label="Reference",
            widget_type="combo",
            values=["average", "vertex"],
            default="average",
        )
        self._ese_reference.pack(fill=X, padx=4, pady=2)

        return sec

    def _build_neighborhood_section(self, parent: tk.Misc) -> CollapsibleSection:
        sec = CollapsibleSection(parent, title="Stage 2: Neighborhood")
        sec.pack(fill=X, padx=8, pady=2)
        sec.collapse()

        self._nhood_radius = LabeledField(sec.body, label="Radius (mm)", default="10.0")
        self._nhood_radius.pack(fill=X, padx=4, pady=2)

        self._k_neighbors = LabeledField(sec.body, label="K neighbors", default="")
        self._k_neighbors.pack(fill=X, padx=4, pady=2)

        self._pca_sigma = LabeledField(sec.body, label="PCA sigma (mm)", default="5.0")
        self._pca_sigma.pack(fill=X, padx=4, pady=2)

        self._min_neighbors = LabeledField(sec.body, label="Min neighbors", default="5")
        self._min_neighbors.pack(fill=X, padx=4, pady=2)

        self._weighted_pca = LabeledField(
            sec.body, label="Weighted PCA", widget_type="check", default="false"
        )
        self._weighted_pca.pack(fill=X, padx=4, pady=2)

        return sec

    def _build_stage3_placeholder(self, parent: tk.Misc) -> CollapsibleSection:
        sec = CollapsibleSection(parent, title="Stage 3: Real Electrode Locations (coming soon)")
        sec.pack(fill=X, padx=8, pady=2)
        sec.collapse()

        lbl = ttk.Label(sec.body, text="Stage 3 is under development.", foreground="gray")
        lbl.pack(padx=4, pady=8)

        return sec

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

        def _int(val: str, default: int | None = None) -> int | None:
            val = val.strip()
            return int(val) if val else default

        def _float(val: str, default: float | None = None) -> float | None:
            val = val.strip()
            return float(val) if val else default

        return Config(
            nifti_path=nifti,
            project_dir=project,
            fiducials_path=fiducials,
            auto_detect_fiducials=False,
            closing_radius=_int(self._closing_radius.get(), 5),
            otsu_scope=self._otsu_scope.get() or "all",  # type: ignore[arg-type]
            otsu_threshold_scale=_float(self._otsu_threshold.get(), 0.6),
            seal_enabled=self._seal_enabled.get() == "true",
            seal_radius=_int(self._seal_radius.get(), 4),
            cleaner_min_vertices=_int(self._cleaner_min.get(), 100),
            cleaner_merge_digits=_int(self._cleaner_digits.get(), 7),
            smoother_type=self._smoother_type.get() or "laplacian",
            smoother_iterations=_int(self._smoother_iters.get(), 5),
            smoother_lamb=_float(self._smoother_lamb.get(), 0.5),
            smoother_nu=_float(self._smoother_nu.get(), -0.53),
            n_electrodes=_int(self._n_electrodes.get()),
            ese_offset_mm=_float(self._ese_offset.get()),
            ese_reference=self._ese_reference.get() or None,
            neighborhood_radius_mm=_float(self._nhood_radius.get(), 10.0),
            k_neighbors=_int(self._k_neighbors.get()),
            use_weighted_pca=self._weighted_pca.get() == "true",
            pca_sigma_mm=_float(self._pca_sigma.get(), 5.0),
            min_neighbors=_int(self._min_neighbors.get(), 5),
        )

    # ------------------------------------------------------------------
    # Pipeline execution (background thread)
    # ------------------------------------------------------------------

    def _on_run(self) -> None:
        try:
            config = self._collect_config()
        except ValueError as exc:
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

        self._pipeline_thread = threading.Thread(
            target=self._run_pipeline, args=(config,), daemon=True
        )
        self._pipeline_thread.start()

    @staticmethod
    def _run_pipeline(config: Config) -> queue.Queue[str | None]:
        """Entry point for the background pipeline thread.

        Returns the queue so the main thread can reference it (unused, kept
        for symmetry).  In practice the queue is stored on ``self._log_queue``.
        """
        # This method is overridden per-instance in ``_on_run`` via the
        # closure; kept here only for type-checking clarity.
        raise NotImplementedError

    def _run_pipeline(self, config: Config) -> None:
        """Background thread: run the pipeline and post results to the queue."""
        try:
            self._log_queue.put("Building configuration...")
            stage1_result, ese_mesh = run(config)

            msg = f"Stage 1: mesh with {len(stage1_result.mesh.vertices)} vertices"
            self._log_queue.put(msg)

            if ese_mesh is not None:
                msg = f"Stage 2: ESE mesh with {len(ese_mesh.vertices)} vertices"
                self._log_queue.put(msg)

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
                if msg is _DONE_SENTINEL:
                    self._on_pipeline_done()
                    break
                if msg is _ERROR_SENTINEL:
                    self._on_pipeline_error()
                    break
                self._log_viewer.append(msg)
        except queue.Empty:
            pass
        self._root.after(100, self._poll_log_queue)

    def _on_pipeline_done(self) -> None:
        self._run_btn.configure(state=tk.NORMAL)
        self._viewer_btn.configure(state=tk.NORMAL)
        self._export_btn.configure(state=tk.NORMAL)
        self._results_viewer_btn.configure(state=tk.NORMAL)
        self._results_export_btn.configure(state=tk.NORMAL)
        self._update_results_info(success=True)

    def _on_pipeline_error(self) -> None:
        self._run_btn.configure(state=tk.NORMAL)
        self._update_results_info(success=False)

    def _update_results_info(self, *, success: bool) -> None:
        self._notebook.select(1)  # switch to Results tab
        if success:
            project = self._last_project_dir or "—"
            self._result_label.configure(
                text=f"Pipeline completed.\nProject directory: {project}",
                foreground="green",
            )
        else:
            self._result_label.configure(
                text="Pipeline failed. Check the log for details.",
                foreground="red",
            )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_open_viewer(self) -> None:
        if not self._last_project_dir:
            return

        project = Path(self._last_project_dir)
        mesh_path = project / "mesh" / "final_mesh.ply"
        fiducials_path = project / "fiducials" / "fiducials.json"
        normals_path = project / "ese" / "normals.npy"

        nifti = self._nifti.get()

        from .viewer import show_viewer

        kwargs: dict[str, Any] = {}
        if nifti:
            kwargs["nifti_path"] = nifti
        if mesh_path.exists():
            kwargs["mesh_path"] = str(mesh_path)
        if fiducials_path.exists():
            kwargs["fiducials_path"] = str(fiducials_path)
        if normals_path.exists():
            kwargs["normals_path"] = str(normals_path)

        if not kwargs:
            Messagebox.show_warning(
                "No mesh or NIfTI file found in the project.", parent=self._root
            )
            return

        threading.Thread(target=show_viewer, kwargs=kwargs, daemon=True).start()

    def _on_export_html(self) -> None:
        if not self._last_project_dir:
            return

        project = Path(self._last_project_dir)
        output = project / "viewer.html"

        self._log_viewer.append(f"Exporting HTML viewer to {output}...")

        def _export() -> None:
            try:
                from .html_export import export_project

                export_project(str(project), output)
                self._log_queue.put(f"HTML exported: {output}")
                self._log_queue.put(_DONE_SENTINEL)
            except Exception as exc:
                self._log_queue.put(f"HTML export failed: {exc}")
                self._log_queue.put(_ERROR_SENTINEL)

        threading.Thread(target=_export, daemon=True).start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the tkinter main loop."""
        self._root.mainloop()


def main() -> None:
    """Entry point for ``virda-gui``."""
    app = VirdaApp()
    app.run()
