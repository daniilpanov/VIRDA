"""Mesh processing tab: base mesh in, postprocessed final mesh out.

The tab holds the *base* scalp mesh (segmentation output, saved as
``mesh/base_mesh.ply``) untouched in memory.  Any change to a smoothing or
density parameter recomputes a preview from the base mesh (or from the saved
final mesh when checked) via the pure ``virda.ops`` atoms (
:func:`virda.ops.atoms.smooth` / :func:`virda.ops.atoms.decimate`), applies no
smoothing when the smoother is set to "none", and auto-saves the result to
``mesh/final_mesh.ply``.  The current density/smoother parameters are GUI-only
and are never written to a pipeline config file.  The base mesh itself is
produced in the Base Generation tab.

The tab is host-agnostic: everything the host needs to render or persist
travels through Qt signals carrying domain objects.  NIfTI volume previews
load on a dedicated worker thread (see :class:`virda_gui.workers.BackgroundWorker`);
only the pure loader executes on the thread.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyvista as pv
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from virda.io.exporters.mesh_arrays import export_mesh_faces, export_mesh_vertices
from virda.io.exporters.scalp_mesh import export_scalp_mesh
from virda.io.importers.scalp_mesh import import_scalp_mesh
from virda.models.scalp_mesh import ScalpMesh
from virda.ops.atoms import decimate, smooth
from virda.ops.options import DecimateOptions, SmoothOptions
from virda_gui.export.mesh_ply import export_mesh_ply
from virda_gui.state import AppState
from virda_gui.viewer.scene import scene_placement
from virda_gui.viewer.viewer_loaders import SceneData, collect_scene_data
from virda_gui.workers import BackgroundWorker

_FINAL_MESH_FILENAME = "final_mesh.ply"
_BASE_MESH_FILENAME = "base_mesh.ply"
_NIFTI_PREVIEW_STRIDE = 2
_MESH_DENSITY_MIN = 1
_MESH_DENSITY_MAX = 100
_DENSITY_THROTTLE_S = 0.5

_SMOOTHER_ITEMS = [
    ("none", "None (keep original)"),
    ("laplacian", "Laplacian"),
    ("taubin", "Taubin"),
]

MeshKind = Literal["scalp", "ese"]
MeshArray = Literal["vertices", "faces"]


class MeshProcessingTab(QWidget):
    """Base mesh in, autosaved postprocessed final mesh out."""

    previewMesh = Signal(object)  # noqa: N815 - a ScalpMesh preview in world coords
    saved = Signal()  # noqa: N815 - fired after Save wrote the mesh to disk
    status = Signal(str)  # noqa: N815 - non-blocking log lines
    baseMeshChanged = Signal(object)  # noqa: N815 - the new base mesh path (Path | None)
    eseRequested = Signal()  # noqa: N815 - jump to the ESE Surface tab
    baseRequested = Signal()  # noqa: N815 - jump to the Base Generation tab

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._base_mesh: ScalpMesh | None = None
        self._base_path: Path | None = None
        self._preview_mesh: ScalpMesh | None = None
        self._final_mesh: ScalpMesh | None = None
        self._active_mesh_label: QLabel | None = None
        self._vertices_export_button: QPushButton | None = None
        self._faces_export_button: QPushButton | None = None
        self._mesh_file_export_button: QPushButton | None = None
        self._cancel_generation_btn: QPushButton | None = None
        self._parameters_box: QGroupBox | None = None
        self._actions_box: QGroupBox | None = None
        self._generation_thread: QThread | None = None
        self._generation_worker: BackgroundWorker | None = None
        self._generation_seq = 0
        self._generation_busy = False
        self._pending_kind: str | None = None
        self._mesh_source_path: Path | None = None
        self._interactor: QtInteractor | None = None
        self._result_actor: Any | None = None
        self._base_actor: Any | None = None
        self._nifti_actor: Any | None = None
        self._nifti_volume: pv.ImageData | None = None
        self._nifti_transform: np.ndarray = np.eye(4)
        self._nifti_mm_scene = True
        self._nifti_path: Path | None = None
        self._result_hidden = False
        self._last_density_recompute = 0.0

        panel = QWidget(self)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.setSpacing(8)

        pipeline_row = QHBoxLayout()
        pipeline_row.setContentsMargins(0, 0, 0, 0)
        self._pipeline_label = QLabel(panel)
        self._pipeline_label.setWordWrap(True)
        pipeline_row.addWidget(self._pipeline_label, 1)
        panel_layout.addLayout(pipeline_row)

        panel_layout.addWidget(self._build_source_box())
        panel_layout.addWidget(self._build_parameters_box())
        panel_layout.addWidget(self._build_actions_box())
        panel_layout.addWidget(self._build_export_box())
        panel_layout.addStretch(1)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        self._preview_timer.timeout.connect(self._recompute_preview)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(1000)
        self._save_timer.timeout.connect(self._autosave_final)
        self._preview_label = QLabel(
            "No base mesh loaded. Generate one in the Base Generation tab.", panel
        )
        panel_layout.addWidget(self._preview_label)

        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._splitter.addWidget(panel)
        # The embedded 3D pane is part of the tab from the moment it is
        # created, so the preview area is always visible next to the controls.
        self._interactor = QtInteractor(parent=self)
        self._splitter.addWidget(self._interactor)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([430, 520])

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._splitter, 1)

        self.baseMeshChanged.connect(self._update_generation_buttons)
        self._connect_parameter_edits()
        self._update_smoother_controls()
        self._update_generation_buttons()
        self._update_export_controls()

    # ---- UI builders ----

    def _build_source_box(self) -> QGroupBox:
        box = QGroupBox("Base scalp mesh", self)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        self._base_label = QLabel("No base mesh loaded", box)
        self._base_label.setWordWrap(True)
        layout.addWidget(self._base_label)
        hint = QLabel("Generate the base mesh in the Base Generation tab.", box)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return box

    def _build_parameters_box(self) -> QGroupBox:
        box = QGroupBox("Mesh parameters", self)
        self._parameters_box = box
        grid = QVBoxLayout(box)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(6)

        smoother_row = QHBoxLayout()
        smoother_row.addWidget(QLabel("Smoother:", box))
        self._smoother_combo = QComboBox(box)
        for value, label in _SMOOTHER_ITEMS:
            self._smoother_combo.addItem(label, value)
        smoother_row.addWidget(self._smoother_combo)
        smoother_row.addWidget(QLabel("Iterations:", box))
        self._iterations_spin = QSpinBox(box)
        self._iterations_spin.setRange(1, 500)
        self._iterations_spin.setValue(5)
        smoother_row.addWidget(self._iterations_spin)
        grid.addLayout(smoother_row)

        lamb_nu_row = QHBoxLayout()
        lamb_nu_row.addWidget(QLabel("Lambda:", box))
        self._lamb_spin = QDoubleSpinBox(box)
        self._lamb_spin.setRange(0.01, 2.0)
        self._lamb_spin.setSingleStep(0.05)
        self._lamb_spin.setValue(0.5)
        lamb_nu_row.addWidget(self._lamb_spin)
        lamb_nu_row.addWidget(QLabel("Nu:", box))
        self._nu_spin = QDoubleSpinBox(box)
        self._nu_spin.setRange(-1.0, 1.0)
        self._nu_spin.setSingleStep(0.05)
        self._nu_spin.setValue(-0.53)
        lamb_nu_row.addWidget(self._nu_spin)
        grid.addLayout(lamb_nu_row)

        density_row = QHBoxLayout()
        density_row.addWidget(QLabel("Density %:", box))
        self._density_slider = QSlider(Qt.Orientation.Horizontal, box)
        self._density_slider.setRange(_MESH_DENSITY_MIN, _MESH_DENSITY_MAX)
        self._density_slider.setValue(100)
        density_row.addWidget(self._density_slider, 1)
        self._density_label = QLabel("100%", box)
        density_row.addWidget(self._density_label)
        grid.addLayout(density_row)

        self._from_final_chk = QCheckBox("Apply on final mesh", box)
        self._from_final_chk.setChecked(False)
        self._from_final_chk.setToolTip(
            "Postprocessing starts from the saved final mesh instead of the base mesh."
        )
        self._from_final_chk.toggled.connect(self._on_parameter_edited)
        grid.addWidget(self._from_final_chk)

        display_row = QHBoxLayout()
        self._show_nifti_chk = QCheckBox("Show NIfTI", box)
        self._show_base_chk = QCheckBox("Show base mesh", box)
        self._show_result_chk = QCheckBox("Show result mesh", box)
        self._show_edges_chk = QCheckBox("Show mesh edges", box)
        self._show_nifti_chk.setChecked(True)
        self._show_base_chk.setChecked(False)
        self._show_result_chk.setChecked(True)
        self._show_edges_chk.setChecked(False)
        self._show_nifti_chk.toggled.connect(self._on_display_toggled)
        self._show_base_chk.toggled.connect(self._on_display_toggled)
        self._show_result_chk.toggled.connect(self._on_display_toggled)
        self._show_edges_chk.toggled.connect(self._on_display_toggled)
        display_row.addWidget(self._show_nifti_chk)
        display_row.addWidget(self._show_base_chk)
        display_row.addWidget(self._show_result_chk)
        display_row.addWidget(self._show_edges_chk)
        display_row.addStretch(1)
        grid.addLayout(display_row)

        return box

    def _build_actions_box(self) -> QGroupBox:
        box = QGroupBox("Actions", self)
        self._actions_box = box
        outer = QVBoxLayout(box)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        base_btn = QPushButton("\u2190 Regenerate base", box)
        base_btn.setToolTip("Go to the Base Generation tab to rebuild the base mesh.")
        base_btn.clicked.connect(self.baseRequested.emit)
        row.addWidget(base_btn)

        ese_next_btn = QPushButton("Generate ESE \u2192", box)
        ese_next_btn.setToolTip("Continue to the ESE Surface tab to build the sensor surface.")
        ese_next_btn.setStyleSheet("font-weight: bold;")
        ese_next_btn.clicked.connect(self.eseRequested.emit)
        row.addWidget(ese_next_btn)

        self._cancel_generation_btn = QPushButton("Cancel", box)
        self._cancel_generation_btn.clicked.connect(self._on_cancel_generation)
        row.addWidget(self._cancel_generation_btn)
        outer.addLayout(row)

        save_row = QHBoxLayout()
        save_row.setContentsMargins(0, 0, 0, 0)
        save_row.setSpacing(6)

        run_all_btn = QPushButton("Run all postprocessings", box)
        run_all_btn.clicked.connect(self._on_run_all)
        run_all_btn.setToolTip("Apply smoothing and density now and save the final mesh.")
        save_row.addWidget(run_all_btn)

        reset_btn = QPushButton("Reset parameters", box)
        reset_btn.clicked.connect(self._on_reset)
        save_row.addWidget(reset_btn)
        outer.addLayout(save_row)
        return box

    def _build_export_box(self) -> QGroupBox:
        box = QGroupBox("Export mesh arrays", self)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(QLabel("Writes active mesh arrays in world coordinates.", box))
        self._active_mesh_label = QLabel(box)
        layout.addWidget(self._active_mesh_label)

        self._vertices_export_button = QPushButton(box)
        self._vertices_export_button.clicked.connect(self._on_export_vertices)
        layout.addWidget(self._vertices_export_button)
        self._faces_export_button = QPushButton(box)
        self._faces_export_button.clicked.connect(self._on_export_faces)
        layout.addWidget(self._faces_export_button)
        self._mesh_file_export_button = QPushButton(box)
        self._mesh_file_export_button.clicked.connect(self._on_export_mesh_file)
        layout.addWidget(self._mesh_file_export_button)
        return box

    def _active_mesh(self) -> ScalpMesh | None:
        return self.current_scalp_mesh()

    def _update_export_controls(self) -> None:
        active = self._active_mesh()
        if self._active_mesh_label is not None:
            self._active_mesh_label.setText(
                "Active mesh: none" if active is None else "Active mesh: scalp"
            )
        if self._vertices_export_button is not None:
            self._vertices_export_button.setEnabled(active is not None)
            self._vertices_export_button.setText("Export scalp vertices (NPY)...")
        if self._faces_export_button is not None:
            self._faces_export_button.setEnabled(active is not None)
            self._faces_export_button.setText("Export scalp faces (NPY)...")
        if self._mesh_file_export_button is not None:
            self._mesh_file_export_button.setEnabled(active is not None)
            self._mesh_file_export_button.setText("Export scalp mesh (PLY)...")

    def _on_export_vertices(self) -> None:
        self._export_active_mesh_array("vertices")

    def _on_export_faces(self) -> None:
        self._export_active_mesh_array("faces")

    def _on_export_mesh_file(self) -> None:
        mesh = self._active_mesh()
        if mesh is None:
            return
        mesh_kind = "scalp"
        project = self._state.last_project_dir
        start = (
            str(Path(project) / "mesh" / f"{mesh_kind}_mesh.ply")
            if project
            else f"{mesh_kind}_mesh.ply"
        )
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            f"Export {mesh_kind} mesh",
            start,
            "PLY (*.ply);;All files (*)",
        )
        if not path:
            return
        try:
            target = export_mesh_ply(path, mesh.vertices, mesh.faces)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(
                self,
                f"Export {mesh_kind} mesh",
                f"Could not export {mesh_kind} mesh:\n{exc}",
            )
            return
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.critical(
                self,
                f"Export {mesh_kind} mesh",
                f"Could not export {mesh_kind} mesh:\n{exc}",
            )
            return
        self.status.emit(f"Exported {mesh_kind} mesh to {target} in world coordinates")

    def _export_start_path(self, mesh_kind: MeshKind, array_kind: MeshArray) -> str:
        """Default location for the NPY export dialog (project folder when known)."""
        filename = f"{mesh_kind}_{array_kind}.npy"
        project = self._state.last_project_dir
        if project:
            return str(Path(project) / filename)
        return filename

    def _export_active_mesh_array(self, array_kind: MeshArray) -> None:
        mesh = self._active_mesh()
        if mesh is None:
            return
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            f"Export scalp {array_kind}",
            self._export_start_path("scalp", array_kind),
            "NumPy array (*.npy);;All files (*)",
        )
        if not path:
            return
        try:
            if array_kind == "vertices":
                target = export_mesh_vertices(path, mesh.vertices)
            elif array_kind == "faces":
                target = export_mesh_faces(path, mesh.faces)
            else:
                raise ValueError(f"unknown mesh array {array_kind!r}")
        except (OSError, ValueError) as exc:
            QMessageBox.critical(
                self,
                f"Export scalp {array_kind}",
                f"Could not export scalp {array_kind}:\n{exc}",
            )
            return
        frame_suffix = " in world coordinates" if array_kind == "vertices" else ""
        self.status.emit(f"Exported scalp {array_kind} to {target}{frame_suffix}")

    # ---- parameter plumbing ----

    def _connect_parameter_edits(self) -> None:
        self._smoother_combo.currentIndexChanged.connect(self._on_parameter_edited)
        self._smoother_combo.currentIndexChanged.connect(self._update_smoother_controls)
        self._iterations_spin.valueChanged.connect(self._on_parameter_edited)
        self._lamb_spin.valueChanged.connect(self._on_parameter_edited)
        self._nu_spin.valueChanged.connect(self._on_parameter_edited)
        self._density_slider.valueChanged.connect(self._on_density_edited)
        self._density_slider.sliderMoved.connect(self._on_density_dragged)
        self._density_slider.sliderReleased.connect(self._on_density_released)

    def _update_smoother_controls(self, *_args: Any) -> None:
        """Enable smoother params only when a real smoother is selected."""
        enabled = self._smoother_combo.currentData() != "none"
        self._iterations_spin.setEnabled(enabled)
        self._lamb_spin.setEnabled(enabled)
        self._nu_spin.setEnabled(enabled)

    def _on_parameter_edited(self, *_args: Any) -> None:
        self._preview_timer.start()

    def _on_density_edited(self, *_args: Any) -> None:
        self._density_label.setText(f"{self._density_slider.value()}%")
        self._preview_timer.start()

    def _on_density_dragged(self, *_args: Any) -> None:
        """Live-recompute while dragging, throttled to at most 2 Hz."""
        self._density_label.setText(f"{self._density_slider.value()}%")
        now = time.monotonic()
        if now - self._last_density_recompute >= _DENSITY_THROTTLE_S:
            self._last_density_recompute = now
            self._recompute_preview()

    def _on_density_released(self, *_args: Any) -> None:
        """Recompute once more at the final thumb position on mouse release."""
        self._last_density_recompute = 0.0
        self._recompute_preview()

    def _on_display_toggled(self, *_args: Any) -> None:
        self._render_scene()

    def _preview_source(self) -> ScalpMesh | None:
        """The mesh postprocessing starts from: saved final if checked, else base."""
        if self._from_final_chk.isChecked() and self._final_mesh is not None:
            return self._final_mesh
        return self._base_mesh

    def _compute_preview(self, base: ScalpMesh) -> ScalpMesh:
        """Run the configured smoother/density on *base* without mutating it."""
        mesh = base
        smoother = self._smoother_combo.currentData()
        if smoother != "none":
            mesh = smooth(
                mesh,
                SmoothOptions(
                    smoother=smoother,
                    iterations=self._iterations_spin.value(),
                    lamb=round(self._lamb_spin.value(), 6),
                    nu=round(self._nu_spin.value(), 6),
                ),
            )
        density = self._density_slider.value()
        if density < 100:
            mesh = decimate(mesh, DecimateOptions(density_percent=density))
        return mesh

    # ---- in-tab live preview ----

    def _ensure_preview_pane(self) -> QtInteractor:
        """Return the embedded 3D pane created in :meth:`__init__`."""
        if self._interactor is None:
            self._interactor = QtInteractor(parent=self)
        return self._interactor

    def _mesh_to_polydata(self, mesh: ScalpMesh) -> pv.PolyData:
        """Build a :class:`pv.PolyData` actor surface from a mesh model."""
        faces = np.column_stack(
            [np.full(len(mesh.faces), 3, dtype=np.int64), np.asarray(mesh.faces, dtype=np.int64)]
        ).ravel()
        return pv.PolyData(np.asarray(mesh.vertices, dtype=np.float64), faces)

    def _display_poly(self, mesh: ScalpMesh) -> pv.PolyData:
        """Build a preview-pane surface with the pane's display transform applied.

        Every layer (base, result) goes through this helper so all of them
        live in the same display frame even when the NIfTI scene is not
        stored in millimetres.
        """
        poly = self._mesh_to_polydata(mesh)
        if not self._nifti_mm_scene:
            poly.transform(self._nifti_transform, inplace=True)
        return poly

    def _render_scene(self) -> None:
        """Compose the preview pane from the visible layers and their flags.

        The checkboxes decide which layer is drawn: the NIfTI volume, the
        original base mesh and the in-RAM working result.  The base layer is
        skipped when it *is* the mesh currently shown as the result.  The
        camera is framed only when the pane was empty, so parameter edits,
        layer toggles and volume loads keep the user's view.
        """
        interactor = self._ensure_preview_pane()
        result = self.current_scalp_mesh()
        was_empty = (
            self._result_actor is None and self._base_actor is None and self._nifti_actor is None
        )
        interactor.clear()
        self._nifti_actor = None
        self._base_actor = None
        self._result_actor = None
        if self._show_nifti_chk.isChecked() and self._nifti_volume is not None:
            self._nifti_actor = interactor.add_volume(
                self._nifti_volume, cmap="bone", opacity="sigmoid", mapper="smart"
            )
        if (
            self._show_base_chk.isChecked()
            and result is not None
            and self._base_mesh is not None
            and result is not self._base_mesh
        ):
            self._base_actor = interactor.add_mesh(
                self._display_poly(self._base_mesh),
                color="lightgray",
                opacity=0.6,
                show_edges=self._show_edges_chk.isChecked(),
            )
        if self._show_result_chk.isChecked() and not self._result_hidden and result is not None:
            self._result_actor = interactor.add_mesh(
                self._display_poly(result),
                color="salmon",
                opacity=0.9,
                show_edges=self._show_edges_chk.isChecked(),
            )
        interactor.add_axes(interactive=False)  # type: ignore[call-arg]
        if was_empty:
            interactor.reset_camera()  # type: ignore[call-arg]
        interactor.render()
        self._refresh_pipeline_badge()

    # ---- preview / actions ----

    def _refresh_pipeline_badge(self) -> None:
        """Show the NIfTI -> base -> final pipeline state as short badges."""

        def _mark(ready: bool) -> str:
            return "[ok]" if ready else "[missing]"

        self._pipeline_label.setText(
            f"NIfTI {_mark(self._nifti_volume is not None)}  "
            f"Base {_mark(self._base_mesh is not None)}  "
            f"Final {_mark(self._final_mesh is not None)}"
        )

    def _describe_mesh(self, mesh: ScalpMesh) -> str:
        """Short vertex/face summary used by the status labels."""
        return f"{len(mesh.vertices)}v/{len(mesh.faces)}f"

    def _recompute_preview(self) -> None:
        """Recompute the parameter-adjusted preview mesh from the chosen source."""
        base = self._preview_source()
        if base is None:
            self._preview_mesh = None
            self._result_hidden = False
            self._render_scene()
            self._update_generation_buttons()
            self._update_export_controls()
            return
        try:
            preview = self._compute_preview(base)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.status.emit(f"Mesh preview failed: {exc}")
            return
        self._preview_mesh = preview
        self._result_hidden = False
        self._preview_label.setText(
            f"Preview: {self._describe_mesh(preview)} (base {self._describe_mesh(base)})"
        )
        self._render_scene()
        self._update_generation_buttons()
        self._update_export_controls()
        self.previewMesh.emit(preview)
        self._save_timer.start()

    def _autosave_final(self) -> None:
        """Write the current preview to final_mesh.ply (debounced postprocessing)."""
        project = self._state.last_project_dir
        preview = self._preview_mesh
        if not project or preview is None:
            return
        final_path = Path(project) / "mesh" / _FINAL_MESH_FILENAME
        try:
            export_scalp_mesh(final_path, preview)
        except (OSError, ValueError) as exc:
            self.status.emit(f"Auto-save failed:\n{exc}")
            return
        self._final_mesh = preview
        self.status.emit(
            f"Auto-saved final mesh ({len(preview.vertices)} vertices) to {final_path}."
        )
        self.saved.emit()

    def _on_run_all(self) -> None:
        """Recompute postprocessing now and flush the auto-save immediately."""
        self._save_timer.stop()
        self._recompute_preview()
        self._autosave_final()

    def _set_base_path(self, path: Path | None) -> None:
        """Record the base mesh path and notify listeners of the change.

        Every mutation of :attr:`_base_path` must go through this setter so the
        ESE button (and any other listener) is re-evaluated exactly when the
        base mesh changes, including when the path is cleared.
        """
        if path == self._base_path:
            return
        self._base_path = Path(path) if path is not None else None
        self.baseMeshChanged.emit(self._base_path)

    def load_base(self, path: Path) -> bool:
        """Load the base mesh from *path*; returns False when it cannot be read."""
        try:
            mesh = import_scalp_mesh(path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.critical(self, "Load scalp mesh", f"Could not load mesh:\n{exc}")
            return False
        self._base_mesh = mesh
        self._set_base_path(path)
        self._preview_mesh = None
        self._final_mesh = None
        self._result_hidden = False
        self._base_label.setText(str(path))
        self._preview_label.setText(f"Base mesh: {self._describe_mesh(mesh)}")
        self._render_scene()
        self._update_generation_buttons()
        self._update_export_controls()
        self.previewMesh.emit(mesh)
        return True

    def _start_generation(
        self,
        fn: Callable[[], object],
        kind: str,
        start_message: str,
        source_path: Path | None = None,
    ) -> None:
        if self._generation_busy:
            return
        self._generation_seq += 1
        seq = self._generation_seq
        self._cancel_running_generation()
        self._generation_busy = True
        self._pending_kind = kind
        self._mesh_source_path = source_path
        self._update_generation_buttons()

        self._generation_thread = QThread(self)
        self._generation_worker = BackgroundWorker(fn, seq)
        self._generation_worker.moveToThread(self._generation_thread)
        self._generation_thread.started.connect(self._generation_worker.run)
        self._generation_worker.done.connect(self._on_generation_done)
        self._generation_worker.failed.connect(self._on_generation_failed)
        self._generation_thread.start()
        self.status.emit(start_message)

    def _on_generation_done(self, seq: int, result: object) -> None:
        if seq != self._generation_seq:
            return
        source_path = self._mesh_source_path
        self._finish_generation()
        scene: SceneData = result  # type: ignore[assignment]
        if scene.volume is not None:
            _, _, transform, mm_scene = scene_placement(scene.affine)
            self._nifti_path = source_path
            self._nifti_volume = scene.volume
            self._nifti_transform = transform
            self._nifti_mm_scene = mm_scene
            self.status.emit(f"NIfTI preview loaded ({scene.volume.dimensions}).")
            self._render_scene()

    def _on_generation_failed(self, seq: int, message: str) -> None:
        if seq != self._generation_seq:
            return
        self._finish_generation()
        self.status.emit(f"NIfTI preview failed: {message}")

    def _finish_generation(self) -> None:
        """Retire the finished generation thread and re-enable the UI."""
        # deleteLater() must be posted while the worker's event loop is still
        # live, otherwise the DeferredDelete event is never processed.
        if self._generation_worker is not None:
            self._generation_worker.deleteLater()
        if self._generation_thread is not None:
            self._generation_thread.quit()
            self._generation_thread.wait()
            if self._generation_thread.isFinished():
                self._generation_thread.deleteLater()
            self._generation_thread = None
        self._generation_worker = None
        self._mesh_source_path = None
        self._pending_kind = None
        self._generation_busy = False
        self._update_generation_buttons()

    def _cancel_running_generation(self) -> None:
        """Stop an in-flight generation so a newer request replaces it.

        A previous generation may still be running its atoms on a worker
        thread; retiring it here prevents the abandoned thread (and its
        ``done``/``failed`` emissions) from lingering after a new call.
        """
        thread = self._generation_thread
        if thread is None:
            return
        worker = self._generation_worker
        if worker is not None:
            worker.done.disconnect(self._on_generation_done)
            worker.failed.disconnect(self._on_generation_failed)
            # deleteLater() must be posted while the worker's event loop is
            # still live, otherwise the DeferredDelete event is never processed.
            worker.deleteLater()
        thread.quit()
        thread.wait(3000)
        # Never delete a thread that is still running (wait timed out);
        # destroying a live QThread is undefined behaviour.  In that case
        # the thread keeps running under its parent until it finishes.
        if thread.isFinished():
            thread.deleteLater()
        self._generation_thread = None
        self._generation_worker = None
        self._mesh_source_path = None
        self._pending_kind = None
        self._generation_busy = False
        self._update_generation_buttons()

    def _on_cancel_generation(self) -> None:
        """Retire the in-flight generation so the UI becomes idle again."""
        if not self._generation_busy:
            return
        self._cancel_running_generation()
        self.status.emit("Generation cancelled.")

    def set_locked(self, locked: bool) -> None:
        """Disable the parameter and action controls while generation runs.

        The Cancel button stays enabled so an in-flight generation can always
        be stopped.
        """
        if self._parameters_box is not None:
            self._parameters_box.setEnabled(not locked)
        if self._actions_box is not None:
            self._actions_box.setEnabled(not locked)
        if locked and self._cancel_generation_btn is not None:
            self._cancel_generation_btn.setEnabled(True)

    def _update_generation_buttons(self) -> None:
        """Reflect the busy flag; nothing can start while a generation runs."""
        ready = not self._generation_busy
        self.set_locked(not ready)
        if self._cancel_generation_btn is not None:
            self._cancel_generation_btn.setEnabled(not ready)

    def shutdown(self) -> None:
        """Stop the generation thread, if any.

        Safe to call at application teardown for tabs whose ``closeEvent`` is
        never delivered (children of a main window).  Finalizes the embedded
        preview pane first so no VTK observer can fire after the GL context
        is gone.
        """
        if self._interactor is not None:
            self._interactor.close()
        self._save_timer.stop()
        self._cancel_running_generation()

    def _on_reset(self) -> None:
        widgets = (
            self._smoother_combo,
            self._iterations_spin,
            self._lamb_spin,
            self._nu_spin,
            self._density_slider,
            self._from_final_chk,
            self._show_nifti_chk,
            self._show_base_chk,
            self._show_result_chk,
            self._show_edges_chk,
        )
        for widget in widgets:
            widget.blockSignals(True)
        self._smoother_combo.setCurrentIndex(0)
        self._iterations_spin.setValue(5)
        self._lamb_spin.setValue(0.5)
        self._nu_spin.setValue(-0.53)
        self._density_slider.setValue(100)
        self._from_final_chk.setChecked(False)
        self._show_nifti_chk.setChecked(True)
        self._show_base_chk.setChecked(False)
        self._show_result_chk.setChecked(True)
        self._show_edges_chk.setChecked(False)
        for widget in widgets:
            widget.blockSignals(False)
        self._density_label.setText("100%")
        self._update_smoother_controls()
        self._recompute_preview()

    # ---- project lifecycle ----

    def prefill_from_project(self, project: str | Path) -> None:
        """Load the project's base and final scalp meshes, if any.

        The base mesh comes only from ``base_mesh.ply``: a final mesh on
        disk is never used as the base, so the base path label cannot point
        at the final file.  Without a base the tab stays empty until the
        user loads or generates one.
        """
        self.clear()
        root = Path(project)
        base_candidate = root / "mesh" / _BASE_MESH_FILENAME
        final_candidate = root / "mesh" / _FINAL_MESH_FILENAME
        if base_candidate.is_file():
            self.load_base(base_candidate)
        if final_candidate.is_file():
            try:
                self._final_mesh = import_scalp_mesh(final_candidate)
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                self.status.emit(f"Final mesh not loaded: {exc}")
        nifti = self._find_project_nifti(root)
        if nifti is not None:
            self._start_nifti_scene(nifti)

    def _find_project_nifti(self, project_root: Path) -> Path | None:
        """The input NIfTI file of *project_root*, if any (used for preview)."""
        for pattern in ("input/*.nii.gz", "input/*.nii"):
            matches = sorted(project_root.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _start_nifti_scene(self, path: str | Path) -> None:
        """Load a NIfTI volume for the preview pane on a background thread."""
        nifti_path = str(path)

        def _run() -> SceneData:
            return collect_scene_data(
                nifti_path=nifti_path,
                downsample_stride=_NIFTI_PREVIEW_STRIDE,
                log=lambda _m: None,
            )

        self._start_generation(
            _run,
            kind="nifti_scene",
            source_path=Path(nifti_path),
            start_message=f"Loading NIfTI preview from {Path(nifti_path).name}...",
        )

    def current_scalp_mesh(self) -> ScalpMesh | None:
        """The in-memory mesh the user is working on (preview, else the base)."""
        return self._preview_mesh if self._preview_mesh is not None else self._base_mesh

    def base_mesh(self) -> ScalpMesh | None:
        """The in-memory base mesh, if one was loaded or generated."""
        return self._base_mesh

    def splitter_state(self) -> Any:
        """Opaque splitter layout for session restore."""
        return self._splitter.saveState()

    def restore_splitter_state(self, state: Any) -> None:
        """Restore a splitter layout saved by :meth:`splitter_state`."""
        self._splitter.restoreState(state)

    def clear(self) -> None:
        """Forget the in-memory mesh state without touching the project."""
        self._save_timer.stop()
        self._cancel_running_generation()
        self._base_mesh = None
        self._set_base_path(None)
        self._preview_mesh = None
        self._final_mesh = None
        self._nifti_volume = None
        self._nifti_transform = np.eye(4)
        self._nifti_mm_scene = True
        self._nifti_path = None
        self._result_hidden = False
        self._base_label.setText("No base mesh loaded")
        self._preview_label.setText("No base mesh loaded. Generate one in the Base Generation tab.")
        self._render_scene()
        self._update_generation_buttons()
        self._update_export_controls()
