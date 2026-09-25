"""Mesh processing tab: trial smoothers/density on the original mesh in memory.

The tab holds the *base* scalp mesh read once from the project and keeps it
untouched in memory.  Any change to a smoothing or density parameter
recomputes a preview from that base mesh via the pure ``virda.ops`` atoms (
:func:`virda.ops.atoms.smooth` / :func:`virda.ops.atoms.decimate`), applies no
smoothing when the smoother is set to "none", and never writes to disk.  The
working mesh is only persisted when the user presses "Save to project";
the current density/smoother parameters are GUI-only and are never written to a
pipeline config file.  The generated ESE mesh
(:func:`virda.ops.atoms.generate_ese` with the chosen offset) is streamed to
the host for overlay, not saved.  A scalp mesh can be produced straight from a
NIfTI scan ("Generate scalp mesh from NIfTI...") through the pure atoms
:func:`virda.ops.atoms.generate_scalp_surface` + :func:`virda.ops.atoms.clean`.

The tab is host-agnostic: everything the host needs to render or persist
travels through Qt signals carrying domain objects.  Mesh generation runs on a
dedicated worker thread (see :meth:`MeshProcessingTab.generate_from_nifti` and
:meth:`MeshProcessingTab.generate_ese`); only the pure atoms execute on the
thread, the resulting
:class:`~virda.models.scalp_mesh.ScalpMesh` /
:class:`~virda.models.ese_mesh.ESEMesh` is applied to the tab and signalled to
the host on the GUI thread.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyvista as pv
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
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
from virda.io.importers.ese_mesh import import_ese_mesh
from virda.io.importers.nifti import import_nifti
from virda.io.importers.scalp_mesh import import_scalp_mesh
from virda.models.ese_mesh import ESEMesh
from virda.models.scalp_mesh import ScalpMesh
from virda.ops.atoms import clean, decimate, generate_ese, generate_scalp_surface, smooth
from virda.ops.options import (
    CleanOptions,
    DecimateOptions,
    EseOptions,
    SealingOptions,
    SmoothOptions,
)
from virda_gui.state import AppState
from virda_gui.viewer.scene import scene_placement
from virda_gui.viewer.viewer_loaders import SceneData, collect_scene_data

_FINAL_MESH_FILENAME = "final_mesh.ply"
_ESE_MESH_FILENAME = "ese_mesh.ply"
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


def generate_mesh_from_nifti(
    path: str | Path,
    sealing: SealingOptions,
    cleaning: CleanOptions,
) -> ScalpMesh:
    """Segment *path* into a scalp mesh via the pure atoms.

    Runs :func:`virda.io.importers.nifti.import_nifti`,
    :func:`virda.ops.atoms.generate_scalp_surface` and
    :func:`virda.ops.atoms.clean` in sequence.  Qt-free; meant to run on a
    worker thread (see :class:`_BackgroundWorker`).
    """
    mri = import_nifti(path)
    surface = generate_scalp_surface(mri, sealing)
    return clean(surface.mesh, cleaning)


class _BackgroundWorker(QObject):
    """Run a pure callable off the GUI thread.

    Lives on a dedicated :class:`QThread`; ``done``/``failed`` are delivered
    back to the main thread because the tab (the receiver) lives there.
    """

    done = Signal(int, object)  # noqa: N815 - seq, result
    failed = Signal(int, str)  # noqa: N815 - seq, error message

    def __init__(self, fn: Callable[[], object], seq: int) -> None:
        super().__init__()
        self._fn = fn
        self._seq = seq

    def run(self) -> None:
        try:
            result = self._fn()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.failed.emit(self._seq, str(exc))
        else:
            self.done.emit(self._seq, result)


class MeshProcessingTab(QWidget):
    """In-memory mesh editing with a single explicit "Save to project" step."""

    previewMesh = Signal(object)  # noqa: N815 - a ScalpMesh preview in world coords
    eseMesh = Signal(object)  # noqa: N815 - the generated ESEMesh in world coords
    saved = Signal()  # noqa: N815 - fired after Save wrote the mesh to disk
    status = Signal(str)  # noqa: N815 - non-blocking log lines
    baseMeshChanged = Signal(object)  # noqa: N815 - the new base mesh path (Path | None)

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._base_mesh: ScalpMesh | None = None
        self._base_path: Path | None = None
        self._preview_mesh: ScalpMesh | None = None
        self._ese_mesh: ESEMesh | None = None
        self._active_mesh_label: QLabel | None = None
        self._vertices_export_button: QPushButton | None = None
        self._faces_export_button: QPushButton | None = None
        self._mesh_generation_btn: QPushButton | None = None
        self._generate_ese_btn: QPushButton | None = None
        self._generation_thread: QThread | None = None
        self._generation_worker: _BackgroundWorker | None = None
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

        panel_layout.addWidget(self._build_source_box())
        panel_layout.addWidget(self._build_parameters_box())
        panel_layout.addWidget(self._build_actions_box())
        panel_layout.addWidget(self._build_export_box())
        panel_layout.addStretch(1)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        self._preview_timer.timeout.connect(self._recompute_preview)
        self._preview_label = QLabel(
            "No base mesh loaded. Load a mesh or generate from NIfTI.", panel
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
        self._update_generation_buttons()
        self._update_export_controls()

    # ---- UI builders ----

    def _build_source_box(self) -> QGroupBox:
        box = QGroupBox("Base scalp mesh", self)
        row = QHBoxLayout(box)
        row.setContentsMargins(4, 4, 4, 4)
        row.setSpacing(6)
        self._base_label = QLabel("No base mesh loaded", box)
        row.addWidget(self._base_label, 1)
        load_btn = QPushButton("Load...", box)
        load_btn.clicked.connect(self._on_load_base)
        row.addWidget(load_btn)
        return box

    def _build_parameters_box(self) -> QGroupBox:
        box = QGroupBox("Mesh parameters", self)
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

        display_row = QHBoxLayout()
        self._show_nifti_chk = QCheckBox("Show NIfTI", box)
        self._show_base_chk = QCheckBox("Show base mesh", box)
        self._show_result_chk = QCheckBox("Show result mesh", box)
        self._show_nifti_chk.setChecked(True)
        self._show_base_chk.setChecked(False)
        self._show_result_chk.setChecked(True)
        self._show_nifti_chk.toggled.connect(self._on_display_toggled)
        self._show_base_chk.toggled.connect(self._on_display_toggled)
        self._show_result_chk.toggled.connect(self._on_display_toggled)
        display_row.addWidget(self._show_nifti_chk)
        display_row.addWidget(self._show_base_chk)
        display_row.addWidget(self._show_result_chk)
        display_row.addStretch(1)
        grid.addLayout(display_row)

        ese_row = QHBoxLayout()
        ese_row.addWidget(QLabel("ESE offset (mm):", box))
        self._ese_offset_spin = QDoubleSpinBox(box)
        self._ese_offset_spin.setRange(0.1, 50.0)
        self._ese_offset_spin.setSingleStep(0.5)
        self._ese_offset_spin.setValue(2.0)
        ese_row.addWidget(self._ese_offset_spin)
        ese_row.addWidget(QLabel("ESE generation uses the current preview mesh.", box))
        ese_row.addStretch(1)
        grid.addLayout(ese_row)

        return box

    def _build_actions_box(self) -> QGroupBox:
        box = QGroupBox("Actions", self)
        row = QHBoxLayout(box)
        row.setContentsMargins(4, 4, 4, 4)
        row.setSpacing(6)

        self._mesh_generation_btn = QPushButton("Generate scalp mesh from NIfTI...", box)
        self._mesh_generation_btn.clicked.connect(self._on_generate_from_nifti)
        row.addWidget(self._mesh_generation_btn)

        self._generate_ese_btn = QPushButton("Generate ESE mesh", box)
        self._generate_ese_btn.clicked.connect(self._on_generate_ese)
        row.addWidget(self._generate_ese_btn)

        save_btn = QPushButton("Save to project", box)
        save_btn.clicked.connect(self._on_save)
        row.addWidget(save_btn)

        reset_btn = QPushButton("Reset parameters", box)
        reset_btn.clicked.connect(self._on_reset)
        row.addWidget(reset_btn)
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
        return box

    def _active_mesh(self) -> tuple[ScalpMesh | ESEMesh, MeshKind] | None:
        if self._ese_mesh is not None:
            return self._ese_mesh, "ese"
        scalp = self.current_scalp_mesh()
        if scalp is not None:
            return scalp, "scalp"
        return None

    def _update_export_controls(self) -> None:
        active = self._active_mesh()
        if self._active_mesh_label is not None:
            self._active_mesh_label.setText(
                "Active mesh: none" if active is None else f"Active mesh: {active[1]}"
            )
        if self._vertices_export_button is not None:
            self._vertices_export_button.setEnabled(active is not None)
            self._vertices_export_button.setText(
                "Export vertices (NPY)..."
                if active is None
                else f"Export {active[1]} vertices (NPY)..."
            )
        if self._faces_export_button is not None:
            self._faces_export_button.setEnabled(active is not None)
            self._faces_export_button.setText(
                "Export faces (NPY)..." if active is None else f"Export {active[1]} faces (NPY)..."
            )

    def _on_export_vertices(self) -> None:
        self._export_active_mesh_array("vertices")

    def _on_export_faces(self) -> None:
        self._export_active_mesh_array("faces")

    def _export_start_path(self, mesh_kind: MeshKind, array_kind: MeshArray) -> str:
        """Default location for the NPY export dialog (project folder when known)."""
        filename = f"{mesh_kind}_{array_kind}.npy"
        project = self._state.last_project_dir
        if project:
            return str(Path(project) / filename)
        return filename

    def _export_active_mesh_array(self, array_kind: MeshArray) -> None:
        active = self._active_mesh()
        if active is None:
            return
        mesh, mesh_kind = active
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            f"Export {mesh_kind} {array_kind}",
            self._export_start_path(mesh_kind, array_kind),
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
                f"Export {mesh_kind} {array_kind}",
                f"Could not export {mesh_kind} {array_kind}:\n{exc}",
            )
            return
        frame_suffix = " in world coordinates" if array_kind == "vertices" else ""
        self.status.emit(f"Exported {mesh_kind} {array_kind} to {target}{frame_suffix}")

    # ---- parameter plumbing ----

    def _connect_parameter_edits(self) -> None:
        self._smoother_combo.currentIndexChanged.connect(self._on_parameter_edited)
        self._iterations_spin.valueChanged.connect(self._on_parameter_edited)
        self._lamb_spin.valueChanged.connect(self._on_parameter_edited)
        self._nu_spin.valueChanged.connect(self._on_parameter_edited)
        self._density_slider.valueChanged.connect(self._on_density_edited)
        self._density_slider.sliderMoved.connect(self._on_density_dragged)
        self._density_slider.sliderReleased.connect(self._on_density_released)

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

    def _mesh_to_polydata(self, mesh: ScalpMesh | ESEMesh) -> pv.PolyData:
        """Build a :class:`pv.PolyData` actor surface from a mesh model."""
        faces = np.column_stack(
            [np.full(len(mesh.faces), 3, dtype=np.int64), np.asarray(mesh.faces, dtype=np.int64)]
        ).ravel()
        return pv.PolyData(np.asarray(mesh.vertices, dtype=np.float64), faces)

    def _render_scene(self) -> None:
        """Compose the preview pane from the visible layers and their flags.

        The three checkboxes decide which layer is drawn: the NIfTI volume, the
        original base mesh and the in-RAM working result.  The base layer is
        skipped when it *is* the mesh currently shown as the result, and the
        result layer disappears after a save until the next edit.  The camera
        is framed only when the pane was empty, so parameter edits, layer
        toggles and volume loads keep the user's view.
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
                self._mesh_to_polydata(self._base_mesh), color="lightgray", opacity=0.6
            )
        if self._show_result_chk.isChecked() and not self._result_hidden and result is not None:
            poly = self._mesh_to_polydata(result)
            if not self._nifti_mm_scene:
                poly.transform(self._nifti_transform, inplace=True)
            self._result_actor = interactor.add_mesh(poly, color="salmon", opacity=0.9)
        interactor.add_axes(interactive=False)  # type: ignore[call-arg]
        if was_empty:
            interactor.reset_camera()  # type: ignore[call-arg]
        interactor.render()

    # ---- preview / actions ----

    def _recompute_preview(self) -> None:
        """Recompute the parameter-adjusted preview mesh whenever a base is set."""
        base = self._base_mesh
        if base is None:
            self._preview_mesh = None
            self._ese_mesh = None
            self._result_hidden = False
            self._render_scene()
            self._update_export_controls()
            return
        try:
            preview = self._compute_preview(base)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.status.emit(f"Mesh preview failed: {exc}")
            return
        self._preview_mesh = preview
        self._ese_mesh = None
        self._result_hidden = False
        self._preview_label.setText(
            f"Preview: {len(preview.vertices)} vertices (base {len(base.vertices)})"
        )
        self._render_scene()
        self._update_export_controls()
        self.previewMesh.emit(preview)

    def _on_load_base(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self, "Load scalp mesh", self._start_dir(), "Meshes (*.ply *.obj);;All files (*)"
        )
        if not path:
            return
        self.load_base(Path(path))

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
        self._ese_mesh = None
        self._result_hidden = False
        self._base_label.setText(str(path))
        self._preview_label.setText(f"Base mesh: {len(mesh.vertices)} vertices")
        self._render_scene()
        self._update_generation_buttons()
        self._update_export_controls()
        self.previewMesh.emit(mesh)
        return True

    def _start_dir(self) -> str:
        if self._base_path is not None:
            return str(self._base_path.parent)
        project = self._state.last_project_dir
        if project:
            return str(Path(project) / "mesh")
        return ""

    def _nifti_start_dir(self) -> str:
        project = self._state.last_project_dir
        if project:
            return str(Path(project) / "input")
        return ""

    def _generation_options(self) -> tuple[SealingOptions, CleanOptions]:
        """Build mesh-generation options from the GUI advanced settings."""
        advanced = self._state.advanced

        def _bool(key: str, default: bool) -> bool:
            return (advanced.get(key, "").strip().lower() == "true") or (
                not advanced.get(key, "").strip() and default
            )

        def _int(key: str, default: int) -> int:
            try:
                value = int(float(advanced.get(key, "")))
            except TypeError, ValueError:
                return default
            return value or default

        return (
            SealingOptions(
                seal_enabled=_bool("seal_enabled", True),
                seal_radius=_int("seal_radius", 4),
            ),
            CleanOptions(
                min_component_vertices=_int("cleaner_min_vertices", 100),
                merge_digits=_int("cleaner_merge_digits", 7),
            ),
        )

    def _on_generate_from_nifti(self) -> None:
        """Pick a NIfTI scan and start generating the scalp mesh in the background."""
        if self._generation_busy:
            return
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Generate scalp mesh from NIfTI",
            self._nifti_start_dir(),
            "NIfTI scans (*.nii.gz *.nii);;All files (*)",
        )
        if not path:
            return
        self.generate_from_nifti(Path(path), self._generation_options(), Path(path).name)

    def generate_from_nifti(
        self,
        path: str | Path,
        options: tuple[SealingOptions, CleanOptions],
        name: str | None = None,
    ) -> None:
        """Segment *path* into a scalp mesh on a background thread.

        The worker thread only runs the pure atoms; the resulting mesh is
        applied to the tab and emitted to the host on the GUI thread.  A new
        generation request retires any in-flight one; stale results are dropped
        via the generation counter.
        """
        if self._generation_busy:
            return
        path = Path(path)
        sealing, cleaning = options

        def _run() -> ScalpMesh:
            return generate_mesh_from_nifti(path, sealing, cleaning)

        self._start_generation(
            _run,
            kind="nifti",
            source_path=path,
            start_message=f"Generating scalp mesh from {name or path.name}...",
        )

    def generate_ese(self, base: ScalpMesh, offset_mm: float) -> None:
        """Offset *base* into an ESE mesh on a background thread."""
        if self._generation_busy:
            return
        offset_mm = round(offset_mm, 6)

        def _run() -> ESEMesh:
            return generate_ese(base, EseOptions(ese_offset_mm=offset_mm))

        self._start_generation(
            _run,
            kind="ese",
            start_message=f"Generating ESE mesh at {offset_mm:g} mm offset...",
        )

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
        self._generation_worker = _BackgroundWorker(fn, seq)
        self._generation_worker.moveToThread(self._generation_thread)
        self._generation_thread.started.connect(self._generation_worker.run)
        self._generation_worker.done.connect(self._on_generation_done)
        self._generation_worker.failed.connect(self._on_generation_failed)
        self._generation_thread.start()
        self.status.emit(start_message)

    def _on_generation_done(self, seq: int, result: object) -> None:
        if seq != self._generation_seq:
            return
        kind = self._pending_kind
        source_path = self._mesh_source_path
        self._finish_generation()
        if kind == "nifti_scene":
            scene: SceneData = result  # type: ignore[assignment]
            if scene.volume is not None:
                _, _, transform, mm_scene = scene_placement(scene.affine)
                self._nifti_path = source_path
                self._nifti_volume = scene.volume
                self._nifti_transform = transform
                self._nifti_mm_scene = mm_scene
                self.status.emit(f"NIfTI preview loaded ({scene.volume.dimensions}).")
                self._render_scene()
            return
        if kind == "ese":
            ese: ESEMesh = result  # type: ignore[assignment]
            self._ese_mesh = ese
            self.status.emit(f"ESE mesh generated: {len(ese.vertices)} vertices.")
            self._update_export_controls()
            self.eseMesh.emit(ese)
            return
        mesh: ScalpMesh = result  # type: ignore[assignment]
        assert source_path is not None
        self._base_mesh = mesh
        self._set_base_path(None)  # generated in memory; gains a path only on Save
        self._preview_mesh = None
        self._ese_mesh = None
        self._base_label.setText(
            f"Generated from {source_path.name}: {len(mesh.vertices)} vertices (unsaved)"
        )
        self._preview_label.setText(f"Base mesh: {len(mesh.vertices)} vertices")
        self._result_hidden = False
        self._render_scene()
        self._update_generation_buttons()
        self._update_export_controls()
        self.previewMesh.emit(mesh)
        self.status.emit(f"Scalp mesh generated: {len(mesh.vertices)} vertices.")

    def _on_generation_failed(self, seq: int, message: str) -> None:
        if seq != self._generation_seq:
            return
        kind = self._pending_kind
        if kind == "nifti_scene":
            self._finish_generation()
            self.status.emit(f"NIfTI preview failed: {message}")
            return
        if kind == "ese":
            title = "Generate ESE"
            subject = "ESE mesh"
            detail = f"ESE generation failed:\n{message}"
        else:
            title = "Generate scalp mesh"
            subject = "Scalp mesh"
            name = (
                self._mesh_source_path.name if self._mesh_source_path is not None else "NIfTI scan"
            )
            detail = f"Generation failed for {name}:\n{message}"
        self._finish_generation()
        QMessageBox.critical(self, title, detail)
        self.status.emit(f"{subject} generation failed: {message}")

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

    def _update_generation_buttons(self) -> None:
        """Reflect the busy flag and ESE's need for a base mesh PLY.

        Nothing can start while a generation runs; the ESE button additionally
        stays disabled until the base mesh has a ``*.ply`` path (set by
        ``Load...`` or by "Save to project").  Any change to the base path re-
        runs this via :attr:`baseMeshChanged`.
        """
        ready = not self._generation_busy
        if self._mesh_generation_btn is not None:
            self._mesh_generation_btn.setEnabled(ready)
        if self._generate_ese_btn is not None:
            base_ready = self._base_path is not None and self._base_path.suffix.lower() == ".ply"
            self._generate_ese_btn.setEnabled(ready and base_ready)

    def shutdown(self) -> None:
        """Stop the generation thread, if any.

        Safe to call at application teardown for tabs whose ``closeEvent`` is
        never delivered (children of a main window).
        """
        self._cancel_running_generation()

    def _on_generate_ese(self) -> None:
        if self._base_path is None:
            QMessageBox.warning(
                self, "Generate ESE", "Generate and save a scalp mesh to the project first."
            )
            return
        base = self.current_scalp_mesh()
        if base is None:
            QMessageBox.warning(self, "Generate ESE", "Load a base mesh first.")
            return
        self.generate_ese(base, self._ese_offset_spin.value())

    def _on_save(self) -> None:
        project = self._state.last_project_dir
        if not project:
            QMessageBox.warning(self, "Save mesh", "Open a project first so the mesh has a home.")
            return
        target = self._preview_mesh if self._preview_mesh is not None else self._base_mesh
        if target is None:
            QMessageBox.warning(self, "Save mesh", "Load a base mesh first.")
            return
        root = Path(project)
        mesh_path = root / "mesh" / _FINAL_MESH_FILENAME
        try:
            export_scalp_mesh(mesh_path, target)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Save mesh", f"Could not save mesh:\n{exc}")
            return
        self.status.emit(f"Saved working mesh ({len(target.vertices)} vertices) to {mesh_path}.")
        self._set_base_path(mesh_path)
        self._base_label.setText(str(mesh_path))
        self._result_hidden = True
        self._render_scene()
        self._update_generation_buttons()
        self.saved.emit()

    def _on_reset(self) -> None:
        widgets = (
            self._smoother_combo,
            self._iterations_spin,
            self._lamb_spin,
            self._nu_spin,
            self._density_slider,
            self._ese_offset_spin,
            self._show_nifti_chk,
            self._show_base_chk,
            self._show_result_chk,
        )
        for widget in widgets:
            widget.blockSignals(True)
        self._smoother_combo.setCurrentIndex(0)
        self._iterations_spin.setValue(5)
        self._lamb_spin.setValue(0.5)
        self._nu_spin.setValue(-0.53)
        self._density_slider.setValue(100)
        self._ese_offset_spin.setValue(2.0)
        self._show_nifti_chk.setChecked(True)
        self._show_base_chk.setChecked(False)
        self._show_result_chk.setChecked(True)
        for widget in widgets:
            widget.blockSignals(False)
        self._density_label.setText("100%")
        self._recompute_preview()

    # ---- project lifecycle ----

    def prefill_from_project(self, project: str | Path) -> None:
        """Load the project's final scalp mesh as the base plus its ESE mesh, if any."""
        self.clear()
        root = Path(project)
        candidate = root / "mesh" / _FINAL_MESH_FILENAME
        if candidate.is_file():
            self.load_base(candidate)
        ese_candidate = root / "ese" / _ESE_MESH_FILENAME
        if ese_candidate.is_file():
            try:
                ese = import_ese_mesh(ese_candidate)
            except ValueError as exc:
                self.status.emit(f"ESE mesh not loaded: {exc}")
            else:
                self._ese_mesh = ese
                self._update_export_controls()
                self.eseMesh.emit(ese)
                self.status.emit(f"ESE mesh loaded: {len(ese.vertices)} vertices.")
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

    def current_ese_mesh(self) -> ESEMesh | None:
        """The in-memory ESE mesh, if one was generated for the current preview."""
        return self._ese_mesh

    def clear(self) -> None:
        """Forget the in-memory mesh state without touching the project."""
        self._cancel_running_generation()
        self._base_mesh = None
        self._set_base_path(None)
        self._preview_mesh = None
        self._ese_mesh = None
        self._nifti_volume = None
        self._nifti_transform = np.eye(4)
        self._nifti_mm_scene = True
        self._nifti_path = None
        self._result_hidden = False
        self._base_label.setText("No base mesh loaded")
        self._preview_label.setText("No base mesh loaded. Load a mesh or generate from NIfTI.")
        self._render_scene()
        self._update_export_controls()
