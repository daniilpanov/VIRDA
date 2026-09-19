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
travels through Qt signals carrying domain objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
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
    QVBoxLayout,
    QWidget,
)

from virda.io.exporters.scalp_mesh import export_scalp_mesh
from virda.io.importers.nifti import import_nifti
from virda.io.importers.scalp_mesh import import_scalp_mesh
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

_FINAL_MESH_FILENAME = "final_mesh.ply"
_MESH_DENSITY_MIN = 1
_MESH_DENSITY_MAX = 100

_SMOOTHER_ITEMS = [
    ("none", "None (keep original)"),
    ("laplacian", "Laplacian"),
    ("taubin", "Taubin"),
]


class MeshProcessingTab(QWidget):
    """In-memory mesh editing with a single explicit "Save to project" step."""

    previewMesh = Signal(object)  # noqa: N815 - a ScalpMesh preview in world coords
    eseMesh = Signal(object)  # noqa: N815 - the generated ESEMesh in world coords
    saved = Signal()  # noqa: N815 - fired after Save wrote the mesh to disk
    status = Signal(str)  # noqa: N815 - non-blocking log lines

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._base_mesh: ScalpMesh | None = None
        self._base_path: Path | None = None
        self._preview_mesh: ScalpMesh | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_source_box())
        layout.addWidget(self._build_parameters_box())
        layout.addWidget(self._build_actions_box())
        layout.addStretch(1)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        self._preview_timer.timeout.connect(self._recompute_preview)
        self._preview_label = QLabel("No base mesh loaded. Load a mesh or generate from NIfTI.", self)
        layout.addWidget(self._preview_label)

        self._connect_parameter_edits()

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

        generate_btn = QPushButton("Generate scalp mesh from NIfTI...", box)
        generate_btn.clicked.connect(self._on_generate_from_nifti)
        row.addWidget(generate_btn)

        ese_btn = QPushButton("Generate ESE mesh", box)
        ese_btn.clicked.connect(self._on_generate_ese)
        row.addWidget(ese_btn)

        save_btn = QPushButton("Save to project", box)
        save_btn.clicked.connect(self._on_save)
        row.addWidget(save_btn)

        reset_btn = QPushButton("Reset parameters", box)
        reset_btn.clicked.connect(self._on_reset)
        row.addWidget(reset_btn)
        return box

    # ---- parameter plumbing ----

    def _connect_parameter_edits(self) -> None:
        self._smoother_combo.currentIndexChanged.connect(self._on_parameter_edited)
        self._iterations_spin.valueChanged.connect(self._on_parameter_edited)
        self._lamb_spin.valueChanged.connect(self._on_parameter_edited)
        self._nu_spin.valueChanged.connect(self._on_parameter_edited)
        self._density_slider.valueChanged.connect(self._on_density_edited)

    def _on_parameter_edited(self, *_args: Any) -> None:
        self._preview_timer.start()

    def _on_density_edited(self, *_args: Any) -> None:
        self._density_label.setText(f"{self._density_slider.value()}%")
        self._preview_timer.start()

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

    # ---- preview / actions ----

    def _recompute_preview(self) -> None:
        base = self._base_mesh
        if base is None:
            return
        try:
            preview = self._compute_preview(base)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.status.emit(f"Mesh preview failed: {exc}")
            return
        self._preview_mesh = preview
        self._preview_label.setText(
            f"Preview: {len(preview.vertices)} vertices (base {len(base.vertices)})"
        )
        self.previewMesh.emit(preview)

    def _on_load_base(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self, "Load scalp mesh", self._start_dir(), "Meshes (*.ply *.obj);;All files (*)"
        )
        if not path:
            return
        self.load_base(Path(path))

    def load_base(self, path: Path) -> bool:
        """Load the base mesh from *path*; returns False when it cannot be read."""
        try:
            mesh = import_scalp_mesh(path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.critical(self, "Load scalp mesh", f"Could not load mesh:\n{exc}")
            return False
        self._base_mesh = mesh
        self._base_path = path
        self._preview_mesh = None
        self._base_label.setText(str(path))
        self._preview_label.setText(f"Base mesh: {len(mesh.vertices)} vertices")
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
            except (TypeError, ValueError):
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
        """Segment a NIfTI scan into a scalp mesh using the pure atoms."""
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Generate scalp mesh from NIfTI",
            self._nifti_start_dir(),
            "NIfTI scans (*.nii.gz *.nii);;All files (*)",
        )
        if not path:
            return
        try:
            mri = import_nifti(path)
            sealing, cleaning = self._generation_options()
            surface = generate_scalp_surface(mri, sealing)
            mesh = clean(surface.mesh, cleaning)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.critical(self, "Generate scalp mesh", f"Generation failed:\n{exc}")
            return
        self._base_mesh = mesh
        self._base_path = Path(path)
        self._preview_mesh = None
        self._base_label.setText(str(path))
        self._preview_label.setText(f"Base mesh: {len(mesh.vertices)} vertices")
        self.previewMesh.emit(mesh)
        self.status.emit(
            f"Scalp mesh generated from {Path(path).name}: {len(mesh.vertices)} vertices."
        )

    def _on_generate_ese(self) -> None:
        base = self.current_scalp_mesh()
        if base is None:
            QMessageBox.warning(self, "Generate ESE", "Load a base mesh first.")
            return
        try:
            ese = generate_ese(
                base, EseOptions(ese_offset_mm=round(self._ese_offset_spin.value(), 6))
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.critical(self, "Generate ESE", f"ESE generation failed:\n{exc}")
            return
        self.status.emit(
            f"ESE mesh generated: {len(ese.vertices)} vertices at "
            f"{self._ese_offset_spin.value():g} mm offset."
        )
        self.eseMesh.emit(ese)

    def _on_save(self) -> None:
        project = self._state.last_project_dir
        if not project:
            QMessageBox.warning(
                self, "Save mesh", "Open a project first so the mesh has a home."
            )
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
        self.saved.emit()

    def _on_reset(self) -> None:
        widgets = (
            self._smoother_combo,
            self._iterations_spin,
            self._lamb_spin,
            self._nu_spin,
            self._density_slider,
            self._ese_offset_spin,
        )
        for widget in widgets:
            widget.blockSignals(True)
        self._smoother_combo.setCurrentIndex(0)
        self._iterations_spin.setValue(5)
        self._lamb_spin.setValue(0.5)
        self._nu_spin.setValue(-0.53)
        self._density_slider.setValue(100)
        self._ese_offset_spin.setValue(2.0)
        for widget in widgets:
            widget.blockSignals(False)
        self._density_label.setText("100%")
        self._recompute_preview()

    # ---- project lifecycle ----

    def prefill_from_project(self, project: str | Path) -> None:
        """Load the project's final scalp mesh, if any, as the base."""
        self.clear()
        root = Path(project)
        candidate = root / "mesh" / _FINAL_MESH_FILENAME
        if candidate.is_file():
            self.load_base(candidate)

    def current_scalp_mesh(self) -> ScalpMesh | None:
        """The in-memory mesh the user is working on (preview, else the base)."""
        return self._preview_mesh if self._preview_mesh is not None else self._base_mesh

    def clear(self) -> None:
        """Forget the in-memory mesh state without touching the project."""
        self._base_mesh = None
        self._base_path = None
        self._preview_mesh = None
        self._base_label.setText("No base mesh loaded")
        self._preview_label.setText("No base mesh loaded. Load a mesh or generate from NIfTI.")