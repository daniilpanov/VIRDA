"""Base scalp-mesh generation tab: NIfTI scan in, base mesh out.

Owns the generation parameters (approximate voxel size driving the
marching-cubes step, head-cavity sealing) and runs
:func:`virda.ops.atoms.generate_scalp_surface` on a worker thread.  On
success the mesh is auto-saved as both ``mesh/base_mesh.ply`` and
``mesh/final_mesh.ply`` together with the ``mesh/source_nifti.sha256``
sidecar, and the host reloads the mesh-processing tab from disk.

Generating while derived artifacts exist replaces the base mesh, so the
Mesh, ESE and Points steps are invalidated; the tab warns about that
before starting.  An embedded 3D pane previews the generated base mesh.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from virda.io.exporters.scalp_mesh import export_scalp_mesh
from virda.io.importers.nifti import import_nifti
from virda.mesh.density import step_size_for_voxel_size
from virda.models.scalp_mesh import ScalpMesh
from virda.ops.atoms import clean, generate_scalp_surface
from virda.ops.options import CleanOptions, SealingOptions
from virda_gui.scan_hash import sha256_file, write_source_hash
from virda_gui.state import AppState
from virda_gui.workers import BackgroundWorker

_BASE_MESH_FILENAME = "base_mesh.ply"
_FINAL_MESH_FILENAME = "final_mesh.ply"
_DEFAULT_SEAL_RADIUS = 4


def generate_mesh_from_nifti(
    path: str | Path,
    sealing: SealingOptions,
    cleaning: CleanOptions,
    voxel_size_mm: float | None = None,
) -> ScalpMesh:
    """Segment *path* into a scalp mesh via the pure atoms.

    Runs :func:`virda.io.importers.nifti.import_nifti`,
    :func:`virda.ops.atoms.generate_scalp_surface` and
    :func:`virda.ops.atoms.clean` in sequence.  Qt-free; meant to run on a
    worker thread (see :class:`virda_gui.workers.BackgroundWorker`).
    """
    mri = import_nifti(path)
    surface = generate_scalp_surface(mri, sealing, voxel_size_mm=voxel_size_mm)
    return clean(surface.mesh, cleaning)


class BaseGenerationTab(QWidget):
    """NIfTI-to-base-mesh generation with immediate save."""

    baseMesh = Signal(object)  # noqa: N815 - the generated ScalpMesh in world coords
    saved = Signal()  # noqa: N815 - fired after auto-save wrote the meshes to disk
    status = Signal(str)  # noqa: N815 - non-blocking log lines
    busyChanged = Signal(bool)  # noqa: N815 - generation running
    meshRequested = Signal()  # noqa: N815 - jump to the Mesh Processing tab

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._source_path: Path | None = None
        self._spacing: tuple[float, float, float] | None = None
        self._base_mesh: ScalpMesh | None = None
        self._generate_btn: QPushButton | None = None
        self._cancel_btn: QPushButton | None = None
        self._generation_thread: QThread | None = None
        self._generation_worker: BackgroundWorker | None = None
        self._generation_seq = 0
        self._generation_busy = False
        self._interactor: QtInteractor | None = None
        self._base_actor: Any | None = None

        panel = QWidget(self)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.setSpacing(8)

        panel_layout.addWidget(self._build_source_box())
        panel_layout.addWidget(self._build_parameters_box())
        panel_layout.addWidget(self._build_actions_box())
        panel_layout.addStretch(1)

        self._status_label = QLabel("Import a brain scan, then generate the base mesh.", panel)
        self._status_label.setWordWrap(True)
        panel_layout.addWidget(self._status_label)

        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._splitter.addWidget(panel)
        self._interactor = QtInteractor(parent=self)
        self._splitter.addWidget(self._interactor)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([430, 520])

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._splitter, 1)

        self._update_buttons()

    # ---- UI builders ----

    def _build_source_box(self) -> QGroupBox:
        box = QGroupBox("Brain scan (NIfTI)", self)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        self._source_label = QLabel("No scan in project", box)
        self._source_label.setWordWrap(True)
        layout.addWidget(self._source_label)
        hint = QLabel("Import scans via File \u2192 Import files.", box)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return box

    def _build_parameters_box(self) -> QGroupBox:
        box = QGroupBox("Generation parameters", self)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        voxel_row = QHBoxLayout()
        voxel_row.addWidget(QLabel("Voxel size (mm):", box))
        self._voxel_spin = QDoubleSpinBox(box)
        self._voxel_spin.setRange(0.1, 10.0)
        self._voxel_spin.setSingleStep(0.1)
        self._voxel_spin.setDecimals(2)
        self._voxel_spin.setValue(1.0)
        self._voxel_spin.setToolTip(
            "Approximate voxel size of the scan; the marching-cubes step "
            "is derived from it. Fractional values are allowed."
        )
        self._voxel_spin.valueChanged.connect(self._refresh_step_labels)
        voxel_row.addWidget(self._voxel_spin)
        voxel_row.addStretch(1)
        layout.addLayout(voxel_row)

        self._detected_label = QLabel("Detected voxel size: --", box)
        layout.addWidget(self._detected_label)
        self._real_label = QLabel("Real voxel size: -- (marching cube step = --)", box)
        layout.addWidget(self._real_label)

        seal_row = QHBoxLayout()
        self._seal_chk = QCheckBox("Seal head cavity", box)
        self._seal_chk.setChecked(True)
        self._seal_chk.setToolTip("Fill the head cavity in the segmentation mask.")
        seal_row.addWidget(self._seal_chk)
        seal_row.addWidget(QLabel("Radius:", box))
        self._seal_radius_spin = QSpinBox(box)
        self._seal_radius_spin.setRange(1, 20)
        self._seal_radius_spin.setValue(_DEFAULT_SEAL_RADIUS)
        self._seal_radius_spin.setToolTip("Sealing neighborhood radius in voxels.")
        seal_row.addWidget(self._seal_radius_spin)
        seal_row.addStretch(1)
        layout.addLayout(seal_row)
        layout.addWidget(
            QLabel(
                "Sealing fills the head cavity so the scalp surface closes "
                "at the neck instead of leaking inside.",
                box,
            )
        )

        clean_row = QHBoxLayout()
        clean_row.addWidget(QLabel("Min component vertices:", box))
        self._cleaner_min_vertices_spin = QSpinBox(box)
        self._cleaner_min_vertices_spin.setRange(1, 100000)
        self._cleaner_min_vertices_spin.setValue(100)
        self._cleaner_min_vertices_spin.setToolTip(
            "Drop mesh components smaller than this after extraction."
        )
        clean_row.addWidget(self._cleaner_min_vertices_spin)
        clean_row.addWidget(QLabel("Merge digits:", box))
        self._cleaner_merge_digits_spin = QSpinBox(box)
        self._cleaner_merge_digits_spin.setRange(1, 12)
        self._cleaner_merge_digits_spin.setValue(7)
        self._cleaner_merge_digits_spin.setToolTip(
            "Coordinate rounding used when merging duplicate vertices."
        )
        clean_row.addWidget(self._cleaner_merge_digits_spin)
        clean_row.addStretch(1)
        layout.addLayout(clean_row)
        return box

    def _build_actions_box(self) -> QGroupBox:
        box = QGroupBox("Actions", self)
        row = QHBoxLayout(box)
        row.setContentsMargins(4, 4, 4, 4)
        row.setSpacing(6)

        self._generate_btn = QPushButton("Generate base mesh", box)
        self._generate_btn.setToolTip("Segment the scan and save the base mesh.")
        self._generate_btn.clicked.connect(self._on_generate)
        row.addWidget(self._generate_btn)

        self._cancel_btn = QPushButton("Cancel", box)
        self._cancel_btn.clicked.connect(self._on_cancel_generation)
        row.addWidget(self._cancel_btn)

        mesh_next_btn = QPushButton("Open in Mesh Processing \u2192", box)
        mesh_next_btn.setToolTip("Continue postprocessing the base mesh.")
        mesh_next_btn.setStyleSheet("font-weight: bold;")
        mesh_next_btn.clicked.connect(self.meshRequested.emit)
        row.addWidget(mesh_next_btn)
        return box

    # ---- source ----

    def _project_scan(self) -> Path | None:
        project = self._state.last_project_dir
        if not project:
            return None
        matches = sorted(Path(project).glob("input/*.nii*"))
        return matches[0] if matches else None

    def set_source(self, path: str | Path) -> None:
        """Use *path* as the generation source and prefill the voxel size."""
        target = Path(path)
        self._source_path = target
        self._source_label.setText(str(target))
        try:
            mri = import_nifti(target)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self._spacing = None
            self.status.emit(f"Could not read scan spacing:\n{exc}")
        else:
            self._spacing = (
                float(mri.spacing[0]),
                float(mri.spacing[1]),
                float(mri.spacing[2]),
            )
            mean_spacing = sum(self._spacing) / 3.0
            self._detected_label.setText(f"Detected voxel size: {mean_spacing:g} mm")
            self._voxel_spin.setValue(round(mean_spacing, 2))
        self._refresh_step_labels()

    def _refresh_step_labels(self) -> None:
        """Recompute the step/real-voxel labels from the voxel spin; no mesh work."""
        if self._spacing is None:
            self._real_label.setText("Real voxel size: -- (marching cube step = --)")
            return
        try:
            step, real = step_size_for_voxel_size(self._voxel_spin.value(), self._spacing)
        except ValueError:
            self._real_label.setText("Real voxel size: -- (marching cube step = --)")
            return
        self._real_label.setText(
            f"Real voxel size: {real[0]:g} x {real[1]:g} x {real[2]:g} mm "
            f"(marching cube step = {step})"
        )

    # ---- generation ----

    def _cleaning_options(self) -> CleanOptions:
        return CleanOptions(
            min_component_vertices=self._cleaner_min_vertices_spin.value(),
            merge_digits=self._cleaner_merge_digits_spin.value(),
        )

    def _derived_artifacts_exist(self) -> bool:
        """Whether generating now would replace saved base/final/ESE meshes."""
        project = self._state.last_project_dir
        if not project:
            return False
        root = Path(project)
        if (root / "mesh" / _BASE_MESH_FILENAME).is_file():
            return True
        if (root / "mesh" / _FINAL_MESH_FILENAME).is_file():
            return True
        return any((root / "ese").glob("*"))

    def _on_generate(self) -> None:
        if self._generation_busy:
            return
        if self._source_path is None:
            source = self._project_scan()
            if source is None:
                QMessageBox.warning(
                    self, "Generate base mesh", "Import a brain scan (NIfTI) first."
                )
                return
            self.set_source(source)
        if self._derived_artifacts_exist() and not self._confirm_overwrite():
            return
        assert self._source_path is not None
        source = self._source_path
        sealing = SealingOptions(
            seal_enabled=self._seal_chk.isChecked(),
            seal_radius=self._seal_radius_spin.value(),
        )
        cleaning = self._cleaning_options()
        voxel_size_mm = round(self._voxel_spin.value(), 6)

        def _run() -> ScalpMesh:
            return generate_mesh_from_nifti(source, sealing, cleaning, voxel_size_mm)

        self._generation_seq += 1
        seq = self._generation_seq
        self._generation_busy = True
        self.busyChanged.emit(True)
        self._update_buttons()

        self._generation_thread = QThread(self)
        self._generation_worker = BackgroundWorker(_run, seq)
        self._generation_worker.moveToThread(self._generation_thread)
        self._generation_thread.started.connect(self._generation_worker.run)
        self._generation_worker.done.connect(self._on_generation_done)
        self._generation_worker.failed.connect(self._on_generation_failed)
        self._generation_thread.start()
        self.status.emit(f"Generating base mesh from {source.name}...")

    def _confirm_overwrite(self) -> bool:
        """Warn that generation invalidates the downstream steps."""
        answer = QMessageBox.question(
            self,
            "Generate base mesh",
            "Generating replaces the base mesh. The Mesh, ESE and Points "
            "steps will be invalidated. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _on_cancel_generation(self) -> None:
        self._cancel_running_generation()
        self.status.emit("Base mesh generation cancelled.")

    def _cancel_running_generation(self) -> None:
        if self._generation_worker is not None:
            self._generation_worker.deleteLater()
        if self._generation_thread is not None:
            self._generation_thread.quit()
            self._generation_thread.wait()
            if self._generation_thread.isFinished():
                self._generation_thread.deleteLater()
            self._generation_thread = None
        self._generation_worker = None
        self._generation_busy = False
        self.busyChanged.emit(False)
        self._update_buttons()

    def _finish_generation(self) -> None:
        if self._generation_worker is not None:
            self._generation_worker.deleteLater()
        if self._generation_thread is not None:
            self._generation_thread.quit()
            self._generation_thread.wait()
            if self._generation_thread.isFinished():
                self._generation_thread.deleteLater()
            self._generation_thread = None
        self._generation_worker = None
        self._generation_busy = False
        self.busyChanged.emit(False)
        self._update_buttons()

    def _on_generation_done(self, seq: int, result: object) -> None:
        if seq != self._generation_seq:
            return
        mesh: ScalpMesh = result  # type: ignore[assignment]
        source = self._source_path
        assert source is not None
        self._finish_generation()
        self._store_result(mesh, source)

    def _store_result(self, mesh: ScalpMesh, source: Path) -> None:
        """Save *mesh* as base+final with the source sidecar and publish it."""
        project = self._state.last_project_dir
        if project:
            mesh_dir = Path(project) / "mesh"
            try:
                export_scalp_mesh(mesh_dir / _BASE_MESH_FILENAME, mesh)
                export_scalp_mesh(mesh_dir / _FINAL_MESH_FILENAME, mesh)
                write_source_hash(mesh_dir, source, sha256_file(source))
            except (OSError, ValueError) as exc:
                self.status.emit(f"Could not auto-save base mesh:\n{exc}")
                return
            self.status.emit(
                f"Base mesh saved ({len(mesh.vertices)} vertices); "
                "Mesh, ESE and Points steps invalidated."
            )
        else:
            self.status.emit(f"Base mesh generated in memory: {len(mesh.vertices)} vertices.")
        self._base_mesh = mesh
        self._status_label.setText(
            f"Base mesh: {len(mesh.vertices)}v/{len(mesh.faces)}f from {source.name}"
        )
        self._render_scene()
        self.baseMesh.emit(mesh)
        self.saved.emit()

    def _on_generation_failed(self, seq: int, message: str) -> None:
        if seq != self._generation_seq:
            return
        self._finish_generation()
        QMessageBox.critical(self, "Generate base mesh", f"Base generation failed:\n{message}")
        self.status.emit(f"Base mesh generation failed: {message}")

    def _update_buttons(self) -> None:
        ready = not self._generation_busy
        if self._generate_btn is not None:
            self._generate_btn.setEnabled(ready)
        if self._cancel_btn is not None:
            self._cancel_btn.setEnabled(not ready)

    # ---- preview pane ----

    def _mesh_to_polydata(self, mesh: ScalpMesh) -> pv.PolyData:
        """Build a :class:`pv.PolyData` actor surface from a mesh model."""
        faces = np.column_stack(
            [np.full(len(mesh.faces), 3, dtype=np.int64), np.asarray(mesh.faces, dtype=np.int64)]
        ).ravel()
        return pv.PolyData(np.asarray(mesh.vertices, dtype=np.float64), faces)

    def _render_scene(self) -> None:
        if self._interactor is None:
            return
        was_empty = self._base_actor is None
        self._interactor.clear()
        self._base_actor = None
        if self._base_mesh is not None:
            self._base_actor = self._interactor.add_mesh(
                self._mesh_to_polydata(self._base_mesh), color="salmon", opacity=0.9
            )
        self._interactor.add_axes(interactive=False)  # type: ignore[call-arg]
        if was_empty:
            self._interactor.reset_camera()  # type: ignore[call-arg]
        self._interactor.render()

    def current_base_mesh(self) -> ScalpMesh | None:
        """The in-memory base mesh, if one was generated or set."""
        return self._base_mesh

    # ---- project lifecycle ----

    def prefill_from_project(self, project: str | Path) -> None:
        """Point the tab at the project's scan, if any."""
        self.clear()
        matches = sorted(Path(project).glob("input/*.nii*"))
        if matches:
            self.set_source(matches[0])

    def clear(self) -> None:
        """Forget the in-memory source and result without touching the project."""
        self._cancel_running_generation()
        if self._interactor is not None:
            self._interactor.clear()
        self._source_path = None
        self._spacing = None
        self._base_mesh = None
        self._base_actor = None
        self._source_label.setText("No scan in project")
        self._detected_label.setText("Detected voxel size: --")
        self._real_label.setText("Real voxel size: -- (marching cube step = --)")
        self._status_label.setText("Import a brain scan, then generate the base mesh.")
        self._update_buttons()

    def shutdown(self) -> None:
        """Stop the generation thread, if any, and release the 3D pane."""
        if self._interactor is not None:
            self._interactor.close()
        if self._generation_worker is not None:
            self._generation_worker.deleteLater()
        if self._generation_thread is not None:
            self._generation_thread.quit()
            self._generation_thread.wait(3000)
            if self._generation_thread.isFinished():
                self._generation_thread.deleteLater()
            self._generation_thread = None
        self._generation_worker = None
        self._generation_busy = False
