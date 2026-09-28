"""ESE surface tab: offset in, immutable surface out.

Owns the offset and normal-estimation parameters, runs
:func:`virda.ops.atoms.generate_ese` on a worker thread and auto-saves
``ese/mesh.ply`` with companions on success.  The result is immutable here:
no smoothing or density controls exist, only NPY export defaulting to
``ese/``.  An embedded 3D pane previews the working surface over the final
scalp mesh.  The host supplies the base mesh through *base_provider* and
locks the mesh tab while generation runs (see ``busyChanged``).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from virda.io.exporters.ese_mesh import export_ese_companions, export_ese_mesh
from virda.io.exporters.mesh_arrays import export_mesh_faces, export_mesh_vertices
from virda.io.importers.ese_mesh import import_ese_mesh
from virda.models.ese_mesh import ESEMesh
from virda.models.scalp_mesh import ScalpMesh
from virda.ops.atoms import generate_ese
from virda.ops.options import EseOptions
from virda_gui.export.mesh_ply import export_mesh_ply
from virda_gui.state import AppState
from virda_gui.workers import BackgroundWorker

_ESE_MESH_FILENAME = "mesh.ply"


class EseTab(QWidget):
    """Offset-to-ESE generation with immediate save and NPY export."""

    eseMesh = Signal(object)  # noqa: N815 - the generated ESEMesh in world coords
    saved = Signal()  # noqa: N815 - fired after auto-save wrote the mesh to disk
    status = Signal(str)  # noqa: N815 - non-blocking log lines
    busyChanged = Signal(bool)  # noqa: N815 - generation running

    def __init__(
        self,
        state: AppState,
        base_provider: Callable[[], ScalpMesh | None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._base_provider = base_provider
        self._final_mesh: ScalpMesh | None = None
        self._ese_mesh: ESEMesh | None = None
        self._generate_btn: QPushButton | None = None
        self._vertices_export_button: QPushButton | None = None
        self._faces_export_button: QPushButton | None = None
        self._mesh_export_button: QPushButton | None = None
        self._generation_thread: QThread | None = None
        self._generation_worker: BackgroundWorker | None = None
        self._generation_seq = 0
        self._generation_busy = False
        self._interactor: QtInteractor | None = None
        self._final_actor: Any | None = None
        self._ese_actor: Any | None = None

        panel = QWidget(self)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.setSpacing(8)

        panel_layout.addWidget(self._build_parameters_box())
        panel_layout.addWidget(self._build_actions_box())
        panel_layout.addWidget(self._build_display_box())
        panel_layout.addWidget(self._build_export_box())
        panel_layout.addStretch(1)

        self._status_label = QLabel("Generate the sensor surface from the final mesh.", panel)
        self._status_label.setWordWrap(True)
        panel_layout.addWidget(self._status_label)

        self._interactor = QtInteractor(parent=self)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.addWidget(panel, 0)
        content.addWidget(self._interactor, 1)
        root.addLayout(content, 1)

        self._update_buttons()

    # ---- UI builders ----

    def _build_parameters_box(self) -> QGroupBox:
        box = QGroupBox("ESE parameters", self)
        grid = QVBoxLayout(box)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(6)

        offset_row = QHBoxLayout()
        offset_row.addWidget(QLabel("Offset (mm):", box))
        self._ese_offset_spin = QDoubleSpinBox(box)
        self._ese_offset_spin.setRange(0.1, 50.0)
        self._ese_offset_spin.setSingleStep(0.5)
        self._ese_offset_spin.setValue(2.0)
        self._ese_offset_spin.setToolTip(
            "Outward offset of the sensor surface (ESE) from the final mesh."
        )
        offset_row.addWidget(self._ese_offset_spin)
        offset_row.addStretch(1)
        grid.addLayout(offset_row)

        adv_row = QHBoxLayout()
        adv_row.addWidget(QLabel("Radius:", box))
        self._ese_radius_spin = QDoubleSpinBox(box)
        self._ese_radius_spin.setRange(1.0, 30.0)
        self._ese_radius_spin.setSingleStep(0.5)
        self._ese_radius_spin.setValue(10.0)
        self._ese_radius_spin.setToolTip("Neighborhood radius in mm.")
        adv_row.addWidget(self._ese_radius_spin)
        adv_row.addWidget(QLabel("k-NN:", box))
        self._ese_k_spin = QSpinBox(box)
        self._ese_k_spin.setRange(0, 500)
        self._ese_k_spin.setValue(0)
        self._ese_k_spin.setToolTip("Neighbors per vertex; 0 picks from the radius.")
        adv_row.addWidget(self._ese_k_spin)
        self._ese_weighted_chk = QCheckBox("Weighted PCA", box)
        self._ese_weighted_chk.setChecked(False)
        adv_row.addWidget(self._ese_weighted_chk)
        adv_row.addWidget(QLabel("Sigma:", box))
        self._ese_sigma_spin = QDoubleSpinBox(box)
        self._ese_sigma_spin.setRange(1.0, 15.0)
        self._ese_sigma_spin.setSingleStep(0.5)
        self._ese_sigma_spin.setValue(5.0)
        self._ese_sigma_spin.setToolTip("Falloff in mm for weighted PCA.")
        adv_row.addWidget(self._ese_sigma_spin)
        adv_row.addWidget(QLabel("Min nbrs:", box))
        self._ese_min_nbrs_spin = QSpinBox(box)
        self._ese_min_nbrs_spin.setRange(1, 50)
        self._ese_min_nbrs_spin.setValue(5)
        adv_row.addWidget(self._ese_min_nbrs_spin)
        adv_row.addStretch(1)
        grid.addLayout(adv_row)

        return box

    def _build_actions_box(self) -> QGroupBox:
        box = QGroupBox("Actions", self)
        row = QHBoxLayout(box)
        row.setContentsMargins(4, 4, 4, 4)
        row.setSpacing(6)

        self._generate_btn = QPushButton("Generate ESE mesh", box)
        self._generate_btn.setToolTip("Build and immediately save the sensor surface (ESE).")
        self._generate_btn.clicked.connect(self._on_generate)
        row.addWidget(self._generate_btn)
        return box

    def _build_display_box(self) -> QGroupBox:
        box = QGroupBox("Display", self)
        row = QHBoxLayout(box)
        row.setContentsMargins(4, 4, 4, 4)
        row.setSpacing(6)
        self._show_final_chk = QCheckBox("Show final mesh", box)
        self._show_ese_chk = QCheckBox("Show ESE surface", box)
        self._show_edges_chk = QCheckBox("Show mesh edges", box)
        self._show_final_chk.setChecked(True)
        self._show_ese_chk.setChecked(True)
        self._show_edges_chk.setChecked(False)
        self._show_final_chk.toggled.connect(self._on_display_toggled)
        self._show_ese_chk.toggled.connect(self._on_display_toggled)
        self._show_edges_chk.toggled.connect(self._on_display_toggled)
        row.addWidget(self._show_final_chk)
        row.addWidget(self._show_ese_chk)
        row.addWidget(self._show_edges_chk)
        row.addStretch(1)
        return box

    def _build_export_box(self) -> QGroupBox:
        box = QGroupBox("Export ESE arrays", self)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(QLabel("Writes ESE arrays in world coordinates.", box))

        self._vertices_export_button = QPushButton(box)
        self._vertices_export_button.clicked.connect(self._on_export_vertices)
        layout.addWidget(self._vertices_export_button)
        self._faces_export_button = QPushButton(box)
        self._faces_export_button.clicked.connect(self._on_export_faces)
        layout.addWidget(self._faces_export_button)
        self._mesh_export_button = QPushButton(box)
        self._mesh_export_button.clicked.connect(self._on_export_mesh_file)
        layout.addWidget(self._mesh_export_button)
        return box

    # ---- preview pane ----

    def _mesh_to_polydata(self, mesh: ScalpMesh | ESEMesh) -> pv.PolyData:
        """Build a :class:`pv.PolyData` surface from a mesh model."""
        faces = np.column_stack(
            [np.full(len(mesh.faces), 3, dtype=np.int64), np.asarray(mesh.faces, dtype=np.int64)]
        ).ravel()
        return pv.PolyData(np.asarray(mesh.vertices, dtype=np.float64), faces)

    def _on_display_toggled(self, *_args: Any) -> None:
        self._render_scene()

    def _render_scene(self) -> None:
        """Draw the final mesh with the ESE surface overlaid."""
        if self._interactor is None:
            return
        edges = self._show_edges_chk.isChecked()
        was_empty = self._final_actor is None and self._ese_actor is None
        self._interactor.clear()
        self._final_actor = None
        self._ese_actor = None
        if self._show_final_chk.isChecked() and self._final_mesh is not None:
            self._final_actor = self._interactor.add_mesh(
                self._mesh_to_polydata(self._final_mesh),
                color="salmon",
                opacity=0.9,
                show_edges=edges,
            )
        if self._show_ese_chk.isChecked() and self._ese_mesh is not None:
            self._ese_actor = self._interactor.add_mesh(
                self._mesh_to_polydata(self._ese_mesh),
                color="royalblue",
                opacity=0.7,
                show_edges=edges,
            )
        self._interactor.add_axes(interactive=False)  # type: ignore[call-arg]
        if was_empty:
            self._interactor.reset_camera()  # type: ignore[call-arg]
        self._interactor.render()

    # ---- inputs from the host ----

    def show_preview(self, mesh: ScalpMesh) -> None:
        """Display the working scalp mesh as the ESE backdrop."""
        self._final_mesh = mesh
        self._render_scene()

    def show_ese(self, ese: ESEMesh) -> None:
        """Display an externally generated ESE mesh."""
        self._ese_mesh = ese
        self._status_label.setText(
            f"ESE mesh: {len(ese.vertices)}v/"
            f"{len(ese.faces)}f (offset {self._ese_offset_spin.value():g} mm)"
        )
        self._render_scene()
        self._update_export_controls()

    def current_ese_mesh(self) -> ESEMesh | None:
        """The in-memory ESE mesh, if one was generated or loaded."""
        return self._ese_mesh

    # ---- generation ----

    def _ese_options(self) -> EseOptions:
        k_value = self._ese_k_spin.value()
        return EseOptions(
            ese_offset_mm=round(self._ese_offset_spin.value(), 6),
            neighborhood_radius_mm=round(self._ese_radius_spin.value(), 6),
            k_neighbors=k_value if k_value > 0 else None,
            use_weighted_pca=self._ese_weighted_chk.isChecked(),
            pca_sigma_mm=round(self._ese_sigma_spin.value(), 6),
            min_neighbors=self._ese_min_nbrs_spin.value(),
        )

    def _on_generate(self) -> None:
        if self._generation_busy:
            return
        base = self._base_provider()
        if base is None:
            QMessageBox.warning(
                self, "Generate ESE", "Prepare a final mesh in Mesh Processing first."
            )
            return
        options = self._ese_options()
        offset_mm = options.ese_offset_mm

        def _run() -> ESEMesh:
            return generate_ese(base, options)

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
        self.status.emit(f"Generating ESE mesh at {offset_mm:g} mm offset...")

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
        ese: ESEMesh = result  # type: ignore[assignment]
        self._finish_generation()
        self._ese_mesh = ese
        self.show_ese(ese)
        self.status.emit(f"ESE mesh generated: {len(ese.vertices)} vertices.")
        self.eseMesh.emit(ese)
        self._autosave_ese()

    def _on_generation_failed(self, seq: int, message: str) -> None:
        if seq != self._generation_seq:
            return
        self._finish_generation()
        QMessageBox.critical(self, "Generate ESE", f"ESE generation failed:\n{message}")
        self.status.emit(f"ESE mesh generation failed: {message}")

    def _autosave_ese(self) -> None:
        """Write the ESE mesh to ese/mesh.ply with companions right away."""
        project = self._state.last_project_dir
        ese = self._ese_mesh
        if not project or ese is None:
            return
        ese_path = Path(project) / "ese" / _ESE_MESH_FILENAME
        try:
            export_ese_mesh(ese_path, ese)
            export_ese_companions(ese_path, ese)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Save ESE mesh", f"Could not save ESE mesh:\n{exc}")
            return
        self.status.emit(f"Saved ESE mesh ({len(ese.vertices)} vertices) to {ese_path}.")
        self.saved.emit()

    def _update_buttons(self) -> None:
        ready = not self._generation_busy
        if self._generate_btn is not None:
            self._generate_btn.setEnabled(ready)
        self._update_export_controls()

    # ---- export ----

    def _update_export_controls(self) -> None:
        ese = self._ese_mesh
        if self._vertices_export_button is not None:
            self._vertices_export_button.setEnabled(ese is not None)
            self._vertices_export_button.setText("Export ESE vertices (NPY)...")
        if self._faces_export_button is not None:
            self._faces_export_button.setEnabled(ese is not None)
            self._faces_export_button.setText("Export ESE faces (NPY)...")
        if self._mesh_export_button is not None:
            self._mesh_export_button.setEnabled(ese is not None)
            self._mesh_export_button.setText("Export ESE mesh (PLY)...")

    def _export_start_path(self, array_kind: str) -> str:
        filename = f"ese_{array_kind}.npy"
        project = self._state.last_project_dir
        if project:
            return str(Path(project) / "ese" / filename)
        return filename

    def _export_array(self, array_kind: str) -> None:
        ese = self._ese_mesh
        if ese is None:
            return
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            f"Export ESE {array_kind}",
            self._export_start_path(array_kind),
            "NumPy array (*.npy);;All files (*)",
        )
        if not path:
            return
        try:
            if array_kind == "vertices":
                target = export_mesh_vertices(path, ese.vertices)
            elif array_kind == "faces":
                target = export_mesh_faces(path, ese.faces)
            else:
                raise ValueError(f"unknown mesh array {array_kind!r}")
        except (OSError, ValueError) as exc:
            QMessageBox.critical(
                self,
                f"Export ESE {array_kind}",
                f"Could not export ESE {array_kind}:\n{exc}",
            )
            return
        self.status.emit(f"Exported ESE {array_kind} to {target} in world coordinates")

    def _on_export_vertices(self) -> None:
        self._export_array("vertices")

    def _on_export_faces(self) -> None:
        self._export_array("faces")

    def _on_export_mesh_file(self) -> None:
        ese = self._ese_mesh
        if ese is None:
            return
        project = self._state.last_project_dir
        start = str(Path(project) / "ese" / "ese_mesh.ply") if project else "ese_mesh.ply"
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export ESE mesh",
            start,
            "PLY (*.ply);;All files (*)",
        )
        if not path:
            return
        try:
            target = export_mesh_ply(path, ese.vertices, ese.faces)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Export ESE mesh", f"Could not export ESE mesh:\n{exc}")
            return
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.critical(self, "Export ESE mesh", f"Could not export ESE mesh:\n{exc}")
            return
        self.status.emit(f"Exported ESE mesh to {target} in world coordinates")

    # ---- project lifecycle ----

    def prefill_from_project(self, project: str | Path) -> None:
        """Load the project's ESE mesh, if any."""
        self.clear()
        ese_candidate = Path(project) / "ese" / _ESE_MESH_FILENAME
        if ese_candidate.is_file():
            try:
                ese = import_ese_mesh(ese_candidate)
            except ValueError as exc:
                self.status.emit(f"ESE mesh not loaded: {exc}")
            else:
                self._ese_mesh = ese
                self.show_ese(ese)
                self.eseMesh.emit(ese)
                self.status.emit(f"ESE mesh loaded: {len(ese.vertices)} vertices.")

    def clear(self) -> None:
        """Forget the in-memory ESE state without touching the project."""
        if self._interactor is not None:
            self._interactor.clear()
        self._final_mesh = None
        self._ese_mesh = None
        self._final_actor = None
        self._ese_actor = None
        self._status_label.setText("Generate the sensor surface from the final mesh.")
        self._update_export_controls()

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
