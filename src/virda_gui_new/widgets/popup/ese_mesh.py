"""ESE mesh popup: normal estimation and scalp offset options."""

from collections.abc import Callable
from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from virda.models.mesh import Mesh
from virda_gui.api.options import EseOptions
from virda_gui_new.api import ops
from virda_gui_new.dto.steps import BASE_MESH, ESE_MESH, MESH_POSTPROCESSING
from virda_gui_new.services import artifact_saving
from virda_gui_new.widgets.popup.abstract import StepPopupBase

_NO_MESH = "ESE mesh needs the postprocessed mesh (run the previous steps first)."


class EseMeshPopup(StepPopupBase):
    """Offset the mesh from earlier steps into the ESE surface.

    The run input is the step-2 ``mesh`` when present, else the step-1
    ``scalp_surface``; it must already be a ready :class:`Mesh` object,
    not a path. The form collects the ESE offset plus the PCA normal
    estimation parameters, then hands the ``ops.generate_ese`` call to
    the base run machinery. On success the mesh and the per-vertex
    normals are stored as ``ese_mesh`` and ``normals`` of step 3; the
    base then advances the pipeline when step 3 is current.
    """

    def _resolve_input(self) -> Mesh | None:
        try:
            mesh = self._manager.get_step_context_item(MESH_POSTPROCESSING.index, "mesh")
            if mesh is None:
                mesh = self._manager.get_step_context_item(BASE_MESH.index, "scalp_surface")
        except ValueError:
            return None
        return mesh if isinstance(mesh, Mesh) else None

    def build_form(self, layout: QVBoxLayout) -> None:
        if self._resolve_input() is None:
            layout.addWidget(QLabel("No in-memory mesh yet."))
            self._run_button.setEnabled(False)

        self._offset_spin = QDoubleSpinBox(self)
        self._offset_spin.setRange(0.1, 10.0)
        self._offset_spin.setSingleStep(0.5)
        self._offset_spin.setDecimals(2)
        self._offset_spin.setValue(1.0)
        layout.addWidget(QLabel("ESE offset (mm)"))
        layout.addWidget(self._offset_spin)

        self._radius_spin = QDoubleSpinBox(self)
        self._radius_spin.setRange(0.1, 100.0)
        self._radius_spin.setSingleStep(0.5)
        self._radius_spin.setDecimals(2)
        self._radius_spin.setValue(10.0)
        layout.addWidget(QLabel("Neighborhood radius (mm)"))
        layout.addWidget(self._radius_spin)

        self._auto_k_check = QCheckBox("Auto K neighbors", self)
        self._auto_k_check.setChecked(True)
        layout.addWidget(self._auto_k_check)

        self._k_spin = QSpinBox(self)
        self._k_spin.setRange(1, 100)
        self._k_spin.setValue(10)
        layout.addWidget(QLabel("K neighbors"))
        layout.addWidget(self._k_spin)

        self._weighted_check = QCheckBox("Weighted PCA", self)
        layout.addWidget(self._weighted_check)

        self._sigma_spin = QDoubleSpinBox(self)
        self._sigma_spin.setRange(0.1, 50.0)
        self._sigma_spin.setSingleStep(0.5)
        self._sigma_spin.setDecimals(2)
        self._sigma_spin.setValue(5.0)
        layout.addWidget(QLabel("PCA sigma (mm)"))
        layout.addWidget(self._sigma_spin)

        self._min_neighbors_spin = QSpinBox(self)
        self._min_neighbors_spin.setRange(1, 50)
        self._min_neighbors_spin.setValue(5)
        layout.addWidget(QLabel("Min neighbors"))
        layout.addWidget(self._min_neighbors_spin)

        self._k_spin.setEnabled(False)
        self._auto_k_check.toggled.connect(lambda checked: self._k_spin.setEnabled(not checked))

    def _collect_run(self) -> Callable[[], tuple[Mesh, Any]]:
        mesh = self._resolve_input()
        if mesh is None:
            raise ValueError(_NO_MESH)
        options = EseOptions(
            ese_offset_mm=self._offset_spin.value(),
            neighborhood_radius_mm=self._radius_spin.value(),
            k_neighbors=None if self._auto_k_check.isChecked() else self._k_spin.value(),
            use_weighted_pca=self._weighted_check.isChecked(),
            pca_sigma_mm=self._sigma_spin.value(),
            min_neighbors=self._min_neighbors_spin.value(),
        )

        def run() -> tuple[Mesh, Any]:
            ese, normals = ops.generate_ese(mesh, options)
            # Normals stay in-memory only; only the ESE surface persists.
            if self._project is not None:
                artifact_saving.save_ese_mesh(self._project, ese)
            return ese, normals

        return run

    def _apply_result(self, result: Any) -> None:
        mesh, normals = result
        self._manager.set_step_context_item(ESE_MESH.index, "ese_mesh", mesh)
        self._manager.set_step_context_item(ESE_MESH.index, "normals", normals)
