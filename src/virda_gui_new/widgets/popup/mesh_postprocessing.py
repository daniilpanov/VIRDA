"""Mesh postprocessing popup: clean, decimate, and smooth options."""

from collections.abc import Callable
from typing import Any, cast

from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from virda.models.mesh import Mesh
from virda_gui.api.options import CleanOptions, DecimateOptions, SmootherKind, SmoothOptions
from virda_gui_new.api import ops
from virda_gui_new.dto.steps import BASE_MESH, MESH_POSTPROCESSING
from virda_gui_new.widgets.popup.abstract import StepPopupBase

_NO_MESH = "Mesh postprocessing needs an in-memory mesh (run the previous steps first)."


class MeshPostprocessingPopup(StepPopupBase):
    """Clean, decimate, and smooth the mesh from earlier steps.

    The run input is the step-2 ``mesh`` when present, else the step-1
    ``scalp_surface``; it must already be a ready :class:`Mesh` object,
    not a path. The form collects component merging plus decimation and
    smoothing parameters, then hands the ``ops.clean`` ->
    ``ops.decimate`` -> ``ops.smooth`` chain to the base run machinery.
    On success the processed mesh is stored as ``mesh`` of step 2; the
    base then advances the pipeline when step 2 is current.
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

        layout.addWidget(QLabel("Clean"))
        self._min_spin = QSpinBox(self)
        self._min_spin.setRange(1, 100000)
        self._min_spin.setValue(100)
        layout.addWidget(QLabel("Min component vertices"))
        layout.addWidget(self._min_spin)

        self._merge_spin = QSpinBox(self)
        self._merge_spin.setRange(1, 12)
        self._merge_spin.setValue(7)
        layout.addWidget(QLabel("Merge digits"))
        layout.addWidget(self._merge_spin)

        layout.addWidget(QLabel("Decimate"))
        self._density_spin = QDoubleSpinBox(self)
        self._density_spin.setRange(0.1, 100.0)
        self._density_spin.setSingleStep(1.0)
        self._density_spin.setDecimals(2)
        self._density_spin.setValue(100.0)
        layout.addWidget(QLabel("Density percent"))
        layout.addWidget(self._density_spin)

        layout.addWidget(QLabel("Smooth"))
        self._smooth_combo = QComboBox(self)
        self._smooth_combo.addItems(["none", "laplacian", "taubin"])
        self._smooth_combo.setCurrentText("laplacian")
        layout.addWidget(QLabel("Smoother"))
        layout.addWidget(self._smooth_combo)

        self._iterations_spin = QSpinBox(self)
        self._iterations_spin.setRange(0, 50)
        self._iterations_spin.setValue(5)
        layout.addWidget(QLabel("Iterations"))
        layout.addWidget(self._iterations_spin)

        self._lamb_spin = QDoubleSpinBox(self)
        self._lamb_spin.setRange(0.05, 5.0)
        self._lamb_spin.setDecimals(3)
        self._lamb_spin.setValue(0.5)
        layout.addWidget(QLabel("Lambda"))
        layout.addWidget(self._lamb_spin)

        self._nu_spin = QDoubleSpinBox(self)
        self._nu_spin.setRange(-5.0, 5.0)
        self._nu_spin.setDecimals(3)
        self._nu_spin.setValue(-0.53)
        layout.addWidget(QLabel("Nu"))
        layout.addWidget(self._nu_spin)

        smoothing = self._smooth_combo.currentText() != "none"
        for widget in (self._iterations_spin, self._lamb_spin, self._nu_spin):
            widget.setEnabled(smoothing)
        self._smooth_combo.currentTextChanged.connect(self._set_smoothing_enabled)

    def _set_smoothing_enabled(self, smoother: str) -> None:
        enable = smoother != "none"
        for widget in (self._iterations_spin, self._lamb_spin, self._nu_spin):
            widget.setEnabled(enable)

    def _collect_run(self) -> Callable[[], Mesh]:
        mesh = self._resolve_input()
        if mesh is None:
            raise ValueError(_NO_MESH)
        cleaning = CleanOptions(
            min_component_vertices=self._min_spin.value(),
            merge_digits=self._merge_spin.value(),
        )
        decimating = DecimateOptions(density_percent=self._density_spin.value())
        smoothing = SmoothOptions(
            smoother=cast(SmootherKind, self._smooth_combo.currentText()),
            iterations=self._iterations_spin.value(),
            lamb=self._lamb_spin.value(),
            nu=self._nu_spin.value(),
        )
        return lambda: ops.smooth(ops.decimate(ops.clean(mesh, cleaning), decimating), smoothing)

    def _apply_result(self, result: Any) -> None:
        self._manager.set_step_context_item(MESH_POSTPROCESSING.index, "mesh", result)
