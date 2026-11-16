"""Mesh postprocessing popup: clean, decimate, and smooth options."""

from collections.abc import Callable
from typing import Any

from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from virda.mesh.mesh_decimator import decimate_mesh
from virda.models.mesh import Mesh
from virda.ops.clean_mesh import clean_mesh
from virda.ops.smooth_mesh import smooth_mesh
from virda_gui_new.dto.steps import MESH_POSTPROCESSING
from virda_gui_new.services import artifact_saving
from virda_gui_new.widgets.popup.abstract import StepPopupBase

_NO_MESH = "Mesh postprocessing needs an in-memory mesh (run the previous steps first)."


class MeshPostprocessingPopup(StepPopupBase):
    """Clean, decimate, and smooth the mesh from earlier steps.

    The run input is the step-2 ``mesh`` when present, else the step-1
    ``scalp_surface``; it must already be a ready :class:`Mesh` object,
    not a path. The form collects component merging plus decimation and
    smoothing parameters, then hands the ``clean_mesh`` ->
    ``decimate_mesh`` -> ``smooth_mesh`` chain to the base run machinery.
    On success the processed mesh is stored as ``mesh`` of step 2; the
    base then advances the pipeline when step 2 is current.
    """

    def _resolve_input(self) -> Mesh | None:
        """The scalp mesh the step works on: the postprocessed one, else the base."""
        return self._manager.active_scalp_mesh()

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
        self._smooth_combo.setCurrentText("taubin")
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

        self._restore_config()
        for widget in (self._min_spin, self._merge_spin, self._iterations_spin):
            widget.valueChanged.connect(self._store_config)
        for widget in (self._density_spin, self._lamb_spin, self._nu_spin):
            widget.valueChanged.connect(self._store_config)
        self._smooth_combo.currentTextChanged.connect(self._store_config)

    def _restore_config(self) -> None:
        """Put the stored settings into the widgets."""
        config = self._load_config()
        numbers: dict[str, QSpinBox | QDoubleSpinBox] = {
            "min_component_vertices": self._min_spin,
            "merge_digits": self._merge_spin,
            "density_percent": self._density_spin,
            "iterations": self._iterations_spin,
            "lamb": self._lamb_spin,
            "nu": self._nu_spin,
        }
        for key, widget in numbers.items():
            value = config.get(key)
            if not isinstance(value, int | float):
                continue
            if isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            else:
                widget.setValue(float(value))
        smoother = config.get("smoother")
        if isinstance(smoother, str):
            self._smooth_combo.setCurrentText(smoother)

    def _store_config(self, *_args: object) -> None:
        """Save the current settings to the project."""
        self._save_config(
            {
                "min_component_vertices": self._min_spin.value(),
                "merge_digits": self._merge_spin.value(),
                "density_percent": self._density_spin.value(),
                "smoother": self._smooth_combo.currentText(),
                "iterations": self._iterations_spin.value(),
                "lamb": self._lamb_spin.value(),
                "nu": self._nu_spin.value(),
            }
        )

    def _set_smoothing_enabled(self, smoother: str) -> None:
        enable = smoother != "none"
        for widget in (self._iterations_spin, self._lamb_spin, self._nu_spin):
            widget.setEnabled(enable)

    def _collect_run(self) -> Callable[[], Mesh]:
        mesh = self._resolve_input()
        if mesh is None:
            raise ValueError(_NO_MESH)
        min_component_vertices = self._min_spin.value()
        merge_digits = self._merge_spin.value()
        density_percent = self._density_spin.value()
        smoother = self._smooth_combo.currentText()
        iterations = self._iterations_spin.value()
        lamb = self._lamb_spin.value()
        nu = self._nu_spin.value()

        def run() -> Mesh:
            processed = smooth_mesh(
                decimate_mesh(
                    clean_mesh(
                        mesh,
                        min_component_vertices=min_component_vertices,
                        merge_digits=merge_digits,
                    ),
                    density_percent=density_percent,
                ),
                smoother_type=smoother,
                iterations=iterations,
                lamb=lamb,
                nu=nu,
            )
            if self._project is not None:
                artifact_saving.save_final_mesh(self._project, processed)
            return processed

        return run

    def _apply_result(self, result: Any) -> None:
        self._manager.set_step_context_item(MESH_POSTPROCESSING.index, "mesh", result)
