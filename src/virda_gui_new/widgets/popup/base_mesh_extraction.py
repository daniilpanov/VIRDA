"""Base mesh popup: scalp segmentation options, run, and context writes."""

from collections.abc import Callable
from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from virda.models.mri_volume import MRIVolume
from virda_gui.api.options import SealingOptions
from virda_gui_new.api import ops
from virda_gui_new.api.ops import ScalpSurface
from virda_gui_new.dto.steps import BASE_MESH
from virda_gui_new.widgets.popup.abstract import StepPopupBase

_NO_VOLUME = "Base mesh needs the NIfTI volume from step 0 (run NIfTI import first)."


class BaseMeshExtractionPopup(StepPopupBase):
    """Extract the scalp surface from the step-0 NIfTI volume.

    The form collects the target voxel size in millimeters (zero means
    native NIfTI resolution) plus mask sealing with its radius, then hands
    the heavy ``ops.generate_scalp_surface`` call to the base run
    machinery. On success the ready mesh is stored as ``scalp_surface``
    and the mask as ``scalp_mask`` of step 1; the base then advances the
    pipeline when step 1 is current.
    """

    def build_form(self, layout: QVBoxLayout) -> None:
        volume = self._manager.get_step_context_item(0, "volume")
        if not isinstance(volume, MRIVolume):
            layout.addWidget(QLabel("Step 0 has no NIfTI yet."))
            self._run_button.setEnabled(False)

        self._voxel_spin = QDoubleSpinBox(self)
        self._voxel_spin.setRange(0.0, 10.0)
        self._voxel_spin.setSingleStep(0.1)
        self._voxel_spin.setDecimals(2)
        self._voxel_spin.setSpecialValueText("native")
        layout.addWidget(QLabel("Voxel size (mm)"))
        layout.addWidget(self._voxel_spin)

        self._seal_box = QCheckBox("Seal mask", self)
        self._seal_box.setChecked(True)
        layout.addWidget(self._seal_box)

        self._radius_spin = QSpinBox(self)
        self._radius_spin.setRange(0, 50)
        self._radius_spin.setValue(4)
        self._radius_spin.setEnabled(self._seal_box.isChecked())
        self._seal_box.toggled.connect(self._radius_spin.setEnabled)
        layout.addWidget(QLabel("Seal radius"))
        layout.addWidget(self._radius_spin)

    def _collect_run(self) -> Callable[[], ScalpSurface]:
        volume = self._manager.get_step_context_item(0, "volume")
        if not isinstance(volume, MRIVolume):
            raise ValueError(_NO_VOLUME)
        voxel_size_mm: float | None = self._voxel_spin.value()
        if voxel_size_mm == 0.0:
            voxel_size_mm = None
        sealing = SealingOptions(
            seal_enabled=self._seal_box.isChecked(),
            seal_radius=self._radius_spin.value(),
        )
        return lambda: ops.generate_scalp_surface(volume, sealing, voxel_size_mm)

    def _apply_result(self, result: Any) -> None:
        self._manager.set_step_context_item(BASE_MESH.index, "scalp_surface", result.mesh)
        self._manager.set_step_context_item(BASE_MESH.index, "scalp_mask", result.mask)
