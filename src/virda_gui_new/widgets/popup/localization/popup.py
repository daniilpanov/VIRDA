"""Concrete localization popup: three tables over the live flow."""

from PySide6.QtWidgets import QLabel, QVBoxLayout

from virda.models.electrode import Electrodes
from virda_gui_new.dto.steps import FIDUCIALS_KEY
from virda_gui_new.dto.viewer3d import LabeledPoints
from virda_gui_new.services.localization import (
    FiducialRows,
    MeasurementTable,
)
from virda_gui_new.widgets.popup.localization.abstract import LocalizationPopupBase
from virda_gui_new.widgets.popup.localization.electrodes_table import ElectrodesTable
from virda_gui_new.widgets.popup.localization.fiducials_table import FiducialsTable
from virda_gui_new.widgets.popup.localization.measurements_table import MeasurementsTable


class LocalizationPopup(LocalizationPopupBase):
    """Localization step editor: tables drive the debounced live runs.

    The form stacks the editable fiducial and measurement grids over
    the read-only electrode grid. Each edit in either input table
    re-validates the whole snapshot and schedules a run; localized
    results are shown in the electrode grid and stored in the step-4
    context by the base.
    """

    def build_form(self, layout: QVBoxLayout) -> None:
        """Arrange the three table sections and wire input changes."""
        super().build_form(layout)
        self._fiducials_table = FiducialsTable(self)
        self._measurements_table = MeasurementsTable(self)
        self._electrodes_table = ElectrodesTable(self)
        layout.addWidget(QLabel("Fiducials", self))
        layout.addWidget(self._fiducials_table)
        layout.addWidget(QLabel("Measurements", self))
        layout.addWidget(self._measurements_table)
        layout.addWidget(QLabel("Electrodes", self))
        layout.addWidget(self._electrodes_table)
        self._fiducials_table.rowsChanged.connect(self._on_inputs_changed)
        self._measurements_table.rowsChanged.connect(self._on_measurement_table_changed)
        self._measurements_table.electrodeEdited.connect(self._on_measurement_edited)
        self._prefill_fiducials()

    def _prefill_fiducials(self) -> None:
        """Fill the fiducial grid from the manager's global store, if any."""
        points = self._manager.get_global_item(FIDUCIALS_KEY)
        if not isinstance(points, LabeledPoints) or points.labels is None:
            return
        rows = [
            (label, [float(value) for value in xyz])
            for label, xyz in zip(points.labels, points.points, strict=True)
        ]
        self._fiducials_table.set_rows(rows)

    def collect_fiducial_rows(self) -> FiducialRows:
        return self._fiducials_table.collect_fiducial_rows()

    def collect_measurement_table(self) -> MeasurementTable:
        return self._measurements_table.collect_measurement_table()

    def show_localized(self, electrodes: Electrodes) -> None:
        self._electrodes_table.show_electrodes(electrodes)

    def _on_measurement_table_changed(self) -> None:
        """Persist structural table edits without starting localization."""
        self._update_relocalize_state()
