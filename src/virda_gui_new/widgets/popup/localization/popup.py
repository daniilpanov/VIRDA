"""Concrete localization popup: three tables over the live flow."""

from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QFileDialog, QLabel, QVBoxLayout

from virda.models.electrode import Electrodes
from virda.models.mri_volume import MRIVolume
from virda_gui_new.dto.steps import NIFTI
from virda_gui_new.dto.viewer3d import LabeledPoints
from virda_gui_new.services import artifact_importing, artifact_loading
from virda_gui_new.services.localization import FiducialRows, MeasurementTable
from virda_gui_new.widgets.popup.localization.abstract import LocalizationPopupBase
from virda_gui_new.widgets.popup.localization.electrodes_table import ElectrodesTable
from virda_gui_new.widgets.popup.localization.fiducials_table import FiducialsTable
from virda_gui_new.widgets.popup.localization.measurements_table import MeasurementsTable

_SECTIONS = ("Fiducials", "Measurements", "Electrodes")
_JSON_FILTER = "JSON (*.json)"
_FIDUCIALS_IMPORT_TITLE = "Import fiducials"
_MEASUREMENTS_IMPORT_TITLE = "Import measurements"


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
        self._fiducials_table = FiducialsTable(self)
        self._measurements_table = MeasurementsTable(self)
        self._electrodes_table = ElectrodesTable(self)
        widgets = (self._fiducials_table, self._measurements_table, self._electrodes_table)
        for caption, widget in zip(_SECTIONS, widgets, strict=True):
            layout.addWidget(QLabel(caption, self))
            layout.addWidget(widget)
        self._fiducials_table.rowsChanged.connect(self._on_inputs_changed)
        self._fiducials_table.cleared.connect(self._clear_fiducial_store)
        self._fiducials_table.importRequested.connect(self._import_fiducials_from_file)
        self._measurements_table.rowsChanged.connect(self._on_inputs_changed)
        self._measurements_table.importRequested.connect(self._import_measurements_from_file)
        self._prefill_fiducials()

    def _prefill_fiducials(self) -> None:
        """Fill the fiducial grid from the manager's global store, if any."""
        points = self._manager.get_global_item("fiducials")
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

    def _import_fiducials_from_file(self) -> None:
        """ "From file": fill the fixed slots from a fiducials/coordsystem JSON.

        The imported rows are canonical world coordinates (voxel rows
        convert through the step-0 MRI affine), so the table and the
        global store stay in sync (the store is refreshed only when the
        import actually changed it). Read or matching errors show in the
        hint label instead of crashing.
        """
        path = self._ask_json_path(_FIDUCIALS_IMPORT_TITLE)
        if path is None:
            return
        try:
            rows = artifact_importing.import_fiducial_rows(path, self._volume_affine())
        except Exception as error:  # noqa: BLE001 - any import error is user-facing
            self._set_hint(f"Failed to import fiducials: {error}")
            return
        self._fiducials_table.set_rows(rows)
        self._refresh_fiducial_store()

    def _volume_affine(self) -> np.ndarray | None:
        """Affine of the step-0 volume, or ``None`` when it is absent."""
        volume = self._read_context_item(NIFTI.index, "volume")
        if isinstance(volume, MRIVolume):
            return volume.affine
        if isinstance(volume, Path):
            return artifact_loading.load_volume(volume).affine
        return None

    def _import_measurements_from_file(self) -> None:
        """ "From file": fill the measurement grid from a measurements JSON."""
        path = self._ask_json_path(_MEASUREMENTS_IMPORT_TITLE)
        if path is None:
            return
        try:
            table = artifact_importing.import_measurement_table(path)
        except Exception as error:  # noqa: BLE001 - any import error is user-facing
            self._set_hint(f"Failed to import measurements: {error}")
            return
        self._measurements_table.set_rows(table)

    def _ask_json_path(self, title: str) -> Path | None:
        """One JSON file chosen in a dialog, or ``None`` when cancelled."""
        path, _ = QFileDialog.getOpenFileName(self, title, "", _JSON_FILTER)
        return Path(path) if path else None

    def show_localized(self, electrodes: Electrodes) -> None:
        self._electrodes_table.show_electrodes(electrodes)
