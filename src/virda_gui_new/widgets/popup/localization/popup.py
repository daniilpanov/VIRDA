"""Concrete localization popup: three tables over the live flow."""

from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QFileDialog, QLabel, QVBoxLayout

from virda.models.electrode import Electrodes
from virda.models.mri_volume import MRIVolume
from virda_gui_new.dto.steps import FIDUCIALS_KEY, LOCALIZATION, MEASUREMENTS_KEY, NIFTI
from virda_gui_new.dto.viewer3d import LabeledPoints
from virda_gui_new.services import artifact_importing, artifact_loading
from virda_gui_new.services.localization import (
    FiducialRows,
    MeasurementTable,
    electrodes_from_points,
    measurement_table_from_model,
)
from virda_gui_new.widgets.popup.localization.abstract import LocalizationPopupBase
from virda_gui_new.widgets.popup.localization.electrodes_table import ElectrodesTable
from virda_gui_new.widgets.popup.localization.fiducials_table import FiducialsTable
from virda_gui_new.widgets.popup.localization.measurements_table import MeasurementsTable

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
        self._fiducials_table.cleared.connect(self._clear_fiducial_store)
        self._fiducials_table.importRequested.connect(self._import_fiducials_from_file)
        self._measurements_table.rowsChanged.connect(self._on_measurement_table_changed)
        self._measurements_table.electrodeEdited.connect(self._on_measurement_edited)
        self._measurements_table.importRequested.connect(self._import_measurements_from_file)
        self._prefill_fiducials()
        self._prefill_measurements()
        self._prefill_electrodes()
        self._sync_frame_selectors()

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

    def _prefill_measurements(self) -> None:
        """Fill the measurement grid from the stored measurements model."""
        measurements = self._manager.get_global_item(MEASUREMENTS_KEY)
        if not isinstance(measurements, Electrodes):
            return
        table = measurement_table_from_model(measurements)
        if table:
            self._measurements_table.set_rows(table)

    def _prefill_electrodes(self) -> None:
        """Fill the electrode grid with the localized results of the project.

        The results live in the step-4 context: the grid shows them as
        they are, and the step computes them when the project holds the
        measured distances but no results yet.
        """
        results = self._stored_results()
        if results is not None and results.items:
            self._electrodes_table.show_electrodes(results)
            return
        scalp_points = self._read_context_item(LOCALIZATION.index, "scalp_points")
        ese_points = self._read_context_item(LOCALIZATION.index, "ese_points")
        measurements = self._manager.get_global_item(MEASUREMENTS_KEY)
        electrodes = electrodes_from_points(
            scalp_points if isinstance(scalp_points, LabeledPoints) else None,
            ese_points if isinstance(ese_points, LabeledPoints) else None,
            measurements if isinstance(measurements, Electrodes) else None,
        )
        if electrodes.items:
            self._electrodes_table.show_electrodes(electrodes)

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
        # The import replaces the table and is stored right away, even
        # when the localization cannot run yet.
        self._save_inputs()

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
        # The import replaces the table and is stored right away, even
        # when the localization cannot run yet.
        self._save_inputs()

    def _ask_json_path(self, title: str) -> Path | None:
        """One JSON file chosen in a dialog, or ``None`` when cancelled."""
        path, _ = QFileDialog.getOpenFileName(self, title, "", _JSON_FILTER)
        return Path(path) if path else None

    def _clear_fiducial_store(self) -> None:
        """Clear the store; the cleared table is the project state now."""
        super()._clear_fiducial_store()
        self._save_inputs()

    def show_localized(self, electrodes: Electrodes) -> None:
        self._electrodes_table.show_electrodes(electrodes)

    def _on_measurement_table_changed(self) -> None:
        """Persist structural table edits without starting localization."""
        self._update_relocalize_state()
