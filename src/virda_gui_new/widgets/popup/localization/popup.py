"""Concrete localization popup: three tables over the live flow."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QAbstractButton,
    QFileDialog,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from virda.geometry import match_fiducial_name
from virda.models.coordinates import CoordinateSystem
from virda.models.electrode import Electrodes
from virda.models.fiducial import FiducialType
from virda.models.measurement import MeasurementPrediction
from virda.models.mri_volume import MRIVolume
from virda_gui_new.constants import SETTLE_DELAY_MS
from virda_gui_new.dto.steps import (
    FIDUCIALS_KEY,
    LOCALIZATION,
    MEASUREMENTS_KEY,
    NIFTI,
    Step,
)
from virda_gui_new.dto.viewer3d import LabeledPoints
from virda_gui_new.services import artifact_importing, artifact_loading
from virda_gui_new.services.localization import (
    FiducialRows,
    MeasurementTable,
    build_fiducials,
    electrodes_from_points,
    fiducial_surface_status,
    measurement_table_from_model,
    predict_missing_measurement,
    predict_missing_measurements,
)
from virda_gui_new.services.steps_manager import StepsManager
from virda_gui_new.services.task_runner import TaskRunner
from virda_gui_new.widgets.popup.localization.abstract import LocalizationPopupBase
from virda_gui_new.widgets.popup.localization.electrodes_table import ElectrodesTable
from virda_gui_new.widgets.popup.localization.fiducials_table import FiducialsTable
from virda_gui_new.widgets.popup.localization.frame_selector import FrameSelector
from virda_gui_new.widgets.popup.localization.measurements_table import MeasurementsTable


@dataclass(frozen=True)
class _Predictions:
    """The predictions of a request and the electrodes without one.

    ``reason`` carries why an electrode failed, so the user learns the
    cause instead of a silent miss.
    """

    predictions: list[tuple[str, str, MeasurementPrediction]] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    reason: str = ""


#: The inputs the predicted measurements depend on: the fiducial rows,
#: the measured distances and the ESE mesh.
PredictionFingerprint = tuple[
    tuple[tuple[str, float, float, float], ...],
    int,
]

_JSON_FILTER = "JSON (*.json)"
_FIDUCIALS_IMPORT_TITLE = "Import fiducials"
_MEASUREMENTS_IMPORT_TITLE = "Import measurements"
#: How long the inputs have to settle before the predictions are checked.


class LocalizationPopup(LocalizationPopupBase):
    """Localization step editor: tables drive the debounced live runs.

    The form stacks the editable fiducial and measurement grids over
    the read-only electrode grid; a coordinate-frame selector (head /
    world / fiducials) sits above the fiducial grid and the electrode
    grid and only changes the displayed numbers -- the canonical rows
    and the store stay world. Each edit in either input table
    re-validates the whole snapshot and schedules a run; localized
    results are shown in the electrode grid and stored in the step-4
    context by the base.
    """

    def __init__(
        self,
        step: Step,
        manager: StepsManager,
        parent: QWidget | None = None,
        *,
        project: Path | None = None,
    ) -> None:
        super().__init__(step, manager, parent, project=project)
        # The geometry is known only after the base built the form.
        self._update_prediction_state()
        self._update_surface_status()
        self._update_pick_state()
        # A prediction is a heavy computation: it runs on a worker
        # thread, so the window stays responsive while it is computed.
        self._prediction_runner = TaskRunner(self)
        self._prediction_runner.done.connect(self._on_prediction_done)
        self._prediction_runner.failed.connect(self._on_prediction_failed)
        self._batch_import_predictions = False
        # Whether a prediction is being computed; the inputs are blocked
        # until it is.
        self._prediction_busy = False

    def shutdown(self) -> None:
        """Stop the prediction worker, then the base machinery."""
        self._prediction_runner.stop()
        super().shutdown()

    def build_form(self, layout: QVBoxLayout) -> None:
        """Arrange the table sections with their frame selectors."""
        super().build_form(layout)
        self._coordinates = self._manager.geometry
        self._fiducials_table = FiducialsTable(self._coordinates, self)
        self._measurements_table = MeasurementsTable(self)
        self._electrodes_table = ElectrodesTable(self._coordinates, self)
        # The fiducials frame is defined by the rows the fiducials table
        # shows, so the table is displayed in the world and the head
        # frame only.
        self._fiducials_frames = FrameSelector(
            self, frames=(CoordinateSystem.HEAD, CoordinateSystem.WORLD)
        )
        self._electrodes_frames = FrameSelector(self)
        self._fiducials_frames.frameChanged.connect(self._fiducials_table.set_display_frame)
        self._electrodes_frames.frameChanged.connect(self._electrodes_table.set_display_frame)
        layout.addWidget(QLabel("Fiducials", self))
        layout.addWidget(self._fiducials_frames)
        layout.addWidget(self._fiducials_table)
        layout.addWidget(QLabel("Measurements", self))
        layout.addWidget(self._measurements_table)
        layout.addWidget(QLabel("Electrodes", self))
        layout.addWidget(self._electrodes_frames)
        layout.addWidget(self._electrodes_table)
        self._fiducials_table.rowsChanged.connect(self._on_inputs_changed)
        self._fiducials_table.cleared.connect(self._clear_fiducial_store)
        self._fiducials_table.importRequested.connect(self._import_fiducials_from_file)
        self._measurements_table.rowsChanged.connect(self._on_measurement_table_changed)
        self._fiducials_table.pickRequested.connect(self._on_pick_requested)
        self._measurements_table.electrodeEdited.connect(self._on_measurement_edited)
        self._measurements_table.electrodeSetChanged.connect(self._on_electrode_set_changed)
        self._measurements_table.importRequested.connect(self._import_measurements_from_file)
        self._measurements_table.predictRequested.connect(self._on_predict_requested)
        self._stale_timer = QTimer(self)
        self._stale_timer.setSingleShot(True)
        self._stale_timer.setInterval(SETTLE_DELAY_MS)
        self._stale_timer.timeout.connect(self._check_stale_predictions)
        self._prediction_state: PredictionFingerprint | None = None
        self._pick_slot: str | None = None
        self._prefill_fiducials()
        self._prefill_measurements()
        self._prefill_electrodes()
        self._sync_frame_selectors()
        # The predictions are up to date: only a change of the fiducials
        # makes them stale.
        self._prediction_state = self._prediction_fingerprint()

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

    def _sync_frame_selectors(self) -> None:
        """Enable available frames and keep each table on its selector.

        Called at build time and whenever the fiducials store changes:
        the fiducials frame is greyed out while fewer than three
        fiducials have coordinates and the head frame while no scalp
        mesh exists; a currently selected frame that became unavailable
        falls back to head (or world without a scalp mesh). The change
        signal keeps the table in step; for an available frame the
        table is set explicitly (a same-frame set is a no-op there).
        """
        head_ok = self._coordinates.is_available(CoordinateSystem.HEAD)
        fiducials_ok = self._coordinates.is_available(CoordinateSystem.FIDUCIALS)
        for selector, table in (
            (self._fiducials_frames, self._fiducials_table),
            (self._electrodes_frames, self._electrodes_table),
        ):
            selector.set_frame_enabled(CoordinateSystem.HEAD, head_ok)
            selector.set_frame_enabled(CoordinateSystem.FIDUCIALS, fiducials_ok)
            frame = selector.frame()
            if not self._coordinates.is_available(frame):
                fallback = CoordinateSystem.HEAD if head_ok else CoordinateSystem.WORLD
                selector.setCurrentIndex(selector.findData(fallback))
                continue  # the change signal switches the table
            table.set_display_frame(frame)

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
        """ "From file": fill the measurement grid from a measurements JSON.

        A file that carries only two distances per electrode gets its
        third one predicted, so the grid arrives complete; an electrode
        the prediction fails for keeps its empty cell.
        """
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
        self._batch_import_predictions = True
        # The predictions of the whole table are a heavy computation:
        # they run on a worker thread.
        self._set_prediction_busy(True)
        self._set_hint(f"Imported {len(table)} electrodes, predicting...")
        self._prediction_runner.start(lambda: self._predict_imported(table))

    def _predict_imported(self, table: MeasurementTable) -> _Predictions:
        """The predictions of an imported table and the electrodes without one."""
        try:
            predictions, failed = self._predict_missing_measurements(table)
        except Exception as exc:  # noqa: BLE001 - any prediction error is user-facing
            return _Predictions(reason=str(exc))
        items = [
            (electrode, fiducial_id, prediction)
            for (electrode, fiducial_id), prediction in predictions.items()
        ]
        return _Predictions(predictions=items, failed=failed)

    def _predict_missing_measurements(
        self, table: MeasurementTable
    ) -> tuple[dict[tuple[str, str], MeasurementPrediction], list[str]]:
        """The predictions of an imported table and the electrodes without one.

        An electrode whose measured distances cannot belong to one point
        gets no prediction: it is reported instead.
        """
        if self._ese_mesh is None:
            return {}, []
        try:
            fiducials = build_fiducials(self.collect_fiducial_rows())
        except ValueError:
            return {}, []
        return predict_missing_measurements(table, fiducials, self._ese_mesh, self._coordinates)

    def _on_predict_requested(self, electrode_id: str, fiducial_id: str) -> None:
        """Predict the missing measurement of one electrode."""
        reason = self._prediction_blocked_reason(electrode_id, FiducialType(fiducial_id))
        if reason is not None:
            self._set_hint(reason)
            return
        missing = FiducialType(fiducial_id)
        self._set_prediction_busy(True)
        self._set_hint("Predicting...")
        self._prediction_runner.start(lambda: self._predict_one(electrode_id, missing))

    def _predict_one(self, electrode_id: str, missing: FiducialType) -> _Predictions:
        """The prediction of one electrode, with the reason of a failure."""
        try:
            prediction = self._predict_measurement(electrode_id, missing)
        except ValueError as exc:
            return _Predictions(failed=[electrode_id], reason=str(exc))
        if prediction is None or prediction.distance_mm is None:
            return _Predictions(failed=[electrode_id])
        return _Predictions(predictions=[(electrode_id, missing.value, prediction)])

    def _on_prediction_done(self, outcome: _Predictions) -> None:
        """Write the predicted measurements; the worker is done."""
        batch_import = self._batch_import_predictions
        self._set_prediction_busy(False)
        self._apply_predictions(outcome)
        if batch_import:
            self._batch_import_predictions = False
            self._on_inputs_changed()
        if len(outcome.predictions) == 1:
            electrode, fiducial_id, prediction = outcome.predictions[0]
            distance = prediction.distance_mm or 0.0
            self._set_hint(f"Predicted {distance:.2f} mm for {electrode} {fiducial_id}.")
        elif outcome.predictions:
            self._set_hint(f"Predicted {len(outcome.predictions)} measurements.")
        if outcome.reason:
            self._set_hint(f"Prediction failed: {outcome.reason}")

    def _apply_predictions(self, outcome: _Predictions) -> None:
        """Write the predictions into the grid and report what failed."""
        ambiguous: list[str] = []
        for electrode, fiducial_id, prediction in outcome.predictions:
            if not self._apply_prediction(electrode, FiducialType(fiducial_id), prediction):
                # The stale value is dropped: the cell stays empty.
                self._measurements_table.clear_predicted(electrode, fiducial_id)
                ambiguous.append(electrode)
        if ambiguous:
            self._report_unpredicted(sorted(set(ambiguous)), "No unambiguous prediction")
        if outcome.failed:
            self._report_unpredicted(sorted(set(outcome.failed)), "No electrode position fits")

    def _on_prediction_failed(self, message: str) -> None:
        """The prediction failed: the reason is shown, the window lives."""
        self._set_prediction_busy(False)
        if self._batch_import_predictions:
            self._batch_import_predictions = False
            self._on_inputs_changed()
        self._set_hint(f"Prediction failed: {message}")

    def _on_measurement_edited(self, electrode_id: str) -> None:
        """Keep imported batch predictions from starting per-row runs."""
        if self._batch_import_predictions:
            self._schedule_save()
            return
        super()._on_measurement_edited(electrode_id)

    def _set_prediction_busy(self, busy: bool) -> None:
        """Block the inputs while a prediction is computed.

        The tables cannot change while the prediction runs: a changed
        measurement would make the prediction stale under the hands of
        the user.
        """
        self._prediction_busy = busy
        self._update_input_tables_enabled()

    def _update_input_tables_enabled(self) -> None:
        """Block every table while localization or prediction is running."""
        blocked = self._localization_busy or getattr(self, "_prediction_busy", False)
        self._measurements_table.setEnabled(not blocked)
        self._fiducials_table.setEnabled(not blocked)
        self._electrodes_table.setEnabled(not blocked)

    def _prediction_blocked_reason(self, electrode_id: str, missing: FiducialType) -> str | None:
        """Why the prediction cannot run, or ``None`` when it can."""
        if self._ese_mesh is None:
            return "The prediction needs the ESE mesh."
        if len(self.collect_fiducial_rows()) < 3:
            return "The prediction needs the three fiducials."
        distances = self._measurements_table.collect_measurement_table().get(electrode_id, {})
        measured = {
            slot for slot in (match_fiducial_name(name) for name in distances) if slot is not None
        }
        if len(measured) - (1 if missing in measured else 0) != 2:
            return "The prediction needs two measured distances."
        return None

    def _predict_measurement(
        self, electrode_id: str, missing: FiducialType
    ) -> MeasurementPrediction | None:
        """Predict one missing measurement, or ``None`` when it cannot run."""
        if self._ese_mesh is None:
            return None
        try:
            fiducials = build_fiducials(self.collect_fiducial_rows())
        except ValueError:
            return None
        table = self._measurements_table.collect_measurement_table()
        return predict_missing_measurement(
            table.get(electrode_id, {}),
            missing,
            fiducials,
            self._ese_mesh,
            self._coordinates,
        )

    def _apply_prediction(
        self, electrode_id: str, missing: FiducialType, prediction: MeasurementPrediction
    ) -> bool:
        """Write a predicted measurement into its cell, marked in italics.

        An ambiguous prediction is not written: the cell stays empty and
        the caller reports the electrode. Returns whether the cell was
        filled.
        """
        if prediction.ambiguous or prediction.distance_mm is None or prediction.residual_mm is None:
            return False
        note = f"predicted · residual {prediction.residual_mm:.2f} mm"
        self._measurements_table.set_predicted(
            electrode_id, missing.value, prediction.distance_mm, note
        )
        self._prediction_state = self._prediction_fingerprint()
        return True

    def _report_unpredicted(self, electrodes: list[str], reason: str) -> None:
        """Report the electrodes without a predicted measurement.

        The box is opened, not shown modally: the localization keeps
        running while the user reads it.
        """
        listed = ", ".join(electrodes)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Prediction")
        box.setText(f"{reason} for: {listed}.")
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.open()

    def _update_prediction_state(self) -> None:
        """Tell the table whether a prediction can run at all."""
        if self._ese_mesh is None:
            self._measurements_table.set_prediction_state(False, "the ESE mesh is needed")
        elif len(self.collect_fiducial_rows()) < 3:
            self._measurements_table.set_prediction_state(False, "the three fiducials are needed")
        else:
            self._measurements_table.set_prediction_state(True)
        # The state above says nothing about a localization in flight:
        # apply the busy block on top of it.
        self._measurements_table.set_prediction_blocked(self._localization_busy)

    def _update_surface_status(self) -> None:
        """Show how far the fiducials are from the scalp surface."""
        if self._scalp_mesh is None:
            self._fiducials_table.set_surface_status({})
            return
        status = fiducial_surface_status(
            self.collect_fiducial_rows(),
            self._scalp_mesh,
            self._coordinates,
            self._options.fiducial_tolerance_mm,
        )
        self._fiducials_table.set_surface_status(status)

    def _update_pick_state(self) -> None:
        """Tell the table whether a point can be picked (the scalp is needed)."""
        self._fiducials_table.set_pick_available(self._scalp_mesh is not None)
        # The availability above says nothing about a localization in
        # flight: apply the busy block on top of it.
        self._fiducials_table.set_pick_blocked(getattr(self, "_localization_busy", False))

    def _on_localization_busy(self, busy: bool) -> None:
        """Block the pick and predict buttons while a localization run goes on."""
        self._localization_busy = busy
        if getattr(self, "_pick_slot", None) is not None:
            self._pick_slot = None
            self._fiducials_table.set_pick_target(None)
            self.pickCancelled.emit()
        self._update_pick_state()
        self._measurements_table.set_prediction_blocked(busy)
        self._update_input_tables_enabled()

    def _prediction_fingerprint(self) -> PredictionFingerprint:
        """The inputs the predicted measurements depend on.

        Only the fiducials: they are what the predictions are derived
        from, while the measured distances are the user's own input and
        do not warrant a dialog.
        """
        rows = tuple(
            (name, float(xyz[0]), float(xyz[1]), float(xyz[2]))
            for name, xyz in self.collect_fiducial_rows()
        )
        return (rows, id(self._ese_mesh))

    def _check_stale_predictions(self) -> None:
        """Offer to recompute the predicted measurements when the inputs moved."""
        if not self._measurements_table.predicted_cells():
            return
        fingerprint = self._prediction_fingerprint()
        if fingerprint == self._prediction_state:
            return
        self._prediction_state = fingerprint
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Recalculate?")
        box.setText("The predicted measurements are out of date. Recalculate them?")
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.buttonClicked.connect(lambda button: self._on_recalculate_choice(box, button))
        box.open()

    def _on_recalculate_choice(self, box: QMessageBox, button: QAbstractButton) -> None:
        """Recompute the predicted measurements when the user says so."""
        box.close()
        if button is not box.button(QMessageBox.StandardButton.Yes):
            return
        cells = self._measurements_table.predicted_cells()
        if not cells:
            return
        self._set_prediction_busy(True)
        self._set_hint("Recalculating...")
        self._prediction_runner.start(lambda: self._recalculate(cells))

    def _recalculate(self, cells: list[tuple[str, str]]) -> _Predictions:
        """The new predictions of the stale cells and the failed electrodes.

        An electrode whose measured distances cannot belong to one point
        fails on its own: the other cells are still recomputed.
        """
        predictions: list[tuple[str, str, MeasurementPrediction]] = []
        failed: list[str] = []
        reason = ""
        for electrode, fiducial_id in cells:
            try:
                prediction = self._predict_measurement(electrode, FiducialType(fiducial_id))
            except ValueError as exc:
                reason = reason or str(exc)
                failed.append(electrode)
                continue
            if prediction is None or prediction.distance_mm is None:
                failed.append(electrode)
                continue
            predictions.append((electrode, fiducial_id, prediction))
        return _Predictions(predictions=predictions, failed=failed, reason=reason)

    def _on_inputs_changed(self) -> None:
        """Schedule the localization, the surface check and the stale check."""
        super()._on_inputs_changed()
        self._update_prediction_state()
        self._update_surface_status()
        self._update_pick_state()
        self._stale_timer.start()

    def _on_options_changed(self, *_args: object) -> None:
        """Store the settings and refresh the surface check."""
        super()._on_options_changed()
        self._update_surface_status()

    def _on_pick_requested(self, slot: str) -> None:
        """Start picking a point for a slot, or stop it for the same slot."""
        if self._pick_slot == slot:
            self._pick_slot = None
            self._fiducials_table.set_pick_target(None)
            self.pickCancelled.emit()
            return
        self._pick_slot = slot
        self._fiducials_table.set_pick_target(FiducialType(slot))
        self.pickRequested.emit(slot)

    def set_picked_point(self, slot: str, point: np.ndarray) -> None:
        """Write a point picked in the 3D view into the fiducials table.

        The store is refreshed right away, so the picked point shows in
        the 3D view even when the localization cannot run yet.
        """
        self._pick_slot = None
        self._fiducials_table.set_pick_target(None)
        self._fiducials_table.set_pick_point(slot, point)
        self._refresh_fiducial_store()

    def _ask_json_path(self, title: str) -> Path | None:
        """One JSON file chosen in a dialog, or ``None`` when cancelled."""
        path, _ = QFileDialog.getOpenFileName(self, title, "", _JSON_FILTER)
        return Path(path) if path else None

    def _clear_fiducial_store(self) -> None:
        """Clear the store, then re-sync frame availability (fiducials lost)."""
        super()._clear_fiducial_store()
        self._sync_frame_selectors()
        # The cleared table is the project state now.
        self._save_inputs()

    def _refresh_fiducial_store(self) -> None:
        """Refresh the store, then re-sync frame availability."""
        super()._refresh_fiducial_store()
        self._sync_frame_selectors()

    def show_localized(self, electrodes: Electrodes) -> None:
        self._electrodes_table.show_electrodes(electrodes)

    def _on_measurement_table_changed(self) -> None:
        """Persist structural table edits without starting localization."""
        self._update_relocalize_state()
