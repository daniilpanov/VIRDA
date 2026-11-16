"""Live localization popup: thin view over the localization service."""

from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QPushButton, QVBoxLayout, QWidget

from virda.models.electrode import Electrode, Electrodes
from virda.models.mesh import Mesh, Normals
from virda.models.stage3_config import Stage3Config
from virda_gui_new.constants import SETTLE_DELAY_MS
from virda_gui_new.dto.steps import (
    ESE_MESH,
    FIDUCIALS_KEY,
    LOCALIZATION,
    MEASUREMENTS_KEY,
    Step,
)
from virda_gui_new.dto.viewer3d import LabeledPoints
from virda_gui_new.services import artifact_saving
from virda_gui_new.services.localization import (
    FiducialRows,
    LiveLocalizationRunner,
    MeasurementTable,
    blocked_reason,
    build_electrodes,
    build_fiducials,
    complete_measurement_table,
    is_complete_measurement,
    labeled_points_from_rows,
    measurement_counts,
    points_from_electrodes,
)
from virda_gui_new.services.steps_manager import StepsManager
from virda_gui_new.widgets.popup.abstract import StepPopupBase

_NEEDS_INPUTS = "Localization needs {} (run the earlier steps first)."
_READY = "Ready."
_NO_GEOMETRY = (
    "Localization needs the ESE mesh and the scalp surface (run the earlier steps first)."
)
_STALE_CONTEXT = "Pipeline changed; results dropped."
_RELOCALIZE = "Re-localization"


def _as_mesh(value: object) -> Mesh | None:
    """Return ``value`` when it is a ready :class:`Mesh`, else ``None``."""
    return value if isinstance(value, Mesh) else None


def _as_normals(value: object) -> Normals | None:
    """Return ``value`` when it is a ready :class:`Normals`, else ``None``."""
    return value if isinstance(value, Normals) else None


def _same_results(left: Electrodes, right: Electrodes) -> bool:
    """Whether two result models hold the same electrodes."""
    if len(left.items) != len(right.items):
        return False
    return all(
        _same_electrode(one, other) for one, other in zip(left.items, right.items, strict=True)
    )


def _same_electrode(left: Electrode, right: Electrode) -> bool:
    """Whether two electrodes hold the same data."""
    return (
        left.electrode_id == right.electrode_id
        and left.measured_distances == right.measured_distances
        and _same_point(left.scalp_coords, right.scalp_coords)
        and _same_point(left.ese_coords, right.ese_coords)
    )


def _same_point(left: np.ndarray | None, right: np.ndarray | None) -> bool:
    """Whether two optional points are the same."""
    if left is None or right is None:
        return left is right
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _same_points(left: LabeledPoints, right: LabeledPoints) -> bool:
    """Structural equality of two labeled point sets."""
    return (
        left.points.shape == right.points.shape
        and bool(np.array_equal(left.points, right.points))
        and left.labels == right.labels
    )


def localization_options(config: dict[str, Any]) -> Stage3Config:
    """The localization settings a stored configuration holds.

    A setting that the configuration does not carry keeps its default.
    The export service reads the settings the same way, so an export
    localizes the way the step does.
    """
    defaults = Stage3Config()
    calibrate = config.get("calibrate_ese_offset")
    threshold = config.get("residual_threshold_mm")
    tolerance = config.get("fiducial_tolerance_mm")
    return Stage3Config(
        calibrate_ese_offset=(
            calibrate if isinstance(calibrate, bool) else defaults.calibrate_ese_offset
        ),
        residual_threshold_mm=(
            float(threshold)
            if isinstance(threshold, int | float)
            else defaults.residual_threshold_mm
        ),
        fiducial_tolerance_mm=(
            float(tolerance)
            if isinstance(tolerance, int | float)
            else defaults.fiducial_tolerance_mm
        ),
    )


def localization_options_payload(options: Stage3Config) -> dict[str, Any]:
    """The localization settings as they are stored in the step configuration."""
    return {
        "calibrate_ese_offset": options.calibrate_ese_offset,
        "residual_threshold_mm": options.residual_threshold_mm,
        "fiducial_tolerance_mm": options.fiducial_tolerance_mm,
    }


class LocalizationPopupBase(StepPopupBase):
    """Base popup for the live localization step.

    The Run button is hidden: edits re-read the table hooks
    (``collect_fiducial_rows`` / ``collect_measurement_table``) and, when
    enough inputs exist, submit a debounced background run through a
    :class:`LiveLocalizationRunner`; results land in the step-4 context
    as ``electrodes``, the current fiducial table rows are mirrored
    into the manager's global ``fiducials`` store and the current
    measurement table into its global ``measurements`` store (both
    ready models for the export popups). Geometry comes from the step-3
    context (with a step-2 mesh or step-1 scalp surface fallback) at
    construction; without it the hint shows the precondition and no
    runner is created. Status text flows through the base hint label and
    the ``on_status`` hook for subclasses.

    The measurements drive the recompute: a changed row recomputes that
    electrode alone, a changed electrode set recomputes all of them. The
    tables stay editable at all times -- only the computation waits for
    the fiducials -- and the "Re-localization" button triggers a full
    recompute of the current tables.
    """

    def __init__(
        self,
        step: Step,
        manager: StepsManager,
        parent: QWidget | None = None,
        *,
        project: Path | None = None,
    ) -> None:
        # The runner is created once the geometry is known, but the
        # attribute must exist from the start: build_form() prefills the
        # fiducial table, and the prefill schedules an input change.
        self._localization_runner: LiveLocalizationRunner | None = None
        self._pending_scope: str | None = None
        self._relocalize_button: QPushButton | None = None
        # The geometry is read after the form is built, but the
        # attributes must exist from the start: the prefill of the
        # tables runs while the form is built.
        self._ese_mesh: Mesh | None = None
        self._normals: Normals | None = None
        self._scalp_mesh: Mesh | None = None
        # The options are read when the form is built: the parent sets
        # the step and the project before it calls build_form().
        self._options: Stage3Config = Stage3Config()
        # The tables are stored once they settle: every edit schedules
        # the save, so a picked point or an edited cell is never lost.
        # The timer has no parent yet: the widget is not initialized.
        # It is destroyed when the popup shuts down.
        self._save_timer: QTimer | None = QTimer()
        timer = self._save_timer
        timer.setSingleShot(True)
        timer.setInterval(SETTLE_DELAY_MS)
        timer.timeout.connect(self._on_inputs_settled)
        # The tables are prefilled while the form is built: those changes
        # are the project's own, not the user's, so they start no run.
        self._loading = True
        super().__init__(step, manager, parent, project=project)
        self._loading = False
        self._run_button.hide()
        self._ese_mesh = _as_mesh(self._read_context_item(ESE_MESH.index, "ese_mesh"))
        self._normals = _as_normals(self._read_context_item(ESE_MESH.index, "normals"))
        self._scalp_mesh = self._manager.active_scalp_mesh()
        missing = self._manager.missing_inputs(self._step.index)
        if missing:
            self._hint_label.setText(_NEEDS_INPUTS.format(", ".join(missing)))
        elif self._ese_mesh is None or self._normals is None or self._scalp_mesh is None:
            # The manager holds the inputs, but the contexts do not hold
            # the geometry the runner needs.
            self._hint_label.setText(_NO_GEOMETRY)
        else:
            self._localization_runner = LiveLocalizationRunner(
                self._ese_mesh, self._normals, self._scalp_mesh, self
            )
            self._localization_runner.localized.connect(self._on_localized)
            self._localization_runner.failed.connect(self._on_run_failed)
            # The step does not run on its own: the first edit (or the
            # re-localization button) starts it.
            reason = self._blocked_reason()
            self._set_hint(f"Blocked: {reason}." if reason else _READY)
        self._update_relocalize_state()
        self._run_on_open()

    def _load_options(self) -> Stage3Config:
        """The stored localization settings (defaults when there are none)."""
        return localization_options(self._load_config())

    def build_form(self, layout: QVBoxLayout) -> None:
        """Add the localization settings and the re-localization button."""
        self._options = self._load_options()
        self._calibrate_box = QCheckBox("Calibrate ESE offset", self)
        self._calibrate_box.setChecked(self._options.calibrate_ese_offset)
        layout.addWidget(self._calibrate_box)

        self._threshold_spin = QDoubleSpinBox(self)
        self._threshold_spin.setRange(0.1, 100.0)
        self._threshold_spin.setDecimals(2)
        self._threshold_spin.setValue(self._options.residual_threshold_mm)
        layout.addWidget(self._threshold_spin)

        self._relocalize_button = QPushButton(_RELOCALIZE, self)
        self._relocalize_button.clicked.connect(self._re_localize)
        layout.addWidget(self._relocalize_button)

        self._calibrate_box.toggled.connect(self._on_options_changed)
        self._threshold_spin.valueChanged.connect(self._on_options_changed)

    def collect_fiducial_rows(self) -> FiducialRows:
        """Hook: return the current ``(name, xyz)`` fiducial rows."""
        return []

    def collect_measurement_table(self) -> MeasurementTable:
        """Hook: return the current electrode -> fiducial -> distance table."""
        return {}

    def show_localized(self, electrodes: Electrodes) -> None:
        """Hook: called after each successful localization."""

    def on_status(self, text: str) -> None:
        """Hook: called with every status line shown in the hint label."""

    def shutdown(self) -> None:
        """Stop the debounced live runs, then the base machinery."""
        timer = self._save_timer
        self._save_timer = None
        if timer is not None:
            timer.stop()
            # The timer has no parent: dropping the last reference lets
            # shiboken delete it. A deferred delete would race the
            # garbage collector and destroy the timer twice.
            del timer
        # The pending save would be lost with the timer: the tables the
        # user entered are written before the popup goes away.
        self._on_inputs_settled()
        if self._localization_runner is not None:
            self._localization_runner.stop()
        super().shutdown()

    def _on_inputs_changed(self) -> None:
        """Rebuild the snapshot and schedule a run when inputs suffice."""
        self._schedule_save()
        self._update_relocalize_state()
        if self._loading or self._localization_runner is None:
            return
        rows = self.collect_fiducial_rows()
        table = complete_measurement_table(self.collect_measurement_table())
        reason = self._blocked_reason()
        if reason is not None:
            self._set_hint(f"Blocked: {reason}.")
            return
        try:
            fiducials = build_fiducials(rows)
            electrodes = build_electrodes(table)
        except ValueError as exc:
            self._set_hint(str(exc))
            return
        assert self._localization_runner is not None
        self._pending_scope = None
        self._localization_runner.submit(fiducials, electrodes, self._options)
        self._set_hint("Running...")

    def _on_electrode_set_changed(self) -> None:
        """The electrode set changed: the results follow the table.

        Adding or removing an electrode does not move the other ones, so
        nothing is recomputed: the results just lose the electrode that
        left the table.
        """
        self._schedule_save()
        self._publish_results()

    def _on_measurement_edited(self, electrode_id: str) -> None:
        """Recompute only a named electrode after its row is complete."""
        if self._loading or self._localization_runner is None:
            return
        # The edited cell is an input of the step: it is stored like
        # every other change of the tables.
        self._schedule_save()
        rows = self.collect_fiducial_rows()
        table = self.collect_measurement_table()
        distances = table.get(electrode_id)
        if not electrode_id or distances is None or not is_complete_measurement(distances):
            self._update_relocalize_state()
            return
        reason = self._blocked_reason()
        if reason is not None:
            self._set_hint(f"Blocked: {reason}.")
            return
        try:
            fiducials = build_fiducials(rows)
            electrodes = build_electrodes({electrode_id: distances})
        except ValueError:
            self._update_relocalize_state()
            return
        assert self._localization_runner is not None
        self._pending_scope = electrode_id
        self._localization_runner.submit(
            fiducials,
            electrodes,
            self._options,
            fixed_offset_shift_mm=self._scoped_offset_shift(),
        )
        self._set_hint("Running...")

    def _scoped_offset_shift(self) -> float | None:
        """Reuse the last full-run calibration during one-electrode updates."""
        if not self._options.calibrate_ese_offset:
            return None
        results = self._stored_results()
        return (
            0.0
            if results is None or results.calibrated_offset_shift_mm is None
            else (results.calibrated_offset_shift_mm)
        )

    def _re_localize(self) -> None:
        """Recompute every electrode from the current tables."""
        self._pending_scope = None
        self._on_inputs_changed()

    def _update_relocalize_state(self) -> None:
        """Enable the re-localization button while the inputs allow a run."""
        if self._relocalize_button is None:
            return
        self._relocalize_button.setEnabled(self._blocked_reason() is None)

    def _blocked_reason(self) -> str | None:
        """Why the localization cannot run right now (None when it can)."""
        if self._localization_runner is None:
            return "no geometry"
        rows = self.collect_fiducial_rows()
        table = complete_measurement_table(self.collect_measurement_table())
        complete, incomplete = measurement_counts(table)
        return blocked_reason(len(rows), complete, incomplete)

    def _stored_results(self) -> Electrodes | None:
        """The results of the step, as its context holds them.

        The step-4 context is the one place the results live: the popup
        reads them there instead of keeping a copy of its own.
        """
        results = self._read_context_item(LOCALIZATION.index, "electrodes")
        return results if isinstance(results, Electrodes) else None

    def _run_on_open(self) -> None:
        """Recompute every complete electrode after the tables load."""
        if self._localization_runner is None:
            return
        if self._blocked_reason() is not None:
            return
        try:
            fiducials = build_fiducials(self.collect_fiducial_rows())
            table = complete_measurement_table(self.collect_measurement_table())
            electrodes = build_electrodes(table)
        except ValueError:
            return
        self._pending_scope = None
        self._localization_runner.submit(fiducials, electrodes, self._options)
        self._set_hint("Running...")

    def _on_options_changed(self, *_args: object) -> None:
        """Store the changed localization settings and use them from now on."""
        self._options = Stage3Config(
            calibrate_ese_offset=self._calibrate_box.isChecked(),
            residual_threshold_mm=self._threshold_spin.value(),
        )
        self._save_config(localization_options_payload(self._options))

    def _on_localized(self, electrodes: object) -> None:
        if not isinstance(electrodes, Electrodes):
            return
        scope = self._pending_scope
        self._pending_scope = None
        if scope is not None:
            electrodes = self._merge_electrode(scope, electrodes)
        try:
            self._publish_results(electrodes)
        except ValueError:
            self._set_hint(_STALE_CONTEXT)
            return
        self._refresh_fiducial_store()
        count = len(electrodes.items)
        unit = "electrode" if count == 1 else "electrodes"
        self._set_hint(f"Localized {count} {unit}.")
        self._save_localization(electrodes)

    def _merge_electrode(self, electrode_id: str, fresh: Electrodes) -> Electrodes:
        """Replace one electrode in the results on screen with the fresh one."""
        items: list[Electrode] = []
        replaced = False
        results = self._stored_results()
        for electrode in results.items if results is not None else []:
            if electrode.electrode_id == electrode_id:
                items.extend(fresh.items)
                replaced = True
            else:
                items.append(electrode)
        if not replaced:
            items.extend(fresh.items)
        offset_shift = (
            fresh.calibrated_offset_shift_mm
            if fresh.calibrated_offset_shift_mm is not None
            else results.calibrated_offset_shift_mm
            if results is not None
            else None
        )
        return Electrodes(items, calibrated_offset_shift_mm=offset_shift)

    def _schedule_save(self) -> None:
        """Store the tables and publish the state once they settle.

        Every change of a table schedules the save, so an edited cell
        or a picked point reaches the project even when the
        localization cannot run. The same timer publishes the current
        state, so the store, the scene and the result grid follow the
        tables.
        """
        if self._save_timer is None:
            return
        self._save_timer.start()

    def _on_inputs_settled(self) -> None:
        """The tables settled: store them and publish the current state."""
        self._save_inputs()
        self._refresh_fiducial_store()
        self._mirror_measurements_into_store()
        self._publish_results()

    def _publish_results(self, results: Electrodes | None = None) -> None:
        """Publish the results to the step context and the result grid.

        The step-4 context is the one place the results live: the popup
        writes them there and reads them from there. Without
        ``results`` the stored ones are published, restricted to the
        electrodes of the current table: a deleted electrode leaves the
        result, so the scene and the result grid follow the table even
        when no localization ran. A step without any result publishes
        nothing: the context keeps what it holds. Results that did not
        change are not published again.
        """
        if results is None:
            stored = self._stored_results()
            if stored is None:
                return
            results = self._results_for_table()
        current = self._stored_results()
        if current is not None and _same_results(current, results):
            return
        # The scene shows the electrodes as two point sets: the ones
        # on the scalp and the ones on the ESE surface.
        scalp_points, ese_points = points_from_electrodes(results)
        # One write of the three keys means one rebuild of the scene.
        self._manager.set_step_context_items(
            LOCALIZATION.index,
            {
                "electrodes": results,
                "scalp_points": scalp_points,
                "ese_points": ese_points,
            },
        )
        self.show_localized(results)

    def _results_for_table(self) -> Electrodes:
        """The stored results, restricted to the electrodes of the table."""
        results = self._stored_results()
        if results is None:
            return Electrodes([])
        ids = set(self.collect_measurement_table())
        return Electrodes([item for item in results.items if item.electrode_id in ids])

    def _save_inputs(self) -> None:
        """Store the current input tables in the project.

        The tables are the inputs of the step, so they are written even
        when the localization cannot run yet (no fiducials, no
        measurements): the project then keeps what the user entered or
        imported.
        """
        if self._project is None:
            return
        try:
            artifact_saving.save_localization_inputs(
                self._project,
                self.collect_fiducial_rows(),
                self.collect_measurement_table(),
            )
        except Exception as error:  # noqa: BLE001 - any save error is user-facing
            self._set_hint(f"Failed to save the project inputs: {error}")

    def _mirror_measurements_into_store(self) -> None:
        """Mirror the current measurement table into the global store.

        Like the fiducial table (mirrored into the global store), the
        measurement table lands in the manager's global store as an
        :class:`Electrodes` model (``measurements``), so the export
        popups can write the last successfully run state without
        touching disk. An invalid table (empty or duplicate electrode id)
        is a hint, not a failure: the electrodes result is kept and
        export falls back to the seeded or last stored state. An
        unchanged store is left alone, so the pull-model listeners are
        not re-triggered.
        """
        try:
            measurements = build_electrodes(self.collect_measurement_table())
        except ValueError as exc:
            self._set_hint(f"Measurements not stored: {exc}")
            return
        if self._manager.get_global_item(MEASUREMENTS_KEY) == measurements:
            return
        self._manager.set_global_item(MEASUREMENTS_KEY, measurements)

    def _save_localization(self, electrodes: Electrodes) -> None:
        """Persist the run's outputs; a save failure is a failed status."""
        if self._project is None:
            return
        try:
            artifact_saving.save_localization(
                self._project,
                self.collect_fiducial_rows(),
                self.collect_measurement_table(),
                electrodes,
            )
        except Exception as error:  # noqa: BLE001 - any save error is user-facing
            self._set_hint(f"Failed to save localization files: {error}")
            self.failed.emit(str(error))

    def _refresh_fiducial_store(self) -> None:
        """Mirror the current fiducial table into the global store.

        The store follows the table, so the scene, the anatomical frame
        and the export all read the same points the user sees. An
        unchanged store is left alone so the pull-model listeners (scene
        rebuild) are not re-triggered needlessly.
        """
        rows = self.collect_fiducial_rows()
        if not rows:
            self._clear_fiducial_store()
            return
        points = labeled_points_from_rows(rows)
        current = self._manager.get_global_item(FIDUCIALS_KEY)
        if isinstance(current, LabeledPoints) and _same_points(current, points):
            return
        self._manager.set_global_item(FIDUCIALS_KEY, points)

    def _clear_fiducial_store(self) -> None:
        """Drop the global fiducials entry so the scene loses the points.

        The table's Empty action clears the entry; the pull-model
        listeners (MainWindow scene rebuild) re-read the store after the
        write, so the rebuilt scene no longer carries fiducial points.
        An entry that is already gone is left alone, so the listeners
        are not re-triggered.
        """
        if self._manager.get_global_item(FIDUCIALS_KEY) is None:
            return
        self._manager.set_global_item(FIDUCIALS_KEY, None)

    def _on_run_failed(self, message: str) -> None:
        self._set_hint(message)

    def _set_hint(self, text: str) -> None:
        self._hint_label.setText(text)
        self.on_status(text)

    def _read_context_item(self, step_n: int, key: str) -> object | None:
        try:
            return self._manager.get_step_context_item(step_n, key)
        except ValueError:
            return None
