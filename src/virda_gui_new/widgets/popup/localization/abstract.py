"""Live localization popup: thin view over the localization service."""

from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QPushButton, QVBoxLayout, QWidget

from virda.models.electrode import Electrode, Electrodes
from virda.models.mesh import Mesh, Normals
from virda.models.stage3_config import Stage3Config
from virda_gui_new.dto.steps import (
    FIDUCIALS_KEY,

    ESE_MESH,
    LOCALIZATION,
    Step,
)
from virda_gui_new.dto.viewer3d import LabeledPoints
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
        if self._localization_runner is not None:
            self._localization_runner.stop()
        super().shutdown()

    def _on_inputs_changed(self) -> None:
        """Rebuild the snapshot and schedule a run when inputs suffice."""
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

    def _on_measurement_edited(self, electrode_id: str) -> None:
        """Recompute only a named electrode after its row is complete."""
        if self._loading or self._localization_runner is None:
            return
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
            self._manager.set_step_context_item(LOCALIZATION.index, "electrodes", electrodes)
            # The scene shows the electrodes as two point sets: the ones
            # on the scalp and the ones on the ESE surface.
            scalp_points, ese_points = points_from_electrodes(electrodes)
            self._manager.set_step_context_item(LOCALIZATION.index, "scalp_points", scalp_points)
            self._manager.set_step_context_item(LOCALIZATION.index, "ese_points", ese_points)
        except ValueError:
            self._set_hint(_STALE_CONTEXT)
            return
        self._refresh_fiducial_store()
        count = len(electrodes.items)
        unit = "electrode" if count == 1 else "electrodes"
        self._set_hint(f"Localized {count} {unit}.")
        self.show_localized(electrodes)

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

    def _refresh_fiducial_store(self) -> None:
        """Mirror the current fiducial table into the global store.

        A successful run proves the table rows are valid, so the store is
        refreshed from them; this keeps store ~= table ~= what a save
        writes. An unchanged store is left alone so the pull-model
        listeners (scene rebuild) are not re-triggered needlessly.
        """
        points = labeled_points_from_rows(self.collect_fiducial_rows())
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
