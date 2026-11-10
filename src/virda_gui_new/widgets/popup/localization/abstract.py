"""Live localization popup: thin view over the localization service."""

from PySide6.QtWidgets import QVBoxLayout, QWidget

from virda.models.electrode import Electrodes
from virda.models.mesh import Mesh, Normals
from virda_gui_new.dto.steps import ESE_MESH, LOCALIZATION, MESH_POSTPROCESSING, Step
from virda_gui_new.services.localization import (
    FiducialRows,
    LiveLocalizationRunner,
    MeasurementTable,
    blocked_reason,
    build_electrodes,
    build_fiducials,
)
from virda_gui_new.services.steps_manager import StepsManager
from virda_gui_new.widgets.popup.abstract import StepPopupBase

_NO_GEOMETRY = "Localization needs the ESE mesh and normals (run the ESE mesh step first)."
_STALE_CONTEXT = "Pipeline changed; results dropped."


def _as_mesh(value: object) -> Mesh | None:
    """Return ``value`` when it is a ready :class:`Mesh`, else ``None``."""
    return value if isinstance(value, Mesh) else None


def _as_normals(value: object) -> Normals | None:
    """Return ``value`` when it is a ready :class:`Normals`, else ``None``."""
    return value if isinstance(value, Normals) else None


class LocalizationPopupBase(StepPopupBase):
    """Base popup for the live localization step.

    The Run button is hidden: edits re-read the table hooks
    (``collect_fiducial_rows`` / ``collect_measurement_table``) and, when
    enough inputs exist, submit a debounced background run through a
    :class:`LiveLocalizationRunner`; results land in the step-4 context
    as ``electrodes``. Geometry comes from the step-3 context (with a
    step-2 mesh or step-1 scalp surface fallback) at construction;
    without it the hint shows the precondition and no runner is created.
    Status text flows through the base hint label and the ``on_status``
    hook for subclasses.
    """

    def __init__(self, step: Step, manager: StepsManager, parent: QWidget | None = None) -> None:
        # The runner is created once the geometry is known, but the
        # attribute must exist from the start: build_form() prefills the
        # fiducial table, and the prefill schedules an input change.
        self._localization_runner: LiveLocalizationRunner | None = None
        super().__init__(step, manager, parent)
        self._run_button.hide()
        self._ese_mesh = _as_mesh(self._read_context_item(ESE_MESH.index, "ese_mesh"))
        self._normals = _as_normals(self._read_context_item(ESE_MESH.index, "normals"))
        self._scalp_mesh = _as_mesh(self._read_context_item(MESH_POSTPROCESSING.index, "mesh"))
        if self._scalp_mesh is None:
            self._scalp_mesh = _as_mesh(self._read_context_item(1, "scalp_surface"))
        if (
            self._ese_mesh is not None
            and self._normals is not None
            and self._scalp_mesh is not None
        ):
            self._localization_runner = LiveLocalizationRunner(
                self._ese_mesh, self._normals, self._scalp_mesh, self
            )
            self._localization_runner.localized.connect(self._on_localized)
            self._localization_runner.failed.connect(self._on_run_failed)
            self._on_inputs_changed()
        else:
            self._hint_label.setText(_NO_GEOMETRY)

    def build_form(self, layout: QVBoxLayout) -> None:
        """Extension hook: add the fiducial and measurement tables here."""

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
        if self._localization_runner is None:
            return
        rows = self.collect_fiducial_rows()
        table = self.collect_measurement_table()
        reason = blocked_reason(len(rows), len(table))
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
        self._localization_runner.submit(fiducials, electrodes)
        self._set_hint("Running...")

    def _on_localized(self, electrodes: object) -> None:
        if not isinstance(electrodes, Electrodes):
            return
        try:
            self._manager.set_step_context_item(LOCALIZATION.index, "electrodes", electrodes)
        except ValueError:
            self._set_hint(_STALE_CONTEXT)
            return
        self._set_hint(f"Localized {len(electrodes.items)} electrodes.")
        self.show_localized(electrodes)

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
