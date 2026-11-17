"""Debounced background runner for live electrode localization."""

from PySide6.QtCore import QObject, QTimer, Signal

from virda.models.electrode import Electrodes
from virda.models.fiducial import Fiducials
from virda.models.mesh import Mesh, Normals
from virda.models.stage3_config import Stage3Config
from virda.ops.localize_electrodes import localize_electrodes
from virda_gui_new.constants import SETTLE_DELAY_MS
from virda_gui_new.services.task_runner import TaskRunner


class LiveLocalizationRunner(QObject):
    """Runs the localizer for the latest fiducial/electrode snapshot.

    The geometric inputs are fixed for one popup lifetime; each
    :meth:`submit` restarts a single-shot debounce timer, and only the
    freshest run ever delivers through :attr:`localized` or
    :attr:`failed` (the :class:`TaskRunner` retires superseded runs).
    """

    localized = Signal(object)  # noqa: N815 - Electrodes
    failed = Signal(str)
    busyChanged = Signal(bool)  # noqa: N815 - whether a run is scheduled or in flight

    def __init__(
        self,
        ese_mesh: Mesh,
        normals: Normals,
        scalp_mesh: Mesh,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if (
            not isinstance(ese_mesh, Mesh)
            or not isinstance(normals, Normals)
            or not isinstance(scalp_mesh, Mesh)
        ):
            raise ValueError(
                "Localization needs the ESE mesh and normals (run the ESE mesh step first)."
            )
        self._ese_mesh = ese_mesh
        self._normals = normals
        self._scalp_mesh = scalp_mesh
        self._options = Stage3Config()
        self._fixed_offset_shift_mm: float | None = None
        self._busy = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._run)
        self._runner: TaskRunner[Electrodes] = TaskRunner(self)
        self._runner.done.connect(self._on_done)
        self._runner.failed.connect(self._on_failed)

    def submit(
        self,
        fiducials: Fiducials,
        electrodes: Electrodes,
        options: Stage3Config | None = None,
        *,
        fixed_offset_shift_mm: float | None = None,
    ) -> None:
        """Schedule one debounced localization run for this snapshot.

        The electrode set may be a subset of the table: the popup
        recomputes a single edited electrode on its own. The options
        travel with the run, so a changed setting takes effect at once.
        """
        self._set_busy(True)
        self._timer.start(SETTLE_DELAY_MS)
        self._fiducials = fiducials
        self._electrodes = electrodes
        self._options = options or Stage3Config()
        self._fixed_offset_shift_mm = fixed_offset_shift_mm

    def stop(self) -> None:
        """Stop the debounce timer and any in-flight run."""
        self._timer.stop()
        self._runner.stop()
        self._set_busy(False)

    def _run(self) -> None:
        self._runner.start(
            lambda: localize_electrodes(
                self._ese_mesh,
                self._normals,
                self._scalp_mesh,
                self._fiducials,
                self._electrodes,
                config=self._options,
                fixed_offset_shift_mm=self._fixed_offset_shift_mm,
            )
        )

    def _on_done(self, result: Electrodes) -> None:
        self.localized.emit(result)
        self._finish_run()

    def _on_failed(self, message: str) -> None:
        self.failed.emit(message)
        self._finish_run()

    def _finish_run(self) -> None:
        """A delivered result ends the busy state unless a new run waits."""
        if not self._timer.isActive():
            self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        """Move the busy flag, announcing every real change."""
        if busy == self._busy:
            return

        self._busy = busy
        self.busyChanged.emit(busy)
