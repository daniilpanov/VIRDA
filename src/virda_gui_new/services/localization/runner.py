"""Debounced background runner for live electrode localization."""

from PySide6.QtCore import QObject, QTimer, Signal

from virda.models.electrode import Electrodes
from virda.models.fiducial import Fiducials
from virda.models.mesh import Mesh, Normals
from virda_gui.api.options import LocalizeOptions
from virda_gui_new.api import ops
from virda_gui_new.services.task_runner import TaskRunner

_DEBOUNCE_MS = 400


class LiveLocalizationRunner(QObject):
    """Runs ``ops.localize`` for the latest fiducial/electrode snapshot.

    The geometric inputs are fixed for one popup lifetime; each
    :meth:`submit` restarts a single-shot debounce timer, and only the
    freshest run ever delivers through :attr:`localized` or
    :attr:`failed` (the :class:`TaskRunner` retires superseded runs).
    """

    localized = Signal(object)  # noqa: N815 - Electrodes
    failed = Signal(str)

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
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._run)
        self._runner: TaskRunner[Electrodes] = TaskRunner(self)
        self._runner.done.connect(self._on_done)
        self._runner.failed.connect(self._on_failed)

    def submit(self, fiducials: Fiducials, electrodes: Electrodes) -> None:
        """Schedule one debounced localization run for this snapshot."""
        self._timer.start(_DEBOUNCE_MS)
        self._fiducials = fiducials
        self._electrodes = electrodes

    def stop(self) -> None:
        """Stop the debounce timer and any in-flight run."""
        self._timer.stop()
        self._runner.stop()

    def _run(self) -> None:
        self._runner.start(
            lambda: ops.localize(
                self._ese_mesh,
                self._normals,
                self._scalp_mesh,
                self._fiducials,
                self._electrodes,
                LocalizeOptions(),
            )
        )

    def _on_done(self, result: Electrodes) -> None:
        self.localized.emit(result)

    def _on_failed(self, message: str) -> None:
        self.failed.emit(message)
