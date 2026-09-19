"""Background pipeline execution and log forwarding.

The GUI never runs the heavy pipeline work on its own thread;
:class:`PipelineRunner` spawns a daemon thread, streams human-readable
progress plus sentinel markers over the shared :class:`queue.Queue`, and the
main thread turns those sentinels into Qt signals via :meth:`PipelineRunner.poll`.
The pipeline itself is orchestrated here from the pure ``virda.ops`` atoms and
``virda.io`` importers/exporters.
"""

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from virda.io.exporters import (
    export_electrodes,
    export_ese_mesh,
    export_fiducials,
    export_head_mask,
    export_scalp_mesh,
)
from virda.io.importers import import_fiducials, import_measurements, import_nifti
from virda.models.coordsystem import Coordsystem
from virda.models.ese_mesh import ESEMesh
from virda.models.fiducial import Fiducials
from virda.models.mri_volume import MRIVolume
from virda.models.scalp_mesh import ScalpMesh
from virda.ops.atoms import clean, decimate, generate_ese, generate_scalp_surface, localize, smooth
from virda.ops.options import (
    CleanOptions,
    DecimateOptions,
    EseOptions,
    LocalizeOptions,
    SealingOptions,
    SmoothOptions,
)
from virda.ops.scalp_surface import ScalpSurface
from virda_gui.state import AppState

_DONE_SENTINEL = "__DONE__"
_ERROR_SENTINEL = "__ERROR__"
_EXPORT_DONE_SENTINEL = "__EXPORT_DONE__"
_EXPORT_ERROR_SENTINEL = "__EXPORT_ERROR__"


@dataclass(frozen=True)
class PipelineRequest:
    """Run parameters collected from the GUI config tab.

    The pure pipeline is configured entirely through the typed dataclasses in
    :mod:`virda.ops.options`; :class:`PipelineRequest` bundles them with the
    three external inputs (the MRI scan, an optional fiducials file and the
    parsed MNE coordsystem) so the runner never touches pydantic or pipeline
    config models.
    """

    nifti_path: Path
    project_dir: Path
    fiducials_path: Path | None = None
    coordsystem: Coordsystem | None = None
    sealing: SealingOptions = SealingOptions()
    cleaning: CleanOptions = CleanOptions()
    smoothing: SmoothOptions = SmoothOptions()
    decimation: DecimateOptions = DecimateOptions()
    ese: EseOptions | None = None
    localization: LocalizeOptions = LocalizeOptions()


class _QueueLogHandler(logging.Handler):
    """Forward ``virda.*`` log records into the GUI log queue.

    ``emit`` runs on whatever thread logged the record (pipeline and viewer
    run in background threads); :class:`queue.Queue` makes the hand-off to
    the main thread safe.
    """

    def __init__(self, log_queue: queue.Queue[str | None]) -> None:
        super().__init__()
        self._log_queue = log_queue
        self.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._log_queue.put(self.format(record))
        except Exception:  # pragma: no cover - logging must never raise
            self.handleError(record)


class PipelineRunner(QObject):
    """Run the virda pipeline off the GUI thread and stream log messages.

    Lives on the main thread so its signals dispatch through the Qt event
    loop; only :meth:`_run` touches background threads.  The queue is shared
    with :class:`virda_gui.widgets.LogViewer` via an attached
    :class:`_QueueLogHandler`.
    """

    finished = Signal()
    failed = Signal()
    exportDone = Signal()  # noqa: N815
    exportFailed = Signal()  # noqa: N815

    def __init__(self, state: AppState) -> None:
        super().__init__()
        self._state = state
        self.log_handler = _QueueLogHandler(state.log_queue)
        self._pipeline_thread: threading.Thread | None = None
        self._export_threads: list[threading.Thread] = []
        self._closed = False

    def submit(self, request: PipelineRequest, measurements_path: str | None) -> None:
        """Start the pipeline on a background thread (daemon)."""
        if self._closed:
            return
        thread = threading.Thread(
            target=self._run, args=(request, measurements_path), daemon=True
        )
        self._pipeline_thread = thread
        thread.start()

    def start_html_export(self, project: Path, output: Path) -> None:
        """Export the HTML viewer for ``project`` on a background thread."""
        if self._closed:
            return

        def _export() -> None:
            try:
                from virda_gui.export.html_export import export_project

                export_project(str(project), output)
                self._state.log_queue.put(f"HTML exported: {output}")
                self._state.log_queue.put(_EXPORT_DONE_SENTINEL)
            except Exception as exc:
                self._state.log_queue.put(f"HTML export failed: {exc}")
                self._state.log_queue.put(_EXPORT_ERROR_SENTINEL)

        thread = threading.Thread(target=_export, daemon=True)
        self._export_threads.append(thread)
        thread.start()

    def shutdown(self, timeout: float = 3.0) -> None:
        """Give background threads a bounded chance to finish at teardown.

        The pipeline and HTML export threads are daemons, so the process can
        exit without them; joining for a bounded time handles the common cases
        (a finished or nearly finished thread) instead of tearing the app down
        while a worker still writes into the log queue.
        """
        deadline = time.monotonic() + timeout
        for thread in [self._pipeline_thread, *self._export_threads]:
            if thread is None or not thread.is_alive():
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(remaining)
        self._closed = True

    def _run(self, request: PipelineRequest, measurements_path: str | None) -> None:
        """Background thread: run the pure-ops pipeline and post results to the queue."""
        try:
            self._state.log_queue.put("Building configuration...")
            project = request.project_dir
            mri = import_nifti(request.nifti_path)

            surface = generate_scalp_surface(mri, request.sealing)
            mesh = clean(surface.mesh, request.cleaning)
            mesh = smooth(mesh, request.smoothing)
            if request.decimation.density_percent < 100.0:
                mesh = decimate(mesh, request.decimation)

            fiducials = self._resolve_fiducials(request)
            self._export_stage1(project, mri, surface, mesh, fiducials)

            msg = f"Stage 1: mesh with {len(mesh.vertices)} vertices"
            self._state.log_queue.put(msg)

            ese_mesh: ESEMesh | None = (
                self._run_stage2(project, request.ese, mesh)
                if request.ese is not None
                else None
            )

            if ese_mesh is not None and measurements_path:
                electrodes = localize(
                    surface=ese_mesh,
                    fiducials=fiducials,
                    electrodes=import_measurements(measurements_path),
                    options=request.localization,
                )
                export_electrodes(project / "localization" / "electrodes.json", electrodes)
                items = electrodes.items
                localized = sum(1 for e in items if e.is_localized)
                flagged = sum(1 for e in items if e.flagged)
                shift = electrodes.calibrated_offset_shift_mm
                msg = f"Stage 3: {localized}/{len(items)} electrodes localized ({flagged} flagged)"
                if shift is not None:
                    msg += f", ESE offset shift {shift:.2f} mm"
                self._state.log_queue.put(msg)
                self._state.stage3_summary = {
                    "total": len(items),
                    "localized": localized,
                    "flagged": flagged,
                    "offset_shift_mm": shift,
                }
            elif measurements_path:
                self._state.log_queue.put(
                    "Stage 3 skipped: ESE mesh or measurements are not available."
                )

            self._state.log_queue.put("Pipeline completed successfully.")
            self._state.log_queue.put(_DONE_SENTINEL)

        except Exception as exc:
            self._state.log_queue.put(f"ERROR: {exc}")
            self._state.log_queue.put(_ERROR_SENTINEL)

    @staticmethod
    def _resolve_fiducials(request: PipelineRequest) -> Fiducials:
        """Pick the fiducials for Stage 3 from the loaded inputs."""
        if request.fiducials_path is not None:
            return import_fiducials(request.fiducials_path)
        if request.coordsystem is not None:
            fiducials = request.coordsystem.to_fiducials()
            if fiducials.items:
                return fiducials
        raise ValueError(
            "No fiducials available: provide a fiducials file or a coordsystem.json "
            "config file."
        )

    @staticmethod
    def _export_stage1(
        project: Path,
        mri: MRIVolume,
        surface: ScalpSurface,
        mesh: ScalpMesh,
        fiducials: Fiducials,
    ) -> None:
        """Persist the Stage 1 artifacts: mask, mesh and fiducials."""
        project.mkdir(parents=True, exist_ok=True)
        export_head_mask(project / "segmentation" / "head_mask.nii.gz", surface.mask, mri)
        export_scalp_mesh(project / "mesh" / "final_mesh.ply", mesh)
        export_fiducials(project / "input" / "fiducials.json", fiducials)

    @staticmethod
    def _run_stage2(project: Path, ese: EseOptions, mesh: ScalpMesh) -> ESEMesh:
        """Generate the ESE mesh and persist its Stage 2 artifact."""
        ese_mesh = generate_ese(mesh, ese)
        ese_dir = project / "ese"
        ese_dir.mkdir(parents=True, exist_ok=True)
        export_ese_mesh(ese_dir / "ese_mesh.ply", ese_mesh)
        return ese_mesh

    def poll(self, append: Callable[[str], None]) -> None:
        """Main thread: drain the log queue and turn sentinels into signals.

        Plain log lines are handed to ``append``, which writes them to the
        visible log widget.  The whole queue is drained per call so a sentinel
        from one producer (e.g. HTML export finishing while the pipeline still
        runs) no longer cuts off the other producer's lines for this tick.
        """
        try:
            while True:
                msg = self._state.log_queue.get_nowait()
                if msg == _DONE_SENTINEL:
                    self.finished.emit()
                elif msg == _ERROR_SENTINEL:
                    self.failed.emit()
                elif msg == _EXPORT_DONE_SENTINEL:
                    self.exportDone.emit()
                elif msg == _EXPORT_ERROR_SENTINEL:
                    self.exportFailed.emit()
                else:
                    append(msg)
        except queue.Empty:
            pass