"""Build and run the VIRDA pipeline for the GUI.

The stage orchestrators (:func:`run_stage1`, :func:`run_stage2`,
:func:`run_stage3`) compose the plain ``virda.ops`` step functions and the
``virda.io.exporters`` writers with explicit arguments and return values —
there is no pipeline context or store registry anymore.  :class:`PipelineRunner`
runs them off the GUI thread and streams human-readable progress plus sentinel
markers over the shared :class:`queue.Queue`; the main thread turns those
sentinels into Qt signals via :meth:`PipelineRunner.poll`.
"""

import logging
import queue
import threading
import time
from collections.abc import Callable
from logging import Logger
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from virda.config import resolve_stage3_config
from virda.io.exporters.ese import export_ese
from virda.io.exporters.localization import export_localization
from virda.io.exporters.stage1 import export_stage1
from virda.io.importers.fiducials import load_manual_fiducials
from virda.io.importers.nifti import load_nifti
from virda.logging_setup import get_stage_logger
from virda.models.config import Config
from virda.models.electrode import Electrodes
from virda.models.ese_mesh import ESEMesh
from virda.models.scalp_mesh import ScalpMesh
from virda.models.stage1_result import Stage1Result
from virda.ops.auto_detect_fiducials import auto_detect_fiducials
from virda.ops.build_ese import build_ese
from virda.ops.clean_mesh import clean_mesh
from virda.ops.extract_mesh import extract_mesh
from virda.ops.load_measurements import load_measurements
from virda.ops.localize_electrodes import localize_electrodes
from virda.ops.resolve_fiducials import resolve_fiducials
from virda.ops.seal_mask import seal_mask
from virda.ops.segment_head import segment_head
from virda.ops.smooth_mesh import smooth_mesh
from virda.ops.stage1_result import build_stage1_result
from virda_gui.state import AppState

_DONE_SENTINEL = "__DONE__"
_ERROR_SENTINEL = "__ERROR__"

_ARTIFACT_SIZE_ATTRS = ("shape", "vertices", "locations", "items")


def _artifact_size_hint(artifact) -> str:
    for attr in _ARTIFACT_SIZE_ATTRS:
        value = getattr(artifact, attr, None)
        if value is not None:
            try:
                return f" {attr}={len(value)}"
            except TypeError:
                return f" {attr}={value}"
    return ""


def _log_artifact(logger: Logger | None, artifact) -> None:
    """Report a produced pipeline artifact through the stage logger."""
    if logger is None:
        return
    logger.info("Artifact '%s' ready.%s", type(artifact).__name__, _artifact_size_hint(artifact))


def run(
    config: Config, measurements_path: str | Path | None = None
) -> tuple[Stage1Result, ESEMesh | None, Electrodes | None]:
    """Run the VIRDA pipeline: Stage 1 -> 2 -> 3.

    Returns the Stage 1 result, the ESE surface when ESE is configured,
    and the localized electrodes when both ESE and measurements are available.
    """
    stage1_result = run_stage1(config)
    if config.to_ese_config() is None:
        return stage1_result, None, None

    ese_mesh = run_stage2(config, stage1_result.mesh)
    electrodes = run_stage3(config, stage1_result, ese_mesh, measurements_path)
    return stage1_result, ese_mesh, electrodes


def run_stage1(config: Config) -> Stage1Result:
    """Segment the head, build the scalp mesh and resolve fiducials (Stage 1)."""
    resolved_nifti_path = config.nifti_path
    if resolved_nifti_path is None:
        raise ValueError(
            "NIfTI path not provided. "
            "Pass it as an argument, set the NIFTI_PATH environment variable,"
            " or add it to an input config file."
        )

    resolved_project_dir = config.project_dir
    if resolved_project_dir is None:
        raise ValueError(
            "Project directory path not provided. "
            "Pass it as an argument, set the PROJECT_DIR environment variable,"
            " or add it to an input config file."
        )

    project_dir = Path(resolved_project_dir)
    resolved_fiducials_path = config.fiducials_path
    nifti_path = Path(resolved_nifti_path)

    logger = get_stage_logger(project_dir, "stage_1")

    volume = load_nifti(nifti_path)
    _log_artifact(logger, volume)

    mask = segment_head(
        volume,
        closing_radius=config.closing_radius,
        otsu_scope=config.otsu_scope,
        threshold_scale=config.otsu_threshold_scale,
    )
    _log_artifact(logger, mask)

    if config.seal_enabled:
        mask = seal_mask(mask, radius=config.seal_radius, logger=logger)
        _log_artifact(logger, mask)

    mesh = extract_mesh(mask, volume)
    _log_artifact(logger, mesh)

    mesh = clean_mesh(
        mesh,
        min_component_vertices=config.cleaner_min_vertices,
        merge_digits=config.cleaner_merge_digits,
    )
    _log_artifact(logger, mesh)
    mesh = smooth_mesh(
        mesh,
        smoother_type=config.smoother_type,
        iterations=config.smoother_iterations,
        lamb=config.smoother_lamb,
        nu=config.smoother_nu,
    )
    _log_artifact(logger, mesh)

    manual = (
        load_manual_fiducials(Path(resolved_fiducials_path)) if resolved_fiducials_path else None
    )
    auto = auto_detect_fiducials(mesh) if config.auto_detect_fiducials else None

    fiducials = resolve_fiducials(manual, auto, config.coordsystem)

    result = build_stage1_result(
        mri_volume=volume,
        segmentation_mask=mask,
        mesh=mesh,
        fiducials=fiducials,
    )
    _log_artifact(logger, result)
    export_stage1(
        result,
        project_dir=project_dir,
        config=config,
        nifti_path=nifti_path,
        logger=logger,
    )

    return result


def run_stage2(config: Config, scalp_mesh: ScalpMesh) -> ESEMesh:
    """Build the ESE surface and export it (Stage 2)."""
    logger = get_stage_logger(Path(config.project_dir), "stage_2")

    ese_mesh = build_ese(
        scalp_mesh,
        config=config.to_stage2_config(),
        ese_offset_mm=config.to_ese_config().ese_offset_mm,
        logger=logger,
    )
    _log_artifact(logger, ese_mesh)
    export_ese(ese_mesh, Path(config.project_dir))
    return ese_mesh


def run_stage3(
    config: Config,
    stage1_result: Stage1Result,
    ese_mesh: ESEMesh | None,
    measurements_path: str | Path | None = None,
) -> Electrodes | None:
    """Localize electrodes from precomputed Stage 1/2 results (Stage 3).

    Allows callers that already have the scalp mesh and ESE surface to run
    localization without repeating Stages 1-2.
    """
    if ese_mesh is None:
        return None

    resolved_measurements_path = measurements_path
    if resolved_measurements_path is None:
        return None

    project_dir = config.project_dir
    if project_dir is None:
        raise ValueError(
            "Project directory path not provided. "
            "Set project_dir in a config file or the PROJECT_DIR environment variable."
        )
    project = Path(project_dir)

    stage3_config = resolve_stage3_config(config)

    electrodes, fiducials = load_measurements(
        Path(resolved_measurements_path),
        stage1_result.fiducials,
    )

    logger = get_stage_logger(project, "stage_3")
    electrodes = localize_electrodes(
        ese_mesh,
        fiducials,
        electrodes,
        config=stage3_config,
        logger=logger,
    )
    _log_artifact(logger, electrodes)
    export_localization(electrodes, project)
    return electrodes


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

    def __init__(self, state: AppState) -> None:
        super().__init__()
        self._state = state
        self.log_handler = _QueueLogHandler(state.log_queue)
        self._pipeline_thread: threading.Thread | None = None
        self._closed = False

    def submit(self, config: Config, measurements_path: str | None) -> None:
        """Start the pipeline on a background thread (daemon)."""
        if self._closed:
            return
        thread = threading.Thread(target=self._run, args=(config, measurements_path), daemon=True)
        self._pipeline_thread = thread
        thread.start()

    def shutdown(self, timeout: float = 3.0) -> None:
        """Give the background thread a bounded chance to finish at teardown.

        The pipeline thread is a daemon, so the process can exit without it;
        joining for a bounded time handles the common cases (a finished or
        nearly finished thread) instead of tearing the app down while the
        worker still writes into the log queue.
        """
        deadline = time.monotonic() + timeout
        thread = self._pipeline_thread
        if thread is None or not thread.is_alive():
            self._closed = True
            return
        remaining = deadline - time.monotonic()
        if remaining > 0:
            thread.join(remaining)
        self._closed = True

    def _run(self, config: Config, measurements_path: str | None) -> None:
        """Background thread: run the pipeline and post results to the queue."""
        try:
            self._state.log_queue.put("Building configuration...")
            stage1_result, ese_mesh, electrodes = run(config, measurements_path)

            msg = f"Stage 1: mesh with {len(stage1_result.mesh.vertices)} vertices"
            self._state.log_queue.put(msg)

            if ese_mesh is not None:
                msg = f"Stage 2: ESE mesh with {len(ese_mesh.vertices)} vertices"
                self._state.log_queue.put(msg)

            if electrodes is not None:
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

    def poll(self, append: Callable[[str], None]) -> None:
        """Main thread: drain the log queue and turn sentinels into signals.

        Plain log lines are handed to ``append``, which writes them to the
        visible log widget.
        """
        try:
            while True:
                msg = self._state.log_queue.get_nowait()
                if msg == _DONE_SENTINEL:
                    self.finished.emit()
                elif msg == _ERROR_SENTINEL:
                    self.failed.emit()
                else:
                    append(msg)
        except queue.Empty:
            pass
