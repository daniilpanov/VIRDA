"""Background pipeline execution and log forwarding.

The GUI never runs the heavy pipeline work on its own thread;
:class:`PipelineRunner` spawns a daemon thread, streams human-readable
progress plus sentinel markers over the shared :class:`queue.Queue`, and the
main thread turns those sentinels into Qt signals via :meth:`PipelineRunner.poll`.
The pipeline itself is orchestrated here from the pure ``virda.ops`` atoms and
``virda.io`` importers/exporters.
"""

import json
import logging
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
from PySide6.QtCore import QObject, Signal

from virda.io.exporters import (
    export_electrodes,
    export_ese_mesh,
    export_fiducials,
    export_head_mask,
    export_scalp_mesh,
)
from virda.io.importers import import_fiducials, import_measurements, import_nifti
from virda.models.config import Config
from virda.models.electrode import Electrodes
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
    SmootherKind,
)
from virda.ops.scalp_surface import ScalpSurface
from virda_gui.state import AppState

_DONE_SENTINEL = "__DONE__"
_ERROR_SENTINEL = "__ERROR__"
_EXPORT_DONE_SENTINEL = "__EXPORT_DONE__"
_EXPORT_ERROR_SENTINEL = "__EXPORT_ERROR__"


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

    def submit(self, config: Config, measurements_path: str | None) -> None:
        """Start the pipeline on a background thread (daemon)."""
        if self._closed:
            return
        thread = threading.Thread(target=self._run, args=(config, measurements_path), daemon=True)
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

    def _run(self, config: Config, measurements_path: str | None) -> None:
        """Background thread: run the pure-ops pipeline and post results to the queue."""
        try:
            self._state.log_queue.put("Building configuration...")
            project = Path(config.project_dir)
            mri = import_nifti(config.nifti_path)

            surface = generate_scalp_surface(
                mri,
                SealingOptions(seal_enabled=config.seal_enabled, seal_radius=config.seal_radius),
            )
            mesh = clean(
                surface.mesh,
                CleanOptions(
                    min_component_vertices=config.cleaner_min_vertices,
                    merge_digits=config.cleaner_merge_digits,
                ),
            )
            mesh = smooth(
                mesh,
                SmoothOptions(
                    smoother=cast(SmootherKind, config.smoother_type),
                    iterations=config.smoother_iterations,
                    lamb=config.smoother_lamb,
                    nu=config.smoother_nu,
                ),
            )
            if config.mesh_density_percent < 100.0:
                mesh = decimate(
                    mesh, DecimateOptions(density_percent=config.mesh_density_percent)
                )

            fiducials = self._resolve_fiducials(config)
            self._export_stage1(project, mri, surface, mesh, fiducials, config)

            msg = f"Stage 1: mesh with {len(mesh.vertices)} vertices"
            self._state.log_queue.put(msg)

            ese_mesh: ESEMesh | None = (
                self._run_stage2(project, config, mesh)
                if config.ese_offset_mm is not None
                else None
            )

            electrodes: Electrodes | None = None
            if ese_mesh is not None and measurements_path:
                electrodes = localize(
                    surface=ese_mesh,
                    fiducials=fiducials,
                    electrodes=import_measurements(measurements_path),
                    options=LocalizeOptions(
                        calibrate_ese_offset=config.calibrate_ese_offset,
                        residual_threshold_mm=config.residual_threshold_mm,
                    ),
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
    def _resolve_fiducials(config: Config) -> Fiducials:
        """Pick the fiducials for Stage 3 from the loaded inputs."""
        if config.auto_detect_fiducials:
            raise ValueError(
                "Auto fiducial detection is not available in the pure library; "
                "provide a fiducials file or a coordsystem.json config file."
            )
        if config.fiducials_path:
            return import_fiducials(config.fiducials_path)
        if config.coordsystem is not None:
            fiducials = config.coordsystem.to_fiducials()
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
        config: Config,
    ) -> None:
        """Persist the Stage 1 artifacts: mesh, mask, fiducials and config."""
        project.mkdir(parents=True, exist_ok=True)
        export_head_mask(project / "segmentation" / "head_mask.nii.gz", surface.mask, mri)
        export_scalp_mesh(project / "mesh" / "final_mesh.ply", mesh)
        np.save(project / "mesh" / "scalp_vertices.npy", mesh.vertices)
        np.save(project / "mesh" / "scalp_faces.npy", mesh.faces)
        np.save(project / "mesh" / "scalp_face_adjacency.npy", mesh.face_adjacency)
        export_fiducials(project / "input" / "fiducials.json", fiducials)
        (project / "input" / "pipeline_config.json").write_text(
            json.dumps(config.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _run_stage2(project: Path, config: Config, mesh: ScalpMesh) -> ESEMesh:
        """Generate the ESE mesh and persist its Stage 2 artifacts."""
        ese = generate_ese(
            mesh,
            EseOptions(
                ese_offset_mm=config.ese_offset_mm,  # type: ignore[arg-type]
                neighborhood_radius_mm=config.neighborhood_radius_mm,
                k_neighbors=config.k_neighbors,
                use_weighted_pca=config.use_weighted_pca,
                pca_sigma_mm=config.pca_sigma_mm,
                min_neighbors=config.min_neighbors,
            ),
        )
        ese_dir = project / "ese"
        ese_dir.mkdir(parents=True, exist_ok=True)
        export_ese_mesh(ese_dir / "ese_mesh.ply", ese)
        np.save(ese_dir / "ese_vertices.npy", ese.vertices)
        np.save(ese_dir / "ese_faces.npy", ese.faces)
        np.save(ese_dir / "normals.npy", ese.normals)
        np.save(ese_dir / "quality.npy", ese.quality)
        return ese

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