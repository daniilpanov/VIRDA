"""Background pipeline execution and log forwarding.

The GUI never runs the heavy :mod:`virda.main.run` work on its own thread;
:class:`PipelineRunner` spawns a daemon thread, streams human-readable
progress plus sentinel markers over the shared :class:`queue.Queue`, and the
main thread turns those sentinels into Qt signals via :meth:`PipelineRunner.poll`.
"""

import logging
import queue
import threading
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from virda.main import run
from virda.models.config import Config
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

    def submit(self, config: Config, measurements_path: str | None) -> None:
        """Start the pipeline on a background thread (daemon)."""
        thread = threading.Thread(target=self._run, args=(config, measurements_path), daemon=True)
        thread.start()

    def start_html_export(self, project: Path, output: Path) -> None:
        """Export the HTML viewer for ``project`` on a background thread."""

        def _export() -> None:
            try:
                from .html_export import export_project

                export_project(str(project), output)
                self._state.log_queue.put(f"HTML exported: {output}")
                self._state.log_queue.put(_EXPORT_DONE_SENTINEL)
            except Exception as exc:
                self._state.log_queue.put(f"HTML export failed: {exc}")
                self._state.log_queue.put(_EXPORT_ERROR_SENTINEL)

        threading.Thread(target=_export, daemon=True).start()

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
                    break
                if msg == _ERROR_SENTINEL:
                    self.failed.emit()
                    break
                if msg == _EXPORT_DONE_SENTINEL:
                    self.exportDone.emit()
                    break
                if msg == _EXPORT_ERROR_SENTINEL:
                    self.exportFailed.emit()
                    break
                append(msg)
        except queue.Empty:
            pass
