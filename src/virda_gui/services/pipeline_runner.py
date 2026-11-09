"""Build and run the VIRDA pipeline for the GUI.

The stage orchestrators (:func:`run_stage1`, :func:`run_stage2`,
:func:`run_stage3`) compose the plain ``virda.ops`` step functions and the
``virda.io.exporters`` writers with explicit arguments and return values —
there is no pipeline context or store registry anymore.  :class:`PipelineRunner`
runs them off the GUI thread and reports completion/failure through Qt signals.
"""

import logging
import threading
import time
from dataclasses import dataclass
from logging import Logger
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from virda.config import resolve_stage3_config
from virda.io.exporters.electrodes import (
    export_electrodes,
    export_electrodes_ese,
    export_electrodes_scalp,
)
from virda.io.exporters.ese_mesh import export_ese_companions, export_ese_mesh
from virda.io.exporters.fiducials import export_fiducials
from virda.io.exporters.mesh_arrays import export_mesh_faces, export_mesh_vertices
from virda.io.exporters.nifti_copy import export_nifti_copy
from virda.io.exporters.pipeline_config import export_pipeline_config
from virda.io.exporters.scalp_mesh import export_scalp_mesh
from virda.io.exporters.segmentation import export_head_mask
from virda.io.importers.fiducials import load_manual_fiducials
from virda.io.importers.measurements import load_measurements
from virda.io.importers.nifti import import_nifti
from virda.models.config import Config
from virda.models.coordsystem import Coordsystem
from virda.models.electrode import Electrodes
from virda.models.fiducial import (
    Fiducials,
    ManualFiducials,
)
from virda.models.mesh import Mesh, Normals
from virda.ops.build_ese import build_ese
from virda.ops.clean_mesh import clean_mesh
from virda.ops.extract_mesh import extract_mesh
from virda.ops.localize_electrodes import localize_electrodes
from virda.ops.seal_mask import seal_mask
from virda.ops.segment_head import segment_head
from virda.ops.smooth_mesh import smooth_mesh
from virda_gui.state import AppState


@dataclass(frozen=True)
class _Stage1Result:
    """GUI-side view of Stage 1: the scalp mesh and the resolved fiducials."""

    mesh: Mesh
    fiducials: Fiducials


def _resolve_stage1_fiducials(
    manual: ManualFiducials | None,
    coordsystem: Coordsystem | None,
) -> Fiducials:
    """Prefer manual fiducials, then coordsystem defaults."""
    for candidate in (
        manual.fiducials if manual is not None else None,
        coordsystem.to_fiducials() if coordsystem is not None else None,
    ):
        if candidate is not None and candidate.items:
            return candidate
    raise ValueError(
        "No fiducials available: provide a manual fiducials file"
        " or supply a coordsystem.json config file"
    )


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
) -> tuple[_Stage1Result, Mesh | None, Electrodes | None]:
    """Run the VIRDA pipeline: Stage 1 -> 2 -> 3.

    Returns the Stage 1 result, the ESE surface when ESE is configured,
    and the localized electrodes when both ESE and measurements are available.
    """
    stage1_result = run_stage1(config)
    if config.to_ese_config() is None:
        return stage1_result, None, None

    ese_mesh, normals = run_stage2(config, stage1_result.mesh)
    electrodes = run_stage3(config, stage1_result, ese_mesh, normals, measurements_path)
    return stage1_result, ese_mesh, electrodes


def run_stage1(config: Config) -> _Stage1Result:
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

    resolved_fiducials_path = config.fiducials_path
    nifti_path = Path(resolved_nifti_path)
    project_dir = Path(resolved_project_dir)

    logger = logging.getLogger("virda_gui.stage_1")

    volume = import_nifti(nifti_path)
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
    export_head_mask(project_dir / "segmentation" / "head_mask.nii.gz", mask, volume)

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

    fiducials = _resolve_stage1_fiducials(manual, config.coordsystem)

    export_scalp_mesh(project_dir / "mesh" / "final_mesh.ply", mesh)
    export_mesh_vertices(project_dir / "mesh" / "scalp_vertices.npy", mesh.vertices)
    export_mesh_faces(project_dir / "mesh" / "scalp_faces.npy", mesh.faces)
    export_fiducials(project_dir / "input" / "fiducials.json", fiducials)
    export_pipeline_config(project_dir / "input" / "pipeline_config.json", config)
    export_nifti_copy(project_dir / "input" / nifti_path.name, nifti_path, logger=logger)

    result = _Stage1Result(mesh=mesh, fiducials=fiducials)
    _log_artifact(logger, result)

    return result


def run_stage2(config: Config, scalp_mesh: Mesh) -> tuple[Mesh, Normals]:
    """Build the ESE surface and export it (Stage 2)."""
    logger = logging.getLogger("virda_gui.stage_2")

    ese_mesh, normals = build_ese(
        scalp_mesh,
        config=config.to_stage2_config(),
        ese_offset_mm=config.to_ese_config().ese_offset_mm,
        logger=logger,
    )
    _log_artifact(logger, ese_mesh)

    ese_mesh_path = Path(config.project_dir) / "ese" / "ese_mesh.ply"
    export_ese_mesh(ese_mesh_path, ese_mesh)
    export_ese_companions(ese_mesh_path, ese_mesh)

    return ese_mesh, normals


def run_stage3(
    config: Config,
    stage1_result: _Stage1Result,
    ese_mesh: Mesh | None,
    normals: Normals | None = None,
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

    stage3_config = resolve_stage3_config(config)

    electrodes, fiducials = load_measurements(
        Path(resolved_measurements_path),
        stage1_result.fiducials,
    )

    logger = logging.getLogger("virda_gui.stage_3")
    electrodes = localize_electrodes(
        ese_mesh,
        normals,
        stage1_result.mesh,
        fiducials,
        electrodes,
        config=stage3_config,
        logger=logger,
    )
    _log_artifact(logger, electrodes)

    localization_dir = Path(project_dir) / "localization"
    export_electrodes(localization_dir / "electrodes.csv", electrodes)
    export_electrodes_scalp(localization_dir / "electrodes_scalp.csv", electrodes)
    export_electrodes_ese(localization_dir / "electrodes_ese.csv", electrodes)

    return electrodes


class PipelineRunner(QObject):
    """Run the virda pipeline off the GUI thread.

    Lives on the main thread so its signals dispatch through the Qt event
    loop; only :meth:`_run` touches background threads.
    """

    finished = Signal()
    failed = Signal()

    def __init__(self, state: AppState) -> None:
        super().__init__()
        self._state = state
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
        nearly finished thread) instead of tearing the app down mid-run.
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
        """Background thread: run the pipeline and report the outcome."""
        try:
            stage1_result, ese_mesh, electrodes = run(config, measurements_path)

            if electrodes is not None:
                items = electrodes.items
                localized = sum(1 for e in items if e.is_localized)
                flagged = sum(1 for e in items if e.flagged)
                shift = electrodes.calibrated_offset_shift_mm
                self._state.stage3_summary = {
                    "total": len(items),
                    "localized": localized,
                    "flagged": flagged,
                    "offset_shift_mm": shift,
                }
            self.finished.emit()

        except Exception:
            self.failed.emit()
