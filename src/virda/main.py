"""Pipeline orchestration for VIRDA.

Keeps the reusable pipeline entry points (:func:`run`, :func:`run_stage3`)
that the GUI imports as ``from virda.main import run``.
"""

from pathlib import Path

from virda.config import resolve_stage3_config
from virda.models.config import Config
from virda.models.electrode import Electrodes
from virda.models.ese_mesh import ESEMesh
from virda.models.stage1_result import Stage1Result
from virda.pipelines.stage1 import Stage1PipelineBuilder
from virda.pipelines.stage2 import Stage2PipelineBuilder

__all__ = ["run", "run_stage3"]


def run(
    config: Config, measurements_path: str | Path | None = None
) -> tuple[Stage1Result, ESEMesh | None, Electrodes | None]:
    """Run the VIRDA pipeline: Stage 1 -> 2 -> 3.

    Returns the Stage 1 result, the ESE surface when ESE is configured,
    and the localized electrodes when both ESE and measurements are available.
    """
    stage1_result = (
        Stage1PipelineBuilder.from_config(config=config)
        .build()
        .run()
        .get_store_notnull(Stage1Result)
    )

    if config.to_ese_config() is None:
        return stage1_result, None, None

    ese_mesh = (
        Stage2PipelineBuilder.from_config(config=config, scalp_mesh=stage1_result.mesh)
        .build()
        .run()
        .get_store_notnull(ESEMesh)
    )
    electrodes = run_stage3(config, stage1_result, ese_mesh, measurements_path)
    return stage1_result, ese_mesh, electrodes


def run_stage3(
    config: Config,
    stage1_result: Stage1Result,
    ese_mesh: ESEMesh | None,
    measurements_path: str | Path | None = None,
) -> Electrodes | None:
    """Localize electrodes (Stage 3) from precomputed Stage 1/2 results.

    Allows callers that already have the scalp mesh and ESE surface to run
    localization without repeating Stages 1-2.
    """
    from virda.io.loader.measurements_loader import MeasurementsLoaderFromJson
    from virda.localization.brute_force_localizer import BruteForceLocalizer
    from virda.models.fiducial import Fiducials
    from virda.models.path import MeasurementsPath
    from virda.pipeline_context import PipelineContext
    from virda.pipelines.helpers import get_stage_logger
    from virda.pipelines.stage3 import Stage3PipelineBuilder

    if ese_mesh is None:
        return None

    resolved_measurements_path = measurements_path
    if resolved_measurements_path is None:
        return None

    project_dir = config.project_dir
    if project_dir is None:
        raise ValueError(
            "Project directory path not provided. "
            "Pass it as an argument or set the PROJECT_DIR environment variable."
        )
    project = Path(project_dir)

    stage3_config = resolve_stage3_config(config)

    load_context = PipelineContext({})
    load_context.stores[MeasurementsPath] = MeasurementsPath(Path(resolved_measurements_path))
    load_context.stores[Fiducials] = stage1_result.fiducials
    electrodes = MeasurementsLoaderFromJson().run(load_context)
    fiducials = load_context.get_store_notnull(Fiducials)

    stage3_pipeline = Stage3PipelineBuilder(
        localizer=BruteForceLocalizer(
            calibrate_ese_offset=stage3_config.calibrate_ese_offset,
            residual_threshold_mm=stage3_config.residual_threshold_mm,
        ),
        stage3_config=stage3_config,
        ese_mesh=ese_mesh,
        electrodes=electrodes,
        fiducials=fiducials,
        project_dir=project,
        logger=get_stage_logger(project, "stage_3"),
    ).build()
    context = stage3_pipeline.run()
    return context.get_store_notnull(Electrodes)
