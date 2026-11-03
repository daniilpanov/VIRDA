from pathlib import Path

from virda.config import resolve_stage3_config
from virda.io.providers.logging_provider import StoreLoggingProvider
from virda.io.providers.stage1_exporter import Stage1Exporter
from virda.io.providers.stage2_exporter import Stage2Exporter
from virda.io.providers.stage3_exporter import Stage3Exporter
from virda.localization.brute_force_localizer import BruteForceLocalizer
from virda.logging_setup import get_stage_logger
from virda.models.config import Config
from virda.models.electrode import Electrodes
from virda.models.ese_mesh import ESEMesh
from virda.models.fiducial import Fiducials, AutoDetectedFiducials, ManualFiducials
from virda.models.mri_volume import MRIVolume
from virda.models.scalp_mesh import ScalpMesh
from virda.models.segmentation_mask import SegmentationMask
from virda.models.stage1_result import Stage1Result
from virda.ops.auto_detect_fiducials import auto_detect_fiducials
from virda.ops.build_ese import build_ese
from virda.ops.clean_mesh import clean_mesh
from virda.ops.extract_mesh import extract_mesh
from virda.ops.load_manual_fiducials import load_manual_fiducials
from virda.ops.load_measurements import load_measurements
from virda.ops.load_nifti import load_nifti
from virda.ops.resolve_fiducials import resolve_fiducials
from virda.ops.seal_mask import seal_mask
from virda.ops.segment_head import segment_head
from virda.ops.smooth_mesh import smooth_mesh
from virda.ops.stage1_result import build_stage1_result
from virda.pipeline import Provider, Step
from virda.pipeline_context import PipelineContext


def _run_step(
        context: PipelineContext,
        step: Step,
        *providers: Provider,
) -> None:
    """Run one step, put its returned store into the context, fire providers.

    This replaces the ``PipelineController`` bookkeeping: a step's return
    value is stored under its runtime type, and every provider bound to it
    (logging, export) runs right after the store is updated.
    """
    new_store = step.run(context)
    context.stores[type(new_store)] = new_store
    for provider in providers:
        provider.provide(new_store, context)


def _put_store(
        context: PipelineContext,
        store,
        *providers: Provider,
) -> None:
    """Put a function result into the context stores and fire providers.

    Same contract as :func:`_run_step` for steps migrated to plain functions.
    """
    context.stores[type(store)] = store
    for provider in providers:
        provider.provide(store, context)


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

    logger = get_stage_logger(project_dir, "stage_1")
    context = PipelineContext({})
    context.logger = logger

    # -- Stores --
    nifti_path = Path(resolved_nifti_path)
    context.stores[Config] = config
    for store_type in (MRIVolume, SegmentationMask, ScalpMesh, Fiducials):
        context.stores[store_type] = None

    # -- Providers --
    log_provider = StoreLoggingProvider()
    ese_config = config.to_ese_config()
    stage1_exporter = Stage1Exporter(
        project_dir=project_dir,
        ese_config=ese_config,
        config=config,
        nifti_path=nifti_path,
    )

    # -- Steps --
    volume = load_nifti(nifti_path)
    _put_store(context, volume, log_provider)

    mask = segment_head(
        volume,
        closing_radius=config.closing_radius,
        otsu_scope=config.otsu_scope,
        threshold_scale=config.otsu_threshold_scale,
    )
    _put_store(context, mask, log_provider)

    if config.seal_enabled:
        mask = seal_mask(mask, radius=config.seal_radius, logger=logger)
        _put_store(context, mask, log_provider)

    mesh = extract_mesh(mask, volume)
    _put_store(context, mesh, log_provider)

    mesh = clean_mesh(
        mesh,
        min_component_vertices=config.cleaner_min_vertices,
        merge_digits=config.cleaner_merge_digits,
    )
    _put_store(context, mesh, log_provider)
    _put_store(
        context,
        smooth_mesh(
            mesh,
            smoother_type=config.smoother_type,
            iterations=config.smoother_iterations,
            lamb=config.smoother_lamb,
            nu=config.smoother_nu,
        ),
        log_provider,
    )

    if resolved_fiducials_path:
        _put_store(context, load_manual_fiducials(Path(resolved_fiducials_path)))

    if config.auto_detect_fiducials:
        _put_store(context, auto_detect_fiducials(mesh))

    fiducials = resolve_fiducials(
        context.get_store(ManualFiducials),
        context.get_store(AutoDetectedFiducials),
        config.coordsystem,
    )
    _put_store(context, fiducials)

    result = build_stage1_result(
        mri_volume=context.get_store_notnull(MRIVolume),
        segmentation_mask=context.get_store_notnull(SegmentationMask),
        mesh=context.get_store_notnull(ScalpMesh),
        fiducials=fiducials,
    )
    _put_store(context, result, log_provider, stage1_exporter)

    return result


def run_stage2(config: Config, scalp_mesh: ScalpMesh) -> ESEMesh:
    logger = get_stage_logger(Path(config.project_dir), "stage_2")
    context = PipelineContext({})
    context.logger = logger
    context.stores[ScalpMesh] = scalp_mesh

    ese_mesh = build_ese(
        scalp_mesh,
        config=config.to_stage2_config(),
        ese_offset_mm=config.to_ese_config().ese_offset_mm,
        logger=logger,
    )
    _put_store(
        context,
        ese_mesh,
        StoreLoggingProvider(),
        Stage2Exporter(project_dir=Path(config.project_dir)),
    )
    return context.get_store_notnull(ESEMesh)


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

    context = PipelineContext({})
    context.logger = get_stage_logger(project, "stage_3")
    context.stores[ESEMesh] = ese_mesh
    context.stores[Fiducials] = fiducials
    _run_step(
        context,
        BruteForceLocalizer(stage3_config),
        StoreLoggingProvider(),
        Stage3Exporter(
            project_dir=project,
            stage3_config=stage3_config,
        ),
    )
    return context.get_store_notnull(Electrodes)
