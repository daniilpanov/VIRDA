from pathlib import Path

from virda.config import resolve_stage3_config
from virda.ese.pca_ese_builder import PCAESEBuilder
from virda.fiducials import AutoFiducialsDetector
from virda.io.loader.manual_fiducials_loader import ManualFiducialsLoader
from virda.io.loader.measurements_loader import MeasurementsLoaderFromJson
from virda.io.loader.nifti_loader import NiftiLoader
from virda.io.providers.logging_provider import StoreLoggingProvider
from virda.io.providers.mesh_versioning_provider import ScalpMeshVersioningProvider
from virda.io.providers.stage1_exporter import Stage1Exporter
from virda.io.providers.stage2_exporter import Stage2Exporter
from virda.io.providers.stage3_exporter import Stage3Exporter
from virda.localization.brute_force_localizer import BruteForceLocalizer
from virda.logging_setup import get_stage_logger
from virda.mesh import MeshPostprocessor
from virda.mesh.laplacian_smoother import LaplacianSmoother
from virda.mesh.mesh_cleaner import TrimeshCleaner
from virda.mesh.mesh_extractor import MarchingCubesExtractor
from virda.mesh.taubin_smoother import TaubinSmoother
from virda.models.config import Config
from virda.models.electrode import Electrodes
from virda.models.ese_mesh import ESEMesh
from virda.models.fiducial import Fiducials, AutoDetectedFiducials, ManualFiducials
from virda.models.mri_volume import MRIVolume
from virda.models.path import FiducialsPath, MeasurementsPath, NiftiPath
from virda.models.scalp_mesh import ScalpMesh
from virda.models.segmentation_mask import SegmentationMask
from virda.models.stage1_result import Stage1Result
from virda.pipeline import PipelineController
from virda.pipeline_context import PipelineContext
from virda.segmentation.head_segmenter import OtsuHeadSegmenter
from virda.segmentation.seal import MaskSealer


class FiducialsRegistrationStep:
    def run(self, context: PipelineContext) -> Fiducials:
        manual = context.get_store(ManualFiducials)
        if manual is not None:
            return manual.fiducials

        auto = context.get_store(AutoDetectedFiducials)
        if auto is not None:
            return auto.fiducials

        config = context.get_store(Config)
        if config is not None and config.coordsystem is not None:
            coordsystem_fiducials = config.coordsystem.to_fiducials()
            if coordsystem_fiducials.items:
                return coordsystem_fiducials

        raise ValueError(
            "No fiducials available: provide a manual fiducials file,"
            " enable auto_detect_fiducials, or supply a coordsystem.json config file"
        )


class OutputGenerator:
    def run(self, context: PipelineContext) -> Stage1Result:
        return Stage1Result(
            mri_volume=context.get_store_notnull(MRIVolume),
            segmentation_mask=context.get_store_notnull(SegmentationMask),
            mesh=context.get_store_notnull(ScalpMesh),
            fiducials=context.get_store_notnull(Fiducials),
        )


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

    smoother: MeshPostprocessor
    if config.smoother_type == "taubin":
        smoother = TaubinSmoother(
            iterations=config.smoother_iterations,
            lamb=config.smoother_lamb,
            nu=config.smoother_nu,
        )
    else:
        smoother = LaplacianSmoother(
            iterations=config.smoother_iterations,
            lamb=config.smoother_lamb,
        )

    stage1_controller = PipelineController(logger=get_stage_logger(project_dir, "stage_1"))

    # -- Steps --
    stage1_controller.register_step(NiftiLoader())
    stage1_controller.register_step(OtsuHeadSegmenter(
        closing_radius=config.closing_radius,
        otsu_scope=config.otsu_scope,
        threshold_scale=config.otsu_threshold_scale,
    ))

    if config.seal_enabled:
        stage1_controller.register_step(MaskSealer(radius=config.seal_radius))

    stage1_controller.register_step(MarchingCubesExtractor())

    stage1_controller.register_step(TrimeshCleaner(
        min_component_vertices=config.cleaner_min_vertices,
        merge_digits=config.cleaner_merge_digits,
    ))
    stage1_controller.register_step(smoother)

    if resolved_fiducials_path:
        fiducials_path = Path(resolved_fiducials_path)
        stage1_controller.register_store(FiducialsPath, FiducialsPath(fiducials_path))
        stage1_controller.register_step(ManualFiducialsLoader())

    if config.auto_detect_fiducials:
        stage1_controller.register_step(AutoFiducialsDetector())

    stage1_controller.register_step(FiducialsRegistrationStep())
    stage1_controller.register_step(OutputGenerator())

    # -- Stores --
    nifti_path = Path(resolved_nifti_path)
    stage1_controller.register_store(NiftiPath, NiftiPath(nifti_path))
    stage1_controller.register_store(MRIVolume)
    stage1_controller.register_store(SegmentationMask)
    stage1_controller.register_store(ScalpMesh)
    stage1_controller.register_store(Fiducials)
    stage1_controller.register_store(Config, config)

    # -- Providers --
    log_provider = StoreLoggingProvider()
    for store_type in (MRIVolume, SegmentationMask, ScalpMesh, Stage1Result):
        stage1_controller.register_provider(log_provider, on_store=store_type)

    ese_config = config.to_ese_config()

    if project_dir:
        stage1_controller.register_provider(
            Stage1Exporter(
                project_dir=project_dir,
                ese_config=ese_config,
                config=config,
                nifti_path=nifti_path,
            ),
            Stage1Result,
        )

        stage1_controller.register_provider(
            ScalpMeshVersioningProvider(project_dir / "mesh" / "versions"),
            on_store=ScalpMesh,
        )

    return stage1_controller.run().get_store_notnull(Stage1Result)


def run_stage2(config: Config, scalp_mesh: ScalpMesh) -> ESEMesh:
    project_dir = config.project_dir

    stage2_controller = PipelineController(logger=get_stage_logger(Path(project_dir), "stage_2"))

    stage2_controller.register_store(ScalpMesh, scalp_mesh)
    stage2_controller.register_step(PCAESEBuilder(
        config=config.to_stage2_config(),
        ese_offset_mm=config.to_ese_config().ese_offset_mm,
    ))

    log_provider = StoreLoggingProvider()
    for store_type in (ScalpMesh, ESEMesh):
        stage2_controller.register_provider(log_provider, on_store=store_type)

    stage2_controller.register_provider(
        Stage2Exporter(project_dir=Path(project_dir)),
        on_store=ESEMesh,
    )
    return stage2_controller.run().get_store_notnull(ESEMesh)


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

    load_context = PipelineContext({})
    load_context.stores[MeasurementsPath] = MeasurementsPath(Path(resolved_measurements_path))
    load_context.stores[Fiducials] = stage1_result.fiducials
    electrodes = MeasurementsLoaderFromJson().run(load_context)
    fiducials = load_context.get_store_notnull(Fiducials)

    stage3_controller = PipelineController(logger=get_stage_logger(project, "stage_3"))

    stage3_controller.register_store(ESEMesh, ese_mesh)
    stage3_controller.register_store(Electrodes, electrodes)
    stage3_controller.register_store(Fiducials, fiducials)
    stage3_controller.register_step(BruteForceLocalizer(stage3_config))

    log_provider = StoreLoggingProvider()
    for store_type in (ESEMesh, Fiducials, Electrodes):
        stage3_controller.register_provider(log_provider, on_store=store_type)

    stage3_controller.register_provider(
        Stage3Exporter(
            project_dir=Path(project_dir),
            stage3_config=stage3_config,
        ),
        on_store=Electrodes,
    )

    context = stage3_controller.run()
    return context.get_store_notnull(Electrodes)
