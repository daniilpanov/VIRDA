from logging import Logger
from pathlib import Path
from typing import Self, cast

from virda.fiducials import AutoFiducialsDetector
from virda.io.loader import MRILoader
from virda.io.loader.manual_fiducials_loader import ManualFiducialsLoader
from virda.io.loader.nifti_loader import NiftiLoader
from virda.io.providers.logging_provider import StoreLoggingProvider
from virda.io.providers.mesh_versioning_provider import ScalpMeshVersioningProvider
from virda.io.providers.stage1_exporter import Stage1Exporter
from virda.mesh import MeshExtractor, MeshPostprocessor
from virda.mesh.mesh_extractor import MarchingCubesExtractor
from virda.models.config import Config
from virda.models.ese_config import ESEConfig
from virda.models.fiducial import AutoDetectedFiducials, Fiducials, ManualFiducials
from virda.models.mri_volume import MRIVolume
from virda.models.path import FiducialsPath, NiftiPath
from virda.models.scalp_mesh import ScalpMesh
from virda.models.segmentation_mask import SegmentationMask
from virda.models.stage1_result import Stage1Result
from virda.pipeline import PipelineController
from virda.pipeline_context import PipelineContext
from virda.pipelines.mesh_generate import SMOOTHER_TYPES, MeshPipeline, MeshPipelineContract
from virda.segmentation import HeadSegmenter, SegmentationMaskPostprocessor
from virda.segmentation.head_segmenter import OtsuHeadSegmenter

from .helpers import get_stage_logger


class MeshStage1Step:
    """Run one mesh stage-1 function (e.g. ``clean``, ``smooth``) inside the
    stage controller, so the resulting mesh store is versioned as usual."""

    def __init__(self, pipeline: MeshPipeline, function_name: str) -> None:
        self._pipeline = pipeline
        self._function_name = function_name

    def run(self, context: PipelineContext) -> ScalpMesh:
        result = cast(ScalpMesh, self._pipeline.run_stage1(self._function_name, context))
        context.stores[ScalpMesh] = result
        return result


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


class Stage1PipelineBuilder:
    def __init__(
        self,
        nifti_path: str | Path,
        mri_loader: MRILoader,
        segmenter: HeadSegmenter,
        extractor: MeshExtractor,
        project_dir: Path | None = None,
        logger: Logger | None = None,
        fiducials_path: Path | str | None = None,
        auto_detect_fiducials: bool = False,
        ese_config: ESEConfig | None = None,
        config: Config | None = None,
    ) -> None:
        self._nifti_path: Path = Path(nifti_path)
        self._loader: MRILoader = mri_loader
        self._segmenter: HeadSegmenter = segmenter
        self._mesh_extractor: MeshExtractor = extractor
        self._mask_postprocessors: list[SegmentationMaskPostprocessor] = []
        self._mesh_postprocessors: list[MeshPostprocessor] = []
        self._project_dir: Path | None = project_dir
        self._logger: Logger | None = logger
        self._fiducials_path: Path | None = Path(fiducials_path) if fiducials_path else None
        self._auto_detect_fiducials: bool = auto_detect_fiducials
        self._ese_config: ESEConfig | None = ese_config
        self._config: Config = config or Config()
        self._atomic_mesh: MeshPipeline | None = None

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Build a Stage 1 pipeline configured from the merged ``config``.

        The mesh generation part is delegated to the atomic mesh pipeline
        (:class:`~virda.pipelines.mesh_generate.MeshPipeline`); cleaning and
        smoothing run as its stage-1 functions, preserving the previous default
        behaviour while keeping them separately invocable.
        """

        resolved_nifti_path = config.nifti_path
        if resolved_nifti_path is None:
            raise ValueError(
                "NIfTI path not provided. "
                "Pass it as an argument, set the NIFTI_PATH environment variable,"
                " or add it to an input config file."
            )
        nifti_path_inst = Path(resolved_nifti_path)

        resolved_project_dir = config.project_dir
        if resolved_project_dir is None:
            raise ValueError(
                "Project directory path not provided. "
                "Pass it as an argument, set the PROJECT_DIR environment variable,"
                " or add it to an input config file."
            )
        project_dir_path_inst = Path(resolved_project_dir)

        resolved_fiducials_path = config.fiducials_path
        fiducials_path_inst = Path(resolved_fiducials_path) if resolved_fiducials_path else None

        logger = get_stage_logger(project_dir_path_inst, "stage_1")

        contract = MeshPipelineContract(
            nifti_path=nifti_path_inst,
            closing_radius=config.closing_radius,
            otsu_scope=config.otsu_scope,
            otsu_threshold_scale=config.otsu_threshold_scale,
            seal_enabled=config.seal_enabled,
            seal_radius=config.seal_radius,
            voxel_size_mm=config.mesh_voxel_size_mm,
            mesh_density_percent=config.mesh_density_percent,
            cleaner_min_vertices=config.cleaner_min_vertices,
            cleaner_merge_digits=config.cleaner_merge_digits,
            smoother_type=cast(SMOOTHER_TYPES, config.smoother_type),
            smoother_iterations=config.smoother_iterations,
            smoother_lamb=config.smoother_lamb,
            smoother_nu=config.smoother_nu,
        )

        builder = cls(
            nifti_path=nifti_path_inst,
            mri_loader=NiftiLoader(),
            segmenter=OtsuHeadSegmenter(
                closing_radius=config.closing_radius,
                otsu_scope=config.otsu_scope,
                threshold_scale=config.otsu_threshold_scale,
            ),
            extractor=MarchingCubesExtractor(),
            project_dir=project_dir_path_inst,
            logger=logger,
            fiducials_path=fiducials_path_inst,
            auto_detect_fiducials=config.auto_detect_fiducials,
            ese_config=config.to_ese_config(),
            config=config,
        )
        builder._atomic_mesh = MeshPipeline(contract=contract, logger=logger)
        return builder

    def setup_mask_postprocessors(
        self, postprocessors: list[SegmentationMaskPostprocessor]
    ) -> Self:
        """Add mask postprocessors (e.g. sealing) between segmentation and extraction."""
        self._mask_postprocessors.extend(postprocessors)
        return self

    def setup_mesh_postprocessors(self, postprocessors: list[MeshPostprocessor]) -> Self:
        """Add mesh cleaners, smoothers, etc."""
        self._mesh_postprocessors.extend(postprocessors)
        return self

    def build(self) -> PipelineController:
        if self._atomic_mesh is not None:
            if self._mask_postprocessors or self._mesh_postprocessors:
                raise ValueError(
                    "Custom postprocessors cannot be combined with an atomic mesh pipeline: "
                    "mesh cleaning and smoothing are governed by the pipeline contract options."
                )
            return self._build_atomic()
        return self._build_manual()

    def _build_atomic(self) -> PipelineController:
        assert self._atomic_mesh is not None
        controller = self._atomic_mesh.build_stage0()

        for function_name in ("clean", "smooth", "decimate"):
            controller.register_step(MeshStage1Step(self._atomic_mesh, function_name))

        if self._fiducials_path:
            controller.register_store(FiducialsPath, FiducialsPath(self._fiducials_path))
            controller.register_step(ManualFiducialsLoader())

        if self._auto_detect_fiducials:
            controller.register_step(AutoFiducialsDetector())

        controller.register_step(FiducialsRegistrationStep())
        controller.register_step(OutputGenerator())

        controller.register_store(Config, self._config)

        log_provider = StoreLoggingProvider()
        for store_type in (MRIVolume, SegmentationMask, ScalpMesh, Stage1Result):
            controller.register_provider(log_provider, on_store=store_type)

        if self._project_dir:
            controller.register_provider(
                Stage1Exporter(
                    project_dir=self._project_dir,
                    ese_config=self._ese_config,
                    config=self._config,
                    nifti_path=self._nifti_path,
                ),
                Stage1Result,
            )

            controller.register_provider(
                ScalpMeshVersioningProvider(self._project_dir / "mesh" / "versions"),
                on_store=ScalpMesh,
            )

        return controller

    def _build_manual(self) -> PipelineController:
        controller = PipelineController(logger=self._logger)

        # -- Steps --
        controller.register_step(self._loader)
        controller.register_step(self._segmenter)

        for mask_postprocessor in self._mask_postprocessors:
            controller.register_step(mask_postprocessor)

        controller.register_step(self._mesh_extractor)

        for mesh_postprocessor in self._mesh_postprocessors:
            controller.register_step(mesh_postprocessor)

        if self._fiducials_path:
            controller.register_store(FiducialsPath, FiducialsPath(self._fiducials_path))
            controller.register_step(ManualFiducialsLoader())

        if self._auto_detect_fiducials:
            controller.register_step(AutoFiducialsDetector())

        controller.register_step(FiducialsRegistrationStep())
        controller.register_step(OutputGenerator())

        # -- Stores --
        controller.register_store(NiftiPath, NiftiPath(self._nifti_path))
        controller.register_store(MRIVolume)
        controller.register_store(SegmentationMask)
        controller.register_store(ScalpMesh)
        controller.register_store(Fiducials)
        controller.register_store(Config, self._config)

        # -- Providers --
        log_provider = StoreLoggingProvider()
        for store_type in (MRIVolume, SegmentationMask, ScalpMesh, Stage1Result):
            controller.register_provider(log_provider, on_store=store_type)

        if self._project_dir:
            controller.register_provider(
                Stage1Exporter(
                    project_dir=self._project_dir,
                    ese_config=self._ese_config,
                    config=self._config,
                    nifti_path=self._nifti_path,
                ),
                Stage1Result,
            )

            controller.register_provider(
                ScalpMeshVersioningProvider(self._project_dir / "mesh" / "versions"),
                on_store=ScalpMesh,
            )

        return controller

