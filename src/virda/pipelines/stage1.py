from logging import Logger
from pathlib import Path
from typing import Self

from virda.fiducials import AutoFiducialsDetector
from virda.io.loader import MRILoader
from virda.io.loader.manual_fiducials_loader import ManualFiducialsLoader
from virda.io.loader.nifti_loader import NiftiLoader
from virda.io.providers.logging_provider import StoreLoggingProvider
from virda.io.providers.mesh_versioning_provider import ScalpMeshVersioningProvider
from virda.io.providers.stage1_exporter import Stage1Exporter
from virda.mesh import MeshExtractor, MeshPostprocessor
from virda.mesh.laplacian_smoother import LaplacianSmoother
from virda.mesh.mesh_cleaner import TrimeshCleaner
from virda.mesh.mesh_extractor import MarchingCubesExtractor
from virda.mesh.taubin_smoother import TaubinSmoother
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
from virda.qc.checks import Stage1QualityControlStep
from virda.segmentation import HeadSegmenter, SegmentationMaskPostprocessor
from virda.segmentation.head_segmenter import OtsuHeadSegmenter
from virda.segmentation.seal import MaskSealer

from .helpers import setup_pipeline_logging


def _build_mask_postprocessors(
    config: Config,
    logger: Logger | None = None,
) -> list[SegmentationMaskPostprocessor]:
    if not config.seal_enabled:
        return []
    return [MaskSealer(radius=config.seal_radius, logger=logger)]


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

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Build a Stage 1 pipeline configured from the merged ``config``.

        Wires the loader, segmenter, mask postprocessors (seal), mesh
        postprocessors (cleaner + smoother), fiducials handling and ESE config.
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

        logger = setup_pipeline_logging(project_dir_path_inst, "stage_1")

        return (
            cls(
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
            .setup_mask_postprocessors(_build_mask_postprocessors(config, logger))
            .setup_mesh_postprocessors(
                [
                    TrimeshCleaner(
                        min_component_vertices=config.cleaner_min_vertices,
                        merge_digits=config.cleaner_merge_digits,
                    ),
                    smoother,
                ]
            )
        )

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
        controller = PipelineController()

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
        if self._logger:
            log_provider = StoreLoggingProvider(self._logger)
            for store_type in (MRIVolume, SegmentationMask, ScalpMesh, Stage1Result):
                controller.register_provider(log_provider, on_store=store_type)

        if self._project_dir:
            controller.register_provider(
                Stage1Exporter(
                    project_dir=self._project_dir,
                    ese_config=self._ese_config,
                    config=self._config,
                    nifti_path=self._nifti_path,
                    logger=self._logger,
                ),
                Stage1Result,
            )

            controller.register_provider(
                ScalpMeshVersioningProvider(self._project_dir / "mesh" / "versions"),
                on_store=ScalpMesh,
            )

            # Quality control runs after the artifacts are exported.
            controller.register_step(
                Stage1QualityControlStep(
                    project_dir=self._project_dir,
                    ese_config=self._ese_config,
                    logger=self._logger,
                )
            )

        return controller
