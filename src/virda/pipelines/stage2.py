from logging import Logger
from pathlib import Path
from typing import Self

from virda.ese.contracts import ESEBuilder
from virda.ese.pca_ese_builder import PCAESEBuilder
from virda.io.providers.logging_provider import StoreLoggingProvider
from virda.io.providers.stage2_exporter import Stage2Exporter
from virda.models.config import Config
from virda.models.ese_mesh import ESEMesh
from virda.models.scalp_mesh import ScalpMesh
from virda.pipeline import PipelineController
from virda.pipeline_context import PipelineContext
from virda.pipelines.ese import ESEPipeline, ESEPipelineContract

from .helpers import get_stage_logger


class Stage2OutputGenerator:
    def __init__(self, ese_builder: ESEBuilder) -> None:
        self._ese_builder = ese_builder

    def run(self, context: PipelineContext) -> ESEMesh:
        return self._ese_builder.run(context)


class Stage2PipelineBuilder:
    def __init__(
        self,
        ese_builder: ESEBuilder,
        scalp_mesh: ScalpMesh,
        project_dir: Path,
        logger: Logger | None = None,
    ) -> None:
        self._ese_builder = ese_builder
        self._scalp_mesh = scalp_mesh
        self._project_dir = Path(project_dir)
        self._logger = logger
        self._ese_pipeline: ESEPipeline | None = None

    @classmethod
    def from_config(cls, config: Config, scalp_mesh: ScalpMesh) -> Self:
        """Build a Stage 2 pipeline configured from the merged ``config``.

        Requires that ESE is fully configured (``config.to_ese_config()`` is not None).
        The ESE mesh generation is delegated to the atomic ESE pipeline
        (:class:`~virda.pipelines.ese.ESEPipeline`); its options are taken from
        the stage-2 configuration, preserving the previous default behaviour.
        """
        ese_config = config.to_ese_config()
        if ese_config is None:
            raise ValueError("ESE is not configured. Provide ese_offset_mm.")

        resolved_project_dir = config.project_dir
        if resolved_project_dir is None:
            raise ValueError(
                "Project directory path not provided. "
                "Pass it as an argument, set the PROJECT_DIR environment variable,"
                " or add it to an input config file."
            )
        project_dir_path = Path(resolved_project_dir)

        stage2_config = config.to_stage2_config()
        logger = get_stage_logger(project_dir_path, "stage_2")

        ese_builder = PCAESEBuilder(
            config=stage2_config,
            ese_offset_mm=ese_config.ese_offset_mm,
        )

        ese_pipeline = ESEPipeline(
            contract=ESEPipelineContract(
                scalp_mesh=scalp_mesh,
                ese_offset_mm=ese_config.ese_offset_mm,
                project_dir=project_dir_path,
                neighborhood_radius_mm=stage2_config.neighborhood_radius_mm,
                k_neighbors=stage2_config.k_neighbors,
                use_weighted_pca=stage2_config.use_weighted_pca,
                pca_sigma_mm=stage2_config.pca_sigma_mm,
                min_neighbors=stage2_config.min_neighbors,
            ),
            logger=logger,
        )

        builder = cls(
            ese_builder=ese_builder,
            scalp_mesh=scalp_mesh,
            project_dir=project_dir_path,
            logger=logger,
        )
        builder._ese_pipeline = ese_pipeline
        return builder

    def build(self) -> PipelineController:
        if self._ese_pipeline is not None:
            return self._build_atomic()
        return self._build_manual()

    def _build_atomic(self) -> PipelineController:
        assert self._ese_pipeline is not None
        controller = self._ese_pipeline.build_stage0()

        log_provider = StoreLoggingProvider()
        for store_type in (ScalpMesh, ESEMesh):
            controller.register_provider(log_provider, on_store=store_type)

        controller.register_provider(
            Stage2Exporter(
                project_dir=self._project_dir,
            ),
            on_store=ESEMesh,
        )

        return controller

    def _build_manual(self) -> PipelineController:
        controller = PipelineController(logger=self._logger)

        controller.register_store(ScalpMesh, self._scalp_mesh)
        controller.register_step(Stage2OutputGenerator(self._ese_builder))

        log_provider = StoreLoggingProvider()
        for store_type in (ScalpMesh, ESEMesh):
            controller.register_provider(log_provider, on_store=store_type)

        controller.register_provider(
            Stage2Exporter(
                project_dir=self._project_dir,
            ),
            on_store=ESEMesh,
        )

        return controller