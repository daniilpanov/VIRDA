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

from .helpers import setup_pipeline_logging


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

    @classmethod
    def from_config(cls, config: Config, scalp_mesh: ScalpMesh) -> Self:
        """Build a Stage 2 pipeline configured from the merged ``config``.

        Requires that ESE is fully configured (``config.to_ese_config()`` is not None).
        """
        ese_config = config.to_ese_config()
        if ese_config is None:
            raise ValueError(
                "ESE is not configured. Provide n_electrodes, ese_offset_mm, and ese_reference."
            )

        resolved_project_dir = config.project_dir
        if resolved_project_dir is None:
            raise ValueError(
                "Project directory path not provided. "
                "Pass it as an argument, set the PROJECT_DIR environment variable,"
                " or add it to an input config file."
            )
        project_dir_path = Path(resolved_project_dir)

        stage2_config = config.to_stage2_config()
        logger = setup_pipeline_logging(project_dir_path, "stage_2")

        ese_builder = PCAESEBuilder(
            config=stage2_config,
            ese_offset_mm=ese_config.ese_offset_mm,
        )

        return cls(
            ese_builder=ese_builder,
            scalp_mesh=scalp_mesh,
            project_dir=project_dir_path,
            logger=logger,
        )

    def build(self) -> PipelineController:
        controller = PipelineController()

        controller.register_store(ScalpMesh, self._scalp_mesh)
        controller.register_step(Stage2OutputGenerator(self._ese_builder))

        if self._logger:
            log_provider = StoreLoggingProvider(self._logger)
            for store_type in (ScalpMesh, ESEMesh):
                controller.register_provider(log_provider, on_store=store_type)

        controller.register_provider(
            Stage2Exporter(
                project_dir=self._project_dir,
                logger=self._logger,
            ),
            on_store=ESEMesh,
        )

        return controller
