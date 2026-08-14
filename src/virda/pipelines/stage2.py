from logging import Logger
from pathlib import Path

from virda.ese.contracts import ESEBuilder
from virda.io.providers.logging_provider import StoreLoggingProvider
from virda.io.providers.stage2_exporter import Stage2Exporter
from virda.models.ese_mesh import ESEMesh
from virda.models.scalp_mesh import ScalpMesh
from virda.models.stage2_config import Stage2Config
from virda.pipeline import PipelineController
from virda.pipeline_context import PipelineContext


class Stage2OutputGenerator:
    def __init__(self, ese_builder: ESEBuilder) -> None:
        self._ese_builder = ese_builder

    def run(self, context: PipelineContext) -> ESEMesh:
        return self._ese_builder.run(context)


class Stage2PipelineBuilder:
    def __init__(
        self,
        ese_builder: ESEBuilder,
        stage2_config: Stage2Config,
        scalp_mesh: ScalpMesh,
        project_dir: Path,
        logger: Logger | None = None,
    ) -> None:
        self._ese_builder = ese_builder
        self._stage2_config = stage2_config
        self._scalp_mesh = scalp_mesh
        self._project_dir = Path(project_dir)
        self._logger = logger

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
                stage2_config=self._stage2_config,
                logger=self._logger,
            ),
            on_store=ESEMesh,
        )

        return controller
