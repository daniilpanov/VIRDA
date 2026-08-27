from logging import Logger
from pathlib import Path

from virda.io.providers.logging_provider import StoreLoggingProvider
from virda.io.providers.stage3_exporter import Stage3Exporter
from virda.localization.contracts import ElectrodeLocalizer
from virda.models.electrode import Electrodes
from virda.models.ese_mesh import ESEMesh
from virda.models.fiducial import Fiducials
from virda.models.stage3_config import Stage3Config
from virda.pipeline import PipelineController
from virda.pipeline_context import PipelineContext


class Stage3LocalizerStep:
    def __init__(self, localizer: ElectrodeLocalizer) -> None:
        self._localizer = localizer

    def run(self, context: PipelineContext) -> Electrodes:
        return self._localizer.run(context)


class Stage3PipelineBuilder:
    def __init__(
        self,
        localizer: ElectrodeLocalizer,
        stage3_config: Stage3Config,
        ese_mesh: ESEMesh,
        electrodes: Electrodes,
        fiducials: Fiducials,
        project_dir: Path,
        logger: Logger | None = None,
    ) -> None:
        self._localizer = localizer
        self._stage3_config = stage3_config
        self._ese_mesh = ese_mesh
        self._electrodes = electrodes
        self._fiducials = fiducials
        self._project_dir = Path(project_dir)
        self._logger = logger

    def build(self) -> PipelineController:
        controller = PipelineController(logger=self._logger)

        controller.register_store(ESEMesh, self._ese_mesh)
        controller.register_store(Electrodes, self._electrodes)
        controller.register_store(Fiducials, self._fiducials)
        controller.register_step(Stage3LocalizerStep(self._localizer))

        if self._logger:
            log_provider = StoreLoggingProvider(self._logger)
            for store_type in (ESEMesh, Fiducials, Electrodes):
                controller.register_provider(log_provider, on_store=store_type)

        controller.register_provider(
            Stage3Exporter(
                project_dir=self._project_dir,
                stage3_config=self._stage3_config,
                logger=self._logger,
            ),
            on_store=Electrodes,
        )

        return controller
