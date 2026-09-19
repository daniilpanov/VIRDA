from abc import ABC, abstractmethod
from logging import Logger

from virda.models.electrode import Electrodes
from virda.models.ese_mesh import ESEMesh
from virda.models.fiducial import Fiducials
from virda.models.scalp_mesh import ScalpMesh
from virda.pipeline_context import PipelineContext


class ElectrodeLocalizer(ABC):
    def __init__(self) -> None:
        self._logger: Logger | None = None

    def run(self, context: PipelineContext) -> Electrodes:
        self._logger = context.get_logger()
        surface = context.get_store(ESEMesh) or context.get_store(ScalpMesh)
        if surface is None:
            raise ValueError("Localization requires a surface store: 'ESEMesh' or 'ScalpMesh'")
        return self._process(
            surface=surface,
            fiducials=context.get_store_notnull(Fiducials),
            electrodes=context.get_store_notnull(Electrodes),
        )

    @abstractmethod
    def _process(
        self,
        surface: ESEMesh | ScalpMesh,
        fiducials: Fiducials,
        electrodes: Electrodes,
    ) -> Electrodes:
        raise NotImplementedError