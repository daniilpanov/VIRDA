from abc import ABC, abstractmethod
from logging import Logger

from virda.models.ese_mesh import ESEMesh
from virda.models.scalp_mesh import ScalpMesh
from virda.pipeline_context import PipelineContext


class ESEBuilder(ABC):
    def __init__(self) -> None:
        self._logger: Logger | None = None

    def run(self, context: PipelineContext) -> ESEMesh:
        self._logger = context.get_logger()
        return self._process(context.get_store_notnull(ScalpMesh))

    @abstractmethod
    def _process(self, scalp_mesh: ScalpMesh) -> ESEMesh:
        raise NotImplementedError
