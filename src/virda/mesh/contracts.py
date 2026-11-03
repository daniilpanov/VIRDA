from abc import ABC, abstractmethod

from virda.models.scalp_mesh import ScalpMesh
from virda.pipeline_context import PipelineContext


class MeshPostprocessor(ABC):
    def run(self, context: PipelineContext) -> ScalpMesh:
        return self._process(context.get_store_notnull(ScalpMesh))

    @abstractmethod
    def _process(self, mesh: ScalpMesh) -> ScalpMesh:
        raise NotImplementedError
