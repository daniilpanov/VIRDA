from abc import ABC, abstractmethod

from virda.models.mri_volume import MRIVolume
from virda.models.path import NiftiPath
from virda.pipeline_context import PipelineContext


class MRILoader(ABC):
    def run(self, context: PipelineContext) -> MRIVolume:
        return self._process(context.get_store_notnull(NiftiPath))

    @abstractmethod
    def _process(self, path: NiftiPath) -> MRIVolume:
        raise NotImplementedError
