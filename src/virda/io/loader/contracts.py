from abc import ABC, abstractmethod

from virda.models.electrode import Electrodes
from virda.models.path import MeasurementsPath
from virda.pipeline_context import PipelineContext


class MeasurementsLoader(ABC):
    def run(self, context: PipelineContext) -> Electrodes:
        return self._process(context, context.get_store_notnull(MeasurementsPath))

    @abstractmethod
    def _process(self, context: PipelineContext, path: MeasurementsPath) -> Electrodes:
        raise NotImplementedError
