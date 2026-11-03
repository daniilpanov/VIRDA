from abc import ABC, abstractmethod
from logging import Logger

from virda.models.electrode import Electrodes
from virda.models.fiducial import ManualFiducials
from virda.models.path import FiducialsPath, MeasurementsPath
from virda.pipeline_context import PipelineContext


class FiducialsLoader(ABC):
    def __init__(self) -> None:
        self._logger: Logger | None = None

    def run(self, context: PipelineContext) -> ManualFiducials:
        self._logger = context.get_logger()
        return self._process(context.get_store_notnull(FiducialsPath))

    @abstractmethod
    def _process(self, path: FiducialsPath) -> ManualFiducials:
        raise NotImplementedError


class MeasurementsLoader(ABC):
    def run(self, context: PipelineContext) -> Electrodes:
        return self._process(context, context.get_store_notnull(MeasurementsPath))

    @abstractmethod
    def _process(self, context: PipelineContext, path: MeasurementsPath) -> Electrodes:
        raise NotImplementedError
