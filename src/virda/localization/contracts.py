from abc import ABC, abstractmethod
from logging import Logger

from virda.models.electrode import Electrodes
from virda.models.ese_mesh import ESEMesh
from virda.models.fiducial import Fiducials
from virda.models.scalp_mesh import ScalpMesh


class ElectrodeLocalizer(ABC):
    def __init__(self) -> None:
        self._logger: Logger | None = None

    @abstractmethod
    def process(
        self,
        surface: ESEMesh | ScalpMesh,
        fiducials: Fiducials,
        electrodes: Electrodes,
    ) -> Electrodes:
        raise NotImplementedError
