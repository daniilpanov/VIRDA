from abc import ABC, abstractmethod
from logging import Logger

from virda.models.ese_mesh import ESEMesh
from virda.models.scalp_mesh import ScalpMesh


class ESEBuilder(ABC):
    def __init__(self) -> None:
        self._logger: Logger | None = None

    @abstractmethod
    def process(self, scalp_mesh: ScalpMesh) -> ESEMesh:
        raise NotImplementedError
