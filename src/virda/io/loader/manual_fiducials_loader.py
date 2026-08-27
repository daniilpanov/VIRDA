from virda.io.fiducial_helpers import load_fiducials
from virda.io.loader.contracts import FiducialsLoader
from virda.models.fiducial import ManualFiducials
from virda.models.path import FiducialsPath


class ManualFiducialsLoader(FiducialsLoader):
    def _process(self, path: FiducialsPath) -> ManualFiducials:
        return ManualFiducials(fiducials=load_fiducials(path.fiducials_path, self._logger))
