from dataclasses import dataclass, field

import numpy as np

from virda.models.fiducial import Fiducial
from virda.models.mri_volume import MRIVolume
from virda.models.scalp_mesh import ScalpMesh


@dataclass(frozen=True)
class Stage1Result:
    mri_volume: MRIVolume
    segmentation_mask: np.ndarray
    mesh: ScalpMesh
    fiducials: list[Fiducial] = field(default_factory=list)
