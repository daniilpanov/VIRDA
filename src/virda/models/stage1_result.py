from dataclasses import dataclass

from virda.models.fiducial import Fiducials
from virda.models.mri_volume import MRIVolume
from virda.models.scalp_mesh import ScalpMesh
from virda.models.segmentation_mask import SegmentationMask


@dataclass(frozen=True)
class Stage1Result:
    mri_volume: MRIVolume
    segmentation_mask: SegmentationMask
    mesh: ScalpMesh
    fiducials: Fiducials
