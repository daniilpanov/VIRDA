"""Step: bundle the Stage 1 artifacts into a single result object."""

from virda.models.fiducial import Fiducials
from virda.models.mri_volume import MRIVolume
from virda.models.scalp_mesh import ScalpMesh
from virda.models.segmentation_mask import SegmentationMask
from virda.models.stage1_result import Stage1Result


def build_stage1_result(
    mri_volume: MRIVolume,
    segmentation_mask: SegmentationMask,
    mesh: ScalpMesh,
    fiducials: Fiducials,
) -> Stage1Result:
    """Collect volume, mask, scalp mesh and fiducials into :class:`Stage1Result`."""
    return Stage1Result(
        mri_volume=mri_volume,
        segmentation_mask=segmentation_mask,
        mesh=mesh,
        fiducials=fiducials,
    )
