"""Export a head segmentation mask as a NIfTI image."""

from pathlib import Path

import nibabel as nib
import numpy as np

from virda.models.mri_volume import MRIVolume
from virda.models.segmentation_mask import SegmentationMask


def export_head_mask(path: str | Path, mask: SegmentationMask, volume: MRIVolume) -> Path:
    """Write *mask* as a uint8 NIfTI preserving *volume*'s affine.

    The affine lets voxel coordinates of the saved mask map 1:1 onto the
    source MRI.  Returns the written path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.asarray(mask.mask, dtype=np.uint8), volume.affine)
    nib.save(image, str(target))
    return target
