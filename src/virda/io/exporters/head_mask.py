"""Export the segmentation mask as a NIfTI volume (preserves the MRI affine)."""

from pathlib import Path

import nibabel as nib
import numpy as np

from virda.models.mri_volume import MRIVolume
from virda.models.segmentation_mask import SegmentationMask


def export_head_mask(path: str | Path, mask: SegmentationMask, mri: MRIVolume) -> Path:
    """Write *mask* as a NIfTI file aligned with the MRI world space."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    seg_nii = nib.Nifti1Image(mask.mask.astype(np.uint8), mri.affine)
    nib.save(seg_nii, str(target))
    return target