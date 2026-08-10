from pathlib import Path

import nibabel as nib
import numpy as np


def export_segmentation(path: str | Path, mask: np.ndarray, affine: np.ndarray) -> Path:
    """Save the binary segmentation mask as a NIfTI file in MRI coordinates."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(mask.astype(np.uint8), affine)
    nib.save(image, out)
    return out
