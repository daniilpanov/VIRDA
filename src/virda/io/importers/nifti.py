"""Import an MRI NIfTI volume from disk into an :class:`MRIVolume` model."""

from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel import aff2axcodes

from virda.models.mri_volume import MRIVolume


def load_nifti(path: Path) -> MRIVolume:
    """Load a NIfTI file into an :class:`MRIVolume`.

    Raises ``FileNotFoundError`` when the file is missing and ``ValueError``
    when the file cannot be parsed or lacks 3 spatial dimensions.
    """
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    try:
        nifti_image = nib.load(path)
    except Exception as error:
        raise ValueError(f"Failed to parse NIfTI file: {error}") from error

    assert isinstance(nifti_image, nib.Nifti1Image)

    volume_data = nifti_image.get_fdata(dtype=np.float32)
    voxel_to_world_affine = nifti_image.affine

    header_zooms = nifti_image.header.get_zooms()
    if len(header_zooms) < 3:
        raise ValueError(
            f"NIfTI header must have at least 3 spatial dimensions, got {len(header_zooms)}"
        )

    zoom_x, zoom_y, zoom_z = header_zooms[:3]
    voxel_spacing_mm: tuple[float, float, float] = (float(zoom_x), float(zoom_y), float(zoom_z))

    return MRIVolume(
        data=volume_data,
        affine=voxel_to_world_affine,
        spacing=voxel_spacing_mm,
        orientation=aff2axcodes(nifti_image.affine),
    )
