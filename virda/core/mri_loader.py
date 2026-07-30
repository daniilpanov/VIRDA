"""MRI volume loading from DICOM and NIfTI formats."""

from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom

from .types import MRIData

logger = logging.getLogger(__name__)


def load_nifti(path: Path) -> MRIData:
    """Load a NIfTI file (.nii or .nii.gz).

    Parameters
    ----------
    path : Path
        Path to the NIfTI file.

    Returns
    -------
    MRIData
        Loaded volume with affine and voxel size.
    """
    img = nib.load(path)
    volume = np.asarray(img.dataobj)
    affine = img.affine
    hdr = img.header

    voxel_size = np.array(hdr.get_zooms()[:3], dtype=np.float64)
    if len(voxel_size) < 3:
        voxel_size = np.pad(voxel_size, (0, 3 - len(voxel_size)), constant_values=1.0)

    header_info = {
        "format": "nifti",
        "shape": list(volume.shape),
        "dtype": str(volume.dtype),
        "qform_code": int(hdr.get_qform_code()) if hasattr(hdr, "get_qform_code") else 0,
        "sform_code": int(hdr.get_sform_code()) if hasattr(hdr, "get_sform_code") else 0,
    }

    logger.info("Loaded NIfTI: shape=%s, voxel_size=%s", volume.shape, voxel_size)
    return MRIData(
        volume=volume,
        affine=affine,
        voxel_size=voxel_size,
        header_info=header_info,
    )


def load_dicom_series(directory: Path) -> MRIData:
    """Load a DICOM series from a directory.

    Parameters
    ----------
    directory : Path
        Directory containing DICOM files.

    Returns
    -------
    MRIData
        Loaded volume with affine and voxel size.

    Raises
    ------
    FileNotFoundError
        If no DICOM files found in the directory.
    """
    directory = Path(directory)
    dicom_files = sorted(
        [f for f in directory.iterdir() if f.is_file() and f.suffix != ".DS_Store"],
        key=lambda f: f.name,
    )

    slices = []
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(f)
            if hasattr(ds, "InstanceNumber"):
                slices.append(ds)
        except pydicom.errors.InvalidDicomError:
            continue

    if not slices:
        raise FileNotFoundError(f"No valid DICOM files found in {directory}")

    slices.sort(key=lambda s: int(s.InstanceNumber))

    volume = np.stack([s.pixel_array for s in slices], axis=-1).astype(np.float64)

    pixel_spacing = getattr(slices[0], "PixelSpacing", [1.0, 1.0])
    slice_thickness = getattr(slices[0], "SliceThickness", 1.0)
    voxel_size = np.array([float(pixel_spacing[0]), float(pixel_spacing[1]), float(slice_thickness)])

    affine = np.eye(4)
    image_position = getattr(slices[0], "ImagePositionPatient", [0.0, 0.0, 0.0])
    image_orientation = getattr(slices[0], "ImageOrientationPatient", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    row_cos = np.array(image_orientation[:3], dtype=np.float64)
    col_cos = np.array(image_orientation[3:], dtype=np.float64)
    slice_cos = np.cross(row_cos, col_cos)

    affine[:3, 0] = row_cos * voxel_size[0]
    affine[:3, 1] = col_cos * voxel_size[1]
    affine[:3, 2] = slice_cos * voxel_size[2]
    affine[:3, 3] = np.array(image_position, dtype=np.float64)

    header_info = {
        "format": "dicom",
        "shape": list(volume.shape),
        "dtype": str(volume.dtype),
        "num_slices": len(slices),
        "patient_id": getattr(slices[0], "PatientID", "unknown"),
    }

    logger.info("Loaded DICOM series: %d slices, shape=%s", len(slices), volume.shape)
    return MRIData(
        volume=volume,
        affine=affine,
        voxel_size=voxel_size,
        header_info=header_info,
    )


def load_mri(path: Path) -> MRIData:
    """Load MRI data from a file or directory.

    Auto-detects format: .nii/.nii.gz -> NIfTI, directory -> DICOM series.

    Parameters
    ----------
    path : Path
        Path to NIfTI file or DICOM directory.

    Returns
    -------
    MRIData
        Loaded volume with spatial metadata.
    """
    path = Path(path)
    if path.is_file() and path.suffix in (".nii", ".gz"):
        return load_nifti(path)
    if path.is_dir():
        return load_dicom_series(path)
    raise ValueError(f"Cannot determine MRI format for: {path}")
