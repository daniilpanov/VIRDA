"""MRI loading adapter — NIfTI/DICOM/MNE → core.MRIData.

This module lives in the API layer because it depends on nibabel,
pydicom, and mne. Core modules never import from this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from core.dataclasses import MRIData

logger = logging.getLogger(__name__)


def load_nifti(path: str | Path) -> MRIData:
    """Load a NIfTI file (.nii or .nii.gz) via nibabel.

    Parameters
    ----------
    path : str or Path
        Path to NIfTI file.

    Returns
    -------
    MRIData
        Loaded MRI data with coordinate metadata.
    """
    import nibabel as nib

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    affine = img.affine.copy()
    header = img.header

    voxel_size = np.array(header.get_zooms()[:3], dtype=np.float64)

    logger.info(
        "Loaded NIfTI: shape=%s, voxel_size=%s, dtype=%s",
        data.shape,
        voxel_size,
        data.dtype,
    )

    return MRIData(
        data=data,
        affine=affine,
        voxel_size=voxel_size,
        source_path=str(path),
    )


def load_dicom_series(directory: str | Path) -> MRIData:
    """Load a DICOM series from a directory via pydicom.

    Parameters
    ----------
    directory : str or Path
        Directory containing DICOM files for one series.

    Returns
    -------
    MRIData
        Loaded MRI data with coordinate metadata.
    """
    import pydicom

    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    dicom_files = sorted(
        [f for f in directory.iterdir() if f.suffix.lower() in (".dcm", "")]
    )

    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found in {directory}")

    slices = []
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(str(f))
            if hasattr(ds, "InstanceNumber"):
                slices.append(ds)
        except Exception:
            continue

    if not slices:
        raise ValueError(f"No valid DICOM files found in {directory}")

    slices.sort(key=lambda s: int(s.InstanceNumber))

    pixel_array = np.stack([s.pixel_array for s in slices], axis=-1)
    pixel_array = pixel_array.astype(np.float64)

    pixel_spacing = np.array(
        [
            float(slices[0].PixelSpacing[1]),
            float(slices[0].PixelSpacing[0]),
            float(slices[0].SliceThickness),
        ],
        dtype=np.float64,
    )

    if hasattr(slices[0], "ImagePositionPatient"):
        position = np.array(
            [float(x) for x in slices[0].ImagePositionPatient], dtype=np.float64
        )
    else:
        position = np.zeros(3)

    affine = np.eye(4)
    affine[:3, :3] = np.diag(pixel_spacing)
    affine[:3, 3] = position

    logger.info(
        "Loaded DICOM series: %d slices, shape=%s, pixel_spacing=%s",
        len(slices),
        pixel_array.shape,
        pixel_spacing,
    )

    return MRIData(
        data=pixel_array,
        affine=affine,
        voxel_size=pixel_spacing,
        source_path=str(directory),
    )


def load_mri(path: str | Path) -> MRIData:
    """Auto-detect format and load MRI.

    Parameters
    ----------
    path : str or Path
        Path to NIfTI file or directory with DICOM series.

    Returns
    -------
    MRIData
        Loaded MRI data.
    """
    path = Path(path)

    if path.is_dir():
        return load_dicom_series(path)

    if path.suffix in (".nii", ".gz") and (
        path.suffix == ".nii" or path.name.endswith(".nii.gz")
    ):
        return load_nifti(path)

    raise ValueError(f"Cannot determine MRI format for: {path}")


def load_mri_with_mne(subject: str, subjects_dir: str | Path) -> MRIData:
    """Load MRI using MNE-Python's read functions.

    This is useful for loading atlas/template MRIs that ship with MNE.

    Parameters
    ----------
    subject : str
        MNE subject name (e.g. 'fsaverage', 'sample').
    subjects_dir : str or Path
        Path to the subjects directory.

    Returns
    -------
    MRIData
        Loaded MRI data.
    """
    import mne

    subjects_dir = Path(subjects_dir)
    t1_path = subjects_dir / subject / "mri" / "T1.mgz"

    if t1_path.exists():
        img = mne.read_fs_mri(str(t1_path))
        data = np.asarray(img.dataobj)
        affine = img.affine

        voxel_size = np.array(
            [np.sqrt(np.sum(affine[:3, i] ** 2)) for i in range(3)],
            dtype=np.float64,
        )

        logger.info("Loaded MRI via MNE: subject=%s, shape=%s", subject, data.shape)

        return MRIData(
            data=data,
            affine=affine,
            voxel_size=voxel_size,
            source_path=str(t1_path),
        )

    raise FileNotFoundError(f"T1.mgz not found for subject {subject} at {t1_path}")
