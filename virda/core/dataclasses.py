"""Shared data types for the VIRDA core.

These dataclasses replace cross-layer imports. Modules in ``core/`` use
these types instead of importing from ``api/`` (e.g. ``mri_loader``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class MRIData:
    """Numpy-only MRI container — no nibabel/mne dependency.

    Attributes
    ----------
    data : np.ndarray
        3D voxel data.
    affine : np.ndarray
        Voxel-to-world transformation matrix (4×4).
    voxel_size : np.ndarray
        Voxel dimensions in mm (3,).
    source_path : str or None
        Original file path for provenance.
    """

    data: np.ndarray
    affine: np.ndarray
    voxel_size: np.ndarray
    source_path: Optional[str] = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def voxel_to_world(self, voxel_coords: np.ndarray) -> np.ndarray:
        """Convert voxel indices to world (mm) coordinates."""
        ones = np.ones((*voxel_coords.shape[:-1], 1))
        coords = np.concatenate([voxel_coords, ones], axis=-1)
        return coords @ self.affine.T

    def world_to_voxel(self, world_coords: np.ndarray) -> np.ndarray:
        """Convert world (mm) coordinates to voxel indices."""
        inv_affine = np.linalg.inv(self.affine)
        ones = np.ones((*world_coords.shape[:-1], 1))
        coords = np.concatenate([world_coords, ones], axis=-1)
        return coords @ inv_affine.T

    def get_voxel_spacing(self) -> np.ndarray:
        """Return voxel spacing in mm along each axis."""
        return np.abs(self.affine[:3, :3].diagonal())


@dataclass
class SegmentationData:
    """Result of head segmentation — no nibabel dependency.

    Attributes
    ----------
    mask : np.ndarray
        Binary 3D segmentation mask.
    voxel_size : np.ndarray
        Voxel dimensions in mm.
    num_components : int
        Number of kept connected components.
    method_name : str
        Name of the segmentation method used.
    """

    mask: np.ndarray
    voxel_size: np.ndarray
    num_components: int = 1
    method_name: str = "unknown"
