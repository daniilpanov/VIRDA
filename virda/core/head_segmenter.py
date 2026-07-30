"""Binary head segmentation from MRI volumes."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from scipy import ndimage
from skimage import measure, morphology

from .types import MRIData, SegmentationResult

logger = logging.getLogger(__name__)


class Segmenter(ABC):
    """Interface for head segmentation strategies."""

    @abstractmethod
    def segment(self, mri: MRIData) -> SegmentationResult:
        """Segment the external head surface from MRI.

        Parameters
        ----------
        mri : MRIData
            Loaded MRI volume.

        Returns
        -------
        SegmentationResult
            Binary mask of the head.
        """


class ThresholdSegmenter(Segmenter):
    """Head segmentation using Otsu thresholding and morphological operations.

    Parameters
    ----------
    min_component_size : int
        Minimum number of voxels for a connected component to be kept.
    closing_iterations : int
        Number of morphological closing iterations.
    opening_iterations : int
        Number of morphological opening iterations.
    """

    def __init__(
        self,
        min_component_size: int = 1000,
        closing_iterations: int = 2,
        opening_iterations: int = 1,
    ) -> None:
        self.min_component_size = min_component_size
        self.closing_iterations = closing_iterations
        self.opening_iterations = opening_iterations

    def segment(self, mri: MRIData) -> SegmentationResult:
        """Segment head using Otsu threshold + morphological cleanup."""
        volume = mri.volume

        threshold = _otsu_threshold(volume)
        mask = (volume > threshold).astype(np.int32)
        logger.info("Otsu threshold: %.2f, initial voxels: %d", threshold, mask.sum())

        mask = _keep_largest_component(mask, self.min_component_size)

        if self.closing_iterations > 0:
            struct = ndimage.generate_binary_structure(3, 2)
            mask = ndimage.binary_closing(mask, structure=struct, iterations=self.closing_iterations)
            mask = mask.astype(np.int32)

        if self.opening_iterations > 0:
            mask = ndimage.binary_opening(mask, iterations=self.opening_iterations)
            mask = mask.astype(np.int32)

        logger.info("Final segmentation: %d voxels", mask.sum())
        return SegmentationResult(mask=mask, affine=mri.affine, method="threshold")


def _otsu_threshold(volume: np.ndarray) -> float:
    """Compute Otsu threshold for a grayscale volume."""
    from skimage.filters import threshold_otsu

    nonzero = volume[volume > 0]
    if len(nonzero) == 0:
        return 0.0
    return float(threshold_otsu(nonzero))


def _keep_largest_component(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Keep only the largest connected components."""
    labeled = measure.label(mask, connectivity=2)
    if labeled.max() == 0:
        return mask

    component_sizes = ndimage.sum(mask, labeled, range(1, labeled.max() + 1))
    large_components = [
        i + 1 for i, size in enumerate(component_sizes) if size >= min_size
    ]

    if not large_components:
        largest_label = int(np.argmax(component_sizes)) + 1
        large_components = [largest_label]

    result = np.isin(labeled, large_components).astype(np.int32)
    logger.info(
        "Kept %d components (out of %d), min_size=%d",
        len(large_components),
        labeled.max(),
        min_size,
    )
    return result


def segment_head(
    mri: MRIData,
    method: str = "threshold",
    **kwargs,
) -> SegmentationResult:
    """Convenience function for head segmentation.

    Parameters
    ----------
    mri : MRIData
        Loaded MRI volume.
    method : str
        Segmentation method. Currently only 'threshold' is supported.
    **kwargs
        Additional arguments passed to the segmenter.

    Returns
    -------
    SegmentationResult
        Binary head mask.
    """
    if method == "threshold":
        segmenter = ThresholdSegmenter(**kwargs)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
    return segmenter.segment(mri)
