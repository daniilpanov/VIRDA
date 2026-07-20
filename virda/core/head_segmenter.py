"""Head segmentation module — extract external head surface from MRI."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .dataclasses import MRIData, SegmentationData

logger = logging.getLogger(__name__)


class SegmenterBase(ABC):
    """Base class for head segmentation methods."""

    @abstractmethod
    def segment(self, mri: MRIData) -> SegmentationData:
        ...


class ThresholdSegmenter(SegmenterBase):
    """Segment head using Otsu thresholding + morphology."""

    def __init__(
        self,
        smooth_sigma: float = 1.0,
        close_radius: int = 3,
        min_component_size: int = 10000,
    ):
        self.smooth_sigma = smooth_sigma
        self.close_radius = close_radius
        self.min_component_size = min_component_size

    def segment(self, mri: MRIData) -> SegmentationData:
        from scipy import ndimage
        from skimage.filters import threshold_otsu
        from skimage.measure import label
        from skimage.morphology import closing, ball

        logger.info("Starting threshold-based head segmentation")

        volume = mri.data.astype(np.float64)

        if self.smooth_sigma > 0:
            volume = ndimage.gaussian_filter(volume, sigma=self.smooth_sigma)

        thresh = threshold_otsu(volume)
        binary = volume > thresh
        logger.info("Otsu threshold: %.2f", thresh)

        if self.close_radius > 0:
            struct = ball(self.close_radius)
            binary = closing(binary, footprint=struct)
            logger.info("Applied morphological closing (radius=%d)", self.close_radius)

        labeled = label(binary)
        component_sizes = np.bincount(labeled.ravel())
        component_sizes[0] = 0

        if len(component_sizes) < 2:
            logger.warning("No components found after segmentation")
            return SegmentationData(
                mask=np.zeros_like(binary, dtype=np.int32),
                voxel_size=mri.get_voxel_spacing(),
                num_components=0,
                method_name="threshold",
            )

        largest_label = component_sizes.argmax()

        small_mask = component_sizes < self.min_component_size
        small_labels = np.where(small_mask)[0]
        if len(small_labels) > 0:
            for lbl in small_labels:
                labeled[labeled == lbl] = 0

        labeled[labeled != largest_label] = 0
        labeled[labeled > 0] = 1
        mask = labeled.astype(np.int32)

        num_valid = int((component_sizes[component_sizes >= self.min_component_size]).sum())

        logger.info(
            "Segmentation complete: %d components, largest label=%d, kept size=%d voxels",
            num_valid,
            largest_label,
            int(mask.sum()),
        )

        return SegmentationData(
            mask=mask,
            voxel_size=mri.get_voxel_spacing(),
            num_components=num_valid,
            method_name="threshold",
        )


class RegionGrowSegmenter(SegmenterBase):
    """Segment head using region growing from image center."""

    def __init__(
        self,
        smooth_sigma: float = 1.0,
        threshold_sigma: float = 2.0,
    ):
        self.smooth_sigma = smooth_sigma
        self.threshold_sigma = threshold_sigma

    def segment(self, mri: MRIData) -> SegmentationData:
        from scipy import ndimage
        from skimage.measure import label

        logger.info("Starting region-growing head segmentation")

        volume = mri.data.astype(np.float64)

        if self.smooth_sigma > 0:
            volume = ndimage.gaussian_filter(volume, sigma=self.smooth_sigma)

        mean_val = volume.mean()
        std_val = volume.std()
        threshold = mean_val + self.threshold_sigma * std_val

        binary = volume > threshold

        labeled = label(binary)
        component_sizes = np.bincount(labeled.ravel())
        component_sizes[0] = 0

        if len(component_sizes) < 2:
            return SegmentationData(
                mask=np.zeros_like(binary, dtype=np.int32),
                voxel_size=mri.get_voxel_spacing(),
                num_components=0,
                method_name="region_grow",
            )

        largest_label = component_sizes.argmax()
        mask = (labeled == largest_label).astype(np.int32)

        logger.info("Region-growing segmentation: kept %d voxels", int(mask.sum()))

        return SegmentationData(
            mask=mask,
            voxel_size=mri.get_voxel_spacing(),
            num_components=int((component_sizes > 0).sum()),
            method_name="region_grow",
        )


class HeadSegmenter:
    """Main head segmentation interface.

    Parameters
    ----------
    method : str
        Segmentation method: 'threshold' or 'region_grow'.
    smooth_sigma : float
        Gaussian smoothing sigma before thresholding.
    close_radius : int
        Morphological closing radius (threshold method only).
    min_component_size : int
        Minimum number of voxels to keep a connected component.
    """

    def __init__(
        self,
        method: str = "threshold",
        smooth_sigma: float = 1.0,
        close_radius: int = 3,
        min_component_size: int = 10000,
        threshold_sigma: float = 2.0,
    ):
        self.method = method
        if method == "threshold":
            self._segmenter = ThresholdSegmenter(
                smooth_sigma=smooth_sigma,
                close_radius=close_radius,
                min_component_size=min_component_size,
            )
        elif method == "region_grow":
            self._segmenter = RegionGrowSegmenter(
                smooth_sigma=smooth_sigma,
                threshold_sigma=threshold_sigma,
            )
        else:
            raise ValueError(f"Unknown segmentation method: {method}")

    def segment(self, mri: MRIData) -> SegmentationData:
        """Run head segmentation on loaded MRI."""
        return self._segmenter.segment(mri)
