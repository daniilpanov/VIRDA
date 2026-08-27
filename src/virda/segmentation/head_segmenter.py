from typing import Literal, cast

import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import ball, closing

from virda.models.mri_volume import MRIVolume
from virda.models.segmentation_mask import SegmentationMask
from virda.segmentation import HeadSegmenter

OtsuScope = Literal["all", "foreground"]


class OtsuHeadSegmenter(HeadSegmenter):
    """Segment the head with an Otsu threshold.

    ``otsu_scope`` chooses which voxels feed the histogram:
      * ``"all"`` (default) uses the whole volume (background air dominates
        and drags the threshold down toward the scalp/air valley, keeping the
        darker scalp shell in the mask);
      * ``"foreground"`` restricts the histogram to head voxels (intensity
        above 10% of the maximum), which pushes the threshold up into the
        bright tissue and isolates the brain. On fat-suppressed T1 (e.g. the
        CTRL PROSET scans, where the scalp sits around the p50 of the head
        intensity while the brain is far brighter) this yields a brain mesh,
        not a scalp mesh, so use it only when tissue isolation is intended.

    ``threshold_scale`` is a multiplier applied to the selected Otsu threshold.
    Values below 1 lower the threshold and retain more low-intensity tissue.
    The default ``0.6`` puts the ``"all"`` threshold in the scalp/air valley:
    across the CTRL/icbm152 datasets it removes the crown holes (no holes
    >=1000 voxels over the top 40% of the head after sealing) while keeping a
    head-sized mesh and improving the genus relative to scale ``1.0``.
    """

    def __init__(
        self,
        closing_radius: int = 5,
        otsu_scope: OtsuScope = "all",
        threshold_scale: float = 0.6,
    ):
        if otsu_scope not in ("all", "foreground"):
            raise ValueError(f"otsu_scope must be 'all' or 'foreground', got {otsu_scope!r}")
        if threshold_scale <= 0:
            raise ValueError(f"threshold_scale must be positive, got {threshold_scale}")
        self._closing_radius: int = closing_radius
        self._otsu_scope: OtsuScope = otsu_scope
        self._threshold_scale: float = threshold_scale
        super().__init__()

    def _process(self, volume: MRIVolume) -> SegmentationMask:
        base_threshold = self._compute_base_threshold(volume.data)
        intensity_threshold = self._threshold_scale * base_threshold
        above_threshold_mask = volume.data > intensity_threshold

        connected_components = label(above_threshold_mask)
        if connected_components.max() < 1:
            return SegmentationMask(mask=np.zeros(volume.data.shape, dtype=bool))

        background_label = connected_components[0, 0, 0]
        largest_component_label = self._find_largest_component_label(
            connected_components, background_label
        )
        largest_component_mask = connected_components == largest_component_label

        structuring_element = ball(self._closing_radius)
        closed_mask = closing(largest_component_mask, structuring_element)

        return SegmentationMask(mask=cast(np.ndarray, closed_mask > 0))

    def _compute_base_threshold(self, data: np.ndarray) -> float:
        if self._otsu_scope == "all":
            return float(threshold_otsu(data))
        return self._threshold_otsu_on_foreground(data)

    @staticmethod
    def _threshold_otsu_on_foreground(data: np.ndarray) -> float:
        foreground = data[data > 0.1 * data.max()]
        if foreground.size < 2:
            return float(threshold_otsu(data))
        threshold = float(threshold_otsu(foreground))
        # Otsu on a single-class foreground returns that class value, and a
        # strict ">" comparison would then drop the whole foreground. Fall back
        # to the full-volume threshold so uniform inputs are not wiped out.
        if threshold >= foreground.max():
            return float(threshold_otsu(data))
        return threshold

    @staticmethod
    def _find_largest_component_label(
        connected_components: np.ndarray, background_label: int
    ) -> int:
        labels, counts = np.unique(connected_components, return_counts=True)
        label_to_count = dict(zip(labels, counts, strict=True))
        label_to_count.pop(background_label, None)
        largest_label = max(
            label_to_count, key=lambda component_label: label_to_count[component_label]
        )
        return int(largest_label)
