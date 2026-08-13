from typing import cast

import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import ball, closing

from virda.models.mri_volume import MRIVolume
from virda.models.segmentation_mask import SegmentationMask
from virda.segmentation import HeadSegmenter


class OtsuHeadSegmenter(HeadSegmenter):
    def __init__(self, closing_radius: int = 5):
        self._closing_radius: int = closing_radius

    def _process(self, volume: MRIVolume) -> SegmentationMask:
        intensity_threshold = threshold_otsu(volume.data)
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
