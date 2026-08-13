"""Post-segmentation sealing of the head mask.

MRI bias-field inhomogeneity can push parts of the scalp (top of the head,
cheeks) below a global intensity threshold, fragmenting the mask and leaving
open channels from the interior cavities (orbits, sinuses, nasal cavity) to the
exterior air. Sealing the mask makes it a solid blob:

  * ``closing`` with a small ball bridges surface openings narrower than the
    ball diameter (nostrils, ear canals, fragmented apex patches);
  * ``binary_fill_holes`` then fills every now-enclosed cavity.

The result has no holes and no internal cavities, so the extracted scalp mesh
is a closed surface and the air-depth cleaner has no walls left to remove.
The output is trimmed to the most voluminous connected component (the head),
so stray fragments are dropped instead of surviving alongside the head.
"""

import logging
from typing import cast

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import ball, closing

from virda.models.segmentation_mask import SegmentationMask
from virda.segmentation.contracts import SegmentationMaskPostprocessor

logger = logging.getLogger(__name__)


def seal_mask(mask: np.ndarray, radius: int = 4, keep_largest: bool = True) -> np.ndarray:
    """Return a solid version of ``mask`` with surface openings bridged.

    ``radius`` (in voxels) is the closing ball radius; it must be large enough
    to bridge the narrowest channels that lead from interior cavities to the
    exterior (nostrils, ear canals), yet small enough not to erode thin scalp
    features. For ~1 mm isotropic data ``radius=4`` works.

    When ``keep_largest`` is set, only the most voluminous connected component
    of the sealed mask is returned; every other fragment is dropped from the
    output.
    """
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")

    sealed = mask.astype(bool, copy=True) if radius < 1 else closing(mask, ball(radius))
    sealed = ndi.binary_fill_holes(sealed)

    if keep_largest:
        sealed = _keep_largest_component(sealed)
    return cast(np.ndarray, sealed)


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labels, component_count = ndi.label(mask)
    if component_count < 1:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    if component_count > 1:
        second_largest = int(np.delete(sizes, largest_label).max())
        if second_largest >= max(100, sizes[largest_label] * 0.01):
            logger.warning(
                "Sealing found %d connected components; keeping the largest (%d voxels) "
                "and dropping %d other(s) (%d voxels in total)",
                component_count,
                sizes[largest_label],
                component_count - 1,
                sizes.sum() - sizes[largest_label],
            )
    return cast(np.ndarray, labels == largest_label)


class MaskSealer(SegmentationMaskPostprocessor):
    def __init__(self, radius: int = 4) -> None:
        self._radius = radius

    def _process(self, mask: SegmentationMask) -> SegmentationMask:
        return SegmentationMask(mask=seal_mask(mask.mask, self._radius))
