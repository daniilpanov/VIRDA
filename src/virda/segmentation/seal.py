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
"""

from typing import cast

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import ball, closing


def seal_mask(mask: np.ndarray, radius: int = 4) -> np.ndarray:
    """Return a solid version of ``mask`` with surface openings bridged.

    ``radius`` (in voxels) is the closing ball radius; it must be large enough
    to bridge the narrowest channels that lead from interior cavities to the
    exterior (nostrils, ear canals), yet small enough not to erode thin scalp
    features. For ~1 mm isotropic data ``radius=4`` works.
    """
    if radius < 1:
        return cast(np.ndarray, ndi.binary_fill_holes(mask))
    sealed = closing(mask, ball(radius))
    sealed = ndi.binary_fill_holes(sealed)
    return cast(np.ndarray, sealed)
