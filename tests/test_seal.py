import numpy as np
from scipy import ndimage as ndi

from virda.segmentation.seal import seal_mask


def _enclosed_air_voxels(mask: np.ndarray) -> int:
    air_labels, _ = ndi.label(~mask)
    exterior_label = air_labels[0, 0, 0]
    sizes = np.bincount(air_labels.ravel())
    labels = range(1, int(air_labels.max()) + 1)
    return int(sum(sizes[label] for label in labels if label != exterior_label))


def _sphere_mask(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radius: float,
) -> np.ndarray:
    grid = np.indices(shape)
    distance_sq = np.sum((grid - np.asarray(center).reshape(-1, 1, 1, 1)) ** 2, axis=0)
    return distance_sq <= radius**2


class TestSealMask:
    def test_fills_internal_cavity(self) -> None:
        solid = _sphere_mask((30, 30, 30), (15.0, 15.0, 15.0), 10.0)
        cavity = _sphere_mask((30, 30, 30), (15.0, 15.0, 15.0), 3.0)
        mask = solid & ~cavity

        assert _enclosed_air_voxels(mask) > 0
        sealed = seal_mask(mask, radius=2)

        assert _enclosed_air_voxels(sealed) == 0

    def test_bridges_small_surface_channel(self) -> None:
        solid = _sphere_mask((30, 30, 30), (15.0, 15.0, 15.0), 10.0)
        cavity = _sphere_mask((30, 30, 30), (15.0, 15.0, 15.0), 3.0)
        channel = np.zeros((30, 30, 30), dtype=bool)
        channel[15, 15, 4:7] = True
        mask = (solid & ~cavity) | channel

        sealed = seal_mask(mask, radius=2)

        assert _enclosed_air_voxels(sealed) == 0
        assert sealed[15, 15, 5]  # the channel is now tissue

    def test_radius_zero_only_fills_enclosed(self) -> None:
        solid = _sphere_mask((30, 30, 30), (15.0, 15.0, 15.0), 10.0)
        cavity = _sphere_mask((30, 30, 30), (15.0, 15.0, 15.0), 3.0)
        mask = solid & ~cavity

        sealed = seal_mask(mask, radius=0)

        assert _enclosed_air_voxels(sealed) == 0

    def test_already_solid_mask_unchanged_volume(self) -> None:
        solid = _sphere_mask((30, 30, 30), (15.0, 15.0, 15.0), 10.0)
        sealed = seal_mask(solid, radius=4)

        assert _enclosed_air_voxels(sealed) == 0
        assert sealed.sum() >= solid.sum()
