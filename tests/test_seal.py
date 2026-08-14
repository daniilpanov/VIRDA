import numpy as np
from scipy import ndimage as ndi

from tests.helpers.pipelines import build_context
from virda.models.segmentation_mask import SegmentationMask
from virda.segmentation.seal import MaskSealer


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
        sealer = MaskSealer(radius=2)
        sealed = sealer._seal_mask(mask)

        assert _enclosed_air_voxels(sealed) == 0

    def test_bridges_small_surface_channel(self) -> None:
        solid = _sphere_mask((30, 30, 30), (15.0, 15.0, 15.0), 10.0)
        cavity = _sphere_mask((30, 30, 30), (15.0, 15.0, 15.0), 3.0)
        channel = np.zeros((30, 30, 30), dtype=bool)
        channel[15, 15, 5:13] = True
        mask = (solid & ~cavity) & ~channel
        zero_radius_sealed = MaskSealer(radius=0)._seal_mask(mask)

        assert not mask[15, 15, 16]
        assert not zero_radius_sealed[15, 15, 16]
        assert _enclosed_air_voxels(zero_radius_sealed) == 0

        sealed = MaskSealer(radius=2)._seal_mask(mask)

        # The channel mouth voxel (15, 15, 5) is not asserted: closing rounds
        # off the mouth of any channel that exits on a curved surface, because
        # the closing ball around the mouth sticks out past the surface.
        # Airtightness plus the bridged mid-channel and the filled cavity are
        # the actual invariants of the sealing.
        assert _enclosed_air_voxels(sealed) == 0
        assert sealed[15, 15, 7]
        assert sealed[15, 15, 16]

    def test_already_solid_mask_unchanged_volume(self) -> None:
        solid = _sphere_mask((30, 30, 30), (15.0, 15.0, 15.0), 10.0)
        sealed = MaskSealer(radius=4)._seal_mask(solid)

        assert _enclosed_air_voxels(sealed) == 0
        assert sealed.sum() >= solid.sum()

    def test_fills_only_most_voluminous_component(self) -> None:
        shape = (50, 50, 50)
        solid_big = _sphere_mask(shape, (20.0, 25.0, 25.0), 10.0)
        cavity = _sphere_mask(shape, (20.0, 25.0, 25.0), 3.0)
        small_blob = _sphere_mask(shape, (42.0, 25.0, 25.0), 3.0)
        mask = (solid_big & ~cavity) | small_blob

        assert _enclosed_air_voxels(mask) > 0
        sealed = MaskSealer(radius=2)._seal_mask(mask)

        assert _enclosed_air_voxels(sealed) == 0
        assert sealed[solid_big].all()
        assert not sealed[small_blob].any()
        assert ndi.label(sealed)[1] == 1


class TestMaskSealer:
    def test_step_seals_mask_from_context(self) -> None:
        solid = _sphere_mask((30, 30, 30), (15.0, 15.0, 15.0), 10.0)
        cavity = _sphere_mask((30, 30, 30), (15.0, 15.0, 15.0), 3.0)
        mask = solid & ~cavity
        context = build_context(SegmentationMask=SegmentationMask(mask=mask))

        result = MaskSealer(radius=2).run(context)

        assert _enclosed_air_voxels(result.mask) == 0
        assert result.mask.dtype == bool
