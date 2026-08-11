import numpy as np
import pytest

from virda.models.fiducial import Fiducial
from virda.segmentation.cap_cut import cut_mask, cut_plane_from_fiducials


def _fiducial(fiducial_id: str, x: float, y: float, z: float) -> Fiducial:
    return Fiducial(
        fiducial_id=fiducial_id,
        name=fiducial_id,
        coordinates=np.array([x, y, z], dtype=np.float64),
        coordinate_system="world",
    )


def _head_like_fiducials(nasion_s: float = 10.0) -> list[Fiducial]:
    return [
        _fiducial("NAS", 10.0, 18.0, nasion_s),
        _fiducial("LPA", 2.0, 12.0, 10.0),
        _fiducial("RPA", 18.0, 12.0, 10.0),
    ]


class TestCutPlaneFromFiducials:
    def test_plane_normal_points_superior(self) -> None:
        normal, point = cut_plane_from_fiducials(_head_like_fiducials(), offset_mm=5.0)

        assert np.dot(normal, [0.0, 0.0, 1.0]) > 0.9
        assert point[2] == pytest.approx(10.0 - 5.0)

    def test_offset_places_plane_below_nasion(self) -> None:
        normal, point = cut_plane_from_fiducials(_head_like_fiducials(), offset_mm=30.0)
        assert point[2] == pytest.approx(10.0 - 30.0)
        assert np.linalg.norm(normal) == pytest.approx(1.0)

    def test_missing_fiducial_raises(self) -> None:
        fiducials = [_fiducial("NAS", 10.0, 18.0, 10.0)]
        with pytest.raises(ValueError, match="LPA"):
            cut_plane_from_fiducials(fiducials)

    def test_collinear_fiducials_raise(self) -> None:
        fiducials = [
            _fiducial("NAS", 10.0, 10.0, 10.0),
            _fiducial("LPA", 0.0, 10.0, 10.0),
            _fiducial("RPA", 20.0, 10.0, 10.0),
        ]
        with pytest.raises(ValueError, match="collinear"):
            cut_plane_from_fiducials(fiducials)


class TestCutMask:
    def test_keeps_voxels_above_plane(self) -> None:
        mask = np.ones((20, 20, 20), dtype=bool)
        cut = cut_mask(mask, np.eye(4), _head_like_fiducials(), offset_mm=5.0)

        assert cut.sum() == 15 * 20 * 20
        assert cut[:, :, :4].sum() == 0
        assert cut[:, :, 5:].sum() == 15 * 20 * 20

    def test_higher_offset_cuts_less(self) -> None:
        mask = np.ones((20, 20, 20), dtype=bool)
        shallow = cut_mask(mask, np.eye(4), _head_like_fiducials(), offset_mm=2.0)
        deep = cut_mask(mask, np.eye(4), _head_like_fiducials(), offset_mm=8.0)

        assert deep.sum() > shallow.sum()

    def test_respects_affine_transform(self) -> None:
        affine = np.eye(4)
        affine[0, 0] = 2.0  # 2 mm voxels along x
        mask = np.ones((20, 20, 20), dtype=bool)
        cut = cut_mask(mask, affine, _head_like_fiducials(), offset_mm=5.0)

        assert cut.sum() == 15 * 20 * 20

    def test_empty_mask_stays_empty(self) -> None:
        mask = np.zeros((10, 10, 10), dtype=bool)
        cut = cut_mask(mask, np.eye(4), _head_like_fiducials(), offset_mm=5.0)

        assert not np.any(cut)
