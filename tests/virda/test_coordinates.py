"""Unit tests for the coordinate models and the frame conversions."""

import numpy as np
import pytest

from virda.geometry import FrameContext, Geometry, convert, convert_point, match_fiducial_name
from virda.models.coordinates import CoordinateSystem, Point
from virda.models.fiducial import FiducialType

# NAS on the +y axis, LPA/RPA symmetric on the x axis: the fiducials
# frame axes come out axis-aligned.
_NAS = np.array([0.0, 100.0, 0.0])
_LPA = np.array([-70.0, 0.0, 0.0])
_RPA = np.array([70.0, 0.0, 0.0])
_CENTROID = np.array([1.0, 2.0, 3.0])
#: A scaling + translation affine, as a NIfTI volume would carry.
_AFFINE = np.array(
    [
        [0.5, 0.0, 0.0, 10.0],
        [0.0, 0.5, 0.0, -5.0],
        [0.0, 0.0, 0.5, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _world(xyz: np.ndarray) -> Point:
    """A point in the world frame."""
    return Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]), system=CoordinateSystem.WORLD)


def _fiducials_ctx() -> FrameContext:
    """A context with all three canonical fiducials, in world."""
    return FrameContext(
        head_origin=_world(_CENTROID),
        fiducials={
            FiducialType.NAS: _world(_NAS),
            FiducialType.LPA: _world(_LPA),
            FiducialType.RPA: _world(_RPA),
        },
    )


class TestPoint:
    def test_as_array_keeps_the_order(self) -> None:
        point = Point(x=1.0, y=2.0, z=3.0, system=CoordinateSystem.WORLD)

        assert np.allclose(point.as_array(), [1.0, 2.0, 3.0])

    def test_system_is_mandatory(self) -> None:
        with pytest.raises(TypeError):
            Point(x=1.0, y=2.0, z=3.0)  # type: ignore[call-arg]

    def test_rejects_non_finite_coordinates(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            Point(x=float("nan"), y=0.0, z=0.0, system=CoordinateSystem.WORLD)
        with pytest.raises(ValueError, match="finite"):
            Point(x=0.0, y=float("inf"), z=0.0, system=CoordinateSystem.WORLD)

    def test_rejects_non_numeric_coordinates(self) -> None:
        with pytest.raises(ValueError, match="must be a number"):
            Point(x="1", y=0.0, z=0.0, system=CoordinateSystem.WORLD)  # type: ignore[arg-type]

    def test_rejects_an_unknown_system(self) -> None:
        with pytest.raises(ValueError, match="CoordinateSystem"):
            Point(x=0.0, y=0.0, z=0.0, system="world")  # type: ignore[arg-type]


class TestMatchFiducialName:
    def test_maps_aliases(self) -> None:
        assert match_fiducial_name("NAS") is FiducialType.NAS
        assert match_fiducial_name("  Nasion ") is FiducialType.NAS
        assert match_fiducial_name("NZ") is FiducialType.NAS
        assert match_fiducial_name("lpa") is FiducialType.LPA
        assert match_fiducial_name("RPA") is FiducialType.RPA

    def test_ignores_unknown_labels(self) -> None:
        assert match_fiducial_name("Extra") is None
        assert match_fiducial_name("") is None


class TestConvert:
    @pytest.mark.parametrize(
        "frame",
        [CoordinateSystem.WORLD, CoordinateSystem.HEAD, CoordinateSystem.FIDUCIALS],
    )
    def test_round_trip_world_frame_world(self, frame: CoordinateSystem) -> None:
        rng = np.random.default_rng(42)
        points = rng.uniform(-100.0, 100.0, size=(7, 3))
        context = _fiducials_ctx()

        converted = convert(points, CoordinateSystem.WORLD, frame, context)
        restored = convert(converted, frame, CoordinateSystem.WORLD, context)

        assert np.allclose(restored, points)

    def test_round_trip_through_the_voxel_frame(self) -> None:
        rng = np.random.default_rng(7)
        points = rng.uniform(-50.0, 50.0, size=(5, 3))
        context = FrameContext(affine=_AFFINE)

        voxels = convert(points, CoordinateSystem.WORLD, CoordinateSystem.VOXEL, context)
        restored = convert(voxels, CoordinateSystem.VOXEL, CoordinateSystem.WORLD, context)

        assert np.allclose(restored, points)

    def test_head_frame_is_a_world_translation(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
        context = FrameContext(head_origin=_world(_CENTROID))

        converted = convert(points, CoordinateSystem.WORLD, CoordinateSystem.HEAD, context)

        assert np.allclose(converted, points - _CENTROID)

    def test_head_frame_needs_the_centroid(self) -> None:
        with pytest.raises(ValueError, match="scalp mesh centroid"):
            convert(np.zeros((1, 3)), CoordinateSystem.WORLD, CoordinateSystem.HEAD, FrameContext())

    def test_voxel_frame_needs_the_affine(self) -> None:
        with pytest.raises(ValueError, match="affine"):
            convert(
                np.zeros((1, 3)),
                CoordinateSystem.WORLD,
                CoordinateSystem.VOXEL,
                FrameContext(),
            )

    def test_fiducials_frame_places_nas_at_the_origin(self) -> None:
        points = np.vstack([_NAS, _LPA, _RPA])

        converted = convert(
            points, CoordinateSystem.WORLD, CoordinateSystem.FIDUCIALS, _fiducials_ctx()
        )

        assert np.allclose(converted[0], [0.0, 0.0, 0.0])
        assert np.allclose(converted[1], [-70.0, -100.0, 0.0])
        assert np.allclose(converted[2], [70.0, -100.0, 0.0])

    def test_fiducials_frame_needs_all_three_fiducials(self) -> None:
        incomplete = FrameContext(
            fiducials={FiducialType.NAS: _world(_NAS), FiducialType.LPA: _world(_LPA)}
        )

        with pytest.raises(ValueError, match="NAS, LPA and RPA"):
            convert(_NAS, CoordinateSystem.WORLD, CoordinateSystem.FIDUCIALS, incomplete)
        with pytest.raises(ValueError, match="NAS, LPA and RPA"):
            convert(_NAS, CoordinateSystem.FIDUCIALS, CoordinateSystem.WORLD, FrameContext())

    def test_fiducials_frame_rejects_collinear_fiducials(self) -> None:
        collinear = FrameContext(
            fiducials={
                FiducialType.NAS: _world(np.zeros(3)),
                FiducialType.LPA: _world(np.array([-1.0, 0.0, 0.0])),
                FiducialType.RPA: _world(np.array([1.0, 0.0, 0.0])),
            }
        )

        with pytest.raises(ValueError, match="zero-length"):
            convert(np.zeros(3), CoordinateSystem.WORLD, CoordinateSystem.FIDUCIALS, collinear)

    def test_fiducials_in_other_frames_are_converted_first(self) -> None:
        context = FrameContext(
            fiducials={
                FiducialType.NAS: _world(_NAS),
                FiducialType.LPA: Point(x=-70.0, y=0.0, z=0.0, system=CoordinateSystem.VOXEL),
                FiducialType.RPA: _world(_RPA),
            },
            affine=_AFFINE,
        )

        converted = convert(_NAS, CoordinateSystem.WORLD, CoordinateSystem.FIDUCIALS, context)

        assert np.allclose(converted, [0.0, 0.0, 0.0], atol=1e-9)

    def test_single_point_is_squeezed(self) -> None:
        converted = convert(
            np.array([1.0, 2.0, 3.0]),
            CoordinateSystem.WORLD,
            CoordinateSystem.WORLD,
            FrameContext(),
        )

        assert converted.shape == (3,)

    def test_point_arrays_are_validated(self) -> None:
        with pytest.raises(ValueError, match="\\(N, 3\\)"):
            convert(np.zeros(4), CoordinateSystem.WORLD, CoordinateSystem.WORLD, FrameContext())
        with pytest.raises(ValueError, match="\\(N, 3\\)"):
            convert(
                np.zeros((2, 2)), CoordinateSystem.WORLD, CoordinateSystem.WORLD, FrameContext()
            )


class TestConvertPoint:
    def test_converts_a_point_into_another_frame(self) -> None:
        point = _world(_NAS)

        converted = convert_point(point, CoordinateSystem.FIDUCIALS, _fiducials_ctx())

        assert converted.system is CoordinateSystem.FIDUCIALS
        assert np.allclose(converted.as_array(), [0.0, 0.0, 0.0])

    def test_keeps_the_source_frame_when_asked_for(self) -> None:
        point = _world(_NAS)

        converted = convert_point(point, CoordinateSystem.WORLD, _fiducials_ctx())

        assert converted.system is CoordinateSystem.WORLD
        assert np.allclose(converted.as_array(), _NAS)


class TestGeometry:
    def test_world_is_always_available(self) -> None:
        geometry = Geometry(FrameContext())

        assert geometry.is_available(CoordinateSystem.WORLD)
        assert np.allclose(
            geometry.from_world(np.array([[1.0, 2.0, 3.0]]), CoordinateSystem.WORLD),
            [[1.0, 2.0, 3.0]],
        )

    def test_head_needs_the_centroid(self) -> None:
        assert not Geometry(FrameContext()).is_available(CoordinateSystem.HEAD)
        geometry = Geometry(FrameContext(head_origin=_world(_CENTROID)))

        assert geometry.is_available(CoordinateSystem.HEAD)

    def test_voxel_needs_the_affine(self) -> None:
        assert not Geometry(FrameContext()).is_available(CoordinateSystem.VOXEL)
        geometry = Geometry(FrameContext(affine=_AFFINE))

        assert geometry.is_available(CoordinateSystem.VOXEL)

    def test_fiducials_need_all_three_slots(self) -> None:
        partial = FrameContext(fiducials={FiducialType.NAS: _world(_NAS)})

        assert not Geometry(partial).is_available(CoordinateSystem.FIDUCIALS)
        assert Geometry(_fiducials_ctx()).is_available(CoordinateSystem.FIDUCIALS)

    def test_fiducials_in_voxel_need_the_affine(self) -> None:
        context = FrameContext(
            fiducials={
                FiducialType.NAS: _world(_NAS),
                FiducialType.LPA: Point(x=-70.0, y=0.0, z=0.0, system=CoordinateSystem.VOXEL),
                FiducialType.RPA: _world(_RPA),
            }
        )

        assert not Geometry(context).is_available(CoordinateSystem.FIDUCIALS)

    def test_the_context_can_be_replaced(self) -> None:
        geometry = Geometry(FrameContext())
        assert not geometry.is_available(CoordinateSystem.HEAD)

        geometry.set_context(FrameContext(head_origin=_world(_CENTROID)))

        assert geometry.is_available(CoordinateSystem.HEAD)
        assert np.allclose(
            geometry.from_world(_CENTROID, CoordinateSystem.HEAD),
            [0.0, 0.0, 0.0],
        )
