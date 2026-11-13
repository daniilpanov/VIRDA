"""Unit tests for the coordinate-frame conversion service."""

from typing import Any, cast

import numpy as np
import pytest

from tests.virda_gui_new.fixtures.steps import advance_to
from virda.models.mesh import Mesh
from virda_gui_new.dto.steps import STEPS
from virda_gui_new.dto.viewer3d import LabeledPoints
from virda_gui_new.services.coordinates import (
    CoordinateFrame,
    CoordinateService,
    FrameContext,
    fiducials_frame,
    from_world,
    match_fiducial_name,
    to_world,
)
from virda_gui_new.services.steps_manager import StepsManager

# NAS on the +y axis, LPA/RPA symmetric on the x axis: the fiducials
# frame axes come out axis-aligned.
_NAS = np.array([0.0, 100.0, 0.0])
_LPA = np.array([-70.0, 0.0, 0.0])
_RPA = np.array([70.0, 0.0, 0.0])
_CENTROID = np.array([1.0, 2.0, 3.0])


def _fiducials_ctx() -> FrameContext:
    return FrameContext(
        head_origin=_CENTROID,
        fiducials={"NAS": _NAS, "LPA": _LPA, "RPA": _RPA},
    )


def test_match_fiducial_name_maps_aliases() -> None:
    assert match_fiducial_name("NAS") == "NAS"
    assert match_fiducial_name("  Nasion ") == "NAS"
    assert match_fiducial_name("NZ") == "NAS"
    assert match_fiducial_name("lpa") == "LPA"
    assert match_fiducial_name("RPA") == "RPA"
    assert match_fiducial_name("Extra") is None
    assert match_fiducial_name("") is None


def test_fiducials_frame_axes_are_axis_aligned_for_symmetric_fiducials() -> None:
    origin, rotation = fiducials_frame(_NAS, _LPA, _RPA)

    assert np.allclose(origin, _NAS)
    assert np.allclose(rotation, np.eye(3), atol=1e-12)


def test_fiducials_frame_rejects_collinear_fiducials() -> None:
    with pytest.raises(ValueError, match="zero-length"):
        fiducials_frame(np.zeros(3), np.array([-1.0, 0, 0]), np.array([1.0, 0, 0]))


@pytest.mark.parametrize(
    ("frame", "ctx"),
    [
        (CoordinateFrame.WORLD, FrameContext()),
        (CoordinateFrame.HEAD, FrameContext(head_origin=_CENTROID)),
        (CoordinateFrame.FIDUCIALS, _fiducials_ctx()),
    ],
)
def test_round_trip_world_frame_world(frame: CoordinateFrame, ctx: FrameContext) -> None:
    rng = np.random.default_rng(42)
    points = rng.uniform(-100.0, 100.0, size=(7, 3))

    converted = from_world(points, frame, ctx)
    restored = to_world(converted, frame, ctx)

    assert np.allclose(restored, points)


def test_head_frame_is_a_world_translation() -> None:
    points = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])

    converted = from_world(points, CoordinateFrame.HEAD, FrameContext(head_origin=_CENTROID))

    assert np.allclose(converted, points - _CENTROID)
    with pytest.raises(ValueError, match="scalp mesh centroid"):
        from_world(points, CoordinateFrame.HEAD, FrameContext())


def test_fiducials_frame_places_nas_at_the_origin() -> None:
    points = np.vstack([_NAS, _LPA, _RPA])

    converted = from_world(points, CoordinateFrame.FIDUCIALS, _fiducials_ctx())

    assert np.allclose(converted[0], [0.0, 0.0, 0.0])
    assert np.allclose(converted[1], [-70.0, -100.0, 0.0])
    assert np.allclose(converted[2], [70.0, -100.0, 0.0])


def test_fiducials_frame_needs_all_three_fiducials() -> None:
    incomplete = FrameContext(fiducials={"NAS": _NAS, "LPA": _LPA})

    with pytest.raises(ValueError, match="NAS, LPA and RPA"):
        from_world(_NAS, CoordinateFrame.FIDUCIALS, incomplete)
    with pytest.raises(ValueError, match="NAS, LPA and RPA"):
        to_world(_NAS, CoordinateFrame.FIDUCIALS, FrameContext())


def test_point_arrays_are_validated() -> None:
    with pytest.raises(ValueError, match="\\(N, 3\\)"):
        from_world(np.zeros(4), CoordinateFrame.WORLD, FrameContext())
    with pytest.raises(ValueError, match="\\(N, 3\\)"):
        from_world(np.zeros((2, 2)), CoordinateFrame.WORLD, FrameContext())


def _make_manager(mesh: Mesh | None = None, fiducials: LabeledPoints | None = None) -> StepsManager:
    """Manager advanced to step 4 with optional mesh and fiducial inputs."""
    steps = StepsManager(tuple(step.required_keys for step in STEPS))
    advance_to(steps, 4)
    if mesh is not None:
        steps.set_step_context_item(1, "scalp_surface", mesh)
        steps.set_step_context_item(2, "mesh", mesh)
    if fiducials is not None:
        steps.set_global_item("fiducials", fiducials)
    return steps


def test_service_head_origin_is_the_mesh_centroid() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    mesh = Mesh(vertices, np.zeros((1, 3), dtype=int))
    service = CoordinateService(_make_manager(mesh=mesh))

    assert np.allclose(cast("Any", service.head_origin()), vertices.mean(axis=0))
    assert service.is_available(CoordinateFrame.HEAD)


def test_service_caches_head_origin_per_mesh() -> None:
    first = Mesh(np.zeros((4, 3)), np.zeros((1, 3), dtype=int))
    second = Mesh(np.ones((4, 3)), np.zeros((1, 3), dtype=int))
    manager = _make_manager(mesh=first)
    service = CoordinateService(manager)

    cached = service.head_origin()
    assert service.head_origin() is cached  # same mesh object: cached value

    manager.set_step_context_item(2, "mesh", second)
    recomputed = service.head_origin()
    assert recomputed is not cached
    assert np.allclose(cast("Any", recomputed), np.ones(3))


def test_service_prefers_step_2_mesh_over_step_1_scalp_surface() -> None:
    base = Mesh(np.zeros((4, 3)), np.zeros((1, 3), dtype=int))
    final = Mesh(np.ones((4, 3)), np.zeros((1, 3), dtype=int))
    manager = _make_manager()
    manager.set_step_context_item(1, "scalp_surface", base)
    service = CoordinateService(manager)

    assert np.allclose(cast("Any", service.head_origin()), np.zeros(3))

    manager.set_step_context_item(2, "mesh", final)
    assert np.allclose(cast("Any", service.head_origin()), np.ones(3))


def test_service_head_needs_a_mesh() -> None:
    service = CoordinateService(_make_manager())

    assert service.head_origin() is None
    assert not service.is_available(CoordinateFrame.HEAD)
    with pytest.raises(ValueError, match="scalp mesh centroid"):
        service.from_world(np.zeros((1, 3)), CoordinateFrame.HEAD)


def test_service_fiducials_match_store_labels() -> None:
    points = LabeledPoints(
        np.array([_NAS, _LPA, _RPA, [9.0, 9.0, 9.0]]),
        ("NASION", "LPA", "rpa", "Extra"),
    )
    service = CoordinateService(_make_manager(fiducials=points))

    coords = service.fiducial_coordinates()
    assert coords is not None
    assert set(coords) == {"NAS", "LPA", "RPA"}
    assert np.allclose(coords["NAS"], _NAS)
    assert service.is_available(CoordinateFrame.FIDUCIALS)
    assert np.allclose(service.from_world(_NAS, CoordinateFrame.FIDUCIALS), [0.0, 0.0, 0.0])


def test_service_fiducials_empty_store_is_unavailable() -> None:
    service = CoordinateService(_make_manager())

    assert service.fiducial_coordinates() is None
    assert not service.is_available(CoordinateFrame.FIDUCIALS)


def test_service_world_is_always_available() -> None:
    service = CoordinateService(_make_manager())

    assert service.is_available(CoordinateFrame.WORLD)
    converted = service.from_world(np.array([[1.0, 2.0, 3.0]]), CoordinateFrame.WORLD)
    assert np.allclose(converted, [[1.0, 2.0, 3.0]])
