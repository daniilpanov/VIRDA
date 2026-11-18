"""Unit tests for the deterministic camera placement."""

import math

import pyvista as pv

from virda_gui_new.services.camera import (
    VIEW_ANGLE_DEG,
    camera_for_bounds,
    mesh_bounds,
)


def test_the_bounds_of_the_meshes_are_merged() -> None:
    first = pv.Box(bounds=(0, 1, 0, 1, 0, 1))
    second = pv.Box(bounds=(2, 3, -1, 0, 4, 5))

    assert mesh_bounds([first, second]) == ((0.0, -1.0, 0.0), (3.0, 1.0, 5.0))


def test_no_mesh_has_no_bounds() -> None:
    assert mesh_bounds([]) is None


def test_the_camera_looks_at_the_center_of_the_bounds() -> None:
    bounds = ((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))

    position, focal_point, up, angle = camera_for_bounds(bounds)

    assert focal_point == (5.0, 5.0, 5.0)
    assert angle == VIEW_ANGLE_DEG
    assert up == (0.0, 0.0, 1.0)
    assert position != focal_point


def test_the_camera_frames_the_whole_bounds() -> None:
    bounds = ((-30.0, -20.0, -10.0), (30.0, 20.0, 10.0))

    position, focal_point, _, angle = camera_for_bounds(bounds)

    radius = math.dist(bounds[0], bounds[1]) / 2.0
    distance = math.dist(position, focal_point)
    assert distance * math.sin(math.radians(angle) / 2.0) >= radius


def test_the_camera_keeps_the_requested_azimuth_and_elevation() -> None:
    bounds = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    position, focal_point, _, _ = camera_for_bounds(bounds, azimuth_deg=0.0, elevation_deg=0.0)

    assert position[0] > focal_point[0]
    assert position[1] == focal_point[1]
    assert position[2] == focal_point[2]
