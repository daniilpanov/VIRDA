"""Deterministic camera placement over the bounds of the scene meshes.

The camera is placed by an explicit formula instead of by the plotter's
own fit: the fit frames every actor, and the NIfTI volume lives in a
different space than the meshes, so it swallows the head. The formula
takes the bounding box of the meshes only and returns the camera that
frames it.
"""

from collections.abc import Sequence

import numpy as np
import pyvista as pv

#: The azimuth (degrees around the vertical axis) the camera looks from.
DEFAULT_AZIMUTH_DEG = 30.0
#: The elevation (degrees above the horizontal plane) the camera looks from.
DEFAULT_ELEVATION_DEG = 20.0
#: The vertical field of view of the camera, in degrees.
VIEW_ANGLE_DEG = 30.0
#: How much room around the bounds the camera keeps.
PADDING = 1.2
#: The closest the camera comes to the bounds, in scene units.
MIN_DISTANCE = 1.0

Bounds = tuple[tuple[float, float, float], tuple[float, float, float]]
"""The bounding box of the scene: its low and its high corner."""

Placement = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    float,
]
"""The camera position, its focal point, its up vector and its view angle."""


def _point(values: np.ndarray) -> tuple[float, float, float]:
    """The three coordinates of a point as a fixed-size tuple."""
    return (float(values[0]), float(values[1]), float(values[2]))


def mesh_bounds(meshes: Sequence[pv.PolyData]) -> Bounds | None:
    """The bounding box of the given meshes; ``None`` when there is none."""
    boxes = [np.asarray(mesh.bounds, dtype=np.float64) for mesh in meshes]
    if not boxes:
        return None
    stacked = np.stack(boxes)
    low = stacked[:, (0, 2, 4)].min(axis=0)
    high = stacked[:, (1, 3, 5)].max(axis=0)
    return (_point(low), _point(high))


def camera_for_bounds(
    bounds: Bounds,
    *,
    azimuth_deg: float = DEFAULT_AZIMUTH_DEG,
    elevation_deg: float = DEFAULT_ELEVATION_DEG,
    padding: float = PADDING,
) -> Placement:
    """The camera that frames the given bounds."""
    low = np.asarray(bounds[0], dtype=np.float64)
    high = np.asarray(bounds[1], dtype=np.float64)
    center = (low + high) / 2.0
    radius = float(np.linalg.norm(high - low)) / 2.0

    half_angle = np.radians(VIEW_ANGLE_DEG) / 2.0
    distance = radius * padding / float(np.sin(half_angle))
    distance = max(distance, MIN_DISTANCE)

    azimuth = np.radians(azimuth_deg)
    elevation = np.radians(elevation_deg)
    direction = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    position = center + distance * direction
    up = (0.0, 0.0, 1.0) if abs(float(direction[2])) < 0.99 else (0.0, 1.0, 0.0)
    return (_point(position), _point(center), up, VIEW_ANGLE_DEG)
