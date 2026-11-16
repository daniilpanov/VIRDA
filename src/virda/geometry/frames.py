"""Coordinate-frame conversions between world, voxel, head and fiducials frames.

World is the canonical frame of the whole application: scanner surface-RAS
millimetres -- every mesh and every stored coordinate lives in it. The
other frames are views of it:

- ``voxel``: the NIfTI voxel index space, reached through the volume
  affine;
- ``head``: the world translated so the origin sits at the head centre of
  mass (the centroid of the active scalp mesh). The axes stay
  world-aligned;
- ``fiducials``: an anatomical frame built from the three fiducials NAS,
  LPA and RPA: the origin is NAS, the right axis is
  ``normalize(RPA - LPA)``, the anterior axis is
  ``normalize(NAS - midpoint(LPA, RPA))`` Gram-Schmidt-orthogonalized
  against the right axis, and the up axis is ``right x anterior``. With
  ``R`` the matrix whose columns are ``(right, anterior, up)``, a world
  point ``p`` maps to ``R.T @ (p - origin)``.

The fiducials that define the anatomical frame must not themselves be
expressed in it -- they *are* its definition.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from virda.geometry.transforms import apply_transform
from virda.models.coordinates import CoordinateSystem, Point
from virda.models.fiducial import SLOT_IDS, FiducialType, match_fiducial_name

__all__ = ["SLOT_IDS", "FrameContext", "convert", "convert_point", "match_fiducial_name"]


@dataclass(frozen=True)
class FrameContext:
    """World-frame anchors the frame conversions need.

    ``head_origin`` is the head centre of mass in world coordinates;
    ``fiducials`` maps the canonical slots to their points, each carrying
    its own coordinate system (they are converted to world on use). A
    missing anchor makes the corresponding frame conversion unavailable
    (``ValueError`` on use).
    """

    head_origin: Point | None = None
    fiducials: Mapping[FiducialType, Point] = field(default_factory=dict)
    affine: np.ndarray | None = None


def convert(
    points: np.ndarray,
    source: CoordinateSystem,
    target: CoordinateSystem,
    context: FrameContext,
) -> np.ndarray:
    """Convert points from one frame into another.

    Accepts ``(N, 3)`` arrays or a single ``(3,)`` point (returned as
    ``(3,)``).
    """
    pts, single = _as_points(points)
    if source is target:
        converted = pts.copy()
    elif target is CoordinateSystem.WORLD:
        converted = _to_world(pts, source, context)
    elif source is CoordinateSystem.WORLD:
        converted = _from_world(pts, target, context)
    else:
        converted = _from_world(_to_world(pts, source, context), target, context)
    return converted[0] if single else converted


def convert_point(point: Point, target: CoordinateSystem, context: FrameContext) -> Point:
    """Convert one point into another frame."""
    coords = convert(point.as_array(), point.system, target, context)
    return Point(x=float(coords[0]), y=float(coords[1]), z=float(coords[2]), system=target)


def _as_points(points: np.ndarray) -> tuple[np.ndarray, bool]:
    """Validate point input: ``(N, 3)`` or a single ``(3,)`` point.

    Returns the points as ``(N, 3)`` plus whether the input was a single
    point (the result is squeezed back to ``(3,)``).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1 and pts.shape == (3,):
        return pts.reshape(1, 3), True
    if pts.ndim == 2 and pts.shape[1] == 3:
        return pts, False
    raise ValueError(f"expected an (N, 3) point array, got shape {pts.shape}")


def _to_world(points: np.ndarray, source: CoordinateSystem, context: FrameContext) -> np.ndarray:
    """World coordinates of points given in ``source``."""
    if source is CoordinateSystem.WORLD:
        return points.copy()
    if source is CoordinateSystem.VOXEL:
        affine = _affine(context)
        return apply_transform(points, affine)
    if source is CoordinateSystem.HEAD:
        return np.asarray(points + _head_origin(context), dtype=np.float64)
    origin, rotation = _fiducials_frame(context)
    return np.asarray(origin + points @ rotation.T, dtype=np.float64)


def _from_world(points: np.ndarray, target: CoordinateSystem, context: FrameContext) -> np.ndarray:
    """Coordinates of world points in ``target``."""
    if target is CoordinateSystem.WORLD:
        return points.copy()
    if target is CoordinateSystem.VOXEL:
        inverse = np.linalg.inv(_affine(context))
        return apply_transform(points, inverse)
    if target is CoordinateSystem.HEAD:
        return np.asarray(points - _head_origin(context), dtype=np.float64)
    origin, rotation = _fiducials_frame(context)
    return np.asarray((points - origin) @ rotation, dtype=np.float64)


def _world_point(point: Point, context: FrameContext) -> np.ndarray:
    """The world coordinates of one point."""
    world = _to_world(point.as_array().reshape(1, 3), point.system, context)
    return np.asarray(world[0], dtype=np.float64)


def _head_origin(context: FrameContext) -> np.ndarray:
    """The head centre of mass in world coordinates."""
    origin = context.head_origin
    if origin is None:
        raise ValueError("head frame needs the scalp mesh centroid")
    return origin.as_array()


def _affine(context: FrameContext) -> np.ndarray:
    """The NIfTI affine of the volume."""
    affine = context.affine
    if affine is None:
        raise ValueError("voxel frame needs the NIfTI volume affine")
    return np.asarray(affine, dtype=np.float64)


def _fiducials_frame(context: FrameContext) -> tuple[np.ndarray, np.ndarray]:
    """The fiducials frame as ``(origin, rotation)`` in world terms.

    The rotation's columns are the unit right, anterior and up axes
    expressed in world coordinates. The fiducials may sit in any frame;
    they are converted to world here.
    """
    missing = [slot.value for slot in SLOT_IDS if slot not in context.fiducials]
    if missing:
        raise ValueError(f"fiducials frame needs NAS, LPA and RPA coordinates (missing {missing})")
    nas, lpa, rpa = (_world_point(context.fiducials[slot], context) for slot in SLOT_IDS)
    origin = np.asarray(nas, dtype=np.float64)
    right = _unit(rpa - lpa)
    anterior = _unit(origin - (lpa + rpa) / 2.0)
    anterior = _unit(anterior - np.dot(anterior, right) * right)  # Gram-Schmidt
    up = np.cross(right, anterior)
    return origin, np.column_stack((right, anterior, up))


def _unit(vector: np.ndarray) -> np.ndarray:
    """Normalize ``vector``; zero vectors raise ``ValueError``."""
    length = float(np.linalg.norm(vector))
    if length == 0.0:
        raise ValueError("cannot normalize a zero-length direction")
    return np.asarray(vector, dtype=np.float64) / length
