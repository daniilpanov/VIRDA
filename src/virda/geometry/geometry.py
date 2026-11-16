"""The geometry service: frame conversions over one fixed context."""

import numpy as np

from virda.geometry.frames import SLOT_IDS, FrameContext, convert, convert_point
from virda.models.coordinates import CoordinateSystem, Point


class Geometry:
    """Frame conversions over one context.

    The context -- the head origin, the fiducials and the voxel affine --
    is held by the object, so callers do not have to thread it through
    every conversion. The GUI keeps one instance alive and refreshes its
    context when the project state changes (:meth:`set_context`); the
    holders of the reference therefore always see the current anchors.
    """

    def __init__(self, context: FrameContext) -> None:
        self._context = context

    @property
    def context(self) -> FrameContext:
        """The anchors this service converts with."""
        return self._context

    def set_context(self, context: FrameContext) -> None:
        """Replace the anchors; the holders of this object see them at once."""
        self._context = context

    def convert(
        self,
        points: np.ndarray,
        source: CoordinateSystem,
        target: CoordinateSystem,
    ) -> np.ndarray:
        """Convert points from one frame into another."""
        return convert(points, source, target, self._context)

    def to_world(self, points: np.ndarray, source: CoordinateSystem) -> np.ndarray:
        """World coordinates of points given in ``source``."""
        return self.convert(points, source, CoordinateSystem.WORLD)

    def from_world(self, points: np.ndarray, target: CoordinateSystem) -> np.ndarray:
        """Coordinates of world points in ``target``."""
        return self.convert(points, CoordinateSystem.WORLD, target)

    def convert_point(self, point: Point, target: CoordinateSystem) -> Point:
        """Convert one point into another frame."""
        return convert_point(point, target, self._context)

    def is_available(self, system: CoordinateSystem) -> bool:
        """Whether conversions into ``system`` can run with this context."""
        if system is CoordinateSystem.WORLD:
            return True
        if system is CoordinateSystem.HEAD:
            return self._context.head_origin is not None
        if system is CoordinateSystem.VOXEL:
            return self._context.affine is not None
        if not all(slot in self._context.fiducials for slot in SLOT_IDS):
            return False
        # A fiducial given in voxel coordinates needs the affine to reach
        # the world frame the anatomical frame is built in.
        needs_affine = any(
            point.system is CoordinateSystem.VOXEL for point in self._context.fiducials.values()
        )
        return not needs_affine or self._context.affine is not None
