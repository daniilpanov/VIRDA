"""The geometry service: frame conversions and mesh queries over one context."""

from collections.abc import Mapping

import numpy as np

from virda.geometry.frames import SLOT_IDS, FrameContext, convert, convert_point
from virda.geometry.measurement import DUPLICATE_TOLERANCE_MM, predict_measurement
from virda.geometry.surface import SurfaceProbe
from virda.models.coordinates import CoordinateSystem, Point
from virda.models.fiducial import FiducialType
from virda.models.measurement import MeasurementPrediction
from virda.models.mesh import Mesh


class Geometry:
    """Frame conversions and mesh queries over one context.

    The context -- the head origin, the fiducials and the voxel affine --
    is held by the object, so callers do not have to thread it through
    every call. The GUI keeps one instance alive and refreshes its
    context when the project state changes (:meth:`set_context`); the
    holders of the reference therefore always see the current anchors.
    """

    def __init__(self, context: FrameContext) -> None:
        self._context = context
        self._probe: SurfaceProbe | None = None
        self._probe_mesh: Mesh | None = None

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

    # --- surface ---

    def distance_to_surface(self, point: Point, mesh: Mesh) -> float:
        """The distance from the point to the mesh surface, in millimetres.

        The point may be expressed in any frame; it is converted into the
        world frame the mesh lives in.
        """
        world = self.convert_point(point, CoordinateSystem.WORLD)
        return self.probe(mesh).distance(world.as_array())

    def point_on_surface(self, point: Point, mesh: Mesh, *, tolerance_mm: float = 2.0) -> bool:
        """Whether the point lies on the mesh surface within the tolerance."""
        return self.distance_to_surface(point, mesh) <= tolerance_mm

    def nearest_vertex(self, point: Point, mesh: Mesh) -> tuple[int, np.ndarray]:
        """The index and the coordinates of the mesh vertex closest to the point."""
        world = self.convert_point(point, CoordinateSystem.WORLD)
        return self.probe(mesh).nearest_vertex(world.as_array())

    def probe(self, mesh: Mesh) -> SurfaceProbe:
        """The proximity probe of a mesh; the last one is kept alive."""
        if self._probe is None or self._probe_mesh is not mesh:
            self._probe = SurfaceProbe(mesh)
            self._probe_mesh = mesh
        return self._probe

    # --- measurements ---

    def predict_measurement(
        self,
        measured: Mapping[FiducialType, float],
        fiducials: Mapping[FiducialType, Point],
        mesh: Mesh,
        missing: FiducialType,
        *,
        approximate: bool = True,
        system: CoordinateSystem = CoordinateSystem.WORLD,
        duplicate_tolerance_mm: float = DUPLICATE_TOLERANCE_MM,
    ) -> MeasurementPrediction:
        """Predict the distance to ``missing`` from two measured distances.

        The electrode lies on the mesh surface: the two measured distances
        put it on a circle, the surface cuts that circle into candidates
        and the electrode side of the fiducial plane picks the right one.
        ``approximate`` snaps the position to the closest mesh vertex --
        the accuracy the mesh density controls.
        """
        return predict_measurement(
            measured,
            fiducials,
            mesh,
            missing,
            self._context,
            approximate=approximate,
            system=system,
            duplicate_tolerance_mm=duplicate_tolerance_mm,
            probe=self.probe(mesh),
        )
