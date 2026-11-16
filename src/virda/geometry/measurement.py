"""Predict the missing measurement of an electrode from two measured ones.

Two measured distances to two known fiducials leave the electrode on a
circle: the intersection of the two spheres around the fiducials. The
mesh surface cuts that circle into a handful of points -- typically two,
on opposite sides of the head -- and the "electrode side" of the
fiducial plane tells which of them is the electrode.

The measurement of the third fiducial then follows from the position:
it is the distance from that position to the third fiducial.
"""

import math
from collections.abc import Mapping

import numpy as np

from virda.geometry.frames import FrameContext, convert_point
from virda.geometry.surface import (
    SurfaceProbe,
    circle_surface_intersections,
)
from virda.models.coordinates import CoordinateSystem, Point
from virda.models.fiducial import FiducialType
from virda.models.measurement import MeasurementPrediction
from virda.models.mesh import Mesh

#: How close two candidates must be to count as the same point.
DUPLICATE_TOLERANCE_MM = 1.0


def predict_measurement(
    measured: Mapping[FiducialType, float],
    fiducials: Mapping[FiducialType, Point],
    mesh: Mesh,
    missing: FiducialType,
    context: FrameContext,
    *,
    approximate: bool = True,
    system: CoordinateSystem = CoordinateSystem.WORLD,
    duplicate_tolerance_mm: float = DUPLICATE_TOLERANCE_MM,
    probe: SurfaceProbe | None = None,
) -> MeasurementPrediction:
    """Predict the distance to ``missing`` from the two measured distances.

    ``measured`` holds the two known distances (by fiducial id), the
    fiducials their positions, and ``mesh`` the surface the electrode
    lies on. ``approximate`` snaps the position to the closest mesh
    vertex -- the accuracy the mesh density controls -- while the exact
    intersection point is kept otherwise. The reported position is
    expressed in ``system``.
    """
    first, second = _measured_ids(measured, fiducials, missing)
    known = _world_points(fiducials, context)
    center, radius, normal = _circle(
        known[first],
        known[second],
        measured[first],
        measured[second],
    )

    points, _faces = circle_surface_intersections(center, radius, normal, mesh)
    points = _deduplicate(points, duplicate_tolerance_mm)
    if points.shape[0] == 0:
        return MeasurementPrediction(
            point=None,
            distance_mm=None,
            ambiguous=False,
            residual_mm=None,
            vertex_index=None,
            approximate=approximate,
        )

    candidates = _electrode_side_candidates(points, known[first], known[second], known[missing])
    ambiguous = candidates.shape[0] > 1
    chosen = _closest_to_origin(candidates, context)

    vertex_index: int | None = None
    if approximate:
        surface = probe if probe is not None else SurfaceProbe(mesh)
        vertex_index, position = surface.nearest_vertex(chosen)
    else:
        position = chosen

    residual = math.sqrt(
        sum(
            (float(np.linalg.norm(position - known[fiducial])) - measured[fiducial]) ** 2
            for fiducial in (first, second)
        )
        / 2.0
    )
    distance = float(np.linalg.norm(position - known[missing]))
    point = convert_point(
        Point(
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
            system=CoordinateSystem.WORLD,
        ),
        system,
        context,
    )
    return MeasurementPrediction(
        point=point,
        distance_mm=distance,
        ambiguous=ambiguous,
        residual_mm=residual,
        vertex_index=vertex_index,
        approximate=approximate,
    )


def _measured_ids(
    measured: Mapping[FiducialType, float],
    fiducials: Mapping[FiducialType, Point],
    missing: FiducialType,
) -> tuple[FiducialType, FiducialType]:
    """The two measured fiducials, checked against the known ones."""
    if len(measured) != 2:
        raise ValueError(f"exactly two measured distances are needed, got {len(measured)}")
    unknown = [fiducial for fiducial in measured if fiducial not in fiducials]
    if unknown:
        raise ValueError(f"no coordinates for the measured fiducials {unknown}")
    if missing not in fiducials:
        raise ValueError(f"no coordinates for the missing fiducial {missing!r}")
    if missing in measured:
        raise ValueError(f"the missing fiducial {missing!r} already has a measurement")
    first, second = tuple(measured)
    return first, second


def _world_points(
    fiducials: Mapping[FiducialType, Point],
    context: FrameContext,
) -> dict[FiducialType, np.ndarray]:
    """The fiducial positions in world coordinates."""
    return {
        fiducial: convert_point(point, CoordinateSystem.WORLD, context).as_array()
        for fiducial, point in fiducials.items()
    }


def _circle(
    first: np.ndarray,
    second: np.ndarray,
    d1: float,
    d2: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """The circle of the points at distance ``d1`` and ``d2`` from the fiducials.

    Returns the centre, the radius and the normal of the circle's plane.
    """
    axis = second - first
    separation = float(np.linalg.norm(axis))
    if separation == 0.0:
        raise ValueError("the two measured fiducials coincide")
    if not abs(d1 - d2) <= separation <= d1 + d2:
        raise ValueError(
            "the measured distances cannot belong to one point: "
            f"d1={d1}, d2={d2}, distance between the fiducials={separation}"
        )
    unit = axis / separation
    offset = (d1**2 - d2**2 + separation**2) / (2.0 * separation)
    squared = d1**2 - offset**2
    if squared < 0:
        raise ValueError("the measured distances cannot belong to one point")
    return first + offset * unit, math.sqrt(squared), unit


def _electrode_side_candidates(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
) -> np.ndarray:
    """The candidates on the electrode side of the fiducial plane.

    The plane through the three fiducials splits the head in two; the
    electrodes sit on the upper side of it. The normal is oriented
    upwards, so the side does not depend on the order the measured
    distances come in.
    """
    normal = np.cross(second - first, third - first)
    length = float(np.linalg.norm(normal))
    if length == 0.0:
        return points
    normal = normal / length
    if normal[2] < 0.0:
        normal = -normal

    sides = np.sign(np.einsum("ij,j->i", points - first, normal))
    selected = points[sides > 0.0]
    return selected if selected.shape[0] else points


def _closest_to_origin(candidates: np.ndarray, context: FrameContext) -> np.ndarray:
    """The candidate closest to the head centre (the tie-break).

    The head centre is the centre of mass of the scalp mesh, which the
    context carries. Without it the candidates are not ordered, and the
    first one is returned.
    """
    origin = context.head_origin
    if origin is None:
        return np.asarray(candidates[0], dtype=np.float64)
    distances = np.linalg.norm(candidates - origin.as_array(), axis=1)
    return np.asarray(candidates[int(np.argmin(distances))], dtype=np.float64)


def _deduplicate(points: np.ndarray, duplicate_tolerance_mm: float) -> np.ndarray:
    """Drop the points that repeat within the tolerance."""
    unique: list[np.ndarray] = []
    for point in points:
        if not any(
            float(np.linalg.norm(point - other)) <= duplicate_tolerance_mm for other in unique
        ):
            unique.append(point)
    return np.asarray(unique, dtype=np.float64).reshape(-1, 3)
