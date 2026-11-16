"""The predicted measurement of an electrode."""

from dataclasses import dataclass

from virda.models.coordinates import Point


@dataclass(frozen=True)
class MeasurementPrediction:
    """The predicted electrode position and the missing measurement.

    ``point`` is the electrode position in the requested frame and
    ``distance_mm`` the predicted distance to the missing fiducial. Both
    are ``None`` when the surface does not meet the circle of the two
    measured distances.

    ``ambiguous`` marks a case where more than one candidate survived the
    electrode-side selection; the first one (the closest to the head
    centre) is the reported one. ``residual_mm`` is the RMS of the two
    measured-distance errors at the reported position -- it is zero for an
    exact intersection and shows what the vertex approximation costs.
    """

    point: Point | None
    distance_mm: float | None
    ambiguous: bool
    residual_mm: float | None
    vertex_index: int | None
    approximate: bool
