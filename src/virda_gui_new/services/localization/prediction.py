"""Predict the missing measurements of the localization popup.

Two measured distances to two fiducials put the electrode on a circle;
the ESE surface cuts that circle into the possible electrode positions
and the third distance follows from the position. The popup uses this
to fill the empty cells of its measurement grid.
"""

from virda.models.measurement import MeasurementPrediction

from virda.geometry import Geometry, match_fiducial_name
from virda.models.coordinates import Point
from virda.models.fiducial import SLOT_IDS, Fiducials, FiducialType
from virda.models.mesh import Mesh
from virda_gui_new.services.localization.build import MeasurementTable


def fiducial_points(fiducials: Fiducials) -> dict[FiducialType, Point]:
    """The canonical fiducial slots with their points."""
    points: dict[FiducialType, Point] = {}
    for fiducial in fiducials.items:
        slot = match_fiducial_name(fiducial.fiducial_id)
        if slot is not None and slot not in points:
            points[slot] = fiducial.point
    return points


def measured_slots(distances: dict[str, float], missing: FiducialType) -> dict[FiducialType, float]:
    """The measured distances of one electrode, keyed by the fiducial slot."""
    measured: dict[FiducialType, float] = {}
    for name, distance in distances.items():
        slot = match_fiducial_name(name)
        if slot is not None and slot is not missing:
            measured[slot] = float(distance)
    return measured


def predict_missing_measurement(
    distances: dict[str, float],
    missing: FiducialType,
    fiducials: Fiducials,
    mesh: Mesh,
    geometry: Geometry,
) -> MeasurementPrediction | None:
    """Predict the distance to ``missing`` from the two measured ones.

    ``None`` when the electrode does not have the two measurements the
    prediction needs, or when the measurements cannot belong to one
    point at all.
    """
    measured = measured_slots(distances, missing)
    if len(measured) != 2:
        return None
    # The measured distances that cannot belong to one point raise: the
    # caller tells the user why there is no prediction.
    return geometry.predict_measurement(
        measured,
        fiducial_points(fiducials),
        mesh,
        missing,
    )


def predict_missing_measurements(
    table: MeasurementTable,
    fiducials: Fiducials,
    mesh: Mesh,
    geometry: Geometry,
) -> tuple[dict[tuple[str, str], MeasurementPrediction], list[str]]:
    """The predictions of every missing measurement of a table.

    The key is the electrode id and the fiducial slot. The second value
    holds the electrodes whose measured distances cannot belong to one
    point: they get no prediction and are reported instead.
    """
    predictions: dict[tuple[str, str], MeasurementPrediction] = {}
    failed: list[str] = []
    for electrode, distances in table.items():
        for slot in SLOT_IDS:
            if slot.value in distances:
                continue
            try:
                prediction = predict_missing_measurement(distances, slot, fiducials, mesh, geometry)
            except ValueError:
                failed.append(electrode)
                continue
            if prediction is None or prediction.distance_mm is None:
                continue
            predictions[(electrode, slot.value)] = prediction
    return predictions, failed
