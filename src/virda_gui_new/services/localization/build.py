"""Build validated fiducial and electrode models from popup table rows."""

import numpy as np

from virda.models.electrode import Electrode, Electrodes
from virda.models.fiducial import Fiducial, Fiducials
from virda_gui_new.dto.viewer3d import LabeledPoints

type FiducialRows = list[tuple[str, np.ndarray]]
"""One ``(name, (3,) xyz)`` row per manual fiducial."""

type MeasurementTable = dict[str, dict[str, float]]
"""Per electrode id: fiducial name -> measured distance in mm."""

_EMPTY_NAME = "Fiducial name must be non-empty."


def labeled_points_from_rows(rows: FiducialRows) -> LabeledPoints:
    """Bundle ``(name, xyz)`` fiducial rows into labeled points.

    Rows come from the same table :func:`build_fiducials` consumes, so
    the points match what the popup shows; an empty row list yields an
    empty point set.
    """
    points = np.asarray([xyz for _name, xyz in rows], dtype=np.float64).reshape(-1, 3)
    labels = tuple(name for name, _xyz in rows)
    return LabeledPoints(points, labels)


def build_fiducials(rows: FiducialRows) -> Fiducials:
    """Build one world-coordinate manual fiducial per row.

    The normalized non-empty name doubles as the id; coordinates must be
    ``(3,)`` float arrays and names unique (mismatched shapes or duplicate
    names raise ``ValueError``).
    """
    items = []
    for name, xyz in rows:
        fiducial_name = str(name).strip()
        if not fiducial_name:
            raise ValueError(_EMPTY_NAME)
        items.append(
            Fiducial(
                fiducial_id=fiducial_name,
                name=fiducial_name,
                coordinates=np.asarray(xyz, dtype=float),
                coordinate_system="world",
                definition_method="manual",
            )
        )
    return Fiducials(items)


def build_electrodes(table: MeasurementTable) -> Electrodes:
    """Build one electrode per table row; validation stays in the model.

    Electrode ids are stripped non-empty strings; duplicate ids raise
    ``ValueError``.
    """
    items = []
    for electrode_id, distances in table.items():
        eid = electrode_id.strip()
        if not eid:
            raise ValueError("Electrode id must be non-empty.")
        items.append(
            Electrode(
                electrode_id=eid,
                measured_distances={name.strip(): value for name, value in distances.items()},
            )
        )
    return Electrodes(items)


def blocked_reason(n_fiducials: int, n_measurements: int) -> str | None:
    """Return why live localization cannot run, or ``None`` when ready.

    At least three fiducials and at least one measurement row are needed.
    """
    if n_fiducials < 3:
        return "not enough fiducials (need 3)"
    if n_measurements < 1:
        return "no measurements"
    return None
