"""Build validated fiducial and electrode models from popup table rows."""

import numpy as np

from virda.geometry import match_fiducial_name
from virda.models.coordinates import CoordinateSystem, Point
from virda.models.electrode import Electrode, Electrodes
from virda.models.fiducial import SLOT_IDS, Fiducial, Fiducials
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

    The normalized non-empty name doubles as the label; the id is the
    canonical slot the name matches (so the measurements and the
    fiducials agree on the id), and the name itself when it matches no
    slot. Coordinates must be ``(3,)`` float arrays and names unique
    (mismatched shapes or duplicate names raise ``ValueError``).
    """
    items = []
    for name, xyz in rows:
        fiducial_name = str(name).strip()
        if not fiducial_name:
            raise ValueError(_EMPTY_NAME)
        coords = np.asarray(xyz, dtype=float)
        if coords.shape != (3,):
            raise ValueError(f"Coordinates must be a (3,) array, got shape {coords.shape}")
        slot = match_fiducial_name(fiducial_name)
        items.append(
            Fiducial(
                fiducial_id=slot.value if slot is not None else fiducial_name,
                name=fiducial_name,
                point=Point(
                    x=float(coords[0]),
                    y=float(coords[1]),
                    z=float(coords[2]),
                    system=CoordinateSystem.WORLD,
                ),
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


def measurement_counts(table: MeasurementTable) -> tuple[int, int]:
    """The rows with all three measured distances and the incomplete ones.

    The distances are counted by their canonical slot: the nasion point
    is named ``NAS`` or ``NASION`` in the project artifacts, and both
    spellings are the same slot.
    """
    complete = 0
    incomplete = 0
    for distances in table.values():
        slots: set[str] = set()
        for name in distances:
            slot = canonical_fiducial_name(name)
            if slot is not None:
                slots.add(slot)
        if len(slots) >= len(SLOT_IDS):
            complete += 1
        elif distances:
            incomplete += 1
    return complete, incomplete


def is_complete_measurement(distances: dict[str, float]) -> bool:
    """Whether one electrode has a distance for each canonical fiducial."""
    slots = {canonical_fiducial_name(name) for name in distances}
    return all(slot in slots for slot in ("NAS", "LPA", "RPA"))


def complete_measurement_table(table: MeasurementTable) -> MeasurementTable:
    """Keep only named electrodes with all three canonical distances."""
    return {
        electrode_id.strip(): distances
        for electrode_id, distances in table.items()
        if electrode_id.strip() and is_complete_measurement(distances)
    }


def canonical_fiducial_name(name: str) -> str | None:
    """The canonical name of a fiducial, or ``None`` when it is unknown.

    The project artifacts name the NAS point either way (``NAS`` or
    ``NASION``), so both spellings are the same slot. The core knows the
    slots, so it holds the alias table.
    """
    slot = match_fiducial_name(name)
    return slot.value if slot is not None else None


def blocked_reason(n_fiducials: int, n_complete: int, n_incomplete: int) -> str | None:
    """Return why live localization cannot run, or ``None`` when ready.

    All the canonical fiducials and all the measured distances per
    electrode are needed: an electrode with fewer distances cannot be
    localized.
    """
    if n_fiducials < len(SLOT_IDS):
        return f"not enough fiducials (need {len(SLOT_IDS)})"
    if n_incomplete:
        return f"{n_incomplete} electrode(s) need all three measured distances"
    if n_complete < 1:
        return "no measurements"
    return None


def points_from_electrodes(electrodes: Electrodes) -> tuple[LabeledPoints, LabeledPoints]:
    """The scalp and the ESE point sets of the localized electrodes."""
    scalp: list[tuple[str, np.ndarray]] = []
    ese: list[tuple[str, np.ndarray]] = []
    for electrode in electrodes.items:
        electrode_id = electrode.electrode_id
        if not electrode_id:
            continue
        if electrode.scalp_coords is not None:
            scalp.append((electrode_id, np.asarray(electrode.scalp_coords)))
        if electrode.ese_coords is not None:
            ese.append((electrode_id, np.asarray(electrode.ese_coords)))
    return _point_set(scalp), _point_set(ese)


def _point_set(items: list[tuple[str, np.ndarray]]) -> LabeledPoints:
    """A labeled point set from the electrode ids and their coordinates."""
    if not items:
        return LabeledPoints(np.zeros((0, 3)), ())
    points = np.asarray([xyz for _electrode_id, xyz in items], dtype=np.float64)
    labels = tuple(electrode_id for electrode_id, _xyz in items)
    return LabeledPoints(points, labels)


def measurement_table_from_model(electrodes: Electrodes) -> MeasurementTable:
    """Build the electrode -> fiducial -> distance table from a model.

    The inverse of :func:`build_electrodes`: the stored measurements are
    shown in the popup grid again, so a reopened project continues with
    the table the user last edited.
    """
    table: MeasurementTable = {}
    for electrode in electrodes.items:
        electrode_id = electrode.electrode_id
        if not electrode_id:
            continue
        table[electrode_id] = dict(electrode.measured_distances)
    return table


def _point_of(points: LabeledPoints | None, electrode_id: str) -> np.ndarray | None:
    """The coordinates of one labeled point, or ``None`` when it is absent."""
    if points is None or points.labels is None:
        return None
    for label, xyz in zip(points.labels, points.points, strict=True):
        if label == electrode_id:
            return np.asarray(xyz, dtype=np.float64)
    return None


def electrodes_from_points(
    scalp_points: LabeledPoints | None,
    ese_points: LabeledPoints | None,
    measurements: Electrodes | None,
) -> Electrodes:
    """Build the electrode model of the seeded scalp and ESE point sets.

    The electrode ids come from the point labels, the coordinates from
    the two point sets and the measured distances from the measurements
    model. An electrode without a measured distance cannot be modelled
    (the model requires at least one distance), so it is left out.
    """
    ids: list[str] = []
    for points in (scalp_points, ese_points):
        if points is None or points.labels is None:
            continue
        ids.extend(label for label in points.labels if label not in ids)

    items: list[Electrode] = []
    for electrode_id in ids:
        distances: dict[str, float] = {}
        if measurements is not None:
            electrode = measurements.get(electrode_id)
            if electrode is not None:
                distances = dict(electrode.measured_distances)
        if not distances:
            continue
        items.append(
            Electrode(
                electrode_id,
                distances,
                ese_coords=_point_of(ese_points, electrode_id),
                scalp_coords=_point_of(scalp_points, electrode_id),
            )
        )
    return Electrodes(items)
