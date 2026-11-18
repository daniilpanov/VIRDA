"""Export Stage 3 electrodes as CSV files (full, scalp and ese variants)."""

import csv
from pathlib import Path

import numpy as np

from virda.models.coordinates import CoordinateSystem
from virda.models.electrode import Electrode, Electrodes


def export_electrodes(path: str | Path, electrodes: Electrodes, system: CoordinateSystem) -> Path:
    """Write the full localization result to *path* as a CSV table.

    Every row carries the electrode id, the coordinate system the numbers
    are given in, its localized ESE body-center and scalp contact
    coordinates, the measured distances to the fiducials (one column per
    fiducial, lowercased), the residual error and the flag; non-localized
    electrodes leave the coordinates and the residual empty. ``system`` is
    mandatory so a CSV is never ambiguous about its numbers. Returns the
    written path.
    """
    fiducial_columns = _fiducial_columns(electrodes)
    fieldnames = [
        "electrode_id",
        "coordinate_system",
        "ese_x",
        "ese_y",
        "ese_z",
        "scalp_x",
        "scalp_y",
        "scalp_z",
        *fiducial_columns,
        "residual_mm",
        "flagged",
    ]
    rows = [
        {
            "electrode_id": electrode.electrode_id or "",
            "coordinate_system": system.value,
            "ese_x": _coordinate(electrode.ese_coords, 0),
            "ese_y": _coordinate(electrode.ese_coords, 1),
            "ese_z": _coordinate(electrode.ese_coords, 2),
            "scalp_x": _coordinate(electrode.scalp_coords, 0),
            "scalp_y": _coordinate(electrode.scalp_coords, 1),
            "scalp_z": _coordinate(electrode.scalp_coords, 2),
            **{
                column: _distance(electrode.measured_distances, column)
                for column in fiducial_columns
            },
            "residual_mm": _residual(electrode),
            "flagged": "yes" if electrode.flagged else "no",
        }
        for electrode in electrodes.items
    ]
    return _export_table(path, fieldnames, rows)


def export_electrodes_scalp(path: str | Path, electrodes: Electrodes) -> Path:
    """Write the scalp table: ``x/y/z`` are scalp contact points in world mm."""
    return _export_coords_table(path, electrodes, "scalp_coords")


def export_electrodes_ese(path: str | Path, electrodes: Electrodes) -> Path:
    """Write the ESE table: ``x/y/z`` are ESE body-center points in world mm."""
    return _export_coords_table(path, electrodes, "ese_coords")


def _export_coords_table(path: str | Path, electrodes: Electrodes, coords_attribute: str) -> Path:
    """Write the common coords-variant table for the given attribute.

    These point CSVs are project artifacts written from world-canonical
    results, so their ``coordinate_system`` column is always ``world``.
    """
    fieldnames = [
        "electrode_id",
        "coordinate_system",
        "x",
        "y",
        "z",
        "residual_mm",
        "flagged",
    ]
    rows = [
        {
            "electrode_id": electrode.electrode_id or "",
            "coordinate_system": CoordinateSystem.WORLD.value,
            "x": _coordinate(getattr(electrode, coords_attribute), 0),
            "y": _coordinate(getattr(electrode, coords_attribute), 1),
            "z": _coordinate(getattr(electrode, coords_attribute), 2),
            "residual_mm": _residual(electrode),
            "flagged": "yes" if electrode.flagged else "no",
        }
        for electrode in electrodes.items
    ]
    return _export_table(path, fieldnames, rows)


def _fiducial_columns(electrodes: Electrodes) -> list[str]:
    """Return one lowercased column per fiducial measured by any electrode."""
    columns: list[str] = []
    for electrode in electrodes.items:
        for fiducial in electrode.measured_distances:
            column = fiducial.lower()
            if column not in columns:
                columns.append(column)
    return columns


def _export_table(
    path: str | Path, fieldnames: list[str], rows: list[dict[str, str | float]]
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return target


def _coordinate(coords: np.ndarray | None, axis: int) -> str | float:
    if coords is None:
        return ""
    return float(np.asarray(coords, dtype=np.float64)[axis])


def _residual(electrode: Electrode) -> str | float:
    if electrode.residual_error is None:
        return ""
    return electrode.residual_error


def _distance(measured_distances: dict[str, float], column: str) -> str | float:
    fiducial = next((key for key in measured_distances if key.lower() == column), None)
    if fiducial is None:
        return ""
    return measured_distances[fiducial]
