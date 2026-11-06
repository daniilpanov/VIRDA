"""Export the live localization preview table to a CSV file.

Qt-free: the widget in ``virda_gui.tabs.editors_tab`` delegates the actual
serialization here so it stays unit-testable without a display.
"""

import csv
from pathlib import Path

import numpy as np

from virda.models.electrode import Electrodes

FIDUCIAL_COLUMNS = ("LPA", "RPA", "NAS")


def export_localization_table(
    path: str | Path,
    electrodes: Electrodes,
    cras_offset: np.ndarray | None = None,
    *,
    fiducial_columns: tuple[str, str, str] = FIDUCIAL_COLUMNS,
) -> Path:
    """Write the localized coordinate table to *path* as CSV.

    Each row carries the electrode name, its localized coordinates in the
    world (scanner RAS) and head (centred) frames, the measured distances to
    the fiducials, the residual error and the flag.  The head frame is
    ``world - cras_offset``; when *cras_offset* is missing (no NIfTI loaded)
    the head columns are left empty.  Returns the written path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "world_x",
        "world_y",
        "world_z",
        "head_x",
        "head_y",
        "head_z",
        *[fiducial.lower() for fiducial in fiducial_columns],
        "residual_mm",
        "flagged",
    ]
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for electrode in electrodes.items:
            head_coords = (
                np.asarray(electrode.ese_coords, dtype=np.float64) - cras_offset
                if electrode.ese_coords is not None and cras_offset is not None
                else None
            )
            row: dict[str, str | int | float] = {
                "name": electrode.electrode_id or "",
                "world_x": _coordinate(electrode.ese_coords, 0),
                "world_y": _coordinate(electrode.ese_coords, 1),
                "world_z": _coordinate(electrode.ese_coords, 2),
                "head_x": _coordinate(head_coords, 0),
                "head_y": _coordinate(head_coords, 1),
                "head_z": _coordinate(head_coords, 2),
                "residual_mm": (
                    electrode.residual_error
                    if electrode.is_localized and electrode.residual_error is not None
                    else ""
                ),
                "flagged": "yes" if electrode.flagged else "no",
            }
            for fiducial in fiducial_columns:
                row[fiducial.lower()] = _distance(electrode.measured_distances, fiducial)
            writer.writerow(row)
    return target


def _coordinate(coords: np.ndarray | None, axis: int) -> float | str:
    if coords is None:
        return ""
    return float(coords[axis])


def _distance(distances: dict[str, float], fiducial_id: str) -> float | str:
    fiducial_lower = fiducial_id.lower()
    for key, value in distances.items():
        if key.lower() == fiducial_lower:
            return value
    return ""
