"""Artifact file import into the localization popup's table inputs.

Two import paths back the localization popup's "From file" buttons:

- fiducial files -- the project ``fiducials.json`` format or an MNE-style
  ``coordsystem.json`` -- are read through
  :func:`virda.io.importers.fiducials.import_fiducials` and matched onto
  the fixed NAS/LPA/RPA table slots case-insensitively (the shared alias
  table in :mod:`virda_gui_new.services.coordinates`); unknown fiducials
  are ignored. The imported coordinates are world millimetres (both
  writers produce them), which is the tables' canonical representation.
- measurements files -- the project ``measurements.json`` format -- are
  read through :func:`virda.io.importers.measurements.import_measurements`
  into an electrode -> fiducial -> distance table; fiducial-name columns
  are whatever the file carries, so no mapping is needed.

Both raise ``ValueError`` (with a user-facing message) on unreadable,
invalid or empty files; the popup surfaces it through its hint label.
Fiducial files carry world or voxel coordinates: voxel rows are
converted to world through the MRI affine, so only world rows ever
reach the tables.
"""

from pathlib import Path

import numpy as np

from virda.io.importers.fiducials import import_fiducials
from virda.io.importers.measurements import import_measurements
from virda_gui_new.services.coordinates import SLOT_IDS, match_fiducial_name
from virda_gui_new.services.localization import FiducialRows, MeasurementTable


def _world_coordinates(
    coordinates: np.ndarray, coordinate_system: str, affine: np.ndarray | None
) -> np.ndarray:
    """One fiducial's world-mm coordinates; voxel rows convert via affine."""
    if coordinate_system == "world":
        return np.asarray(coordinates, dtype=np.float64)
    if coordinate_system == "voxel":
        if affine is None:
            raise ValueError(
                "voxel-coordinate fiducials need the NIfTI volume (import it in step 0 first)"
            )
        homogeneous = np.asarray([*coordinates, 1.0], dtype=np.float64)
        return (affine @ homogeneous)[:3]
    raise ValueError(f"unsupported coordinate system: {coordinate_system!r}")


def import_fiducial_rows(path: Path, affine: np.ndarray | None) -> FiducialRows:
    """Read a fiducials/coordsystem JSON into fixed-slot ``(name, xyz)`` rows.

    Fiducial labels (and ids, as a fallback) are matched
    case-insensitively onto the NAS/LPA/RPA slots; the first match per
    slot wins and unmatched fiducials are ignored. Rows come back in the
    fixed slot order in world millimetres: world-coordinate rows pass
    through, voxel-coordinate rows convert through ``affine`` (``None``
    or a missing affine raises), any other coordinate system raises.
    No known fiducial at all raises ``ValueError``.
    """
    fiducials = import_fiducials(path)
    matched: dict[str, tuple[str, np.ndarray]] = {}
    for fiducial in fiducials.items:
        label = fiducial.name or fiducial.fiducial_id
        slot = match_fiducial_name(label) or match_fiducial_name(fiducial.fiducial_id)
        if slot is not None and slot not in matched:
            xyz = _world_coordinates(fiducial.coordinates, fiducial.coordinate_system, affine)
            matched[slot] = (label, xyz)
    rows = [matched[slot] for slot in SLOT_IDS if slot in matched]
    if not rows:
        raise ValueError("no NAS/LPA/RPA fiducials found in file")
    return rows


def import_measurement_table(path: Path) -> MeasurementTable:
    """Read a measurements JSON into an electrode -> fiducial -> distance table.

    The file format is the project ``measurements.json`` (electrode ids
    with their measured distances); the fiducial names are taken as the
    file carries them. An empty electrode list raises ``ValueError``.
    """
    electrodes = import_measurements(path)
    table: MeasurementTable = {
        electrode.electrode_id: {
            name: float(value) for name, value in electrode.measured_distances.items()
        }
        for electrode in electrodes.items
        if electrode.electrode_id
    }
    if not table:
        raise ValueError("no electrodes found in file")
    return table
