"""Export an :class:`Electrodes` model to the Stage 3 measurements JSON."""

import json
from pathlib import Path

from virda.models.electrode import Electrodes


def export_measurements(path: str | Path, electrodes: Electrodes) -> Path:
    """Write *electrodes* to *path* as a Stage 3 measurements JSON file.

    The document is exactly what
    :func:`virda.io.importers.measurements.import_measurements` reads
    back: one entry per electrode with its id and measured distances.
    ``fiducial_weights`` is never written (weights live on the
    :class:`~virda.models.fiducial.Fiducials` model). Electrodes without
    an id are written with an empty ``electrode_id``; the importer then
    assigns sequential ``E001``-style ids, so the round trip of id-less
    electrodes is not name-stable.  Returns the written path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "electrodes": [
            {
                "electrode_id": electrode.electrode_id or "",
                "measured_distances": dict(electrode.measured_distances),
            }
            for electrode in electrodes.items
        ]
    }
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return target
