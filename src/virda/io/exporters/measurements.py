"""Export Stage 3 measurements to a JSON file."""

import json
from pathlib import Path

from virda.models.electrode import Electrodes


def export_measurements(
    path: str | Path,
    electrodes: Electrodes,
    fiducial_weights: dict[str, float] | None = None,
) -> Path:
    """Write *electrodes* (and optional *fiducial_weights*) as measurements JSON.

    The output schema is the same document that
    :func:`virda.io.importers.import_measurements` reads back, so the two are
    round-trip compatible.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {
        "electrodes": [
            {
                "electrode_id": electrode.electrode_id,
                "measured_distances": dict(electrode.measured_distances),
            }
            for electrode in electrodes.items
        ]
    }
    if fiducial_weights:
        data["fiducial_weights"] = dict(fiducial_weights)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return target