"""Import a Stage 3 measurements JSON into an :class:`Electrodes` model.

Format::

    {
      "electrodes": [
        {"electrode_id": "Fz", "measured_distances": {"NAS": 120.5, "LPA": 131.2}}
      ],
      "fiducial_weights": {"NAS": 1.5}
    }

``electrode_id`` is optional; when missing or empty, sequential ids
(``E001``, ``E002``, ...) are assigned in file order so that downstream
consumers always see stable identifiers.  ``fiducial_weights`` is ignored by
the importer (fiducial weights live on the :class:`Fiducials` model).
"""

import json
from pathlib import Path
from typing import Any, cast

from virda.models.electrode import Electrode, Electrodes


def import_measurements(path: str | Path) -> Electrodes:
    """Read a measurements JSON file into an :class:`Electrodes` model."""
    data = cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))
    items = [
        Electrode(
            electrode_id=item.get("electrode_id") or f"E{index + 1:03d}",
            measured_distances={
                str(fiducial_id): float(distance)
                for fiducial_id, distance in item["measured_distances"].items()
            },
        )
        for index, item in enumerate(data["electrodes"])
    ]
    return Electrodes(items=items)