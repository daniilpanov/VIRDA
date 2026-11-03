"""Step: load Stage 3 measurements from a JSON file."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from virda.models.electrode import Electrode, Electrodes
from virda.models.fiducial import Fiducials


def load_measurements(
    path: Path,
    fiducials: Fiducials | None = None,
) -> tuple[Electrodes, Fiducials | None]:
    """Load a Stage 3 measurements JSON into an :class:`Electrodes` set.

    Format::

        {
          "electrodes": [
            {"electrode_id": "Fz", "measured_distances": {"NAS": 120.5, "LPA": 131.2}}
          ],
          "fiducial_weights": {"NAS": 1.5}
        }

    ``electrode_id`` is optional; when missing or empty, sequential ids
    (``E001``, ``E002``, ...) are assigned in file order so that downstream
    consumers always see stable identifiers.  ``fiducial_weights`` is
    optional; when present it overrides the weights of ``fiducials`` used by
    the localizer.

    Returns the electrodes together with the (possibly re-weighted)
    :class:`Fiducials` copy.
    """
    data = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
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
    return Electrodes(items=items), _apply_weights(fiducials, data)


def _apply_weights(
    fiducials: Fiducials | None,
    data: dict[str, Any],
) -> Fiducials | None:
    """Return ``fiducials`` with file-level weights applied (an untouched copy)."""
    raw = data.get("fiducial_weights")
    if not raw:
        return fiducials
    if fiducials is None:
        raise KeyError("'Fiducials' store is not initialized")
    return Fiducials(
        items=[
            replace(
                fiducial,
                weight=(
                    float(raw[fiducial.fiducial_id])
                    if fiducial.fiducial_id in raw
                    else fiducial.weight
                ),
            )
            for fiducial in fiducials.items
        ]
    )
