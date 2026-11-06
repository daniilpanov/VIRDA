"""Import a Stage 3 measurements JSON into :class:`Electrodes` models.

Two callers live here:

* :func:`import_measurements` is the plain reader used by the GUI project
  loading; it returns only the :class:`Electrodes` set and ignores the
  file-level ``fiducial_weights`` (fiducial weights live on the
  :class:`Fiducials` model);
* :func:`load_measurements` is the legacy entry point used by the old
  pipeline runner; it also re-weights an optionally provided
  :class:`Fiducials` copy.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from virda.models.electrode import Electrode, Electrodes
from virda.models.fiducial import Fiducials


def import_measurements(path: str | Path) -> Electrodes:
    """Read a measurements JSON file into an :class:`Electrodes` model.

    Format::

        {
          "electrodes": [
            {"electrode_id": "Fz", "measured_distances": {"NAS": 120.5, "LPA": 131.2}}
          ],
          "fiducial_weights": {"NAS": 1.5}
        }

    ``electrode_id`` is optional; when missing or empty, sequential ids
    (``E001``, ``E002``, ...) are assigned in file order so that downstream
    consumers always see stable identifiers.
    """
    data = cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))
    return Electrodes(items=_parse_electrodes(data))


def load_measurements(
    path: Path,
    fiducials: Fiducials | None = None,
) -> tuple[Electrodes, Fiducials | None]:
    """Load a Stage 3 measurements JSON into an :class:`Electrodes` set.

    The document format is the same as for :func:`import_measurements`.
    ``fiducial_weights`` is optional; when present it overrides the weights of
    ``fiducials`` used by the localizer.

    Returns the electrodes together with the (possibly re-weighted)
    :class:`Fiducials` copy.
    """
    data = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    return Electrodes(items=_parse_electrodes(data)), _apply_weights(fiducials, data)


def _parse_electrodes(data: dict[str, Any]) -> list[Electrode]:
    """Build the electrode list from the parsed measurements document."""
    return [
        Electrode(
            electrode_id=item.get("electrode_id") or f"E{index + 1:03d}",
            measured_distances={
                str(fiducial_id): float(distance)
                for fiducial_id, distance in item["measured_distances"].items()
            },
        )
        for index, item in enumerate(data["electrodes"])
    ]


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
