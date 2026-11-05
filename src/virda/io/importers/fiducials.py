"""Import fiducials from JSON into typed models.

Two callers live here:

* :func:`import_fiducials` reads a project fiducials JSON (or an MNE-style
  ``coordsystem.json``) into a :class:`Fiducials` model;
* :func:`load_manual_fiducials` is the legacy entry point used by the old
  pipeline runner; it wraps :func:`virda.io.fiducial_helpers.load_fiducials`.
"""

import json
from logging import Logger
from pathlib import Path
from typing import Any, cast

import numpy as np

from virda.io.fiducial_helpers import load_fiducials
from virda.models.coordsystem import Coordsystem
from virda.models.fiducial import Fiducial, Fiducials, ManualFiducials


def import_fiducials(path: str | Path) -> Fiducials:
    """Read a JSON file into a :class:`Fiducials` model.

    Accepts both the format written by :func:`virda.io.exporters.export_fiducials`
    and MNE-style ``coordsystem.json`` files (detected via
    :meth:`Coordsystem.is_coordsystem_dict`); the latter are converted via
    :meth:`Coordsystem.to_fiducials`.
    """
    data = cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))
    if Coordsystem.is_coordsystem_dict(data):
        coordsystem = Coordsystem.model_validate(data)
        return coordsystem.to_fiducials()
    items = [
        Fiducial(
            fiducial_id=item["fiducial_id"],
            name=item["name"],
            coordinates=np.asarray(item["coordinates"], dtype=np.float64),
            coordinate_system=item["coordinate_system"],
            definition_method=item["definition_method"],
            weight=float(item.get("weight", 1.0)),
        )
        for item in data["fiducials"]
    ]
    return Fiducials(items=items)


def load_manual_fiducials(path: Path, *, logger: Logger | None = None) -> ManualFiducials:
    """Load a :class:`ManualFiducials` from a fiducials JSON file."""
    return ManualFiducials(fiducials=load_fiducials(path, logger))
