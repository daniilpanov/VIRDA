"""Import fiducials from JSON into a :class:`Fiducials` model."""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from virda.models.coordsystem import Coordsystem
from virda.models.fiducial import Fiducial, Fiducials


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