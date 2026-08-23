"""Helpers for persisting and restoring the :class:`Fiducials` model."""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from virda.models.fiducial import Fiducial, Fiducials


def save_fiducials(path: Path, fiducials: Fiducials) -> None:
    """Write a :class:`Fiducials` model to ``path`` as JSON."""
    data = {
        "fiducials": [
            {
                "fiducial_id": fiducial.fiducial_id,
                "name": fiducial.name,
                "coordinates": fiducial.coordinates.tolist(),
                "coordinate_system": fiducial.coordinate_system,
                "definition_method": fiducial.definition_method,
                "weight": fiducial.weight,
            }
            for fiducial in fiducials.items
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_fiducials(path: Path) -> Fiducials:
    """Read a JSON file written by :func:`save_fiducials` into a :class:`Fiducials` model."""
    data = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
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
