"""Export a :class:`Fiducials` model to a JSON file."""

import json
from pathlib import Path

from virda.models.fiducial import Fiducials


def export_fiducials(path: str | Path, fiducials: Fiducials) -> Path:
    """Write *fiducials* to ``path`` as JSON; returns the written path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
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
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return target