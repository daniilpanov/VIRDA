"""Helpers for persisting and restoring the :class:`Fiducials` model."""

import json
from logging import Logger
from pathlib import Path
from typing import Any, cast

import numpy as np

from virda.models.coordsystem import _SURFACE_RAS_FRAMES, Coordsystem
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


def load_fiducials(path: Path, logger: Logger | None = None) -> Fiducials:
    """Read a JSON file into a :class:`Fiducials` model.

    Accepts both the format written by :func:`save_fiducials` and MNE-style
    ``coordsystem.json`` files (detected via
    :meth:`Coordsystem.is_coordsystem_dict`); the latter are converted via
    :meth:`Coordsystem.to_fiducials`.
    """
    data = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if Coordsystem.is_coordsystem_dict(data):
        coordsystem = Coordsystem.model_validate(data)
        units = (coordsystem.coordinate_units or coordsystem.eeg_coordinate_units or "m").strip()
        scale = coordsystem.mri_unit_scale_mm()
        if units.lower() != "mm" and logger is not None:
            logger.debug("coordsystem units %r converted to mm (x%s)", units, scale)
        frame = coordsystem.coordinate_system or coordsystem.eeg_coordinate_system
        if frame and frame.strip().lower() not in _SURFACE_RAS_FRAMES and logger is not None:
            logger.debug(
                "coordsystem frame %r is not surface RAS; assuming FreeSurfer surface RAS anyway",
                frame,
            )
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
