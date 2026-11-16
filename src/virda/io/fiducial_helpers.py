"""Helpers for persisting and restoring the :class:`Fiducials` model."""

import json
from logging import Logger
from pathlib import Path
from typing import Any, cast

import numpy as np

from virda.models.coordinates import CoordinateSystem, Point
from virda.models.coordsystem import _SURFACE_RAS_FRAMES, Coordsystem
from virda.models.fiducial import Fiducial, Fiducials

#: The frames a stored fiducial may be expressed in.
_STORED_SYSTEMS = (CoordinateSystem.WORLD, CoordinateSystem.VOXEL)


def save_fiducials(path: Path, fiducials: Fiducials) -> None:
    """Write a :class:`Fiducials` model to ``path`` as JSON."""
    data = {
        "fiducials": [
            {
                "fiducial_id": fiducial.fiducial_id,
                "name": fiducial.name,
                "coordinates": fiducial.point.as_array().tolist(),
                "coordinate_system": fiducial.point.system.value,
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
            point=_point_from_record(item),
            definition_method=item["definition_method"],
            weight=float(item.get("weight", 1.0)),
        )
        for item in data["fiducials"]
    ]
    return Fiducials(items=items)


def _point_from_record(record: dict[str, Any]) -> Point:
    """The point of one stored fiducial record.

    Only the world and the voxel frame are stored; any other frame is
    rejected, and so is a coordinate list that is not a triple.
    """
    coordinates = np.asarray(record["coordinates"], dtype=np.float64)
    if coordinates.shape != (3,):
        raise ValueError(f"Coordinates must be a (3,) array, got shape {coordinates.shape}")
    try:
        system = CoordinateSystem(record["coordinate_system"])
    except ValueError as error:
        raise ValueError(
            f"Invalid coordinate_system: {record['coordinate_system']!r}, "
            "expected 'world' or 'voxel'"
        ) from error
    if system not in _STORED_SYSTEMS:
        raise ValueError(
            f"Invalid coordinate_system: {record['coordinate_system']!r}, "
            "expected 'world' or 'voxel'"
        )
    return Point(
        x=float(coordinates[0]),
        y=float(coordinates[1]),
        z=float(coordinates[2]),
        system=system,
    )
