from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np

FiducialCoordinateSystem = Literal["world", "voxel"]
DefinitionMethod = Literal["auto", "manual", "imported"]


class FiducialType(StrEnum):
    """The canonical anatomical fiducial slots.

    The values are the strings the project artifacts carry, so a stored
    ``"NAS"`` id and ``FiducialType.NAS`` are the same thing. Manual
    fiducials may carry any other name; these are the three slots the
    anatomical frame and the localization tables are built from.
    """

    NAS = "NAS"
    LPA = "LPA"
    RPA = "RPA"


#: The three canonical fiducial slots, in fixed row order.
SLOT_IDS: tuple[FiducialType, ...] = (
    FiducialType.NAS,
    FiducialType.LPA,
    FiducialType.RPA,
)

#: Lowercased label -> canonical slot. Checked case-insensitively.
_ALIASES: dict[str, FiducialType] = {
    "nas": FiducialType.NAS,
    "nasion": FiducialType.NAS,
    "nz": FiducialType.NAS,
    "lpa": FiducialType.LPA,
    "rpa": FiducialType.RPA,
}


def match_fiducial_name(name: str) -> FiducialType | None:
    """Return the canonical slot of a fiducial label, or ``None``.

    The project artifacts name the NAS point either way
    (``fiducial_id="NAS"`` with ``name="NASION"``), so the alias table
    accepts both spellings.
    """
    return _ALIASES.get(name.strip().lower())


@dataclass(frozen=True)
class Fiducial:
    fiducial_id: str
    name: str
    coordinates: np.ndarray
    coordinate_system: FiducialCoordinateSystem
    definition_method: DefinitionMethod = "manual"
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.coordinates.shape != (3,):
            raise ValueError(f"Coordinates must be (3,) array, got shape {self.coordinates.shape}")
        if self.coordinate_system not in {"world", "voxel"}:
            raise ValueError(
                f"Invalid coordinate_system: {self.coordinate_system!r}, "
                "expected 'world' or 'voxel'"
            )
        if self.definition_method not in {"auto", "manual", "imported"}:
            raise ValueError(
                f"Invalid definition_method: {self.definition_method!r}, "
                "expected 'auto', 'manual' or 'imported'"
            )
        if self.weight <= 0:
            raise ValueError(f"weight must be positive, got {self.weight}")


@dataclass(frozen=True)
class Fiducials:
    items: list[Fiducial]

    def __post_init__(self) -> None:
        ids = [fiducial.fiducial_id for fiducial in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Fiducial ids must be unique, got {ids}")

    @property
    def ids(self) -> list[str]:
        return [fiducial.fiducial_id for fiducial in self.items]

    def get(self, fiducial_id: str) -> Fiducial | None:
        for fiducial in self.items:
            if fiducial.fiducial_id == fiducial_id:
                return fiducial
        return None


@dataclass(frozen=True)
class ManualFiducials:
    fiducials: Fiducials


@dataclass(frozen=True)
class AutoDetectedFiducials:
    fiducials: Fiducials
