from dataclasses import dataclass
from typing import Literal

import numpy as np

CoordinateSystem = Literal["world", "voxel"]
DefinitionMethod = Literal["auto", "manual", "imported"]


@dataclass(frozen=True)
class Fiducial:
    fiducial_id: str
    name: str
    coordinates: np.ndarray
    coordinate_system: CoordinateSystem
    definition_method: DefinitionMethod = "manual"

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
