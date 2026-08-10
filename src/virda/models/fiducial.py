from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class Fiducial:
    fiducial_id: str
    name: str
    coordinates: np.ndarray
    coordinate_system: Literal["world", "voxel"]
    definition_method: Literal["auto", "manual", "imported"] = "manual"

    def __post_init__(self) -> None:
        if self.coordinates.shape != (3,):
            raise ValueError(f"Coordinates must be (3,) array, got shape {self.coordinates.shape}")
        if self.coordinate_system not in {"world", "voxel"}:
            raise ValueError(
                f"Invalid coordinate_system: {self.coordinate_system!r}, "
                "expected 'world' or 'voxel'"
            )
        if self.definition_method not in {"manual", "auto", "imported"}:
            raise ValueError(
                f"Invalid definition_method: {self.definition_method!r}, "
                "expected 'manual', 'auto' or 'imported'"
            )
