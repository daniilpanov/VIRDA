"""Coordinate-system models: the frames a point can be expressed in."""

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class CoordinateSystem(Enum):
    """The coordinate frames a point can be expressed in.

    ``WORLD`` is the canonical frame: scanner surface-RAS millimetres.
    ``VOXEL`` is the NIfTI voxel index space, ``HEAD`` the world
    translated to the head centre of mass and ``FIDUCIALS`` the
    anatomical frame built from NAS, LPA and RPA.
    """

    WORLD = "world"
    VOXEL = "voxel"
    HEAD = "head"
    FIDUCIALS = "fiducials"


@dataclass(frozen=True)
class Point:
    """A three-dimensional point together with the frame it is expressed in.

    The frame is mandatory: a bare coordinate triple is ambiguous, and the
    geometry services convert between frames. The units are millimetres
    unless the frame says otherwise (the voxel frame counts voxels).
    """

    x: float
    y: float
    z: float
    system: CoordinateSystem

    def __post_init__(self) -> None:
        for name in ("x", "y", "z"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} must be a number, got {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if not isinstance(self.system, CoordinateSystem):
            raise ValueError(f"system must be a CoordinateSystem, got {self.system!r}")

    def as_array(self) -> np.ndarray:
        """The coordinates as a ``(3,)`` float array."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)
