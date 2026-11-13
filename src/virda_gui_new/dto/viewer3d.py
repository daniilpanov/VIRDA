"""Display-ready data structures for 3D scenes."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyvista as pv

VolumeSource = Any  # MRIVolume | (data, affine)
MeshSource = Any  # PolyData | (vertices, faces) | virda Mesh model


@dataclass(frozen=True)
class LabeledPoints:
    """An (N, 3) point set with an optional label per point.

    ``labels`` may be omitted for unnamed point sets.
    """

    points: np.ndarray
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Points must be (N, 3) arrays, got shape {points.shape}")

        labels = self.labels
        if labels is not None:
            labels = tuple(str(label) for label in labels)
            if len(labels) != len(points):
                raise ValueError(
                    f"Got {len(labels)} labels for {len(points)} points; counts must match"
                )

        object.__setattr__(self, "points", points)
        object.__setattr__(self, "labels", labels)

    def __len__(self) -> int:
        return len(self.points)

    @classmethod
    def from_arrays(cls, points: np.ndarray, labels: Sequence[str] | None = None) -> LabeledPoints:
        return cls(points, tuple(labels) if labels is not None else None)


PointSource = LabeledPoints  # ready display DTO


@dataclass(frozen=True)
class MeshWithPoints:
    """A mesh together with labeled points lying on it."""

    mesh: MeshSource
    points: PointSource | None = None


@dataclass
class MeshBundle:
    mesh: pv.PolyData | None
    points: LabeledPoints | None


@dataclass
class ViewerScene:
    """Fully resolved 3D scene, ready to be rendered."""

    volume: pv.ImageData | None = None
    mm_scene: bool = True
    fiducials: LabeledPoints | None = None
    scalp_mesh: pv.PolyData | None = None
    scalp_points: LabeledPoints | None = None
    ese_mesh: pv.PolyData | None = None
    ese_points: LabeledPoints | None = None
