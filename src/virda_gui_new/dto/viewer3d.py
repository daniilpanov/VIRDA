"""Plain display DTOs for the ``viewer3d`` widgets."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyvista as pv

VolumeSource = Any  # MRIVolume | (data, affine)
MeshSource = Any  # PolyData | (vertices, faces) | ScalpMesh | ESEMesh


@dataclass(frozen=True)
class LabeledPoints:
    """An (N, 3) point set with an optional label per point.

    The universal representation of "points drawn over a mesh or a volume":
    fiducials, electrode contact points, and so on.  ``labels`` may be omitted
    for unnamed point sets.  The widget consumes ready instances of this DTO;
    reading or converting source data into it belongs to the import layer
    (``virda`` core ops behind the ``virda_gui_new.api`` facades).
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
        """Build the DTO from already-loaded data (no I/O)."""
        return cls(points, tuple(labels) if labels is not None else None)


PointSource = LabeledPoints  # ready display DTO


@dataclass(frozen=True)
class MeshWithPoints:
    """A mesh together with labeled points lying on it (``points`` optional)."""

    mesh: MeshSource
    points: PointSource | None = None


@dataclass
class MeshBundle:
    mesh: pv.PolyData | None
    points: LabeledPoints | None


@dataclass
class ViewerScene:
    """Fully resolved advanced-viewer scene, ready to be rendered.

    Everything is already placed in the scene frame: in a world-millimeter
    scene (``mm_scene``) the volume keeps its spacing/origin from the affine,
    otherwise every array was moved into voxel index space.
    """

    volume: pv.ImageData | None = None
    mm_scene: bool = True
    fiducials: LabeledPoints | None = None
    scalp_mesh: pv.PolyData | None = None
    scalp_points: LabeledPoints | None = None
    ese_mesh: pv.PolyData | None = None
    ese_points: LabeledPoints | None = None
