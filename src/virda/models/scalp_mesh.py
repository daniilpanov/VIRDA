from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScalpMesh:
    vertices: np.ndarray
    faces: np.ndarray
    face_adjacency: np.ndarray | None = None
    coordinate_system: str = "world"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"Vertices must be (N, 3) array, got shape {self.vertices.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"Faces must be (M, 3) array, got shape {self.faces.shape}")
        if self.face_adjacency is not None and (
            self.face_adjacency.ndim != 2 or self.face_adjacency.shape[1] != 2
        ):
            raise ValueError(
                f"Face adjacency must be (E, 2) array, got shape {self.face_adjacency.shape}"
            )
