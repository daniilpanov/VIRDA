from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScalpMesh:
    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"Vertices must be (N, 3) array, got shape {self.vertices.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"Faces must be (M, 3) array, got shape {self.faces.shape}")
